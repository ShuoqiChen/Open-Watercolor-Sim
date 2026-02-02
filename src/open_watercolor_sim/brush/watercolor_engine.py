"""
This engine is a stylized, realtime watercolor-like simulator.

High-level approach:
- Eulerian grid fields for wetness, pigment, and velocity
- Semi-Lagrangian advection and simple diffusion for transport
- Separate wet pigment vs stained pigment layers with drying/absorption
- Edge-biased pigment deposition to create rim darkening
- Procedural noise fields for turbulence and paper texture
- Exponential attenuation style rendering to map pigment mass to color

Inspired by classic building blocks from:
- Stam 1999 (stable fluids, semi-Lagrangian advection)
- Curtis et al. 1997 (computer-generated watercolor)
- Deegan et al. 1997 (coffee-ring motivation for edge deposition)
- Bridson et al. 2007 (noise-driven flow stylization)
"""


import time
import random
from typing import Optional, Tuple, List

import numpy as np
import taichi as ti

from .configs import SimParams

# =============================================================================
# GLOBAL CONSTANTS & SIMULATION LIMITS
# =============================================================================
REFERENCE_RESOLUTION = 512.0  # Baseline resolution for parameter scaling
NOISE_TEXTURE_RES = 512       # Resolution of the pre-computed noise buffer

_GLOBAL_TAICHI_INITIALIZED = False

def _initialize_taichi_backend(arch: str, use_profiler: bool = False):
    """Initializes the Taichi runtime with the best available backend."""
    global _GLOBAL_TAICHI_INITIALIZED
    if _GLOBAL_TAICHI_INITIALIZED:
        return
    
    # Initialization settings
    init_kwargs = {
        "offline_cache": True,
        "kernel_profiler": use_profiler,
    }
    
    # Strategy for selecting backend
    ti_arch = ti.cpu
    if arch == "gpu":
        if ti.core.with_cuda():
            ti_arch = ti.cuda
        elif ti.core.with_metal():
            ti_arch = ti.metal
        elif ti.core.with_vulkan():
            ti_arch = ti.vulkan
        else:
            ti_arch = ti.cpu
    elif arch == "vulkan":
        ti_arch = ti.vulkan
    elif arch == "metal":
        ti_arch = ti.metal
    elif arch == "cuda":
        ti_arch = ti.cuda
    
    print(f"[WatercolorEngine] Initializing Taichi with backend: {ti_arch}")
    ti.init(arch=ti_arch, **init_kwargs)
    
    print(f"[WatercolorEngine] Taichi initialized. Backend: {ti.cfg.arch} | Profiler: {use_profiler}")
    _GLOBAL_TAICHI_INITIALIZED = True


@ti.func
def _clamp_01(x: ti.f32) -> ti.f32:
    """Clamps a scalar value to the range [0.0, 1.0]."""
    return ti.min(1.0, ti.max(0.0, x))


@ti.func
def _clamp_vec3_01(v: ti.template()) -> ti.template():
    """Clamps a 3-component vector to the range [0.0, 1.0]."""
    return ti.Vector([_clamp_01(v.x), _clamp_01(v.y), _clamp_01(v.z)])


@ti.func
def _hash21(p: ti.math.vec2) -> ti.f32:
    q = ti.math.fract(p * 0.1031)
    q += ti.math.dot(q, q.yx + 33.33)
    return ti.math.fract((q.x + q.y) * q.x)


@ti.data_oriented
class WatercolorEngine:
    """Taichi watercolor-ish wet media simulation.

    Fields on grid:
    - W: wetness [0,1]
    - P: wet pigment RGB [0,1]
    - A: stained pigment (paper-bound) RGB [0,1]
    - V: velocity field (grid-aligned)
    - T: paper texture noise [0,1]

    Uses ping-pong buffers for W/P/V.
    """

    def __init__(self, res: int = 384, dt: float = 1.0 / 60.0, seed: int = 0, arch: str = "cpu", preview_res: int = 256, use_profiler: bool = False):
        try:
            _initialize_taichi_backend(arch, use_profiler=use_profiler)
        except Exception as e:
            print(f"[WatercolorEngine] GPU Init failed: {e}. Falling back to CPU.")
            _initialize_taichi_backend("cpu", use_profiler=use_profiler)

        self.res = int(res)
        self.dt = float(dt)
        self.seed = int(seed)
        self.preview_res = int(preview_res)
        
        self.render_every = 2
        self.timing_mode = False
        self._frame_count = 0
        self._last_img = None
        
        self.activity_threshold_w = 1e-4
        self.activity_threshold_p = 1e-4

        self.res_scale = self.res / REFERENCE_RESOLUTION

        # Ping-pong index: 0/1
        self._ping = ti.field(dtype=ti.i32, shape=())

        # User properties
        self.p = SimParams()
        self._set_params_fields()

        # Simulation fields (ping-pong in leading dimension)
        self.W = ti.field(dtype=ti.f32, shape=(2, self.res, self.res))
        self.P = ti.Vector.field(4, dtype=ti.f32, shape=(2, self.res, self.res)) # RGBA (A is mass)
        self.V = ti.Vector.field(2, dtype=ti.f32, shape=(2, self.res, self.res))
        self.A = ti.Vector.field(4, dtype=ti.f32, shape=(self.res, self.res)) # RGBA (A is mass)
        self.T = ti.field(dtype=ti.f32, shape=(self.res, self.res))
        self.Age = ti.field(dtype=ti.f32, shape=(self.res, self.res)) # Per-pixel stroke timer

        self._mask = ti.field(dtype=ti.f32, shape=(self.res, self.res))
        self._mask_w = ti.field(dtype=ti.f32, shape=(self.res, self.res))
        self._mask2 = ti.field(dtype=ti.f32, shape=(self.res, self.res))
        self._img = ti.Vector.field(3, dtype=ti.f32, shape=(self.res, self.res))
        self._img_u8 = ti.field(dtype=ti.u8, shape=(self.res, self.res, 3))
        
        self._img_preview_u8 = ti.field(dtype=ti.u8, shape=(self.preview_res, self.preview_res, 3))
        
        self._time = ti.field(dtype=ti.f32, shape=())
        self._stamp_debug = ti.field(dtype=ti.i32, shape=())

        # Pre-calculated Noise Buffer for optimizations
        self._noise_res = NOISE_TEXTURE_RES
        self._noise_tex = ti.field(dtype=ti.f32, shape=(self._noise_res, self._noise_res))
        self.VNoise = ti.Vector.field(2, dtype=ti.f32, shape=(self._noise_res, self._noise_res))

        self._ping[None] = 0
        self._time[None] = 0.0

        self._build_paper_texture(2.0)
        self._init_noise_tex()
        self.reset()
        self.warmup()

    @ti.kernel
    def _init_noise_tex(self):
        for i, j in self._noise_tex:
            n = _hash21(ti.Vector([ti.cast(i, ti.f32), ti.cast(j, ti.f32)]))
            self._noise_tex[i, j] = n
            # Precompute unit vectors for turbulence to avoid sin/cos in sim loop
            ang = 6.2831853 * n
            self.VNoise[i, j] = ti.Vector([ti.cos(ang), ti.sin(ang)])

    @ti.func
    def _sample_noise(self, i: ti.i32, j: ti.i32, offset: ti.f32) -> ti.f32:
        # Fast sampling from pre-computed noise with tiling
        ii = (i + ti.cast(offset * 123.4, ti.i32)) % self._noise_res
        jj = (j + ti.cast(offset * 567.8, ti.i32)) % self._noise_res
        return self._noise_tex[ii, jj]

    def _set_params_fields(self):
        self._brush_radius = ti.field(dtype=ti.f32, shape=())
        self._pigment_load = ti.field(dtype=ti.f32, shape=())
        self._water_load = ti.field(dtype=ti.f32, shape=())
        self._color = ti.Vector.field(3, dtype=ti.f32, shape=())

        self._water_diffusion = ti.field(dtype=ti.f32, shape=())
        self._pigment_diffusion = ti.field(dtype=ti.f32, shape=())
        self._gravity_strength = ti.field(dtype=ti.f32, shape=())
        self._drip_threshold = ti.field(dtype=ti.f32, shape=())
        self._drip_rate = ti.field(dtype=ti.f32, shape=())
        self._lateral_turbulence = ti.field(dtype=ti.f32, shape=())
        self._flow_advection = ti.field(dtype=ti.f32, shape=())
        self._velocity_damping = ti.field(dtype=ti.f32, shape=())
        self._k_pressure = ti.field(dtype=ti.f32, shape=())
        self._v_max = ti.field(dtype=ti.f32, shape=())

        self._drying_rate = ti.field(dtype=ti.f32, shape=())
        self._absorption_rate = ti.field(dtype=ti.f32, shape=())
        self._pigment_settle = ti.field(dtype=ti.f32, shape=())
        self._granulation_strength = ti.field(dtype=ti.f32, shape=())
        self._edge_darkening = ti.field(dtype=ti.f32, shape=())
        self._wet_darken = ti.field(dtype=ti.f32, shape=())
        self._max_wet_pigment = ti.field(dtype=ti.f32, shape=())
        self._max_stain_pigment = ti.field(dtype=ti.f32, shape=())
        self._pigment_absorb_floor = ti.field(dtype=ti.f32, shape=())
        self._pigment_neutral_density = ti.field(dtype=ti.f32, shape=())
        self._stroke_life = ti.field(dtype=ti.f32, shape=())
        self._fade_rate = ti.field(dtype=ti.f32, shape=())

    def set_params(self, params: SimParams):
        self.p = params
        self._upload_params()

    def update_params(self, **kwargs):
        prev_scale = float(self.p.paper_texture_scale)
        for k, v in kwargs.items():
            if hasattr(self.p, k):
                setattr(self.p, k, v)
        self._upload_params()
        if abs(float(self.p.paper_texture_scale) - prev_scale) > 1e-6:
            self._build_paper_texture(self.p.paper_texture_scale)

    def _upload_params(self):
        """Synchronizes Python-side parameters to Taichi-side fields with resolution scaling."""
        self._brush_radius[None] = float(self.p.brush_radius)
        self._pigment_load[None] = float(self.p.pigment_load)
        self._water_load[None] = float(self.p.water_release)
        self._color[None] = ti.Vector([float(self.p.color_rgb[0]), float(self.p.color_rgb[1]), float(self.p.color_rgb[2])])

        # Map simplified params to internal engine physics with resolution scaling
        # Velocity and Gravity scale linearly with res_scale.
        # Diffusion scales quadratically (res_scale^2) to maintain consistent look across resolutions.
        s = self.res_scale
        s2 = s * s
        
        self._water_diffusion[None] = float(self.p.diffusion) * float(self.p.water_diffusion_coeff) * s2
        self._pigment_diffusion[None] = float(self.p.diffusion) * float(self.p.pigment_diffusion_coeff) * s2
        self._gravity_strength[None] = float(self.p.gravity) * float(self.p.gravity_coeff) * s
        self._drying_rate[None] = float(self.p.canvas_evaporation) * float(self.p.drying_coeff)
        self._granulation_strength[None] = float(self.p.granulation) * 1.5 
        
        self._drip_threshold[None] = float(self.p.drip_threshold)
        self._drip_rate[None] = float(self.p.drip_rate_coeff) * s
        self._lateral_turbulence[None] = float(self.p.turbulence_coeff) * s
        self._flow_advection[None] = float(self.p.advection_coeff) * s
        self._velocity_damping[None] = float(self.p.velocity_damping)
        self._k_pressure[None] = float(self.p.pressure_coeff) * s
        self._v_max[None] = float(self.p.max_velocity_coeff) * s
        self._stroke_life[None] = float(self.p.stroke_life)
        self._absorption_rate[None] = float(self.p.absorption_coeff)
        self._pigment_settle[None] = float(self.p.settle_coeff)
        self._edge_darkening[None] = float(self.p.edge_darkening) * 0.5 
        self._wet_darken[None] = float(self.p.wet_darken_coeff)
        self._max_wet_pigment[None] = float(self.p.max_wet_pigment)
        self._max_stain_pigment[None] = float(self.p.max_stain_pigment)
        self._pigment_absorb_floor[None] = float(self.p.pigment_absorb_floor)
        self._pigment_neutral_density[None] = float(self.p.pigment_neutral_density)
        
        ft = float(self.p.fade_time)
        if ft > 0:
            self._fade_rate[None] = 4.0 / (ft + 1e-6)
        else:
            self._fade_rate[None] = 0.0

    def reset(self):
        self._upload_params()
        self._clear()

    @staticmethod
    def _subdivide_polygon_edges(p1, p2, depth, variance, vdiv=2.0):
        """Recursively subdivides a line segment with random jitter for organic shapes."""
        if depth < 0:
            return []
        mid = (p1 + p2) / 2.0
        nx = mid[0] + random.gauss(0, variance)
        ny = mid[1] + random.gauss(0, variance)
        new_pt = np.array([nx, ny])
        
        res = []
        res.extend(WatercolorEngine._subdivide_polygon_edges(p1, new_pt, depth - 1, random.uniform(0, variance / vdiv), vdiv))
        res.append(new_pt)
        res.extend(WatercolorEngine._subdivide_polygon_edges(new_pt, p2, depth - 1, random.uniform(0, variance / vdiv), vdiv))
        return res

    def _create_deformed_polygon(self, x, y, r, nsides=10, depth=5, variance=15.0):
        """Creates a circular polygon and deforms its edges for a natural look."""
        angles = np.linspace(0, 2*np.pi, nsides, endpoint=False)
        return [(x + np.cos(a)*r, y + np.sin(a)*r) for a in angles]

    def apply_mask(self, mask01: np.ndarray):
        """Applies a normalized numpy mask to the canvas."""
        if mask01 is None: return
        self._mask.from_numpy(mask01.astype(np.float32))
        self._mask_w.from_numpy(mask01.astype(np.float32))
        self._apply_pigment_and_water_mask(self.dt)

    @ti.kernel
    def _mask_diffuse_step(self, k: ti.f32):
        """Internal helper to diffuse a mask stencil."""
        for i, j in self._mask:
            m = self._mask[i, j]
            ml = self._mask[ti.max(0, i - 1), j]
            mr = self._mask[ti.min(self.res - 1, i + 1), j]
            mu = self._mask[i, ti.max(0, j - 1)]
            md = self._mask[i, ti.min(self.res - 1, j + 1)]
            m_diag = (
                self._mask[ti.max(0, i-1), ti.max(0, j-1)] +
                self._mask[ti.min(self.res-1, i+1), ti.max(0, j-1)] +
                self._mask[ti.max(0, i-1), ti.min(self.res-1, j+1)] +
                self._mask[ti.min(self.res-1, i+1), ti.min(self.res-1, j+1)]
            )
            lap = (ml + mr + mu + md) * 0.2 + m_diag * 0.05 - m * 1.0
            self._mask2[i, j] = _clamp_01(m + k * lap)
        for i, j in self._mask:
            self._mask[i, j] = self._mask2[i, j]

    def debug_stamp_center(self, radius: int = 18):
        """Utility to place a single round stamp in the center of the canvas."""
        r = int(max(1, radius))
        cx = int(self.res // 2)
        cy = int(self.res // 2)
        self._render_circular_brush_stamp(cx, cy, r, 0.5, 0)

    @ti.kernel
    def _render_circular_brush_stamp(self, cx: ti.i32, cy: ti.i32, r: ti.i32, dryness: ti.f32, brush_id: ti.i32):
        """Renders/stamps a circular brush shape onto the simulation fields."""
        ping = self._ping[None]
        col = self._color[None]
        life = self._stroke_life[None]
        t = self._time[None]

        # EXTREME MAPPING - Improved to avoid hollow centers
        # We increase base pigment so wet brushes aren't too faint
        water_to_add = (1.0 - dryness) * 4.0 * self._water_load[None]
        pigment_to_add = (0.4 + dryness * 0.8) * self._pigment_load[None]
        
        # Bounded iteration
        x_start, x_end = ti.max(0, cx - r), ti.min(self.res, cx + r + 1)
        y_start, y_end = ti.max(0, cy - r), ti.min(self.res, cy + r + 1)
        
        for i, j in ti.ndrange((x_start, x_end), (y_start, y_end)):
            dx = ti.cast(i - cx, ti.f32)
            dy = ti.cast(j - cy, ti.f32)
            d = ti.sqrt(dx * dx + dy * dy)
            f_r = ti.cast(r, ti.f32)

            mask = 0.0
            if brush_id == 0: # SOFT ROUND
                if d <= f_r:
                    # Use a softer falloff (linear instead of squared) for fuller centers
                    mask = ti.max(0.0, 1.0 - d / (f_r + 1e-6))
            elif brush_id == 1: # SPONGE
                if d <= f_r:
                    # Multi-scale noise for sponge texture
                    uv = ti.Vector([ti.cast(i, ti.f32)*0.3, ti.cast(j, ti.f32)*0.3])
                    n1 = _hash21(uv + t)
                    n2 = _hash21(uv * 0.5 + t * 0.7)
                    n3 = _hash21(uv * 2.0 + t * 1.3)
                    sponge_mask = (n1 * 0.5 + n2 * 0.3 + n3 * 0.2)
                    if sponge_mask > 0.5: 
                        mask = (1.0 - d/f_r) * sponge_mask * 1.5

            if mask > 0.0:
                self.Age[i, j] = dryness * (life + 1.0) 
                a_mask = _clamp_01(mask)
                
                old_w = self.W[ping, i, j]
                self.W[ping, i, j] = _clamp_01(old_w + water_to_add * a_mask)
                
                old_p = self.P[ping, i, j]
                new_m = pigment_to_add * a_mask
                
                total_m_raw = old_p.w + new_m
                total_m = ti.min(total_m_raw, MAX_WET_PIGMENT_MASS)
                
                kept = total_m / (total_m_raw + 1e-6)
                new_m = new_m * kept
                
                new_amt = old_p.xyz + col * new_m
                self.P[ping, i, j] = ti.Vector([new_amt.x, new_amt.y, new_amt.z, total_m])

    def step(self, steps: int = 1):
        """Advances the simulation by the specified number of time steps."""
        for _ in range(int(steps)):
            t0 = time.perf_counter() if self.timing_mode else 0
            
            self._time[None] += self.dt
            
            # Phase 1: Physical simulation (drying, settling, gravity, velocity)
            self._apply_fluid_physics_step(self.dt)
            
            t1 = time.perf_counter() if self.timing_mode else 0
            
            # Phase 2: Flow simulation (advection and diffusion)
            self._apply_advection_diffusion_step(self.dt)
            
            self._frame_count += 1
            
            if self.timing_mode and self._frame_count % 30 == 0:
                ti.sync()
                t2 = time.perf_counter()
                print(f"[Watercolor] Sim Step: {(t1-t0)*1000:4.1f}ms (phys) + {(t2-t1)*1000:4.1f}ms (advect)")

    def clear(self):
        """Clears all simulation fields."""
        self._clear()

    def set_color(self, r, g, b):
        """Sets the current brush color."""
        self._color[None] = ti.Vector([r, g, b])

    def paint_brush(self, x, y, r, brush_id=0, dryness=0.5, **kwargs):
        """Applies a circular brush stroke at (x, y)."""
        self._render_circular_brush_stamp(int(x), int(y), int(r), float(dryness), int(brush_id))

    def paint(self, x, y, r, dryness=0.5):
        """Applies a default round brush stroke at (x, y)."""
        self._render_circular_brush_stamp(int(x), int(y), int(r), float(dryness), 0)

    def paint_mask(self, pigment_mask, water_mask=None):
        """Applies an arbitrary mask to provide irregular pigment/water deposits."""
        self._mask.from_numpy(pigment_mask)
        if water_mask is not None:
            self._mask_w.from_numpy(water_mask)
        else:
            self._mask_w.from_numpy(pigment_mask)
        self._apply_pigment_and_water_mask(self.dt)

    def render(self, debug_wet: bool = False, full_res: bool = False) -> np.ndarray:
        """Composites and returns the current canvas as a numpy array."""
        if self._frame_count % self.render_every == 0 or self._last_img is None or full_res:
            t0 = time.perf_counter() if self.timing_mode else 0
            
            # Always update the full-resolution image field for the Taichi GUI (canvas.set_image)
            self._draw_full_canvas(int(debug_wet))
            
            if not full_res:
                # Update low-res preview if full resolution isn't requested for the CPU return
                self._draw_preview_canvas(int(debug_wet))
            
            if self.timing_mode: ti.sync()
            t1 = time.perf_counter() if self.timing_mode else 0
            
            # Return the appropriate resolution to the caller
            if full_res:
                self._last_img = self._img_u8.to_numpy()
            else:
                self._last_img = self._img_preview_u8.to_numpy()
            
            if self.timing_mode and self._frame_count % 30 == 0:
                t2 = time.perf_counter()
                print(f"[Watercolor] Render: {(t1-t0)*1000:4.1f}ms (kernel) + {(t2-t1)*1000:4.1f}ms (transfer)")
                
        return self._last_img

    def warmup(self):
        """Trigger JIT compilation of all kernels by running a small dummy simulation."""
        self.clear()
        self._build_paper_texture(2.0)
        self._init_noise_tex()
        self._render_circular_brush_stamp(self.res//2, self.res//2, 10, 0.5, 0)
        self._apply_fluid_physics_step(self.dt)
        self._ping[None] = 1 - self._ping[None]
        self._apply_advection_diffusion_step(self.dt)
        self._ping[None] = 1 - self._ping[None]
        self._draw_preview_canvas(0)
        self._draw_full_canvas(0)
        self.clear()
        ti.sync()
        print(f"[WatercolorEngine] Warmup complete.")

    def test_integrity(self):
        """Verifies simulation state for stability (NaN checks)."""
        self.clear()
        self._render_circular_brush_stamp(self.res//2, self.res//2, 10, 0.5, 0)
        for i in range(10):
            self.step(1)
        
        w = self.W.to_numpy()
        if np.any(np.isnan(w)):
            print("[WatercolorEngine] INTEGRITY ERROR: NaN detected in wetness field!")
        if np.any((w < -1e-5) | (w > 1.0 + 1e-5)):
            max_w = w.max()
            min_w = w.min()
            print(f"[WatercolorEngine] INTEGRITY ERROR: Wetness out of range: [{min_w}, {max_w}]")
        else:
            print("[WatercolorEngine] Integrity test passed.")

    # ===============================
    # Taichi kernels
    # ===============================

    @ti.kernel
    def _clear(self):
        self._ping[None] = 0
        self._time[None] = 0.0
        for i, j in self.A:
            self.A[i, j] = ti.Vector([0.0, 0.0, 0.0, 0.0])
            self.Age[i, j] = 0.0
        for b, i, j in self.W:
            self.W[b, i, j] = 0.0
            self.P[b, i, j] = ti.Vector([0.0, 0.0, 0.0, 0.0])
            self.V[b, i, j] = ti.Vector([0.0, 0.0])
        for i, j in self._mask:
            self._mask[i, j] = 0.0
            self._mask_w[i, j] = 0.0
            self._mask2[i, j] = 0.0
        for i, j in self._img:
            self._img[i, j] = ti.Vector([1.0, 1.0, 1.0])
        for i, j, k in self._img_u8:
            self._img_u8[i, j, k] = ti.cast(255, ti.u8)

    @ti.kernel
    def _mask_diffuse(self, dt: ti.f32, k: ti.f32):
        # A cheap diffusion step to widen the sketch mask into a soft brush stamp.
        # Writes into _mask2 then swaps back into _mask.
        for i, j in self._mask:
            m = self._mask[i, j]
            ml = self._mask[ti.max(0, i - 1), j]
            mr = self._mask[ti.min(self.res - 1, i + 1), j]
            mu = self._mask[i, ti.max(0, j - 1)]
            md = self._mask[i, ti.min(self.res - 1, j + 1)]
            lap = (ml + mr + mu + md - 4.0 * m)
            self._mask2[i, j] = _clamp_01(m + k * lap)

        for i, j in self._mask:
            self._mask[i, j] = self._mask2[i, j]

    @ti.kernel
    def _build_paper_texture(self, scale: ti.f32):
        """Generates a procedural multi-octave noise texture to simulate paper roughness."""
        for i, j in self.T:
            p = ti.Vector([ti.cast(i, ti.f32), ti.cast(j, ti.f32)])
            
            # Multi-octave noise for organic paper feel
            n0 = _hash21(p * (0.015 * scale) + 7.1)
            n1 = _hash21(p * (0.050 * scale) + 19.7)
            n2 = _hash21(p * (0.120 * scale) + 41.3)
            
            n = 0.50 * n0 + 0.35 * n1 + 0.15 * n2
            
            # Contrast mapping based on scale to maintain texture visibility
            roughness_contrast = 1.0 + 0.1 * scale + (1.0 / (scale + 0.1))
            
            n = (n - 0.5) * roughness_contrast + 0.5
            self.T[i, j] = _clamp_01(n)

    @ti.func
    def _sample_scalar(self, f: ti.template(), x: ti.f32, y: ti.f32, b: ti.i32) -> ti.f32:
        # Bilinear sampling in grid coords
        x = ti.max(0.0, ti.min(ti.cast(self.res - 1, ti.f32), x))
        y = ti.max(0.0, ti.min(ti.cast(self.res - 1, ti.f32), y))
        x0 = ti.cast(ti.floor(x), ti.i32)
        y0 = ti.cast(ti.floor(y), ti.i32)
        x1 = ti.min(self.res - 1, x0 + 1)
        y1 = ti.min(self.res - 1, y0 + 1)
        tx = x - ti.cast(x0, ti.f32)
        ty = y - ti.cast(y0, ti.f32)

        v00 = f[b, x0, y0]
        v10 = f[b, x1, y0]
        v01 = f[b, x0, y1]
        v11 = f[b, x1, y1]

        v0 = v00 * (1.0 - tx) + v10 * tx
        v1 = v01 * (1.0 - tx) + v11 * tx
        return v0 * (1.0 - ty) + v1 * ty

    @ti.func
    def _sample_vec3(self, f: ti.template(), x: ti.f32, y: ti.f32, b: ti.i32) -> ti.math.vec3:
        x = ti.max(0.0, ti.min(ti.cast(self.res - 1, ti.f32), x))
        y = ti.max(0.0, ti.min(ti.cast(self.res - 1, ti.f32), y))
        x0 = ti.cast(ti.floor(x), ti.i32)
        y0 = ti.cast(ti.floor(y), ti.i32)
        x1 = ti.min(self.res - 1, x0 + 1)
        y1 = ti.min(self.res - 1, y0 + 1)
        tx = x - ti.cast(x0, ti.f32)
        ty = y - ti.cast(y0, ti.f32)

        v00 = f[b, x0, y0]
        v10 = f[b, x1, y0]
        v01 = f[b, x0, y1]
        v11 = f[b, x1, y1]

        v0 = v00 * (1.0 - tx) + v10 * tx
        v1 = v01 * (1.0 - tx) + v11 * tx
        return v0 * (1.0 - ty) + v1 * ty

    @ti.func
    def _sample_vec4(self, f: ti.template(), x: ti.f32, y: ti.f32, b: ti.i32) -> ti.math.vec4:
        x = ti.max(0.0, ti.min(ti.cast(self.res - 1, ti.f32), x))
        y = ti.max(0.0, ti.min(ti.cast(self.res - 1, ti.f32), y))
        x0 = ti.cast(ti.floor(x), ti.i32)
        y0 = ti.cast(ti.floor(y), ti.i32)
        x1 = ti.min(self.res - 1, x0 + 1)
        y1 = ti.min(self.res - 1, y0 + 1)
        tx = x - ti.cast(x0, ti.f32)
        ty = y - ti.cast(y0, ti.f32)

        v00 = f[b, x0, y0]
        v10 = f[b, x1, y0]
        v01 = f[b, x0, y1]
        v11 = f[b, x1, y1]

        v0 = v00 * (1.0 - tx) + v10 * tx
        v1 = v01 * (1.0 - tx) + v11 * tx
        return v0 * (1.0 - ty) + v1 * ty

    @ti.func
    def _sample_vec2(self, f: ti.template(), x: ti.f32, y: ti.f32, b: ti.i32) -> ti.math.vec2:
        x = ti.max(0.0, ti.min(ti.cast(self.res - 1, ti.f32), x))
        y = ti.max(0.0, ti.min(ti.cast(self.res - 1, ti.f32), y))
        x0 = ti.cast(ti.floor(x), ti.i32)
        y0 = ti.cast(ti.floor(y), ti.i32)
        x1 = ti.min(self.res - 1, x0 + 1)
        y1 = ti.min(self.res - 1, y0 + 1)
        tx = x - ti.cast(x0, ti.f32)
        ty = y - ti.cast(y0, ti.f32)

        v00 = f[b, x0, y0]
        v10 = f[b, x1, y0]
        v01 = f[b, x0, y1]
        v11 = f[b, x1, y1]

        v0 = v00 * (1.0 - tx) + v10 * tx
        v1 = v01 * (1.0 - tx) + v11 * tx
        return v0 * (1.0 - ty) + v1 * ty

    @ti.kernel
    def _apply_pigment_and_water_mask(self, dt: ti.f32):
        """Applies a custom mask (usually fractal/organic) to the pigment and water fields."""
        ping = self._ping[None]
        water_add = self._water_load[None] * 0.8
        pig_add = self._pigment_load[None] * 0.08
        col = self._color[None]

        for i, j in self._mask:
            m_p = _clamp_01(self._mask[i, j])
            m_w = _clamp_01(self._mask_w[i, j])
            
            if m_p > 0.0 or m_w > 0.0:
                self.Age[i, j] = 0.0
                
                old_p = self.P[ping, i, j]

                bonus_w = 0.35 * _clamp_01(self.W[ping, i, j] + old_p.w * 2.5)
                w = _clamp_01(self.W[ping, i, j] + (water_add + bonus_w) * m_w)
                
                hn = _hash21(ti.Vector([ti.cast(i, ti.f32), ti.cast(j, ti.f32)]) * 2.3 + 91.7)
                hole = 1.0
                if hn > 0.94: 
                    hole = 0.0
                
                n = _hash21(ti.Vector([ti.cast(i, ti.f32), ti.cast(j, ti.f32)]) * 8.0 + 13.7)
                grain = 0.85 + 0.3 * (n - 0.5)
                m_p = _clamp_01(m_p * hole * grain)
                m_w = _clamp_01(m_w * (0.95 + 0.05 * hole))
                
                sat_stain = self.A[i, j].w / (MAX_STAIN_PIGMENT_MASS + 1e-6)
                sat_wet = old_p.w / (MAX_WET_PIGMENT_MASS + 1e-6)
                sat = ti.max(sat_stain, sat_wet)
                sat = ti.min(1.0, ti.max(0.0, sat))
                m_p = m_p * (1.0 - 0.98 * ti.pow(sat, 3.5))
                
                new_m = pig_add * m_p
                total_m_raw = old_p.w + new_m

                total_m = ti.min(total_m_raw, MAX_WET_PIGMENT_MASS)

                kept = total_m / (total_m_raw + 1e-6)
                new_m = new_m * kept
                total_m = old_p.w + new_m

                new_amt = old_p.xyz + col * new_m
                
                self.W[ping, i, j] = w
                self.P[ping, i, j] = ti.Vector([new_amt.x, new_amt.y, new_amt.z, total_m])

                self._mask[i, j] = 0.0
                self._mask_w[i, j] = 0.0

    @ti.kernel
    def _apply_fluid_physics_step(self, dt: ti.f32):
        """Simulation phase 1: Handles drying, settling, gravity, and velocity updates."""
        ping = self._ping[None]
        pong = 1 - ping

        life = self._stroke_life[None]
        dry = self._drying_rate[None]
        absorb = self._absorption_rate[None]
        settle = self._pigment_settle[None]
        g = self._gravity_strength[None]
        kpress = self._k_pressure[None]
        turb = self._lateral_turbulence[None]
        damp = self._velocity_damping[None]
        vmax = self._v_max[None]

        for i, j in self.A:
            w_prev = self.W[ping, i, j]
            p_prev = self.P[ping, i, j]

            if w_prev < 1e-5 and p_prev.w < 1e-5:
                # Early exit for inactive pixels
                self.W[pong, i, j] = 0.0
                self.P[pong, i, j] = ti.Vector([0.0, 0.0, 0.0, 0.0])
                self.V[pong, i, j] = ti.Vector([0.0, 0.0])
                continue

            # --- Settling & Drying ---
            self.Age[i, j] += dt
            tex = self.T[i, j]
            
            tex_impact = 0.2 + 1.2 * tex # Hardcoded feel multiplier
            sink = dt * (dry + absorb * tex_impact)
            w_new = ti.max(0.0, w_prev - sink)
            dw = ti.max(0.0, w_prev - w_new)
            
            # Gradients for pressure and edge detection
            wl = self.W[ping, ti.max(0, i-1), j]
            wr = self.W[ping, ti.min(self.res-1, i+1), j]
            wu = self.W[ping, i, ti.max(0, j-1)]
            wd = self.W[ping, i, ti.min(self.res-1, j+1)]
            gradw = ti.Vector([(wr - wl) * 0.5, (wd - wu) * 0.5])
            
            edge = ti.sqrt(gradw.x * gradw.x + gradw.y * gradw.y)
            
            # Organic jagged edge logic
            en = self._sample_noise(i, j, 8.1)
            edge_j = edge * (0.6 + 0.8 * en) 
            edge_cap = 1.2
            edge_factor = 2.0 + 6.0 * self._edge_darkening[None]
            
            bn1 = self._sample_noise(i, j, self._time[None] * 1.5)
            bn2 = self._sample_noise(i, j, -self._time[None] * 0.9)
            branch_n = (bn1 * bn1 * bn1 * ti.sqrt(bn1)) * (0.5 + 0.5 * bn2)
            
            gran = self._granulation_strength[None]
            branch_mod = 1.0 + 8.0 * branch_n * _clamp_01(edge_j * 16.0) * (0.1 + 3.9 * tex * gran)
            
            edge_term = 0.08 + edge_factor * (edge_j * (1.0 - edge_j / edge_cap) if edge_j < edge_cap else edge_cap / 4.0)
            s = _clamp_01(settle * dw * edge_term * branch_mod)
            
            old_a = self.A[i, j]
            stain_sat = _clamp_01(old_a.w / (MAX_STAIN_PIGMENT_MASS + 1e-6))
            
            one_minus_sat = 1.0 - stain_sat
            s = s * (one_minus_sat * one_minus_sat * ti.sqrt(one_minus_sat))
            
            settled_amt = p_prev.xyz * s
            settled_m = p_prev.w * s
            
            total_am_raw = old_a.w + settled_m
            total_am = ti.min(total_am_raw, MAX_STAIN_PIGMENT_MASS)
            kept_a = total_am / (total_am_raw + 1e-6)
            
            res_c = (old_a.xyz + settled_amt) * kept_a
            
            # Optional automatic fading
            fr = self._fade_rate[None]
            if fr > 0.0:
                fade_factor = ti.max(0.0, 1.0 - fr * dt)
                res_c *= fade_factor
                total_am *= fade_factor
            
            self.A[i, j] = ti.Vector([res_c.x, res_c.y, res_c.z, total_am])
            self.P[pong, i, j] = ti.Vector([p_prev.x - settled_amt.x, p_prev.y - settled_amt.y, p_prev.z - settled_amt.z, p_prev.w - settled_m])
            self.W[pong, i, j] = _clamp_01(w_new)

            # --- Velocity Field Update ---
            v = self.V[ping, i, j]
            v.y -= g * w_prev * dt
            v += (-kpress * gradw) * dt
            
            # Sample turbulence unit vectors
            off = ti.cast(self._time[None] * 60, ti.i32)
            nv_i = (ti.cast(i, ti.i32) + off) % self._noise_res
            nv_j = (ti.cast(j, ti.i32) + off * 3) % self._noise_res
            v_noise = self.VNoise[nv_i, nv_j]
            v += v_noise * (turb * w_prev * dt)
            
            v *= (1.0 - damp * dt)
            v_sq = v.x * v.x + v.y * v.y
            if v_sq > vmax * vmax:
                v *= (vmax / (ti.sqrt(v_sq) + 1e-6))
            self.V[pong, i, j] = v

    @ti.kernel
    def _apply_advection_diffusion_step(self, dt: ti.f32):
        """Simulation phase 2: Handles advection (flow) and diffusion of water and pigment."""
        ping = self._ping[None]
        pong = 1 - ping

        adv = self._flow_advection[None]
        wd_rate = self._water_diffusion[None]
        pd_rate = self._pigment_diffusion[None]

        for i, j in self.A:
            v = self.V[pong, i, j]
            
            # Semi-Lagrangian Advection
            back_x = ti.cast(i, ti.f32) - v.x * dt * adv
            back_y = ti.cast(j, ti.f32) - v.y * dt * adv
            
            w_adv = self._sample_scalar(self.W, back_x, back_y, pong)
            p_adv = self._sample_vec4(self.P, back_x, back_y, pong)
            v_adv = self._sample_vec2(self.V, back_x, back_y, pong)

            # Diffusion (using 5-point laplacian with edge preservation)
            w = w_adv
            wl = self.W[pong, ti.max(0, i-1), j]
            wr = self.W[pong, ti.min(self.res-1, i+1), j]
            wu = self.W[pong, i, ti.max(0, j-1)]
            wd = self.W[pong, i, ti.min(self.res-1, j + 1)]
            lap_w = (wl + wr + wu + wd - 4.0 * w)
            edge_preserve = w * (1.0 - w)
            w_diff = _clamp_01(w + wd_rate * dt * lap_w * (0.10 + 2.2 * edge_preserve))

            p = p_adv
            pl = self.P[pong, ti.max(0, i-1), j]
            pr = self.P[pong, ti.min(self.res-1, i+1), j]
            pu = self.P[pong, i, ti.max(0, j-1)]
            pd = self.P[pong, i, ti.min(self.res-1, j+1)]
            lap_p = (pl + pr + pu + pd - 4.0 * p)
            p_diff = p + (pd_rate * dt * _clamp_01(w_diff)) * lap_p
            
            self.W[ping, i, j] = _clamp_01(w_diff)
            self.P[ping, i, j] = p_diff
            self.V[ping, i, j] = v_adv 
    @ti.kernel
    def _draw_full_canvas(self, debug_wet: ti.i32):
        """Composites the fluid and stain layers into the final 8-bit image field."""
        ping = self._ping[None]

        for i, j in self._img:
            # Paper base
            paper = ti.Vector([1.0, 1.0, 1.0])

            # Wet pigment layer
            p = self.P[ping, i, j]
            wet_m = p.w
            wet_rgb = ti.Vector([0.0, 0.0, 0.0])
            if wet_m > 1e-8:
                wet_rgb = p.xyz / (wet_m + 1e-10)

            # Stain pigment layer
            a = self.A[i, j]
            stain_m = a.w
            stain_rgb = ti.Vector([0.0, 0.0, 0.0])
            if stain_m > 1e-8:
                stain_rgb = a.xyz / (stain_m + 1e-10)

            # Convert masses to opacities
            wet_alpha = _clamp_01(wet_m / (self._max_wet_pigment[None] + 1e-6))
            stain_alpha = _clamp_01(stain_m / (self._max_stain_pigment[None] + 1e-6))

            # Rendering strengths
            wet_strength = 0.85 
            stain_strength = 0.85 * (0.5 + 2.0 * self._edge_darkening[None])

            # Optical density compositing
            absorb_floor = _clamp_01(self._pigment_absorb_floor[None])
            neutral = _clamp_01(self._pigment_neutral_density[None])

            absorb_wet = ti.max(absorb_floor, _clamp_vec3_01(1.0 - wet_rgb))
            absorb_stain = ti.max(absorb_floor, _clamp_vec3_01(1.0 - stain_rgb))

            OD_stain = stain_strength * stain_alpha * (absorb_stain + neutral)
            OD_wet = wet_strength * wet_alpha * (absorb_wet + neutral)

            col = paper
            col = col * ti.exp(-OD_stain)
            col = col * ti.exp(-OD_wet)

            # Paper texture overlay
            tex = self.T[i, j]
            col = col * (0.94 + 0.06 * tex)

            # Wetening darkening effect
            col = col * (1.0 - self._wet_darken[None] * _clamp_01(self.W[ping, i, j]))

            if debug_wet != 0:
                w = self.W[ping, i, j]
                col = ti.Vector([w, w, w])

            col = ti.max(0.0, ti.min(1.0, col))
            self._img[i, j] = col
            self._img_u8[i, j, 0] = ti.cast(col.x * 255.0, ti.u8)
            self._img_u8[i, j, 1] = ti.cast(col.y * 255.0, ti.u8)
            self._img_u8[i, j, 2] = ti.cast(col.z * 255.0, ti.u8)

    @ti.kernel
    def _draw_preview_canvas(self, debug_wet: ti.i32):
        """Low-resolution optimized composition for real-time preview."""
        ping = self._ping[None]
        scale = ti.cast(self.res, ti.f32) / ti.cast(self.preview_res, ti.f32)

        for i, j in ti.ndrange(self.preview_res, self.preview_res):
            paper = ti.Vector([1.0, 1.0, 1.0])
            
            fx = (ti.cast(i, ti.f32) + 0.5) * scale
            fy = (ti.cast(j, ti.f32) + 0.5) * scale
            
            p = self._sample_vec4(self.P, fx, fy, ping)
            a = self._sample_vec4_2d(self.A, fx, fy)
            w = self._sample_scalar(self.W, fx, fy, ping)
            
            wet_m = p.w
            wet_rgb = p.xyz / (wet_m + 1e-10) if wet_m > 1e-8 else ti.Vector([0.0, 0.0, 0.0])
            
            stain_m = a.w
            stain_rgb = a.xyz / (stain_m + 1e-10) if stain_m > 1e-8 else ti.Vector([0.0, 0.0, 0.0])
            
            wet_alpha = _clamp_01(wet_m / (self._max_wet_pigment[None] + 1e-6))
            stain_alpha = _clamp_01(stain_m / (self._max_stain_pigment[None] + 1e-6))
            
            wet_strength = 0.85
            stain_strength = 0.85 * (0.5 + 2.0 * self._edge_darkening[None])

            absorb_floor = _clamp_01(self._pigment_absorb_floor[None])
            neutral = _clamp_01(self._pigment_neutral_density[None])

            absorb_wet = ti.max(absorb_floor, _clamp_vec3_01(1.0 - wet_rgb))
            absorb_stain = ti.max(absorb_floor, _clamp_vec3_01(1.0 - stain_rgb))

            OD_stain = stain_strength * stain_alpha * (absorb_stain + neutral)
            OD_wet = wet_strength * wet_alpha * (absorb_wet + neutral)

            col = paper
            col = col * ti.exp(-OD_stain)
            col = col * ti.exp(-OD_wet)
            col = col * (1.0 - self._wet_darken[None] * _clamp_01(w))
            
            if debug_wet != 0:
                col = ti.Vector([w, w, w])

            col = ti.max(0.0, ti.min(1.0, col))
            self._img_preview_u8[i, j, 0] = ti.cast(col.x * 255.0, ti.u8)
            self._img_preview_u8[i, j, 1] = ti.cast(col.y * 255.0, ti.u8)
            self._img_preview_u8[i, j, 2] = ti.cast(col.z * 255.0, ti.u8)

    @ti.func
    def _sample_vec4_2d(self, f: ti.template(), x: ti.f32, y: ti.f32) -> ti.math.vec4:
        x = ti.max(0.0, ti.min(ti.cast(self.res - 1, ti.f32), x))
        y = ti.max(0.0, ti.min(ti.cast(self.res - 1, ti.f32), y))
        x0 = ti.cast(ti.floor(x), ti.i32)
        y0 = ti.cast(ti.floor(y), ti.i32)
        x1 = ti.min(self.res - 1, x0 + 1)
        y1 = ti.min(self.res - 1, y0 + 1)
        tx = x - ti.cast(x0, ti.f32)
        ty = y - ti.cast(y0, ti.f32)

        v00 = f[x0, y0]
        v10 = f[x1, y0]
        v01 = f[x0, y1]
        v11 = f[x1, y1]

        v0 = v00 * (1.0 - tx) + v10 * tx
        v1 = v01 * (1.0 - tx) + v11 * tx
        return v0 * (1.0 - ty) + v1 * ty
