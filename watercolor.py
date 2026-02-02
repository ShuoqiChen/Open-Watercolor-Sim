import taichi as ti
import numpy as np
import time
from brush import WatercolorEngine

def hsv_to_rgb(h, s, v):
    h = h % 1.0
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i %= 6
    if i == 0: return (v, t, p)
    if i == 1: return (q, v, p)
    if i == 2: return (p, v, t)
    if i == 3: return (p, q, v)
    if i == 4: return (t, p, v)
    if i == 5: return (v, p, q)
    return (0, 0, 0)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Watercolor Simulator: High Quality GPU Mode")
    parser.add_argument("-r", "--res", type=int, default=1024, help="Simulation resolution (default: 1024)")
    parser.add_argument("-f", "--fps", type=int, default=60, help="Target FPS cap (default: 60)")
    parser.add_argument("-s", "--substeps", type=int, default=2, help="Max simulation substeps when painting (default: 2)")
    parser.add_argument("-d", "--decimate", type=int, default=1, help="Render decimation (default: 1)")
    args = parser.parse_args()

    RES = args.res
    arch = "gpu"
    
    print(f"\n[SightOfSound] Starting Watercolor (High Quality GPU Mode)")
    print(f" - Resolution: {RES}x{RES}")
    print(f" - Backend:    {arch.upper()}")
    print(f" - FPS Cap:    {args.fps}")
    print(f" - Substeps:   {args.substeps} (adaptive)")
    print(f"--------------------------------")

    # The engine now supports GPU/CPU selection properly
    engine = WatercolorEngine(res=RES, arch=arch)
    
    window = ti.ui.Window("Watercolor: GPU (Fluid Simulation)", (RES, RES))
    canvas = window.get_canvas()
    gui = window.get_gui()
    
    p_color = (0.4, 0.25, 0.15)
    brush_radius = 40.0 * (RES / 1024.0)
    brush_id = 0
    dryness = 0.75

    # Slider-Adjustable Parameters (initialized to engine defaults)
    p_pigment_load = 0.25 # 1.0
    p_water_release = 1.25
    p_diffusion = 0.125
    p_canvas_evaporation = 0.75
    p_gravity = 0.1 # 1.0
    p_fade_time = 30.0
    p_edge_darkening = 0.8
    p_paper_scale = 25.0
    p_granulation = 0.5 # 0.55
    
    last_paint_pos = None
    last_paint_time = 0
    paint_interval = 1.0 / 60.0
    stroke_dist_accum = 0.0
    
    # Kernel pacing stats
    total_stamps = 0
    last_stat_time = time.time()
    frame_idx = 0

    print("\n[Controls]")
    print(" - Mouse Left (LMB): Paint with bleeding/dripping water")
    print(" - Space: Clear Canvas")
    print(" - S: Save Screenshot")
    print(" - UI: Use the 'Controls' sidebar (top-left) for all sliders")
    print(" - Shortcuts: [ / ] for Size, D / F for Dryness")

    start_time = time.time()
    fps_limit = args.fps

    while window.running:
        frame_start = time.time()
        curr_time = frame_start - start_time
        stamps_this_frame = 0 # Throttling cap
        
        is_painting = window.is_pressed(ti.ui.LMB)

        # Handle events
        events = window.get_events(ti.ui.PRESS)
        for e in events:
            if e.key == ti.ui.SPACE:
                engine.clear()
            elif e.key == 'b':
                brush_id = (brush_id + 1) % 2
                print(f"Brush ID: {brush_id}")
            elif e.key == 'd':
                dryness = max(0.0, dryness - 0.1)
            elif e.key == 'f':
                dryness = min(1.0, dryness + 0.1)
            elif e.key == '[':
                brush_radius = max(5, brush_radius - 5)
            elif e.key == ']':
                brush_radius = min(200, brush_radius + 5)
            elif e.key == 's':
                # Force full render for screenshot
                engine.render(full_res=True)
                img = engine._img_u8.to_numpy()
                import PIL.Image
                PIL.Image.fromarray(img).save(f"render_{int(time.time())}.png")
                print("Saved screenshot.")
            elif e.key == ti.ui.ESCAPE:
                window.running = False

        if is_painting:
            # ti.ui.Window (GGUI) uses [0,1] with origin at BOTTOM-LEFT.
            mx, my = window.get_cursor_pos()
            
            # Hit-test for control panel: [0.05, 0.35] x [0.05, 0.95]
            if not (0.05 <= mx <= 0.35 and 0.05 <= my <= 0.95):
                px, py = mx * RES, my * RES
                
                if last_paint_pos is not None:
                    lx, ly = last_paint_pos
                    dx, dy = px - lx, py - ly
                    dist = (dx*dx + dy*dy)**0.5
                    
                    spacing = max(1.0, brush_radius * 0.8)
                    stroke_dist_accum += dist
                    
                    # 1. Throttling: distance-based and time-based
                    if dist > 1e-4: # Ignore absolute stillness
                        if stroke_dist_accum >= spacing or (curr_time - last_paint_time > paint_interval):
                            engine.set_color(*p_color)
                            
                            # Use a single stamp per event to keep stamps/sec <= 60
                            if stamps_this_frame < 1:
                                engine.paint_brush(px, py, brush_radius, brush_id=brush_id, dryness=dryness)
                                stamps_this_frame += 1
                                total_stamps += 1
                            
                            stroke_dist_accum = 0.0
                            last_paint_time = curr_time
                    
                    last_paint_pos = (px, py)
                else:
                    # Initial click stamp
                    engine.set_color(*p_color)
                    engine.paint_brush(px, py, brush_radius, brush_id=brush_id, dryness=dryness)
                    stamps_this_frame += 1
                    total_stamps += 1
                    last_paint_pos = (px, py)
                    last_paint_time = curr_time
            else:
                last_paint_pos = None # Reset when moving into UI
        else:
            last_paint_pos = None
            stroke_dist_accum = 0.0

        # UI Sidebar (Overlay)
        with gui.sub_window("Controls", 0.05, 0.05, 0.3, 0.9) as w:
            gui.text("Simulation Controls")
            if gui.button("Clear Canvas"): engine.clear()
            
            gui.text("Brush Settings")
            brush_id = gui.slider_int("Brush Type", brush_id, 0, 1)
            gui.text("0: Round | 1: Sponge")
            
            brush_radius = gui.slider_float("Brush Radius", brush_radius, 5, 200)
            dryness = gui.slider_float("Brush Dryness", dryness, 0.0, 1.0)
            p_color = gui.color_edit_3("Color Picker", p_color)
            
            gui.text("Physics & Rendering")
            p_pigment_load = gui.slider_float("Pigment Load", p_pigment_load, 0.02, 0.80)
            p_water_release = gui.slider_float("Water Release", p_water_release, 0.10, 3.00)
            p_diffusion = gui.slider_float("Diffusion", p_diffusion, 0.00, 0.50)
            p_canvas_evaporation = gui.slider_float("Canvas Drying", p_canvas_evaporation, 0.00, 1.00)
            p_gravity = gui.slider_float("Gravity", p_gravity, 0.00, 2.00)
            p_edge_darkening = gui.slider_float("Edge Darken", p_edge_darkening, 0.00, 2.00)
            
            if gui.button("Save Screenshot"):
                engine.render(full_res=True)
                img = engine._img_u8.to_numpy()
                import PIL.Image
                PIL.Image.fromarray(img).save(f"render_{int(time.time())}.png")

        # Sync GUI to Engine
        engine.update_params(
            pigment_load=p_pigment_load,
            water_release=p_water_release,
            diffusion=p_diffusion,
            canvas_evaporation=p_canvas_evaporation,
            gravity=p_gravity,
            fade_time=0.0, # Disable Clean Cycle (Fade)
            edge_darkening=p_edge_darkening,
            paper_texture_scale=p_paper_scale,
            granulation=p_granulation
        )

        # Performance stats monitor
        now = time.time()
        if now - last_stat_time > 2.0:
            fps_val = 1.0/elapsed if 'elapsed' in locals() and elapsed > 0 else 0
            print(f"[Stats] Stamps/sec: {total_stamps//2} | FPS: {fps_val:.1f} | Substeps: {current_substeps}")
            total_stamps = 0
            last_stat_time = now

        # Consistent Simulation Substeps (Fixed speed regardless of interaction)
        current_substeps = args.substeps
        for _ in range(current_substeps):
            engine.step()

        # Render directly from field with decimation
        if (frame_idx % args.decimate == 0):
            engine.render()
            canvas.set_image(engine._img)
        
        window.show()
        frame_idx += 1

        # Enforce FPS cap to prevent resource hogging
        elapsed = time.time() - frame_start
        if elapsed < 1.0 / fps_limit:
            time.sleep(1.0 / fps_limit - elapsed)

if __name__ == "__main__":
    main()
