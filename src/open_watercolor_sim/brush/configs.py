from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class SimParams:
    """User-adjustable parameters for the watercolor simulation."""
    
    # --- [NORMAL] Brush Properties ---
    brush_radius: float = field(default=40.0, metadata={"help": "The radius of the brush in pixels.", "category": "Normal", "min": 5.0, "max": 200.0})
    pigment_load: float = field(default=1.0, metadata={"help": "Amount of pigment deposited with each stroke.", "category": "Normal", "min": 0.0, "max": 2.0})
    water_release: float = field(default=0.4, metadata={"help": "Amount of water deposited with each stroke.", "category": "Normal", "min": 0.0, "max": 2.0})
    
    # --- [NORMAL] Physics Behavior ---
    diffusion: float = field(default=0.15, metadata={"help": "How fast pigment and water spread across the paper.", "category": "Normal", "min": 0.0, "max": 1.0})
    canvas_evaporation: float = field(default=0.2, metadata={"help": "Rate at which the canvas dries (water loss).", "category": "Normal", "min": 0.0, "max": 1.0})
    gravity: float = field(default=0.5, metadata={"help": "Downward force affecting wet paint flow.", "category": "Normal", "min": 0.0, "max": 5.0})
    granulation: float = field(default=0.55, metadata={"help": "Strength of pigment settling into paper grain.", "category": "Normal", "min": 0.0, "max": 1.0})
    
    # --- [NORMAL] Visuals ---
    color_rgb: Tuple[float, float, float] = field(default=(0.1, 0.2, 0.8), metadata={"help": "Current brush color (normalized RGB).", "category": "Normal"})
    edge_darkening: float = field(default=0.65, metadata={"help": "Strength of the wet-edge darkening effect.", "category": "Normal", "min": 0.0, "max": 2.0})
    fade_time: float = field(default=30.0, metadata={"help": "Seconds until canvas fully clears (0 to disable).", "category": "Normal", "min": 0.0, "max": 300.0})

    # --- [ADVANCED] Internal Media Physics ---
    max_wet_pigment: float = field(default=0.18, metadata={"help": "Max pigment mass suspended in water.", "category": "Advanced", "min": 0.01, "max": 1.0})
    max_stain_pigment: float = field(default=0.5, metadata={"help": "Max pigment mass absorbed by paper.", "category": "Advanced", "min": 0.1, "max": 2.0})
    water_diffusion_coeff: float = field(default=0.4, metadata={"help": "Base coefficient for water spread.", "category": "Advanced", "min": 0.0, "max": 2.0})
    pigment_diffusion_coeff: float = field(default=0.22, metadata={"help": "Base coefficient for pigment spread.", "category": "Advanced", "min": 0.0, "max": 2.0})
    gravity_coeff: float = field(default=8.0, metadata={"help": "Base coefficient for gravity force.", "category": "Advanced", "min": 0.0, "max": 20.0})
    drying_coeff: float = field(default=1.5, metadata={"help": "Base coefficient for drying speed.", "category": "Advanced", "min": 0.0, "max": 5.0})
    
    # --- [ADVANCED] Flow Dynamics ---
    drip_threshold: float = field(default=0.45, metadata={"help": "Wetness threshold before paint begins to drip.", "category": "Advanced", "min": 0.0, "max": 1.0})
    drip_rate_coeff: float = field(default=2.5, metadata={"help": "Speed of dripping once threshold is met.", "category": "Advanced", "min": 0.0, "max": 10.0})
    turbulence_coeff: float = field(default=0.8, metadata={"help": "Strength of noise-driven flow direction.", "category": "Advanced", "min": 0.0, "max": 5.0})
    advection_coeff: float = field(default=10.0, metadata={"help": "Strength of pigment transport by water flow.", "category": "Advanced", "min": 0.0, "max": 20.0})
    velocity_damping: float = field(default=0.15, metadata={"help": "How fast water flow slows down.", "category": "Advanced", "min": 0.0, "max": 1.0})
    pressure_coeff: float = field(default=3.5, metadata={"help": "Internal pressure affecting water movement.", "category": "Advanced", "min": 0.0, "max": 10.0})
    max_velocity_coeff: float = field(default=80.0, metadata={"help": "Maximum flow velocity allowed.", "category": "Advanced", "min": 10.0, "max": 200.0})
    
    # --- [ADVANCED] Paper & Interaction ---
    absorption_coeff: float = field(default=0.12, metadata={"help": "Base rate of water absorption into paper.", "category": "Advanced", "min": 0.0, "max": 1.0})
    settle_coeff: float = field(default=0.32, metadata={"help": "Base rate of pigment settling (staining).", "category": "Advanced", "min": 0.0, "max": 1.0})
    paper_texture_scale: float = field(default=2.2, metadata={"help": "Scale of the procedural grain.", "category": "Advanced", "min": 0.1, "max": 10.0})
    wet_darken_coeff: float = field(default=0.05, metadata={"help": "Visual darkening of wet areas.", "category": "Advanced", "min": 0.0, "max": 0.5})
    pigment_absorb_floor: float = field(default=0.10, metadata={"help": "Minimum opacity for absorbed pigment.", "category": "Advanced", "min": 0.0, "max": 0.5})
    pigment_neutral_density: float = field(default=0.35, metadata={"help": "Lightness bias for pigment rendering.", "category": "Advanced", "min": 0.0, "max": 1.0})
    
    # --- Internal State (Not for direct UI control but persistent) ---
    bloom_variance: float = field(default=12.0, metadata={"help": "Inner randomness for brush edges.", "category": "Advanced", "min": 0.0, "max": 50.0})
    stroke_life: float = field(default=8.0, metadata={"help": "Active simulation time after a stroke.", "category": "Advanced", "min": 1.0, "max": 60.0})

