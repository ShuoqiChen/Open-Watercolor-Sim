"""
Open Watercolor Sim.

Copyright (c) 2026 Shuoqi Chen
SPDX-License-Identifier: MIT OR Apache-2.0
"""
import taichi as ti
import numpy as np
import time
from .brush import WatercolorEngine, SimParams

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

def launch_viewer():
    import argparse
    from dataclasses import fields
    
    parser = argparse.ArgumentParser(description="Watercolor Simulator: High Quality GPU Mode")
    parser.add_argument("-r", "--res", type=int, default=1024, help="Simulation resolution (default: 1024)")
    parser.add_argument("-f", "--fps", type=int, default=60, help="Target FPS cap (default: 60)")
    parser.add_argument("-s", "--substeps", type=int, default=2, help="Max simulation substeps when painting (default: 2)")
    parser.add_argument("-d", "--decimate", type=int, default=1, help="Render decimation (default: 1)")
    
    # Add SimParams as arguments automatically
    for f in fields(SimParams):
        if f.name == 'color_rgb': continue # Skip complex types for CLI for now
        arg_name = f.name.replace('_', '-')
        # Use float for numeric types to be safe
        arg_type = type(f.default) if f.default is not None else float
        parser.add_argument(f"--{arg_name}", type=arg_type, default=f.default, help=f.metadata.get('help', ''))
        
    args = parser.parse_args()

    RES = args.res
    arch = "gpu"
    
    print(f"\n[SightOfSound] Starting Watercolor (High Quality GPU Mode)")
    print(f" - Resolution: {RES}x{RES}")
    print(f" - Backend:    {arch.upper()}")
    print(f" - FPS Cap:    {args.fps}")
    print(f" - Substeps:   {args.substeps} (adaptive)")
    print(f"--------------------------------")

    # Initialize Engine
    engine = WatercolorEngine(res=RES, arch=arch)
    
    # Update engine params from CLI
    cli_params = {}
    for f in fields(SimParams):
        if hasattr(args, f.name):
            cli_params[f.name] = getattr(args, f.name)
    engine.update_params(**cli_params)
    
    window = ti.ui.Window("Watercolor: GPU (Fluid Simulation)", (RES, RES))
    canvas = window.get_canvas()
    gui = window.get_gui()
    
    brush_id = 0
    dryness = 0.5
    
    last_paint_pos = None
    last_paint_time = 0
    paint_interval = 1.0 / 60.0
    stroke_dist_accum = 0.0
    
    # Kernel pacing stats
    total_stamps = 0
    last_stat_time = time.time()
    frame_idx = 0
    show_advanced = False

    print("\n[Controls]")
    print(" - Mouse Left (LMB): Paint with bleeding/dripping water")
    print(" - Space: Clear Canvas")
    print(" - S: Save Screenshot")
    print(" - UI: Use the 'Controls' sidebar (top-left) for all sliders")
    print(" - Shortcuts: [ / ] for Size, D / F for Dryness")

    start_time = time.time()
    fps_limit = args.fps

    # UI State
    show_ui = True
    paint_enabled = True

    while window.running:
        frame_start = time.time()
        curr_time = frame_start - start_time
        stamps_this_frame = 0 # Throttling cap
        
        is_painting = window.is_pressed(ti.ui.LMB)
        safe_mode = window.is_pressed(ti.ui.SHIFT)

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
                engine.update_params(brush_radius=max(SimParams.__dataclass_fields__['brush_radius'].metadata['min'], engine.p.brush_radius - 5))
            elif e.key == ']':
                engine.update_params(brush_radius=min(SimParams.__dataclass_fields__['brush_radius'].metadata['max'], engine.p.brush_radius + 5))
            elif e.key == 's':
                # Force full render for screenshot
                engine.render(full_res=True)
                img = engine._img_u8.to_numpy()
                import PIL.Image
                PIL.Image.fromarray(img).save(f"render_{int(time.time())}.png")
                print("Saved screenshot.")
            elif e.key == 'p':
                paint_enabled = not paint_enabled
            elif e.key == ti.ui.TAB:
                show_ui = not show_ui
            elif e.key == ti.ui.ESCAPE:
                window.running = False

        if is_painting and paint_enabled and not safe_mode:
            # ti.ui.Window (GGUI) uses [0,1] with origin at BOTTOM-LEFT.
            mx, my = window.get_cursor_pos()
            
            # Hit-test logic removed to allow movable UI.
            # Use 'Shift' guard or 'Tab' toggle to manage UI interactions.
            px, py = mx * RES, my * RES
            
            if last_paint_pos is not None:
                lx, ly = last_paint_pos
                dx, dy = px - lx, py - ly
                dist = (dx*dx + dy*dy)**0.5
                
                spacing = max(1.0, engine.p.brush_radius * 0.8)
                stroke_dist_accum += dist
                
                # 1. Throttling: distance-based and time-based
                if dist > 1e-4: # Ignore absolute stillness
                    if stroke_dist_accum >= spacing or (curr_time - last_paint_time > paint_interval):
                        # Use a single stamp per event to keep stamps/sec <= 60
                        if stamps_this_frame < 1:
                            engine.paint_brush(px, py, engine.p.brush_radius, brush_id=brush_id, dryness=dryness)
                            stamps_this_frame += 1
                            total_stamps += 1
                        
                        stroke_dist_accum = 0.0
                        last_paint_time = curr_time
                
                last_paint_pos = (px, py)
            else:
                # Initial click stamp
                engine.paint_brush(px, py, engine.p.brush_radius, brush_id=brush_id, dryness=dryness)
                stamps_this_frame += 1
                total_stamps += 1
                last_paint_pos = (px, py)
                last_paint_time = curr_time
        else:
            last_paint_pos = None
            stroke_dist_accum = 0.0

        # UI Sidebar (Overlay)
        if show_ui:
            with gui.sub_window("Controls", 0.05, 0.05, 0.3, 0.9) as w:
                gui.text("Simulation Controls")
                gui.text(f"Brush: {'ON' if paint_enabled else 'PAUSED'} [P]")
                gui.text("Toggle UI: [Tab]")
                gui.text("Safe Move: [Hold Shift]")
                
                if gui.button("Clear Canvas"): engine.clear()
                
                gui.text("--- Brush & Color ---")
                brush_id = gui.slider_int("Brush Type", brush_id, 0, 1)
                dryness = gui.slider_float("Brush Dryness", dryness, 0.0, 1.0)
                engine.p.color_rgb = gui.color_edit_3("Color Picker", engine.p.color_rgb)
                
                # Use dataclass fields to build the UI dynamically
                from dataclasses import fields
                
                for f in fields(SimParams):
                    cat = f.metadata.get("category", "Normal")
                    if cat == "Normal":
                        if f.name in ["color_rgb"]: continue # Handled specially
                        
                        display_name = f.name.replace("_", " ").title()
                        val = getattr(engine.p, f.name)
                        new_val = gui.slider_float(display_name, val, f.metadata.get('min', 0.0), f.metadata.get('max', 1.0))
                        setattr(engine.p, f.name, new_val)
                
                gui.text("--- Advanced ---")
                show_advanced = gui.checkbox("Advanced Settings", show_advanced)
                
                if show_advanced:
                    for f in fields(SimParams):
                        cat = f.metadata.get("category")
                        if cat == "Advanced":
                            display_name = f.name.replace("_", " ").title()
                            val = getattr(engine.p, f.name)
                            new_val = gui.slider_float(display_name, val, f.metadata.get('min', 0.0), f.metadata.get('max', 1.0))
                            setattr(engine.p, f.name, new_val)

                if gui.button("Save Screenshot"):
                    engine.render(full_res=True)
                    img = engine._img_u8.to_numpy()
                    import PIL.Image
                    PIL.Image.fromarray(img).save(f"render_{int(time.time())}.png")

        # Sync GUI to Engine
        engine.update_params()

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
    launch_viewer()
