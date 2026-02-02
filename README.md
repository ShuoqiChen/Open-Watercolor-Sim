# Open Watercolor Sim

Current Version: **v1.0.0**

A high-quality, GPU-accelerated watercolor simulation engine built using Taichi. This simulator models fluid dynamics, pigment diffusion, and edge darkening to create realistic-looking watercolor effects in real-time.

## Features

- **Real-time Fluid Simulation**: High-performance GPU simulation of water and pigment.
- **Dynamic Brushes**: Round and Sponge brush types with adjustable properties.
- **Physical Parameters**: Control diffusion, evaporation, gravity, and pigment load.
- **Cross-Platform**: Runs on any system supported by Taichi (CUDA, Vulkan, Metal, etc.).

## Prerequisites

- Python 3.7+
- A GPU with Vulkan, CUDA, or Metal support (recommended)

## Installation

1. Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Running the Simulator

Launch the main simulation with:

```bash
python watercolor.py
```

### Command Line Arguments

- `-r`, `--res`: Simulation resolution (default: 1024)
- `-f`, `--fps`: Target FPS cap (default: 60)
- `-s`, `--substeps`: Simulation substeps per frame (default: 2)
- `-d`, `--decimate`: Render decimation factor (default: 1)

## Controls

### Mouse
- **Left Click (LMB)**: Paint on the canvas. Moving the mouse while clicking applies water and pigment.

### Keyboard Shortcuts
- **Space**: Clear the canvas.
- **S**: Save a screenshot (saved as `render_[timestamp].png`).
- **[ / ]**: Decrease / Increase brush radius.
- **D / F**: Decrease / Increase brush dryness.
- **B**: Toggle between brush types (Round / Sponge).
- **Escape**: Exit the simulator.

### UI Panel
A control panel is available in the top-left corner to adjust:
- **Brush Settings**: Type, Radius, Dryness, and Color.
- **Physics**: Pigment Load, Water Release, Diffusion, Canvas Drying, Gravity, and Edge Darkening.

## Project Structure

- [watercolor.py](watercolor.py): Main entry point and GUI handling.
- [brush/](brush/): Core simulation logic.
    - [brush/watercolor_engine.py](brush/watercolor_engine.py): The Taichi-based simulation engine.
    - [brush/__init__.py](brush/__init__.py): Module initialization.
- [requirements.txt](requirements.txt): List of dependencies.
