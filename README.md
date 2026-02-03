# Open-Watercolor-Sim

**Open-Watercolor-Sim** is a real-time GPU-accelerated watercolor simulation framework in Python. It provides a physically inspired simulation engine designed for both artistic creation and as a research toolbox for computer graphics and fluid dynamics exploration.

## Demo

![Watercolor Simulation](https://raw.githubusercontent.com/shuoqichen/Open-Watercolor-Sim/main/demo/demo.png)

## Features

- **GPU Acceleration**: Built on [Taichi Lang](https://github.com/taichi-dev/taichi) for high-performance physics-based simulation.
- **Interactive Controls**: Real-time brush interaction with adjustable radius, pressure (pigment load), and water release.
- **Physically Inspired Effects**: Models complex watercolor behaviors including diffusion, evaporation, edge darkening (coffee-ring effect), and granulation.
- **Cross-Platform**: Support for multiple GPU backends (CUDA, Metal, Vulkan) enabling real-time performance on Windows, macOS, and Linux.
- **Research Toolbox**: Easily extensible architecture for testing new fluid advection schemes or pigment interaction models.

## Installation

**Open-Watercolor-Sim** requires Python 3.10 or newer.

```bash
pip install open-watercolor-sim
```

## Quickstart

You can launch the interactive viewer using the console entry point:

```bash
watercolor-sim
```

Alternatively, run the module directly:

```bash
python -m open_watercolor_sim.viewer
```

## Project Structure

```text
demo/                      # Previews and example outputs
src/open_watercolor_sim/   # Main package
├── viewer.py              # Interactive GGUI-based viewer
└── brush/
    ├── configs.py         # Simulation and artistic parameters
    └── watercolor_engine.py # Core Taichi-based simulation logic
```

## Citation

If you use this framework in your research, please cite it using the following BibTeX:

```bibtex
@software{open_watercolor_sim,
  title  = {Open-Watercolor-Sim: A Real-Time GPU-Accelerated Watercolor Simulation Framework},
  author = {Chen, Shuoqi},
  year   = {2026},
  url    = {https://github.com/shuoqichen/Open-Watercolor-Sim}
}
```

*Note: A DOI-backed citation for the paper-aligned release will be added in a future update.*

## License

This project is licensed under the **Apache License 2.0**.
