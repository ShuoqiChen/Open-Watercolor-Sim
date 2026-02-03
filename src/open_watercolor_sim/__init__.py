"""
Open Watercolor Sim.

Copyright (c) 2026 Shuoqi Chen
SPDX-License-Identifier: MIT OR Apache-2.0
"""
from .brush.watercolor_engine import WatercolorEngine
from .brush.configs import SimParams
from .viewer import launch_viewer

__version__ = "1.0.0"
__author__ = "Shuoqi Chen"
__license__ = "MIT"
__all__ = ["WatercolorEngine", "SimParams", "launch_viewer"]
