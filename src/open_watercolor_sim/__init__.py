"""
Open Watercolor Sim.

Copyright (c) 2026 Shuoqi Chen
SPDX-License-Identifier: Apache-2.0
"""
from .brush.watercolor_engine import WatercolorEngine
from .brush.configs import SimParams
from .viewer import launch_viewer

__version__ = "0.1.0"
__author__ = "Shuoqi Chen"
__license__ = "Apache-2.0"
__all__ = ["WatercolorEngine", "SimParams", "launch_viewer"]
