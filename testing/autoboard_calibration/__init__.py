"""
Chess Board Auto-Calibration System

A device-independent auto-calibration system for chess board detection
and UI element positioning.
"""

__version__ = "1.0.0"
__author__ = "Auto-generated calibration system"

from .board_detector import BoardDetector
from .coordinate_mapper import CoordinateMapper
from .auto_calibrator import AutoCalibrator
from .visualizer import CalibrationVisualizer

__all__ = [
    "BoardDetector",
    "CoordinateMapper", 
    "AutoCalibrator",
    "CalibrationVisualizer"
]
