#!/usr/bin/env python3
"""
Auto-Calibration Module for Chess Board Detection

This module provides automatic detection of chess board position and UI elements
(clocks, notation, ratings) from screen captures. It supports both live calibration
and offline fitting from saved screenshots.

Main Components:
    - BoardDetector: Detects chess board using colour segmentation
    - ClockDetector: Finds clock positions using OCR validation
    - CoordinateCalculator: Derives notation/rating positions from board/clock data
    - Visualiser: Creates debug visualisations
    - Config: Loads and saves calibration configurations

Usage:
    # Live calibration
    python calibrator.py --live
    
    # Offline fitting from screenshots
    python calibrator.py --offline ./calibration_screenshots/
    
    # In code
    from auto_calibration.config import ChessConfig
    config = ChessConfig()
    board_x, board_y, step = config.get_board_info()
"""

from .config import ChessConfig, get_config

__version__ = "2.0.0"
__all__ = ["ChessConfig", "get_config"]
