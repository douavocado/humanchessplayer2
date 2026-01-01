#!/usr/bin/env python3
"""
Auto-Calibration Module for Chess Board Detection

This module provides automatic detection of chess board position and UI elements
(clocks, notation, ratings) from saved screenshots.

Main Components:
    - BoardDetector: Detects chess board using colour segmentation
    - ClockDetector: Finds clock positions using OCR validation
    - CoordinateCalculator: Derives notation/rating positions from board/clock data
    - Visualiser: Creates debug visualisations
    - Config: Loads and saves calibration configurations
    - TemplateExtractor: Extracts digit/piece/result templates
    - OfflineFitter: Fits calibration from saved screenshots
    - ButtonDetector: Dynamically detects Lichess UI buttons

Usage:
    # Fit calibration from screenshots
    python -m auto_calibration.offline_fitter --dir ./screenshots/ --profile my_profile --extract-all

    # Test calibration accuracy
    python -m auto_calibration.calibration_readback_test --screenshots ./screenshots/ --profile my_profile

    # In code
    from auto_calibration.config import ChessConfig
    config = ChessConfig()
    board_x, board_y, step = config.get_board_info()
"""

from .config import ChessConfig, get_config

__version__ = "3.0.0"
__all__ = [
    "ChessConfig", 
    "get_config",
]
