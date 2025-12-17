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
    - TemplateExtractor: Extracts digit/piece/result templates
    - ShadowCalibrator: Passive template extraction during gameplay
    - InteractiveCalibrator: Manual template capture with GUI

Usage:
    # Live calibration
    python -m auto_calibration.calibrator --live
    
    # Offline fitting from screenshots
    python -m auto_calibration.calibrator --offline ./calibration_screenshots/
    
    # Shadow calibration (passive, during gameplay)
    python -m auto_calibration.shadow_calibrator --duration 5
    
    # Interactive calibration (manual template capture)
    python -m auto_calibration.interactive_calibrator --guided
    
    # In code
    from auto_calibration.config import ChessConfig
    config = ChessConfig()
    board_x, board_y, step = config.get_board_info()
"""

from .config import ChessConfig, get_config

__version__ = "2.1.0"
__all__ = [
    "ChessConfig", 
    "get_config",
    # Lazy imports for optional components
    # from auto_calibration.template_extractor import TemplateExtractor
    # from auto_calibration.shadow_calibrator import ShadowCalibrator
    # from auto_calibration.interactive_calibrator import InteractiveCalibrator
]
