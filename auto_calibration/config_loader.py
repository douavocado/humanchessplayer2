#!/usr/bin/env python3
"""
Configuration Loader

Loads and provides access to auto-calibrated coordinates.
This module replaces hardcoded coordinates in the main chess system.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

class ChessConfig:
    """Loads and provides access to chess board configuration."""
    
    def __init__(self, config_file: str = "chess_config.json"):
        """
        Initialize configuration loader.
        
        Args:
            config_file: Path to configuration file (relative to auto_calibration folder)
        """
        self.config_file = config_file
        self.config = None
        self.fallback_used = False
        
        # Fallback coordinates (your original hardcoded values as backup)
        self.fallback_coordinates = {
            'board': {'x': 543, 'y': 179, 'width': 848, 'height': 848},
            'bottom_clock': {
                'play': {'x': 1420, 'y': 742, 'width': 147, 'height': 44},
                'start1': {'x': 1420, 'y': 756, 'width': 147, 'height': 44},
                'start2': {'x': 1420, 'y': 770, 'width': 147, 'height': 44},
                'end1': {'x': 1420, 'y': 811, 'width': 147, 'height': 44},
                'end2': {'x': 1420, 'y': 747, 'width': 147, 'height': 44},
                'end3': {'x': 1420, 'y': 776, 'width': 147, 'height': 44}
            },
            'top_clock': {
                'play': {'x': 1420, 'y': 424, 'width': 147, 'height': 44},
                'start1': {'x': 1420, 'y': 396, 'width': 147, 'height': 44},
                'start2': {'x': 1420, 'y': 410, 'width': 147, 'height': 44},
                'end1': {'x': 1420, 'y': 355, 'width': 147, 'height': 44},
                'end2': {'x': 1420, 'y': 420, 'width': 147, 'height': 44}
            },
            'notation': {'x': 1458, 'y': 591, 'width': 166, 'height': 104},
            'rating': {
                'opp_white': {'x': 1755, 'y': 458, 'width': 40, 'height': 24},
                'own_white': {'x': 1755, 'y': 706, 'width': 40, 'height': 24},
                'opp_black': {'x': 1755, 'y': 473, 'width': 40, 'height': 24},
                'own_black': {'x': 1755, 'y': 691, 'width': 40, 'height': 24}
            }
        }
        
        self.load_config()
    
    def load_config(self) -> bool:
        """
        Load configuration from file.
        
        Returns:
            True if config loaded successfully, False if using fallback
        """
        # Try to find config file in auto_calibration directory
        config_paths = [
            Path(__file__).parent / self.config_file,  # Same directory as this file
            Path(__file__).parent.parent / self.config_file,  # Project root
            Path(self.config_file)  # Current working directory
        ]
        
        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        self.config = json.load(f)
                    print(f"✅ Loaded chess configuration from: {config_path}")
                    self.fallback_used = False
                    return True
                except Exception as e:
                    print(f"⚠️  Error loading config from {config_path}: {e}")
                    continue
        
        print("⚠️  No configuration file found. Using fallback coordinates.")
        print("   Run auto_calibration/calibrator.py to generate device-specific coordinates.")
        self.fallback_used = True
        return False
    
    def get_coordinates(self) -> Dict:
        """Get coordinate configuration (auto-calibrated or fallback)."""
        if self.config and 'coordinates' in self.config:
            return self.config['coordinates']
        else:
            return self.fallback_coordinates
    
    def get_board_position(self) -> Tuple[int, int, int, int]:
        """Get board position as (x, y, width, height)."""
        coords = self.get_coordinates()
        board = coords['board']
        return (board['x'], board['y'], board['width'], board['height'])
    
    def get_clock_position(self, clock_type: str, state: str = "play") -> Tuple[int, int, int, int]:
        """
        Get clock position as (x, y, width, height).
        
        Args:
            clock_type: 'bottom_clock' or 'top_clock'
            state: Game state ('play', 'start1', 'start2', 'end1', 'end2', 'end3')
            
        Returns:
            Tuple of (x, y, width, height)
        """
        coords = self.get_coordinates()
        
        if clock_type in coords and state in coords[clock_type]:
            clock = coords[clock_type][state]
            return (clock['x'], clock['y'], clock['width'], clock['height'])
        else:
            # Fallback to play state if specific state not found
            if clock_type in coords and 'play' in coords[clock_type]:
                clock = coords[clock_type]['play']
                return (clock['x'], clock['y'], clock['width'], clock['height'])
            else:
                # Ultimate fallback
                fallback = self.fallback_coordinates[clock_type]['play']
                return (fallback['x'], fallback['y'], fallback['width'], fallback['height'])
    
    def get_notation_position(self) -> Tuple[int, int, int, int]:
        """Get notation area position as (x, y, width, height)."""
        coords = self.get_coordinates()
        notation = coords['notation']
        return (notation['x'], notation['y'], notation['width'], notation['height'])
    
    def get_rating_position(self, rating_type: str) -> Tuple[int, int, int, int]:
        """
        Get rating position as (x, y, width, height).
        
        Args:
            rating_type: 'opp_white', 'own_white', 'opp_black', 'own_black'
        """
        coords = self.get_coordinates()
        rating = coords['rating'][rating_type]
        return (rating['x'], rating['y'], rating['width'], rating['height'])
    
    def get_step_size(self) -> int:
        """
        Calculate step size (size of one chess square) from board dimensions.
        """
        board_x, board_y, board_width, board_height = self.get_board_position()
        # Chess board is 8x8, so step size is board_width / 8
        return board_width // 8
    
    def get_start_position(self) -> Tuple[int, int]:
        """Get the top-left corner of the chess board."""
        board_x, board_y, _, _ = self.get_board_position()
        return (board_x, board_y)
    
    def is_using_fallback(self) -> bool:
        """Check if using fallback coordinates (no auto-calibration loaded)."""
        return self.fallback_used
    
    def get_calibration_info(self) -> Optional[Dict]:
        """Get calibration metadata if available."""
        if self.config and 'calibration_info' in self.config:
            return self.config['calibration_info']
        return None
    
    def print_status(self):
        """Print current configuration status."""
        if self.is_using_fallback():
            print("📍 Using fallback coordinates (hardcoded values)")
            print("   Run 'python auto_calibration/calibrator.py' to auto-calibrate")
        else:
            info = self.get_calibration_info()
            if info:
                print(f"📍 Using auto-calibrated coordinates")
                print(f"   Calibrated: {info.get('timestamp', 'Unknown')}")
                print(f"   Success rate: {info.get('validation_success_rate', 0):.1%}")
                print(f"   Detection method: {info.get('board_detection', {}).get('method', 'Unknown')}")

# Global configuration instance
chess_config = ChessConfig()

# Convenience functions for direct access
def get_board_position() -> Tuple[int, int, int, int]:
    """Get board position (x, y, width, height)."""
    return chess_config.get_board_position()

def get_clock_position(clock_type: str, state: str = "play") -> Tuple[int, int, int, int]:
    """Get clock position (x, y, width, height)."""
    return chess_config.get_clock_position(clock_type, state)

def get_step_size() -> int:
    """Get chess square step size."""
    return chess_config.get_step_size()

def get_start_position() -> Tuple[int, int]:
    """Get board start position (x, y)."""
    return chess_config.get_start_position()

def reload_config():
    """Reload configuration from file."""
    chess_config.load_config()

def print_config_status():
    """Print current configuration status."""
    chess_config.print_status()
