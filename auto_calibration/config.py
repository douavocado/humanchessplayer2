#!/usr/bin/env python3
"""
Configuration loading and saving for auto-calibration.

Provides the ChessConfig class that loads calibrated coordinates
and falls back to hardcoded values if no configuration exists.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from datetime import datetime


class ChessConfig:
    """
    Loads and provides access to chess board calibration configuration.
    
    Falls back to hardcoded coordinates if no configuration file exists.
    """
    
    # Default configuration file name
    DEFAULT_CONFIG_FILE = "chess_config.json"
    
    # Fallback coordinates (for when no config exists)
    FALLBACK_COORDINATES = {
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
    
    # Standard dimensions for UI elements
    CLOCK_WIDTH = 147
    CLOCK_HEIGHT = 44
    NOTATION_WIDTH = 166
    NOTATION_HEIGHT = 104
    RATING_WIDTH = 40
    RATING_HEIGHT = 24
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialise configuration loader.
        
        Args:
            config_file: Path to configuration file. If None, uses default location.
        """
        self.config_file = config_file or self.DEFAULT_CONFIG_FILE
        self.config: Optional[Dict] = None
        self.using_fallback = True
        self._load_config()
    
    def _load_config(self) -> bool:
        """
        Load configuration from file.
        
        Returns:
            True if loaded successfully, False if using fallback.
        """
        # Try multiple possible locations
        config_paths = [
            Path(__file__).parent / self.config_file,  # Same directory
            Path(__file__).parent.parent / self.config_file,  # Project root
            Path(self.config_file)  # Absolute or relative path
        ]
        
        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        self.config = json.load(f)
                    self.using_fallback = False
                    return True
                except Exception as e:
                    print(f"Warning: Error loading config from {config_path}: {e}")
                    continue
        
        self.using_fallback = True
        return False
    
    def reload(self) -> bool:
        """Reload configuration from file."""
        return self._load_config()
    
    def get_coordinates(self) -> Dict:
        """Get all coordinates (from config or fallback)."""
        if self.config and 'coordinates' in self.config:
            return self.config['coordinates']
        return self.FALLBACK_COORDINATES
    
    def get_board_info(self) -> Tuple[int, int, int]:
        """
        Get board position and step size.
        
        Returns:
            (x, y, step) tuple where step is the size of one square.
        """
        coords = self.get_coordinates()
        board = coords['board']
        step = board['width'] // 8
        return board['x'], board['y'], step
    
    def get_board_position(self) -> Tuple[int, int, int, int]:
        """
        Get board position as (x, y, width, height).
        """
        coords = self.get_coordinates()
        board = coords['board']
        return board['x'], board['y'], board['width'], board['height']
    
    def get_clock_position(self, clock_type: str, state: str = "play") -> Tuple[int, int, int, int]:
        """
        Get clock position as (x, y, width, height).
        
        Args:
            clock_type: 'bottom_clock' or 'top_clock'
            state: Game state ('play', 'start1', 'start2', 'end1', 'end2', 'end3')
        
        Returns:
            (x, y, width, height) tuple.
        """
        coords = self.get_coordinates()
        
        if clock_type in coords and state in coords[clock_type]:
            clock = coords[clock_type][state]
            return clock['x'], clock['y'], clock['width'], clock['height']
        
        # Fallback to 'play' state
        if clock_type in coords and 'play' in coords[clock_type]:
            clock = coords[clock_type]['play']
            return clock['x'], clock['y'], clock['width'], clock['height']
        
        # Ultimate fallback
        fallback = self.FALLBACK_COORDINATES[clock_type]['play']
        return fallback['x'], fallback['y'], fallback['width'], fallback['height']
    
    def get_notation_position(self) -> Tuple[int, int, int, int]:
        """Get notation area position as (x, y, width, height)."""
        coords = self.get_coordinates()
        notation = coords.get('notation', self.FALLBACK_COORDINATES['notation'])
        return notation['x'], notation['y'], notation['width'], notation['height']
    
    def get_rating_position(self, rating_type: str) -> Tuple[int, int, int, int]:
        """
        Get rating position as (x, y, width, height).
        
        Args:
            rating_type: 'opp_white', 'own_white', 'opp_black', 'own_black'
        """
        coords = self.get_coordinates()
        rating = coords.get('rating', self.FALLBACK_COORDINATES['rating'])
        r = rating.get(rating_type, self.FALLBACK_COORDINATES['rating'][rating_type])
        return r['x'], r['y'], r['width'], r['height']
    
    def get_step_size(self) -> int:
        """Get size of one chess square in pixels."""
        _, _, step = self.get_board_info()
        return step
    
    def is_using_fallback(self) -> bool:
        """Check if using fallback coordinates."""
        return self.using_fallback
    
    def get_calibration_info(self) -> Optional[Dict]:
        """Get calibration metadata if available."""
        if self.config and 'calibration_info' in self.config:
            return self.config['calibration_info']
        return None
    
    def print_status(self):
        """Print current configuration status."""
        if self.using_fallback:
            print("📍 Using fallback coordinates (no config file found)")
            print("   Run 'python -m auto_calibration.calibrator --live' to calibrate")
        else:
            info = self.get_calibration_info()
            if info:
                print("📍 Using auto-calibrated coordinates")
                print(f"   Calibrated: {info.get('timestamp', 'Unknown')}")
                print(f"   Confidence: {info.get('board_confidence', 0):.1%}")
                if 'clock_states_detected' in info:
                    print(f"   Clock states: {info['clock_states_detected']}")


def save_config(config_data: Dict, output_path: Optional[str] = None) -> str:
    """
    Save calibration configuration to file.
    
    Args:
        config_data: Configuration dictionary to save.
        output_path: Output file path. If None, uses default location.
    
    Returns:
        Path to saved configuration file.
    """
    if output_path is None:
        output_path = Path(__file__).parent / ChessConfig.DEFAULT_CONFIG_FILE
    else:
        output_path = Path(output_path)
    
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp
    if 'calibration_info' not in config_data:
        config_data['calibration_info'] = {}
    config_data['calibration_info']['timestamp'] = datetime.now().isoformat()
    
    # Convert numpy types to native Python types
    config_data = _convert_to_native_types(config_data)
    
    with open(output_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    return str(output_path)


def _convert_to_native_types(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialisation."""
    import numpy as np
    
    if isinstance(obj, dict):
        return {k: _convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_native_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return _convert_to_native_types(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


# Global configuration instance (lazy-loaded)
_global_config: Optional[ChessConfig] = None


def get_config() -> ChessConfig:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = ChessConfig()
    return _global_config


def reload_config() -> ChessConfig:
    """Reload the global configuration from file."""
    global _global_config
    _global_config = ChessConfig()
    return _global_config
