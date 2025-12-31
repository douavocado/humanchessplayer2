#!/usr/bin/env python3
"""
Configuration loading and saving for auto-calibration.

Provides the ChessConfig class that loads calibrated coordinates
and falls back to hardcoded values if no configuration exists.
"""

import json
import os
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

    # Environment variables for selecting an alternative calibration
    # - HCP_CALIBRATION_FILE: explicit path (absolute or relative) to a calibration JSON
    # - HCP_CALIBRATION_PROFILE: profile name resolved to auto_calibration/calibrations/{profile}.json
    ENV_CALIBRATION_FILE = "HCP_CALIBRATION_FILE"
    ENV_CALIBRATION_PROFILE = "HCP_CALIBRATION_PROFILE"
    
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
        },
        'result_region': {'x': 1480, 'y': 522, 'width': 50, 'height': 30},
        'resign_button': {'x': 1540, 'y': 640, 'width': 40, 'height': 40}
    }
    
    # Standard dimensions for UI elements
    CLOCK_WIDTH = 147
    CLOCK_HEIGHT = 44
    NOTATION_WIDTH = 166
    NOTATION_HEIGHT = 104
    RATING_WIDTH = 40
    RATING_HEIGHT = 24
    
    # Default colour scheme (Lichess green theme - BGR format)
    FALLBACK_COLOUR_SCHEME = {
        'light_square': [214, 235, 238],      # Cream/beige
        'dark_square': [86, 150, 118],         # Green
        'highlight_light': [177, 209, 205],    # Light square with last move highlight
        'highlight_dark': [100, 151, 144],     # Dark square with last move highlight
        'premove_light': [160, 165, 170],      # Premove on light square
        'premove_dark': [135, 140, 147],       # Premove on dark square
    }
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialise configuration loader.
        
        Args:
            config_file: Path to configuration file. If None, uses default location.
        """
        self.config_file = config_file or self.DEFAULT_CONFIG_FILE
        self.config: Optional[Dict] = None
        self.using_fallback = True
        self._profile_name: Optional[str] = None
        self._load_config()
    
    def get_profile_name(self) -> Optional[str]:
        """
        Get the name of the current profile.
        
        Extracts from config file path (e.g., 'calibrations/laptop.json' -> 'laptop').
        
        Returns:
            Profile name string, or None if using default/fallback.
        """
        if self._profile_name:
            return self._profile_name
        
        if self.config_file and self.config_file != self.DEFAULT_CONFIG_FILE:
            # Extract profile name from path like 'calibrations/laptop.json'
            path = Path(self.config_file)
            if path.stem and path.stem != 'chess_config':
                self._profile_name = path.stem
                return self._profile_name
        
        return None
    
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
    
    def get_result_region_position(self) -> Tuple[int, int, int, int]:
        """
        Get result region position as (x, y, width, height).
        
        This region shows the game result ("0-1", "1-0", "½-½") when game ends.
        """
        coords = self.get_coordinates()
        result = coords.get('result_region', self.FALLBACK_COORDINATES['result_region'])
        return result['x'], result['y'], result['width'], result['height']
    
    def get_resign_button_position(self) -> Tuple[int, int]:
        """
        Get resign button centre position as (x, y).
        
        The resign button is the flag icon below the notation panel.
        Returns the centre of the button for clicking.
        """
        coords = self.get_coordinates()
        resign = coords.get('resign_button', self.FALLBACK_COORDINATES['resign_button'])
        # Return centre of button
        centre_x = resign['x'] + resign.get('width', 40) // 2
        centre_y = resign['y'] + resign.get('height', 40) // 2
        return centre_x, centre_y
    
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
    
    def get_digit_positions(self) -> Optional[Dict]:
        """
        Get clock digit positions as fractions of clock width.
        
        Returns:
            Dictionary with d1_start, d1_end, d2_start, d2_end, etc. as fractions [0-1],
            or None if not calibrated.
        """
        if self.config and 'digit_positions' in self.config:
            return self.config['digit_positions']
        return None
    
    def get_template_dir(self) -> Path:
        """
        Get the directory containing extracted templates for the current profile.
        
        Returns profile-specific directory if a profile is active,
        otherwise returns the generic templates directory.
        
        Returns:
            Path to templates directory
        """
        base_template_dir = Path(__file__).parent / "templates"
        
        profile = self.get_profile_name()
        if profile:
            profile_template_dir = base_template_dir / profile
            if profile_template_dir.exists():
                return profile_template_dir
        
        return base_template_dir
    
    def get_colour_scheme(self) -> Dict:
        """
        Get the colour scheme for board detection.
        
        Returns colours in BGR format for OpenCV compatibility.
        
        Returns:
            Dictionary with keys: light_square, dark_square, highlight_light,
            highlight_dark, premove_light, premove_dark
        """
        if self.config and 'colour_scheme' in self.config:
            return self.config['colour_scheme']
        return self.FALLBACK_COLOUR_SCHEME.copy()
    
    def get_highlight_colours(self) -> list:
        """
        Get all highlight colours as a list for move detection.
        
        Returns:
            List of BGR colour tuples that indicate last move highlights.
        """
        scheme = self.get_colour_scheme()
        colours = [
            scheme.get('highlight_light', self.FALLBACK_COLOUR_SCHEME['highlight_light']),
            scheme.get('highlight_dark', self.FALLBACK_COLOUR_SCHEME['highlight_dark']),
            scheme.get('premove_light', self.FALLBACK_COLOUR_SCHEME['premove_light']),
            scheme.get('premove_dark', self.FALLBACK_COLOUR_SCHEME['premove_dark']),
        ]
        return colours
    
    def get_piece_template_size(self) -> int:
        """
        Get the native piece template size for this profile.
        
        Returns:
            Piece size in pixels (width = height for square pieces)
        """
        if self.config and 'template_info' in self.config:
            return self.config['template_info'].get('piece_size', 106)
        
        # Calculate from board size
        coords = self.get_coordinates()
        board = coords.get('board', {})
        board_width = board.get('width', 848)
        return board_width // 8
    
    def has_calibrated_templates(self) -> bool:
        """
        Check if we have calibrated templates available for the current profile.
        
        Requires ALL piece templates (12) and ALL digit templates (10) to be present.
        
        Returns:
            True if complete templates exist
        """
        template_dir = self.get_template_dir()
        if not template_dir.exists():
            return False
        
        # Check for all digit templates (0-9)
        digit_dir = template_dir / "digits"
        if not digit_dir.exists():
            return False
        digits_found = len(list(digit_dir.glob("*.png")))
        
        # Check for all piece templates (12 pieces)
        piece_dir = template_dir / "pieces"
        if not piece_dir.exists():
            return False
        pieces_found = len(list(piece_dir.glob("*.png")))
        
        # Require at least partial templates
        return digits_found >= 1 and pieces_found >= 12
    
    def has_complete_templates(self) -> bool:
        """
        Check if we have COMPLETE templates (all digits and all pieces).
        
        Returns:
            True only if all 10 digits and 12 pieces are present.
        """
        template_dir = self.get_template_dir()
        if not template_dir.exists():
            return False
        
        digit_dir = template_dir / "digits"
        piece_dir = template_dir / "pieces"
        
        digits_found = len(list(digit_dir.glob("*.png"))) if digit_dir.exists() else 0
        pieces_found = len(list(piece_dir.glob("*.png"))) if piece_dir.exists() else 0
        
        return digits_found >= 10 and pieces_found >= 12
    
    def get_template_status(self) -> Dict:
        """
        Get detailed status of template extraction for the current profile.
        
        Returns:
            Dictionary with extraction progress
        """
        template_dir = self.get_template_dir()
        progress_file = template_dir / "extraction_progress.json"
        
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        # Build status from what files actually exist
        status = {
            "digits": {str(i): False for i in range(10)},
            "pieces": {p: False for p in "RNBQKPrnbqkp"},
            "results": {"white_win": False, "black_win": False, "draw": False}
        }
        
        # Check actual files
        digit_dir = template_dir / "digits"
        if digit_dir.exists():
            for i in range(10):
                if (digit_dir / f"{i}.png").exists():
                    status["digits"][str(i)] = True
        
        piece_dir = template_dir / "pieces"
        if piece_dir.exists():
            piece_files = {
                'R': 'w_rook.png', 'N': 'w_knight.png', 'B': 'w_bishop.png',
                'Q': 'w_queen.png', 'K': 'w_king.png', 'P': 'w_pawn.png',
                'r': 'b_rook.png', 'n': 'b_knight.png', 'b': 'b_bishop.png',
                'q': 'b_queen.png', 'k': 'b_king.png', 'p': 'b_pawn.png'
            }
            for piece, filename in piece_files.items():
                if (piece_dir / filename).exists():
                    status["pieces"][piece] = True
        
        return status
    
    def print_status(self):
        """Print current configuration status."""
        profile = self.get_profile_name()
        
        if self.using_fallback:
            print("📍 Using fallback coordinates (no config file found)")
            print("   Run 'python -m auto_calibration.calibrator --live' to calibrate")
        else:
            info = self.get_calibration_info()
            if info:
                profile_str = f" (profile: {profile})" if profile else ""
                print(f"📍 Using auto-calibrated coordinates{profile_str}")
                print(f"   Calibrated: {info.get('timestamp', 'Unknown')}")
                print(f"   Confidence: {info.get('board_confidence', 0):.1%}")
                if 'clock_states_detected' in info:
                    print(f"   Clock states: {info['clock_states_detected']}")
        
        # Template status
        template_dir = self.get_template_dir()
        if self.has_calibrated_templates():
            status = self.get_template_status()
            digits_done = sum(1 for v in status.get("digits", {}).values() if v)
            pieces_done = sum(1 for v in status.get("pieces", {}).values() if v)
            results_done = sum(1 for v in status.get("results", {}).values() if v)
            print(f"📋 Templates ({template_dir.name}): {digits_done}/10 digits, {pieces_done}/12 pieces, {results_done}/3 results")
        else:
            print(f"📋 No calibrated templates found in {template_dir}")
            print("   Run 'python -m auto_calibration.offline_fitter --dir <screenshots_dir> --profile <name> --extract-all'")
        
        # Colour scheme status
        if self.config and 'colour_scheme' in self.config:
            print("🎨 Colour scheme: Calibrated")
        else:
            print("🎨 Colour scheme: Using fallback (may cause detection issues)")


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


def _resolve_selected_config_file() -> str:
    """
    Resolve the selected config file from environment variables.

    Returns:
        A config file string suitable for ChessConfig(config_file=...).
        This can be a filename, relative path, or absolute path.
    """
    explicit = os.getenv(ChessConfig.ENV_CALIBRATION_FILE)
    if explicit:
        return explicit

    profile = os.getenv(ChessConfig.ENV_CALIBRATION_PROFILE)
    if profile:
        # Stored under auto_calibration/calibrations/{profile}.json by default.
        return str(Path("calibrations") / f"{profile}.json")

    return ChessConfig.DEFAULT_CONFIG_FILE


def select_config(*, config_file: Optional[str] = None, profile: Optional[str] = None) -> ChessConfig:
    """
    Select which calibration config file should be used globally.

    This is mainly a convenience wrapper around environment-variable selection,
    and it forces the global config singleton to reload.
    """
    if config_file and profile:
        raise ValueError("Provide only one of config_file or profile")

    # Update env vars so downstream imports (e.g. chessimage/image_scrape_utils.py)
    # can pick the selection up at import-time.
    if config_file is not None:
        os.environ[ChessConfig.ENV_CALIBRATION_FILE] = config_file
        os.environ.pop(ChessConfig.ENV_CALIBRATION_PROFILE, None)
    elif profile is not None:
        os.environ[ChessConfig.ENV_CALIBRATION_PROFILE] = profile
        os.environ.pop(ChessConfig.ENV_CALIBRATION_FILE, None)
    else:
        # Reset to default behaviour
        os.environ.pop(ChessConfig.ENV_CALIBRATION_FILE, None)
        os.environ.pop(ChessConfig.ENV_CALIBRATION_PROFILE, None)

    return reload_config()


def get_config() -> ChessConfig:
    """Get the global configuration instance."""
    global _global_config
    desired_file = _resolve_selected_config_file()

    # Recreate the singleton if it hasn't been created yet, or if selection changed.
    if _global_config is None or getattr(_global_config, "config_file", None) != desired_file:
        _global_config = ChessConfig(config_file=desired_file)
    return _global_config


def reload_config() -> ChessConfig:
    """Reload the global configuration from file."""
    global _global_config
    _global_config = ChessConfig(config_file=_resolve_selected_config_file())
    return _global_config
