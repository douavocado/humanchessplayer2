#!/usr/bin/env python3
"""
Chess Clock Detection Module

Detects clock positions using text block detection.

Primary detection method:
- Find dark text blocks on light background (Lichess theme)
- Identify clocks by their characteristic wide aspect ratio (~6:1)

Fallback method:
- Y-axis sweep with OCR validation using read_clock()
"""

import cv2
import numpy as np
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, List

from .utils import extract_region, cluster_values, cluster_mean
from .config import ChessConfig
from .panel_detector import ClockTextDetector


# Add parent directory for imports
PARENT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PARENT_DIR))


# Standard clock dimensions (for reference board size of 848px)
REFERENCE_BOARD_SIZE = 848
REFERENCE_CLOCK_WIDTH = 147
REFERENCE_CLOCK_HEIGHT = 44


def get_read_clock_function():
    """
    Import the read_clock function from image_scrape_utils.
    
    Returns:
        The read_clock function, or a fallback if import fails.
    """
    try:
        from chessimage.image_scrape_utils import read_clock
        return read_clock
    except ImportError:
        # Return a fallback function that always fails
        def fallback_read_clock(clock_image):
            return None
        return fallback_read_clock


class ClockDetector:
    """
    Detects clock positions using text block detection.
    
    Primary method (text-based):
    1. Find dark text blocks on light background
    2. Identify clock-like blocks (wide aspect ratio ~6:1)
    3. Select largest blocks at top and bottom of board area
    
    Fallback method (OCR sweep):
    1. Calculate expected X position relative to board
    2. Sweep Y-axis in small increments
    3. At each Y, extract clock region and try read_clock()
    4. Cluster successful Y positions
    
    Clock dimensions are scaled based on detected board size.
    """
    
    # Y-sweep parameters (for fallback OCR method)
    SWEEP_STEP = 3  # Pixels between Y positions to test
    SWEEP_PADDING = 100  # Extra pixels to search above/below board
    
    # Clustering threshold for grouping nearby detections
    CLUSTER_THRESHOLD = 15
    
    # Expected gap between board and clock panel (relative to board size)
    CLOCK_GAP_RATIO_MIN = 0.01  # ~1% of board size
    CLOCK_GAP_RATIO_MAX = 0.08  # ~8% of board size
    
    def __init__(self, board_detection: Optional[Dict] = None):
        """
        Initialise clock detector.
        
        Args:
            board_detection: Board detection result from BoardDetector.
        """
        self.board = board_detection
        self.read_clock = get_read_clock_function()
        self.text_detector = ClockTextDetector(board_detection)
        
        # Calculate scaled clock dimensions
        self._update_clock_dimensions()
    
    def _update_clock_dimensions(self):
        """Update clock dimensions based on board size."""
        if self.board:
            board_size = self.board['size']
            scale = board_size / REFERENCE_BOARD_SIZE
            self.clock_width = int(REFERENCE_CLOCK_WIDTH * scale)
            self.clock_height = int(REFERENCE_CLOCK_HEIGHT * scale)
            self.scale = scale
        else:
            self.clock_width = REFERENCE_CLOCK_WIDTH
            self.clock_height = REFERENCE_CLOCK_HEIGHT
            self.scale = 1.0
    
    def set_board(self, board_detection: Dict):
        """Set the board detection result and update clock dimensions."""
        self.board = board_detection
        self.text_detector.set_board(board_detection)
        self._update_clock_dimensions()
    
    def detect(self, image: np.ndarray) -> Optional[Dict]:
        """
        Detect clock positions in an image.
        
        Uses text-based detection as primary method, with OCR sweep as fallback.
        
        Args:
            image: BGR image to search.
        
        Returns:
            Dictionary with detection results:
            {
                'bottom_clock': {state: {'x', 'y', 'width', 'height'}, ...},
                'top_clock': {state: {'x', 'y', 'width', 'height'}, ...},
                'clock_x': int,  # Common X position
                'detection_count': int,
                'detection_method': str  # 'text' or 'ocr_sweep'
            }
            Or None if no clocks found.
        """
        if self.board is None:
            print("Error: Board detection required before clock detection")
            return None
        
        # Try text-based detection first (faster and more reliable)
        result = self._detect_by_text(image)
        
        if result:
            return result
        
        # Fallback to OCR sweep method
        print("Text detection failed, trying OCR sweep...")
        return self._detect_by_ocr_sweep(image)
    
    def _detect_by_text(self, image: np.ndarray) -> Optional[Dict]:
        """
        Detect clocks using text block detection.
        
        Args:
            image: BGR image.
        
        Returns:
            Detection result or None.
        """
        text_result = self.text_detector.detect(image)
        
        if text_result is None:
            return None
        
        top_clock = text_result.get('top_clock')
        bottom_clock = text_result.get('bottom_clock')
        
        if not top_clock and not bottom_clock:
            return None
        
        # Determine common X position
        if top_clock and bottom_clock:
            clock_x = (top_clock['x'] + bottom_clock['x']) // 2
        elif top_clock:
            clock_x = top_clock['x']
        else:
            clock_x = bottom_clock['x']
        
        # Format as game state dictionary (default to 'play' state)
        result = {
            'clock_x': clock_x,
            'detection_count': (1 if top_clock else 0) + (1 if bottom_clock else 0),
            'detection_method': 'text'
        }
        
        if top_clock:
            result['top_clock'] = {
                'play': {
                    'x': top_clock['x'],
                    'y': top_clock['y'],
                    'width': top_clock['width'],
                    'height': top_clock['height']
                }
            }
        
        if bottom_clock:
            result['bottom_clock'] = {
                'play': {
                    'x': bottom_clock['x'],
                    'y': bottom_clock['y'],
                    'width': bottom_clock['width'],
                    'height': bottom_clock['height']
                }
            }
        
        return result
    
    def _detect_by_ocr_sweep(self, image: np.ndarray) -> Optional[Dict]:
        """
        Detect clocks using OCR sweep method (fallback).
        
        Args:
            image: BGR image.
        
        Returns:
            Detection result or None.
        """
        
        if image is None or image.size == 0:
            return None
        
        img_h, img_w = image.shape[:2]
        board_x = self.board['x']
        board_y = self.board['y']
        board_size = self.board['size']
        
        # Calculate scaled gap values
        gap_min = int(board_size * self.CLOCK_GAP_RATIO_MIN)
        gap_max = int(board_size * self.CLOCK_GAP_RATIO_MAX)
        
        print(f"Clock detection: board={board_size}px, scale={self.scale:.2f}, "
              f"clock_size={self.clock_width}x{self.clock_height}")
        
        # Calculate search region
        # Clock X is to the right of board
        clock_x_start = board_x + board_size + gap_min
        clock_x_end = min(board_x + board_size + gap_max, img_w - self.clock_width)
        
        if clock_x_start >= img_w - self.clock_width:
            print("Error: No space for clock to right of board")
            return None
        
        # Find the correct X position
        clock_x = self._find_clock_x(image, clock_x_start, clock_x_end, board_y, board_size)
        
        if clock_x is None:
            print("Warning: Could not find clock X position, using estimate")
            # Use scaled default gap
            clock_x = board_x + board_size + int(29 * self.scale)
        
        # Define Y search ranges for top and bottom clocks
        # Scale the sweep padding too
        scaled_padding = int(self.SWEEP_PADDING * self.scale)
        top_y_start = max(0, board_y - scaled_padding)
        top_y_end = board_y + board_size // 3
        
        bottom_y_start = board_y + 2 * board_size // 3
        bottom_y_end = min(img_h - self.clock_height, board_y + board_size + scaled_padding)
        
        # Sweep for top clock
        top_detections = self._sweep_y_range(image, clock_x, top_y_start, top_y_end)
        
        # Sweep for bottom clock
        bottom_detections = self._sweep_y_range(image, clock_x, bottom_y_start, bottom_y_end)
        
        if not top_detections and not bottom_detections:
            return None
        
        # Cluster and assign states
        top_clock = self._cluster_and_assign_states(top_detections, 'top')
        bottom_clock = self._cluster_and_assign_states(bottom_detections, 'bottom')
        
        return {
            'bottom_clock': bottom_clock,
            'top_clock': top_clock,
            'clock_x': clock_x,
            'detection_count': len(top_detections) + len(bottom_detections)
        }
    
    def _extract_and_resize_clock(self, image: np.ndarray, x: int, y: int) -> Optional[np.ndarray]:
        """
        Extract clock region at scaled size and resize to standard 147x44 for OCR.
        
        Args:
            image: Source image.
            x: X position.
            y: Y position.
        
        Returns:
            Resized clock region (147x44) or None if extraction failed.
        """
        # Extract at scaled size
        region = extract_region(image, x, y, self.clock_width, self.clock_height)
        
        if region is None:
            return None
        
        # Resize to standard size for read_clock()
        # read_clock() expects exactly 147x44 with digits at specific positions
        resized = cv2.resize(region, (REFERENCE_CLOCK_WIDTH, REFERENCE_CLOCK_HEIGHT),
                            interpolation=cv2.INTER_AREA)
        
        return resized
    
    def _find_clock_x(self, image: np.ndarray, x_start: int, x_end: int,
                      board_y: int, board_size: int) -> Optional[int]:
        """
        Find the correct X position for clocks.
        
        Tests multiple X positions and returns the one with most successful reads.
        
        Args:
            image: Source image.
            x_start: Start of X search range.
            x_end: End of X search range.
            board_y: Board Y position.
            board_size: Board size.
        
        Returns:
            Best X position, or None if none found.
        """
        # Test Y positions where clocks are likely to be (scaled)
        test_y_positions = [
            board_y + int(250 * self.scale),  # Approximate top clock position
            board_y + board_size - int(100 * self.scale),  # Approximate bottom clock position
        ]
        
        best_x = None
        best_count = 0
        
        # Use scaled step for X search
        x_step = max(3, int(5 * self.scale))
        
        # Test X positions
        for x in range(x_start, x_end, x_step):
            success_count = 0
            
            for y in test_y_positions:
                # Sweep small Y range around test position (scaled)
                y_range = int(30 * self.scale)
                y_step = max(3, int(5 * self.scale))
                
                for dy in range(-y_range, y_range + 1, y_step):
                    test_y = y + dy
                    if test_y < 0:
                        continue
                    
                    region = self._extract_and_resize_clock(image, x, test_y)
                    if region is None:
                        continue
                    
                    time_value = self.read_clock(region)
                    if time_value is not None:
                        success_count += 1
            
            if success_count > best_count:
                best_count = success_count
                best_x = x
            
            # Early termination if we found a good position
            if success_count >= 3:
                return x
        
        return best_x
    
    def _sweep_y_range(self, image: np.ndarray, clock_x: int,
                       y_start: int, y_end: int) -> List[Dict]:
        """
        Sweep Y range and find all positions where clock can be read.
        
        Args:
            image: Source image.
            clock_x: X position of clock.
            y_start: Start of Y range.
            y_end: End of Y range.
        
        Returns:
            List of successful detections.
        """
        detections = []
        
        # Scale the sweep step
        sweep_step = max(2, int(self.SWEEP_STEP * self.scale))
        
        for y in range(y_start, y_end, sweep_step):
            region = self._extract_and_resize_clock(image, clock_x, y)
            
            if region is None:
                continue
            
            time_value = self.read_clock(region)
            
            if time_value is not None:
                detections.append({
                    'x': clock_x,
                    'y': y,
                    'width': self.clock_width,
                    'height': self.clock_height,
                    'time_value': time_value
                })
        
        return detections
    
    def _cluster_and_assign_states(self, detections: List[Dict],
                                    clock_type: str) -> Dict[str, Dict]:
        """
        Cluster detections and assign game states.
        
        Args:
            detections: List of detection dicts.
            clock_type: 'top' or 'bottom'.
        
        Returns:
            Dictionary mapping state names to coordinates.
        """
        if not detections:
            return {}
        
        # Extract Y values
        y_values = [d['y'] for d in detections]
        
        # Cluster nearby Y values (scaled threshold)
        cluster_threshold = int(self.CLUSTER_THRESHOLD * self.scale)
        clusters = cluster_values(y_values, cluster_threshold)
        
        # Get representative detection for each cluster
        cluster_detections = []
        
        for cluster in clusters:
            # Find detection closest to cluster mean
            mean_y = cluster_mean(cluster)
            best_detection = min(detections, key=lambda d: abs(d['y'] - mean_y))
            cluster_detections.append({
                **best_detection,
                'y': mean_y  # Use cluster mean for stability
            })
        
        # Sort by Y position
        cluster_detections.sort(key=lambda d: d['y'])
        
        # Assign states based on position
        # For bottom clock: first (lowest Y) is 'play', then start1, start2, end1, end2, end3
        # For top clock: first (lowest Y) is end1, then higher Y values
        
        state_order = ['play', 'start1', 'start2', 'end1', 'end2', 'end3']
        
        result = {}
        
        for i, detection in enumerate(cluster_detections):
            if i < len(state_order):
                state = state_order[i]
            else:
                state = f'extra_{i}'
            
            result[state] = {
                'x': detection['x'],
                'y': detection['y'],
                'width': detection['width'],
                'height': detection['height'],
                'time_value': detection.get('time_value')
            }
        
        return result
    
    def validate_position(self, image: np.ndarray, x: int, y: int,
                          width: Optional[int] = None, 
                          height: Optional[int] = None) -> Tuple[bool, Optional[int]]:
        """
        Validate a specific clock position.
        
        Args:
            image: Source image.
            x: X coordinate.
            y: Y coordinate.
            width: Clock width (uses scaled default if None).
            height: Clock height (uses scaled default if None).
        
        Returns:
            (is_valid, time_value) tuple.
        """
        w = width if width is not None else self.clock_width
        h = height if height is not None else self.clock_height
        
        region = extract_region(image, x, y, w, h)
        
        if region is None:
            return False, None
        
        # Resize to standard dimensions for read_clock
        resized = cv2.resize(region, (REFERENCE_CLOCK_WIDTH, REFERENCE_CLOCK_HEIGHT),
                            interpolation=cv2.INTER_AREA)
        
        time_value = self.read_clock(resized)
        
        return time_value is not None, time_value
    
    def validate_all_states(self, image: np.ndarray,
                           clock_positions: Dict) -> Dict[str, Dict]:
        """
        Validate all clock state positions.
        
        Args:
            image: Source image.
            clock_positions: Dictionary of clock positions from config.
        
        Returns:
            Validation results for each clock type and state.
        """
        results = {}
        
        for clock_type in ['bottom_clock', 'top_clock']:
            if clock_type not in clock_positions:
                continue
            
            results[clock_type] = {}
            
            for state, coords in clock_positions[clock_type].items():
                is_valid, time_value = self.validate_position(
                    image, coords['x'], coords['y'],
                    coords.get('width'), coords.get('height')
                )
                
                results[clock_type][state] = {
                    'valid': is_valid,
                    'time_value': time_value,
                    'coordinates': coords
                }
        
        return results


def detect_clocks(image: np.ndarray, board_detection: Dict) -> Optional[Dict]:
    """
    Convenience function to detect clocks.
    
    Args:
        image: BGR image to search.
        board_detection: Board detection result.
    
    Returns:
        Clock detection result dict or None.
    """
    detector = ClockDetector(board_detection)
    return detector.detect(image)


def detect_clocks_from_screenshot(board_detection: Dict) -> Optional[Dict]:
    """
    Detect clocks from current screen.
    
    Args:
        board_detection: Board detection result.
    
    Returns:
        Clock detection result dict or None.
    """
    from .utils import capture_screenshot
    
    screenshot = capture_screenshot()
    if screenshot is None:
        return None
    
    return detect_clocks(screenshot, board_detection)
