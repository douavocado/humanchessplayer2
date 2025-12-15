#!/usr/bin/env python3
"""
Offline Fitting Module

Fits calibration from saved screenshots instead of live capture.
Useful for:
- Capturing rare game states (stalemate, etc.)
- Calibrating on a different machine than where screenshots were taken
- Re-running calibration without an active game
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import json
import re

from .utils import load_image, get_screenshots_directory, remove_background_colours
from .board_detector import BoardDetector
from .clock_detector import ClockDetector
from .coordinate_calculator import CoordinateCalculator
from .visualiser import CalibrationVisualiser
from .config import save_config


def detect_digit_positions(clock_image: np.ndarray) -> Optional[Dict[str, float]]:
    """
    Detect digit positions within a clock image.
    
    Analyses the clock to find where the 4 time digits (MM:SS) are located,
    returning positions as fractions of clock width for resolution independence.
    
    Args:
        clock_image: BGR or grayscale clock image
        
    Returns:
        Dictionary with d1_start, d1_end, d2_start, d2_end, d3_start, d3_end, d4_start, d4_end
        as fractions of width (0.0 to 1.0), or None if detection failed.
    """
    if clock_image is None:
        return None
    
    # Convert to grayscale and process
    if clock_image.ndim == 3:
        processed = remove_background_colours(clock_image, thresh=1.6).astype(np.uint8)
    else:
        processed = clock_image.copy()
    
    if processed.size == 0 or processed.ndim != 2:
        return None
    
    img_height, img_width = processed.shape
    
    # Find dark content (digits are dark on light background)
    dark_mask = processed < 128
    col_sums = dark_mask.sum(axis=0)
    
    # Find content regions
    threshold = 2
    content = col_sums > threshold
    transitions = np.where(np.diff(content.astype(int)) != 0)[0]
    
    if len(transitions) < 6:
        return None
    
    # Extract and split regions (handle "1:" being detected as single region)
    raw_regions = []
    for i in range(0, len(transitions) - 1, 2):
        start = transitions[i] + 1
        end = transitions[i + 1] + 1 if i + 1 < len(transitions) else img_width
        if end - start > 3:
            raw_regions.append((start, end))
    
    # Split regions where there's a big jump in column sums (digit vs colon)
    regions = []
    for start, end in raw_regions:
        region_sums = col_sums[start:end]
        if len(region_sums) == 0:
            continue
        
        max_sum = region_sums.max()
        min_sum = region_sums[region_sums > 0].min() if (region_sums > 0).any() else 0
        
        if max_sum > 20 and min_sum < 10 and max_sum > 3 * min_sum:
            # Split thin digit from colon/decimal
            for j in range(len(region_sums) - 1):
                if region_sums[j] < 10 and region_sums[j+1] > 20:
                    regions.append((start, start + j + 1))
                    regions.append((start + j + 1, end))
                    break
            else:
                regions.append((start, end))
        else:
            regions.append((start, end))
    
    # Filter for digit-sized regions
    digit_regions = []
    for start, end in regions:
        width = end - start
        if width < 3:
            continue
        avg_col_sum = col_sums[start:end].mean()
        if width >= 6 and avg_col_sum < 25:
            digit_regions.append((start, end))
    
    if len(digit_regions) < 4:
        return None
    
    # Add small padding and convert to fractions
    padding_left = 3
    padding_right = 2
    
    def to_fraction(start, end):
        s = max(0, start - padding_left)
        e = min(img_width, end + padding_right)
        return s / img_width, e / img_width
    
    d1_frac = to_fraction(digit_regions[0][0], digit_regions[0][1])
    d2_frac = to_fraction(digit_regions[1][0], digit_regions[1][1])
    d3_frac = to_fraction(digit_regions[2][0], digit_regions[2][1])
    d4_frac = to_fraction(digit_regions[3][0], digit_regions[3][1])
    
    return {
        'd1_start': d1_frac[0], 'd1_end': d1_frac[1],
        'd2_start': d2_frac[0], 'd2_end': d2_frac[1],
        'd3_start': d3_frac[0], 'd3_end': d3_frac[1],
        'd4_start': d4_frac[0], 'd4_end': d4_frac[1]
    }


class OfflineFitter:
    """
    Fits calibration from saved screenshots.
    
    Supports multiple screenshots with different game states,
    combining them into a single comprehensive configuration.
    """
    
    def __init__(self, screenshots_dir: Optional[Path] = None):
        """
        Initialise offline fitter.
        
        Args:
            screenshots_dir: Directory containing screenshots.
                           If None, uses default location.
        """
        if screenshots_dir is None:
            screenshots_dir = get_screenshots_directory()
        self.screenshots_dir = Path(screenshots_dir)
        
        self.board_detector = BoardDetector()
        self.clock_detector = ClockDetector()
        self.calculator = CoordinateCalculator()
    
    def fit_from_screenshots(self, 
                            screenshot_paths: Optional[List[str]] = None,
                            state_hints: Optional[Dict[str, str]] = None) -> Optional[Dict]:
        """
        Fit calibration from multiple screenshots.
        
        Args:
            screenshot_paths: List of paths to screenshots.
                            If None, uses all images in screenshots_dir.
            state_hints: Optional mapping of filename to state hint.
                        e.g., {'screenshot_001.png': 'play', 'screenshot_002.png': 'start1'}
        
        Returns:
            Complete calibration configuration, or None if failed.
        """
        # Get screenshot paths
        if screenshot_paths is None:
            screenshot_paths = self._find_screenshots()
        
        if not screenshot_paths:
            print("No screenshots found")
            return None
        
        print(f"Found {len(screenshot_paths)} screenshots")
        
        # Process each screenshot
        all_detections = []
        board_detection = None
        
        for path in screenshot_paths:
            print(f"\nProcessing: {Path(path).name}")
            
            # Load image
            image = load_image(path)
            if image is None:
                print(f"  Failed to load image")
                continue
            
            # Detect board (use first successful detection)
            if board_detection is None:
                board_detection = self.board_detector.detect(image)
                if board_detection:
                    print(f"  Board detected: ({board_detection['x']}, {board_detection['y']}) "
                          f"size={board_detection['size']} conf={board_detection['confidence']:.2f}")
            
            if board_detection is None:
                print(f"  Skipping (no board detected)")
                continue
            
            # Detect clocks
            self.clock_detector.set_board(board_detection)
            clock_detection = self.clock_detector.detect(image)
            
            if clock_detection:
                # Get state hint from filename or provided hints
                filename = Path(path).name
                state_hint = None
                
                if state_hints and filename in state_hints:
                    state_hint = state_hints[filename]
                else:
                    state_hint = self._extract_state_from_filename(filename)
                
                all_detections.append({
                    'path': path,
                    'image': image,
                    'board': board_detection,
                    'clocks': clock_detection,
                    'state_hint': state_hint
                })
                
                print(f"  Clocks detected: {clock_detection['detection_count']} positions")
                if state_hint:
                    print(f"  State hint: {state_hint}")
            else:
                print(f"  No clocks detected")
        
        if not all_detections:
            print("\nNo valid detections from any screenshot")
            return None
        
        # Combine detections
        combined = self._combine_detections(all_detections, board_detection)
        
        return combined
    
    def _find_screenshots(self) -> List[str]:
        """Find all screenshot files in the screenshots directory."""
        if not self.screenshots_dir.exists():
            return []
        
        extensions = ['.png', '.jpg', '.jpeg']
        screenshots = []
        
        for ext in extensions:
            screenshots.extend(self.screenshots_dir.glob(f'*{ext}'))
            screenshots.extend(self.screenshots_dir.glob(f'*{ext.upper()}'))
        
        return sorted([str(p) for p in screenshots])
    
    def _extract_state_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract game state hint from filename.
        
        Looks for patterns like:
        - screenshot_play.png
        - calibration_20250115_start1.png
        - game_end_resign.png
        """
        states = ['play', 'start1', 'start2', 'end1', 'end2', 'end3',
                  'start', 'end', 'resign', 'draw', 'stalemate', 'checkmate']
        
        filename_lower = filename.lower()
        
        for state in states:
            if state in filename_lower:
                # Map generic states to specific ones
                if state == 'start':
                    return 'start1'
                elif state == 'end' or state in ['resign', 'draw', 'stalemate', 'checkmate']:
                    return 'end1'
                else:
                    return state
        
        return None
    
    def _combine_detections(self, detections: List[Dict],
                           board_detection: Dict) -> Dict:
        """
        Combine multiple screenshot detections into single config.
        
        Args:
            detections: List of detection results.
            board_detection: Board detection to use.
        
        Returns:
            Combined configuration dictionary.
        """
        # Use calculator to get base coordinates
        self.calculator.set_board(board_detection)
        
        # Collect all clock positions
        bottom_clocks = {}
        top_clocks = {}
        clock_x = None
        
        for detection in detections:
            clocks = detection['clocks']
            state_hint = detection['state_hint']
            
            if 'clock_x' in clocks and clock_x is None:
                clock_x = clocks['clock_x']
            
            for clock_type, positions in [('bottom_clock', bottom_clocks), 
                                          ('top_clock', top_clocks)]:
                if clock_type in clocks:
                    for state, coords in clocks[clock_type].items():
                        # If we have a state hint, use it
                        if state_hint and state == 'play':
                            actual_state = state_hint
                        else:
                            actual_state = state
                        
                        # Keep the position with highest time value (most likely valid)
                        if actual_state not in positions:
                            positions[actual_state] = coords
                        elif coords.get('time_value', 0) > positions[actual_state].get('time_value', 0):
                            positions[actual_state] = coords
        
        # Set combined clocks
        combined_clocks = {
            'bottom_clock': bottom_clocks,
            'top_clock': top_clocks,
            'clock_x': clock_x or (board_detection['x'] + board_detection['size'] + 29),
            'detection_count': len(bottom_clocks) + len(top_clocks)
        }
        
        self.calculator.set_clocks(combined_clocks)
        
        # Calculate all coordinates
        coordinates = self.calculator.calculate_all()
        
        # Estimate any missing clock states
        estimated_clocks = self.calculator.estimate_missing_clock_states(combined_clocks)
        
        # Merge estimated into coordinates
        for clock_type in ['bottom_clock', 'top_clock']:
            if clock_type in estimated_clocks:
                if clock_type not in coordinates:
                    coordinates[clock_type] = {}
                for state, coords in estimated_clocks[clock_type].items():
                    if state not in coordinates[clock_type]:
                        coordinates[clock_type][state] = coords
        
        # Build final config
        config = {
            'calibration_info': {
                'method': 'offline_fitting',
                'screenshots_used': len(detections),
                'board_confidence': board_detection.get('confidence', 0),
                'clock_states_detected': list(bottom_clocks.keys()) + list(top_clocks.keys())
            },
            'coordinates': coordinates
        }
        
        # Try to detect digit positions from the first valid clock
        # This is stored as fractions so it works regardless of clock width
        digit_positions = self._detect_digit_positions_from_detections(detections, coordinates)
        if digit_positions:
            config['digit_positions'] = digit_positions
            print(f"✓ Digit positions calibrated")
        else:
            print("⚠ Could not calibrate digit positions, will use fallback")
        
        return config
    
    def _detect_digit_positions_from_detections(self, detections: List[Dict],
                                                 coordinates: Dict) -> Optional[Dict]:
        """
        Detect digit positions from the best available clock image.
        
        Args:
            detections: List of detection results with images.
            coordinates: Calculated coordinates.
            
        Returns:
            Digit positions as fractions, or None if detection failed.
        """
        # Try to find a 'start1' or 'start2' detection as these show initial time
        for detection in detections:
            if detection['state_hint'] in ['start1', 'start2']:
                image = detection.get('image')
                clocks = detection.get('clocks', {})
                
                if image is not None and 'bottom_clock' in clocks:
                    # Extract clock region from image
                    for state, clock_coords in clocks['bottom_clock'].items():
                        if 'x' in clock_coords and 'y' in clock_coords:
                            x = clock_coords['x']
                            y = clock_coords['y']
                            w = clock_coords.get('width', 220)
                            h = clock_coords.get('height', 40)
                            
                            clock_img = image[y:y+h, x:x+w]
                            if clock_img.size > 0:
                                positions = detect_digit_positions(clock_img)
                                if positions:
                                    return positions
        
        # Fallback: try any detection
        for detection in detections:
            image = detection.get('image')
            clocks = detection.get('clocks', {})
            
            if image is not None and 'bottom_clock' in clocks:
                for state, clock_coords in clocks['bottom_clock'].items():
                    if 'x' in clock_coords and 'y' in clock_coords:
                        x = clock_coords['x']
                        y = clock_coords['y']
                        w = clock_coords.get('width', 220)
                        h = clock_coords.get('height', 40)
                        
                        clock_img = image[y:y+h, x:x+w]
                        if clock_img.size > 0:
                            positions = detect_digit_positions(clock_img)
                            if positions:
                                return positions
        
        return None
    
    def fit_from_single_screenshot(self, screenshot_path: str,
                                   state_hint: Optional[str] = None,
                                   visualise: bool = True) -> Optional[Dict]:
        """
        Fit calibration from a single screenshot.
        
        Args:
            screenshot_path: Path to screenshot.
            state_hint: Optional state hint for the screenshot.
            visualise: Whether to create debug visualisations.
        
        Returns:
            Calibration configuration, or None if failed.
        """
        # Load image
        image = load_image(screenshot_path)
        if image is None:
            print(f"Failed to load: {screenshot_path}")
            return None
        
        print(f"Processing: {screenshot_path}")
        
        # Detect board
        board_detection = self.board_detector.detect(image)
        if board_detection is None:
            print("Board detection failed")
            return None
        
        print(f"Board: ({board_detection['x']}, {board_detection['y']}) "
              f"size={board_detection['size']} conf={board_detection['confidence']:.2f}")
        
        # Detect clocks
        self.clock_detector.set_board(board_detection)
        clock_detection = self.clock_detector.detect(image)
        
        if clock_detection:
            print(f"Clocks: {clock_detection['detection_count']} positions detected")
        else:
            print("Clock detection failed")
        
        # Calculate coordinates
        self.calculator.set_board(board_detection)
        self.calculator.set_clocks(clock_detection)
        coordinates = self.calculator.calculate_all()
        
        # Estimate missing states
        if clock_detection:
            estimated = self.calculator.estimate_missing_clock_states(clock_detection)
            for clock_type in ['bottom_clock', 'top_clock']:
                if clock_type in estimated:
                    for state, coords in estimated[clock_type].items():
                        if state not in coordinates.get(clock_type, {}):
                            if clock_type not in coordinates:
                                coordinates[clock_type] = {}
                            coordinates[clock_type][state] = coords
        
        # Build config
        config = {
            'calibration_info': {
                'method': 'offline_single',
                'source_screenshot': screenshot_path,
                'board_confidence': board_detection.get('confidence', 0),
                'state_hint': state_hint
            },
            'coordinates': coordinates
        }
        
        # Visualise
        if visualise:
            visualiser = CalibrationVisualiser()
            outputs = visualiser.visualise_all(image, board_detection, 
                                               clock_detection, coordinates)
            print(f"\nDebug output saved to: {visualiser.get_output_dir()}")
        
        return config


def fit_from_screenshots(screenshots_dir: Optional[str] = None,
                        save_to_config: bool = True) -> Optional[Dict]:
    """
    Convenience function to fit from screenshots directory.
    
    Args:
        screenshots_dir: Directory with screenshots.
        save_to_config: Whether to save result to chess_config.json.
    
    Returns:
        Configuration dictionary, or None if failed.
    """
    fitter = OfflineFitter(Path(screenshots_dir) if screenshots_dir else None)
    config = fitter.fit_from_screenshots()
    
    if config and save_to_config:
        output_path = save_config(config)
        print(f"\nConfiguration saved to: {output_path}")
    
    return config


def fit_from_single(screenshot_path: str,
                   state_hint: Optional[str] = None,
                   save_to_config: bool = True) -> Optional[Dict]:
    """
    Convenience function to fit from single screenshot.
    
    Args:
        screenshot_path: Path to screenshot.
        state_hint: Optional state hint.
        save_to_config: Whether to save result to chess_config.json.
    
    Returns:
        Configuration dictionary, or None if failed.
    """
    fitter = OfflineFitter()
    config = fitter.fit_from_single_screenshot(screenshot_path, state_hint)
    
    if config and save_to_config:
        output_path = save_config(config)
        print(f"\nConfiguration saved to: {output_path}")
    
    return config


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Offline calibration fitter")
    parser.add_argument("--dir", type=str, help="Screenshots directory")
    parser.add_argument("--file", type=str, help="Single screenshot file")
    parser.add_argument("--state", type=str, help="State hint for single file")
    parser.add_argument("--no-save", action="store_true", help="Don't save to config")
    
    args = parser.parse_args()
    
    if args.file:
        fit_from_single(args.file, args.state, not args.no_save)
    elif args.dir:
        fit_from_screenshots(args.dir, not args.no_save)
    else:
        # Use default screenshots directory
        fit_from_screenshots(None, not args.no_save)
