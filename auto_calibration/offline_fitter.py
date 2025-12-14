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

from .utils import load_image, get_screenshots_directory
from .board_detector import BoardDetector
from .clock_detector import ClockDetector
from .coordinate_calculator import CoordinateCalculator
from .visualiser import CalibrationVisualiser
from .config import save_config


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
        
        return config
    
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
