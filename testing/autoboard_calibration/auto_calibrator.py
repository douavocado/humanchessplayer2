#!/usr/bin/env python3
"""
Auto Calibration System

Main interface for automatically calibrating chess board detection and UI element positioning.
This creates a device-independent configuration that can be used by the main chess client.
"""

import cv2
import numpy as np
import json
import time
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
from fastgrab import screenshot

# Add parent directories to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.append(str(parent_dir))
sys.path.append(str(parent_dir / "chessimage"))

from board_detector import BoardDetector
from coordinate_mapper import CoordinateMapper
from utils import simple_read_clock

class AutoCalibrator:
    """Main calibration system for device-independent chess board detection."""
    
    def __init__(self):
        self.detector = BoardDetector()
        self.mapper = CoordinateMapper()
        self.screen_capture = screenshot.Screenshot()
        self.config = None
        
    def run_full_calibration(self, save_config: bool = True) -> Optional[Dict]:
        """
        Run complete calibration process.
        
        Returns:
            Configuration dictionary or None if failed
        """
        print("=" * 60)
        print("CHESS BOARD AUTO-CALIBRATION SYSTEM")
        print("=" * 60)
        print()
        
        # Step 1: Detect board
        print("STEP 1: Detecting chess board...")
        print("Please ensure a chess game is open and visible on your screen.")
        print("Starting detection in 5 seconds...")
        
        for i in range(5, 0, -1):
            print(f"  {i}...")
            time.sleep(1)
        
        board_data = self.detector.find_chess_board()
        
        if not board_data:
            print("❌ CALIBRATION FAILED: Could not detect chess board")
            print("\nTroubleshooting tips:")
            print("1. Make sure a chess game is open (Chess.com or Lichess)")
            print("2. Ensure the board is fully visible and not obstructed")
            print("3. Try adjusting your browser zoom level")
            print("4. Ensure good contrast between board and background")
            return None
        
        print(f"✅ Board detected using {board_data['method']}")
        print(f"   Position: {board_data['position']}")
        print(f"   Confidence: {board_data['confidence']:.3f}")
        print()
        
        # Step 2: Map coordinates
        print("STEP 2: Mapping UI element coordinates...")
        self.mapper.set_board_position(board_data)
        
        # Capture current screenshot for analysis
        screenshot_img = self.screen_capture.capture()
        if screenshot_img is None:
            print("❌ Failed to capture screenshot for coordinate mapping")
            return None
        
        # Step 3: Detect optimal template scale with state-specific analysis
        print("STEP 3: Detecting optimal template scale and state-specific coordinates...")
        optimal_scale = self.mapper.auto_detect_template_scale(screenshot_img)
        
        # Step 4: Generate configuration
        print("STEP 4: Generating configuration...")
        config = self.mapper.generate_coordinate_config(screenshot_img)
        config['template_scale'] = optimal_scale
        
        # Step 5: Validate configuration
        print("STEP 5: Validating configuration...")
        validation_results = self.validate_configuration(config, screenshot_img)
        
        if validation_results['success']:
            print("✅ Configuration validation passed")
            print(f"   Clock detection success rate: {validation_results['clock_success_rate']:.1%}")
        else:
            print("⚠️  Configuration validation had issues")
            print(f"   Clock detection success rate: {validation_results['clock_success_rate']:.1%}")
            if validation_results['clock_success_rate'] < 0.5:
                print("   Warning: Low clock detection rate - results may be unreliable")
        
        # Step 6: Save configuration
        if save_config:
            print("STEP 6: Saving configuration...")
            self.mapper.save_config(config, "auto_calibration_config.json")
            print("✅ Configuration saved")
        
        self.config = config
        
        print()
        print("=" * 60)
        print("CALIBRATION COMPLETE")
        print("=" * 60)
        self.print_calibration_summary(config)
        
        return config
    
    def validate_configuration(self, config: Dict, screenshot_img: np.ndarray) -> Dict:
        """
        Validate the generated configuration by testing clock detection.
        
        Returns:
            Dictionary with validation results
        """
        validation = {
            'success': False,
            'clock_success_rate': 0.0,
            'tested_positions': 0,
            'successful_reads': 0,
            'errors': []
        }
        
        try:
            # Test clock positions
            for clock_type in ['bottom_clock', 'top_clock']:
                if clock_type in config['ui_elements']:
                    for state, coords in config['ui_elements'][clock_type].items():
                        validation['tested_positions'] += 1
                        
                        # Extract clock region
                        x, y = coords['x'], coords['y']
                        w, h = coords['width'], coords['height']
                        
                        # Ensure coordinates are within screenshot bounds
                        if (x >= 0 and y >= 0 and 
                            x + w <= screenshot_img.shape[1] and 
                            y + h <= screenshot_img.shape[0]):
                            
                            clock_region = screenshot_img[y:y+h, x:x+w]
                            
                            # Test clock reading
                            try:
                                result = simple_read_clock(clock_region)
                                if result is not None:
                                    validation['successful_reads'] += 1
                            except Exception as e:
                                validation['errors'].append(f"Clock read error at {clock_type}/{state}: {str(e)}")
                        else:
                            validation['errors'].append(f"Clock position {clock_type}/{state} out of bounds")
            
            # Calculate success rate
            if validation['tested_positions'] > 0:
                validation['clock_success_rate'] = validation['successful_reads'] / validation['tested_positions']
                validation['success'] = validation['clock_success_rate'] > 0.3  # At least 30% success rate
            
        except Exception as e:
            validation['errors'].append(f"Validation error: {str(e)}")
        
        return validation
    
    def test_clock_detection(self, config: Optional[Dict] = None) -> Dict:
        """
        Test clock detection using current or provided configuration.
        
        Returns:
            Dictionary with test results
        """
        if config is None:
            config = self.config
        
        if not config:
            return {'error': 'No configuration available'}
        
        print("Testing clock detection...")
        
        screenshot_img = self.screen_capture.capture()
        if screenshot_img is None:
            return {'error': 'Failed to capture screenshot'}
        
        results = {}
        
        for clock_type in ['bottom_clock', 'top_clock']:
            if clock_type in config['ui_elements']:
                results[clock_type] = {}
                
                for state, coords in config['ui_elements'][clock_type].items():
                    x, y = coords['x'], coords['y']
                    w, h = coords['width'], coords['height']
                    
                    # Extract and test clock region
                    clock_region = screenshot_img[y:y+h, x:x+w]
                    
                    try:
                        time_value = simple_read_clock(clock_region)
                        results[clock_type][state] = {
                            'success': time_value is not None,
                            'value': time_value,
                            'position': (x, y, w, h)
                        }
                    except Exception as e:
                        results[clock_type][state] = {
                            'success': False,
                            'error': str(e),
                            'position': (x, y, w, h)
                        }
        
        return results
    
    def print_calibration_summary(self, config: Dict):
        """Print a summary of the calibration results."""
        print("\nCalibration Summary:")
        print(f"Board Position: {config['board_detection']['position']}")
        print(f"Detection Method: {config['board_detection']['method']}")
        print(f"Confidence: {config['board_detection']['confidence']:.3f}")
        print(f"Template Scale: {config['template_scale']:.2f}")
        
        # Show state-specific analysis if available
        if 'state_analysis' in config:
            print(f"State Analysis: {config['state_analysis']['method']}")
            print(f"Game States Detected: {', '.join(config['state_analysis']['states_detected'])}")
        
        print()
        
        print("UI Element Positions (with state variations):")
        for element_type, element_data in config['ui_elements'].items():
            print(f"  {element_type}:")
            if isinstance(element_data, dict):
                for sub_type, coords in element_data.items():
                    if isinstance(coords, dict) and 'x' in coords:
                        print(f"    {sub_type}: ({coords['x']}, {coords['y']}) [{coords['width']}x{coords['height']}]")
                    else:
                        print(f"    {sub_type}: {coords}")
            else:
                print(f"    {element_data}")
                
        print("\n✅ Configuration accounts for UI position shifts during different game states")
        print("   (start1/start2 for new games, end1/end2/end3 for game over, play for normal gameplay)")
    
    def create_test_interface(self):
        """Create an interactive test interface for the calibration."""
        if not self.config:
            print("No configuration loaded. Please run calibration first.")
            return
        
        print("\n" + "=" * 60)
        print("INTERACTIVE TEST INTERFACE")
        print("=" * 60)
        print()
        print("Available commands:")
        print("  'test' - Test clock detection with current configuration")
        print("  'validate' - Validate configuration accuracy")
        print("  'save' - Save current configuration")
        print("  'show' - Show configuration summary")
        print("  'quit' - Exit")
        print()
        
        while True:
            try:
                command = input("Enter command: ").strip().lower()
                
                if command == 'quit':
                    break
                elif command == 'test':
                    results = self.test_clock_detection()
                    self._print_test_results(results)
                elif command == 'validate':
                    screenshot_img = self.screen_capture.capture()
                    validation = self.validate_configuration(self.config, screenshot_img)
                    self._print_validation_results(validation)
                elif command == 'save':
                    filename = input("Enter filename (default: auto_calibration_config.json): ").strip()
                    if not filename:
                        filename = "auto_calibration_config.json"
                    self.mapper.save_config(self.config, filename)
                elif command == 'show':
                    self.print_calibration_summary(self.config)
                else:
                    print("Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("Goodbye!")
    
    def _print_test_results(self, results: Dict):
        """Print formatted test results."""
        if 'error' in results:
            print(f"Test failed: {results['error']}")
            return
        
        print("\nClock Detection Test Results:")
        for clock_type, clock_data in results.items():
            print(f"\n{clock_type.replace('_', ' ').title()}:")
            for state, result in clock_data.items():
                status = "✅" if result['success'] else "❌"
                if result['success']:
                    time_str = f"{result['value']//60}:{result['value']%60:02d}" if result['value'] else "Unknown"
                    print(f"  {status} {state}: {time_str}")
                else:
                    error = result.get('error', 'Unknown error')
                    print(f"  {status} {state}: {error}")
    
    def _print_validation_results(self, validation: Dict):
        """Print formatted validation results."""
        print(f"\nValidation Results:")
        print(f"Success Rate: {validation['clock_success_rate']:.1%}")
        print(f"Tested Positions: {validation['tested_positions']}")
        print(f"Successful Reads: {validation['successful_reads']}")
        print(f"Overall Success: {'✅' if validation['success'] else '❌'}")
        
        if validation['errors']:
            print("\nErrors encountered:")
            for error in validation['errors']:
                print(f"  ❌ {error}")


def main():
    """Main function for running calibration."""
    calibrator = AutoCalibrator()
    
    # Run full calibration
    config = calibrator.run_full_calibration()
    
    if config:
        # Start interactive test interface
        calibrator.create_test_interface()
    else:
        print("\nCalibration failed. Please check the troubleshooting tips above.")


if __name__ == "__main__":
    main()
