#!/usr/bin/env python3
"""
Chess Board Auto-Calibration Tool

Main calibration tool that detects chess board and generates coordinate configuration.
This tool creates a config file that replaces hardcoded coordinates in the main system.
"""

import json
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from board_detector import BoardDetector
from coordinate_calculator import CoordinateCalculator
from clock_detector import ClockDetector
from calibration_utils import SCREEN_CAPTURE

class ChessBoardCalibrator:
    """Main calibration system for generating device-independent coordinates."""
    
    def __init__(self, search_region=None):
        """
        Initialise calibrator.
        
        Args:
            search_region: Optional (x, y, width, height) to limit board search.
                          Use (0, 0, 1920, 1080) for left monitor only.
        """
        self.board_detector = BoardDetector(search_region=search_region)
        self.clock_detector = ClockDetector()
        self.calculator = CoordinateCalculator()  # Keep for non-clock coordinates
        self.screen_capture = SCREEN_CAPTURE
        self.search_region = search_region
        
    def run_calibration(self, output_file: str = "chess_config.json") -> Optional[Dict]:
        """
        Run complete calibration process and save configuration.
        
        Args:
            output_file: Path to save the configuration file
            
        Returns:
            Configuration dictionary or None if failed
        """
        print("=" * 60)
        print("CHESS BOARD AUTO-CALIBRATION")
        print("=" * 60)
        print()
        print("This tool will automatically detect your chess board and generate")
        print("device-independent coordinates for clocks and UI elements.")
        print()
        print("Instructions:")
        print("1. Open a chess game in your browser (Chess.com or Lichess)")
        print("2. Make sure the board and clocks are visible")
        print("3. The calibration will start in 5 seconds...")
        print()
        
        # Countdown
        for i in range(5, 0, -1):
            print(f"Starting in {i}...")
            time.sleep(1)
        
        print("\nStep 1: Detecting chess board...")
        
        # Detect board position
        board_data = self.board_detector.find_chess_board()
        if not board_data:
            print("❌ FAILED: Could not detect chess board")
            print("\nTroubleshooting:")
            print("• Ensure a chess game is open and fully visible")
            print("• Try adjusting browser zoom to 100%")
            print("• Make sure the board is not obstructed")
            return None
        
        print(f"✅ Board detected: {board_data['method']} (confidence: {board_data['confidence']:.3f})")
        print(f"   Position: {board_data['position']}")
        
        print("\nStep 2: Detecting clock positions...")
        
        # Set board position for clock detector and detect clocks
        self.clock_detector.set_board_position(board_data)
        clock_data = self.clock_detector.find_clocks()
        
        if not clock_data:
            print("❌ FAILED: Could not detect any clocks")
            print("\nTroubleshooting:")
            print("• Ensure a chess game is OPEN and ACTIVE")
            print("• Clocks must show actual time values (not '--:--')")
            print("• Try during active gameplay when times are displayed")
            print("• Make sure clock area is not obstructed")
            print("• The board detector may need recalibration if coordinates are far off")
            return None
        
        print(f"✅ Clocks detected: {clock_data['total_detections']} positions found")
        for clock in clock_data['clocks']:
            pos = clock['position']
            print(f"   {clock['clock_type']} ({clock['state']}): ({pos[0]}, {pos[1]}) - "
                  f"{clock['time_value']}s (confidence: {clock['confidence']:.3f})")
        
        print("\nStep 3: Calculating remaining UI coordinates...")
        
        # Calculate non-clock coordinates using the existing calculator
        screenshot_img = self.screen_capture.capture()
        if screenshot_img is None:
            print("❌ FAILED: Could not capture screenshot for coordinate calculation")
            return None
        
        self.calculator.set_board_position(board_data)
        base_coordinates = self.calculator.calculate_ui_coordinates(screenshot_img)
        
        # Merge OCR-detected clocks with calculated coordinates
        coordinates = self._merge_clock_coordinates(base_coordinates, clock_data)
        
        print("\nStep 4: Validating coordinates...")
        
        # Validate the final coordinates using OCR-based validation
        validation = self.clock_detector.validate_existing_coordinates(coordinates, screenshot_img)
        
        if validation['overall_success_rate'] < 0.3:
            print(f"⚠️  WARNING: Low success rate ({validation['overall_success_rate']:.1%})")
            print("   The calibration may not be accurate. Consider:")
            print("   • Running calibration during active gameplay")
            print("   • Ensuring clocks are clearly visible")
            print("   • Trying a different browser zoom level")
        else:
            print(f"✅ Validation successful: {validation['overall_success_rate']:.1%} success rate")
        
        print("\nStep 5: Generating configuration...")
        
        # Create final configuration
        config = {
            'calibration_info': {
                'timestamp': datetime.now().isoformat(),
                'board_detection': board_data,
                'clock_detection': {
                    'method': clock_data['detection_method'],
                    'total_detections': clock_data['total_detections'],
                    'clocks_found': len(clock_data['clocks'])
                },
                'validation_success_rate': validation['overall_success_rate'],
                'total_positions_tested': validation['total_tested'],
                'successful_positions': validation['total_successful']
            },
            'coordinates': coordinates,
            'validation_results': validation,
            'clock_detection_results': clock_data
        }
        
        # Save configuration to file
        config_path = Path(__file__).parent / output_file
        try:
            # Convert numpy types to native Python types for JSON serialization
            config_json = self._convert_numpy_types(config)
            with open(config_path, 'w') as f:
                json.dump(config_json, f, indent=2)
            print(f"✅ Configuration saved to: {config_path}")
        except Exception as e:
            print(f"❌ FAILED to save configuration: {e}")
            return None
        
        print("\n" + "=" * 60)
        print("CALIBRATION COMPLETE!")
        print("=" * 60)
        
        self.print_summary(config)
        
        print(f"\nNext steps:")
        print(f"1. The configuration has been saved to: {config_path}")
        print(f"2. Your main.py will now use these coordinates instead of hardcoded values")
        print(f"3. To recalibrate, simply run this tool again")
        
        return config
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        import numpy as np
        
        if isinstance(obj, dict):
            return {self._convert_numpy_types(key): self._convert_numpy_types(value) 
                    for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._convert_numpy_types(obj.tolist())
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    def _merge_clock_coordinates(self, base_coordinates: Dict, clock_data: Dict) -> Dict:
        """
        Merge OCR-detected clock positions with calculated coordinates.
        
        Args:
            base_coordinates: Coordinates calculated by CoordinateCalculator
            clock_data: Clock detection results from ClockDetector
            
        Returns:
            Updated coordinates dictionary with OCR-detected clocks
        """
        coordinates = base_coordinates.copy()
        
        # Initialize clock dictionaries if they don't exist
        if 'bottom_clock' not in coordinates:
            coordinates['bottom_clock'] = {}
        if 'top_clock' not in coordinates:
            coordinates['top_clock'] = {}
        
        # Process each detected clock
        for clock in clock_data['clocks']:
            clock_type = clock['clock_type'] + '_clock'  # Convert 'bottom' -> 'bottom_clock'
            state = clock['state']
            x, y, width, height = clock['position']
            
            # Add the detected clock position
            coordinates[clock_type][state] = {
                'x': x,
                'y': y,
                'width': width,
                'height': height,
                'detection_confidence': clock['confidence'],
                'time_value_detected': clock['time_value']
            }
        
        # Ensure we have all expected states for each clock type
        # If OCR detection missed some states, keep the calculated ones as fallback
        expected_states = ['play', 'start1', 'start2', 'end1', 'end2', 'end3']
        
        for clock_type in ['bottom_clock', 'top_clock']:
            detected_states = set(coordinates[clock_type].keys())
            
            # If we have at least one OCR detection for this clock type,
            # we can estimate missing states based on the pattern
            if detected_states:
                # Find the 'play' state or use the first available as reference
                reference_state = 'play' if 'play' in detected_states else list(detected_states)[0]
                reference_coord = coordinates[clock_type][reference_state]
                
                # Fill in missing states using small Y offsets from reference
                state_offsets = {
                    'play': 0,
                    'start1': 14,
                    'start2': 28, 
                    'end1': 69,
                    'end2': 5,
                    'end3': 34
                }
                
                reference_offset = state_offsets.get(reference_state, 0)
                
                for state in expected_states:
                    if state not in detected_states:
                        # Estimate position based on reference
                        y_offset = state_offsets[state] - reference_offset
                        
                        coordinates[clock_type][state] = {
                            'x': reference_coord['x'],
                            'y': reference_coord['y'] + y_offset,
                            'width': reference_coord['width'],
                            'height': reference_coord['height'],
                            'detection_confidence': 0.5,  # Lower confidence for estimated
                            'estimated': True
                        }
        
        return coordinates
    
    def print_summary(self, config: Dict):
        """Print a summary of calibration results."""
        board_pos = config['calibration_info']['board_detection']['position']
        success_rate = config['calibration_info']['validation_success_rate']
        
        print(f"\nSummary:")
        print(f"• Board Position: ({board_pos[0]}, {board_pos[1]}) [{board_pos[2]}x{board_pos[3]}]")
        print(f"• Board Detection Method: {config['calibration_info']['board_detection']['method']}")
        print(f"• Board Confidence: {config['calibration_info']['board_detection']['confidence']:.3f}")
        
        # Show clock detection info
        if 'clock_detection' in config['calibration_info']:
            clock_info = config['calibration_info']['clock_detection']
            print(f"• Clock Detection Method: {clock_info['method']}")
            print(f"• Clocks Found: {clock_info['clocks_found']} positions")
        
        print(f"• Validation Success Rate: {success_rate:.1%}")
        
        print(f"\nDetected UI Elements:")
        coords = config['coordinates']
        
        # Show board position
        if 'board' in coords:
            b = coords['board']
            print(f"• Board: ({b['x']}, {b['y']}) [{b['width']}x{b['height']}]")
        
        # Show clock positions with OCR info
        if 'bottom_clock' in coords and 'play' in coords['bottom_clock']:
            bc = coords['bottom_clock']['play']
            confidence = bc.get('detection_confidence', 'N/A')
            time_val = bc.get('time_value_detected', 'N/A')
            estimated = " (estimated)" if bc.get('estimated', False) else ""
            print(f"• Bottom Clock (play): ({bc['x']}, {bc['y']}) - "
                  f"confidence: {confidence}, time: {time_val}s{estimated}")
        
        if 'top_clock' in coords and 'play' in coords['top_clock']:
            tc = coords['top_clock']['play']
            confidence = tc.get('detection_confidence', 'N/A')
            time_val = tc.get('time_value_detected', 'N/A')
            estimated = " (estimated)" if tc.get('estimated', False) else ""
            print(f"• Top Clock (play): ({tc['x']}, {tc['y']}) - "
                  f"confidence: {confidence}, time: {time_val}s{estimated}")
        
        # Show state coverage
        bottom_states = len(coords.get('bottom_clock', {}))
        top_states = len(coords.get('top_clock', {}))
        ocr_detected = 0
        estimated = 0
        
        for clock_type in ['bottom_clock', 'top_clock']:
            if clock_type in coords:
                for state_data in coords[clock_type].values():
                    if state_data.get('estimated', False):
                        estimated += 1
                    elif 'detection_confidence' in state_data:
                        ocr_detected += 1
        
        print(f"• Game States: {bottom_states} bottom, {top_states} top positions")
        print(f"• OCR Detected: {ocr_detected}, Estimated: {estimated}")

def main():
    """Main function for running calibration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Chess Board Auto-Calibration Tool")
    parser.add_argument("-o", "--output", type=str, default="chess_config.json",
                        help="Output configuration file name")
    parser.add_argument("-l", "--left-monitor", action="store_true",
                        help="Search only the left monitor (first 1920 pixels, or scaled)")
    parser.add_argument("-r", "--region", type=str, default=None,
                        help="Search region as 'x,y,width,height' (e.g., '0,0,1920,1080')")
    parser.add_argument("-w", "--monitor-width", type=int, default=1920,
                        help="Monitor width for --left-monitor option (default: 1920)")
    parser.add_argument("-H", "--monitor-height", type=int, default=None,
                        help="Monitor height for --left-monitor option (default: full height)")
    parser.add_argument("-s", "--scale", type=float, default=1.0,
                        help="Screenshot scale factor (e.g., 1.5 if HiDPI scaling affects capture)")
    
    args = parser.parse_args()
    
    print("Chess Board Auto-Calibration Tool")
    print("=====================================")
    
    # Determine search region
    search_region = None
    
    if args.region:
        # Parse custom region
        try:
            parts = [int(x.strip()) for x in args.region.split(',')]
            if len(parts) == 4:
                search_region = tuple(parts)
                print(f"Using custom search region: {search_region}")
            else:
                print("Error: --region must be 'x,y,width,height'")
                return 1
        except ValueError:
            print("Error: --region values must be integers")
            return 1
    elif args.left_monitor:
        # Use left monitor
        # Apply scale factor for HiDPI/mixed scaling setups
        scaled_width = int(args.monitor_width * args.scale)
        height = args.monitor_height if args.monitor_height else 3000  # Large default
        scaled_height = int(height * args.scale)
        search_region = (0, 0, scaled_width, scaled_height)
        if args.scale != 1.0:
            print(f"Searching left monitor (logical {args.monitor_width}px, scaled to {scaled_width}px at {args.scale}x)")
        else:
            print(f"Searching left monitor only (first {args.monitor_width} pixels)")
    
    calibrator = ChessBoardCalibrator(search_region=search_region)
    
    try:
        config = calibrator.run_calibration(args.output)
        
        if config:
            print("\n🎉 Calibration successful!")
            return 0
        else:
            print("\n❌ Calibration failed!")
            return 1
            
    except KeyboardInterrupt:
        print("\n\nCalibration cancelled by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Calibration error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
