#!/usr/bin/env python3
"""
Quick Calibration Verification Tool

Tests your calibrated coordinates by attempting to read clocks
and showing you exactly what regions are being captured.
"""

import cv2
import numpy as np
import json
import sys
from pathlib import Path
from calibration_utils import SCREEN_CAPTURE, simple_clock_test, analyze_clock_region_quality

def load_config():
    """Load the calibration configuration."""
    config_path = Path(__file__).parent / "chess_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        print(f"❌ No configuration found at {config_path}")
        print("Run 'python calibrator.py' first!")
        return None

def test_clock_detection():
    """Test clock detection with current calibration."""
    config = load_config()
    if not config:
        return
    
    print("Testing Clock Detection")
    print("=" * 30)
    
    coordinates = config['coordinates']
    screen_capture = SCREEN_CAPTURE
    
    # Test each clock type and state
    for clock_type in ['bottom_clock', 'top_clock']:
        if clock_type in coordinates:
            print(f"\n{clock_type.upper()}:")
            
            for state, coords in coordinates[clock_type].items():
                x, y, w, h = coords['x'], coords['y'], coords['width'], coords['height']
                
                # Capture the clock region
                try:
                    clock_region = screen_capture.capture((x, y, w, h))
                    
                    if clock_region is not None:
                        # Test clock detection
                        can_detect = simple_clock_test(clock_region)
                        confidence = analyze_clock_region_quality(clock_region)
                        
                        status = "✅" if can_detect else "❌"
                        print(f"  {state:8} ({x:4}, {y:3}): {status} confidence={confidence:.3f}")
                        
                        # Save region for inspection
                        filename = f"verify_{clock_type}_{state}.png"
                        save_path = Path(__file__).parent / filename
                        cv2.imwrite(str(save_path), clock_region)
                        
                    else:
                        print(f"  {state:8} ({x:4}, {y:3}): ❌ Failed to capture")
                        
                except Exception as e:
                    print(f"  {state:8} ({x:4}, {y:3}): ❌ Error: {e}")

def show_coordinate_info():
    """Show detailed coordinate information."""
    config = load_config()
    if not config:
        return
    
    print("Calibration Coordinate Information")
    print("=" * 40)
    
    # Show calibration metadata
    if 'calibration_info' in config:
        info = config['calibration_info']
        print(f"Timestamp: {info.get('timestamp', 'Unknown')}")
        print(f"Detection Method: {info.get('board_detection', {}).get('method', 'Unknown')}")
        print(f"Board Confidence: {info.get('board_detection', {}).get('confidence', 0):.3f}")
        print(f"Validation Success Rate: {info.get('validation_success_rate', 0):.1%}")
        print()
    
    coordinates = config['coordinates']
    
    # Show board info
    if 'board' in coordinates:
        board = coordinates['board']
        print(f"BOARD:")
        print(f"  Position: ({board['x']}, {board['y']})")
        print(f"  Size: {board['width']}x{board['height']}")
        print(f"  Step size: {board['width'] // 8}")
        print()
    
    # Show clock coordinates
    for clock_type in ['bottom_clock', 'top_clock']:
        if clock_type in coordinates:
            print(f"{clock_type.upper()}:")
            
            for state, coords in coordinates[clock_type].items():
                print(f"  {state:8}: ({coords['x']:4}, {coords['y']:3}) [{coords['width']}x{coords['height']}]")
            print()
    
    # Show other UI elements
    for element in ['notation', 'rating']:
        if element in coordinates:
            print(f"{element.upper()}:")
            if element == 'rating':
                for rating_type, coords in coordinates[element].items():
                    print(f"  {rating_type:10}: ({coords['x']:4}, {coords['y']:3}) [{coords['width']}x{coords['height']}]")
            else:
                coords = coordinates[element]
                print(f"  Position: ({coords['x']}, {coords['y']}) [{coords['width']}x{coords['height']}]")
            print()

def compare_with_original():
    """Compare calibrated coordinates with original hardcoded values."""
    config = load_config()
    if not config:
        return
    
    print("Comparison with Original Hardcoded Coordinates")
    print("=" * 50)
    
    # Original hardcoded values
    original = {
        'board': {'x': 543, 'y': 179, 'width': 848, 'height': 848},
        'bottom_clock_play': {'x': 1420, 'y': 742},
        'bottom_clock_start1': {'x': 1420, 'y': 756},
        'bottom_clock_start2': {'x': 1420, 'y': 770},
        'top_clock_play': {'x': 1420, 'y': 424},
        'top_clock_start1': {'x': 1420, 'y': 396},
    }
    
    coordinates = config['coordinates']
    
    # Compare board
    if 'board' in coordinates:
        board = coordinates['board']
        orig_board = original['board']
        
        print("BOARD:")
        print(f"  Original: ({orig_board['x']}, {orig_board['y']}) [{orig_board['width']}x{orig_board['height']}]")
        print(f"  Detected: ({board['x']}, {board['y']}) [{board['width']}x{board['height']}]")
        
        dx = board['x'] - orig_board['x']
        dy = board['y'] - orig_board['y']
        dw = board['width'] - orig_board['width']
        dh = board['height'] - orig_board['height']
        
        print(f"  Difference: ({dx:+}, {dy:+}) [{dw:+}x{dh:+}]")
        print()
    
    # Compare clocks
    for clock_type in ['bottom_clock', 'top_clock']:
        if clock_type in coordinates:
            print(f"{clock_type.upper()}:")
            
            for state in ['play', 'start1', 'start2']:
                if state in coordinates[clock_type]:
                    detected = coordinates[clock_type][state]
                    orig_key = f"{clock_type}_{state}"
                    
                    if orig_key in original:
                        orig = original[orig_key]
                        dx = detected['x'] - orig['x']
                        dy = detected['y'] - orig['y']
                        
                        print(f"  {state:8}: Original=({orig['x']}, {orig['y']}) Detected=({detected['x']}, {detected['y']}) Diff=({dx:+}, {dy:+})")
            print()

def main():
    """Main function with command line interface."""
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        print("Chess Calibration Verification Tool")
        print("=" * 35)
        print()
        print("Available modes:")
        print("1. Test clock detection")
        print("2. Show coordinate info") 
        print("3. Compare with original")
        print("4. All of the above")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            mode = "test"
        elif choice == "2": 
            mode = "info"
        elif choice == "3":
            mode = "compare"
        elif choice == "4":
            mode = "all"
        else:
            mode = "test"
    
    if mode == "test" or mode == "all":
        test_clock_detection()
        print()
    
    if mode == "info" or mode == "all":
        show_coordinate_info()
        print()
    
    if mode == "compare" or mode == "all":
        compare_with_original()
    
    print("Verification complete!")
    print(f"Check saved images in: {Path(__file__).parent}")

if __name__ == "__main__":
    main()
