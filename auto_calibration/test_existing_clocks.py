#!/usr/bin/env python3
"""
Test script to check if read_clock works with existing coordinates.
"""

import os
import sys
from pathlib import Path
import cv2

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

# Change to parent directory for relative path imports
original_cwd = os.getcwd()
try:
    os.chdir(parent_dir)
    from chessimage.image_scrape_utils import read_clock, capture_bottom_clock, capture_top_clock
finally:
    os.chdir(original_cwd)

from calibration_utils import SCREEN_CAPTURE

def test_existing_clock_positions():
    """Test if we can read clocks from existing configured positions."""
    print("Testing existing clock positions...")
    
    # Test bottom clock with different states
    states_to_test = ['play', 'start1', 'start2', 'end1', 'end2', 'end3']
    
    print("\nTesting bottom clock positions:")
    for state in states_to_test:
        try:
            print(f"\nTesting bottom clock state '{state}':")
            
            # Capture clock region using existing function
            clock_img = capture_bottom_clock(state)
            
            if clock_img is not None:
                print(f"  Captured image shape: {clock_img.shape}")
                
                # Try to read the clock
                time_value = read_clock(clock_img)
                
                if time_value is not None:
                    print(f"  ✅ Successfully read time: {time_value} seconds")
                    
                    # Save the image for inspection
                    debug_path = f"debug_bottom_clock_{state}.png"
                    cv2.imwrite(debug_path, clock_img)
                    print(f"  Saved debug image: {debug_path}")
                else:
                    print(f"  ❌ Failed to read time (returned None)")
                    
                    # Save failed image for inspection
                    debug_path = f"debug_bottom_clock_{state}_failed.png"
                    cv2.imwrite(debug_path, clock_img)
                    print(f"  Saved failed image: {debug_path}")
            else:
                print(f"  ❌ Failed to capture clock image")
                
        except Exception as e:
            print(f"  ❌ Error testing state '{state}': {e}")
    
    print("\nTesting top clock positions:")
    for state in states_to_test:
        try:
            print(f"\nTesting top clock state '{state}':")
            
            # Capture clock region using existing function
            clock_img = capture_top_clock(state)
            
            if clock_img is not None:
                print(f"  Captured image shape: {clock_img.shape}")
                
                # Try to read the clock
                time_value = read_clock(clock_img)
                
                if time_value is not None:
                    print(f"  ✅ Successfully read time: {time_value} seconds")
                    
                    # Save the image for inspection
                    debug_path = f"debug_top_clock_{state}.png"
                    cv2.imwrite(debug_path, clock_img)
                    print(f"  Saved debug image: {debug_path}")
                else:
                    print(f"  ❌ Failed to read time (returned None)")
                    
                    # Save failed image for inspection
                    debug_path = f"debug_top_clock_{state}_failed.png"
                    cv2.imwrite(debug_path, clock_img)
                    print(f"  Saved failed image: {debug_path}")
            else:
                print(f"  ❌ Failed to capture clock image")
                
        except Exception as e:
            print(f"  ❌ Error testing state '{state}': {e}")

def test_manual_region():
    """Test clock reading from a manual region near expected positions."""
    print("\n" + "="*50)
    print("Testing manual clock regions...")
    
    # Capture full screenshot
    screenshot = SCREEN_CAPTURE.capture()
    if screenshot is None:
        print("❌ Failed to capture screenshot")
        return
    
    print(f"Screenshot shape: {screenshot.shape}")
    
    # Test some positions around where clocks should be
    test_positions = [
        (1391, 928),  # Bottom clock play from config
        (1391, 393),  # Top clock play from config
        (1380, 920),  # Slightly offset positions
        (1400, 400),
    ]
    
    clock_width, clock_height = 147, 44
    
    for i, (x, y) in enumerate(test_positions):
        print(f"\nTesting manual position {i+1}: ({x}, {y})")
        
        # Extract region
        if (x + clock_width <= screenshot.shape[1] and 
            y + clock_height <= screenshot.shape[0]):
            
            clock_region = screenshot[y:y+clock_height, x:x+clock_width]
            
            try:
                time_value = read_clock(clock_region)
                
                if time_value is not None:
                    print(f"  ✅ Read time: {time_value} seconds")
                    debug_path = f"debug_manual_{i+1}_success.png"
                    cv2.imwrite(debug_path, clock_region)
                    print(f"  Saved: {debug_path}")
                else:
                    print(f"  ❌ No time detected")
                    debug_path = f"debug_manual_{i+1}_failed.png"
                    cv2.imwrite(debug_path, clock_region)
                    print(f"  Saved: {debug_path}")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
        else:
            print(f"  ❌ Position out of bounds")

if __name__ == "__main__":
    print("Clock Reading Test Script")
    print("=" * 40)
    
    test_existing_clock_positions()
    test_manual_region()
    
    print("\n" + "="*50)
    print("Test complete! Check debug images in the current directory.")
