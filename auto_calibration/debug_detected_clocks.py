#!/usr/bin/env python3
"""
Debug script to examine the detected clock positions and see why OCR is failing.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

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

def test_detected_positions():
    """Test the specific positions that were detected to see what's actually there."""
    
    # Positions from the latest calibration output
    detected_positions = [
        ('top_play', (1351, 386, 147, 44)),
        ('top_start1', (1407, 386, 147, 44)),
        ('top_start2', (1463, 386, 147, 44)),
        ('top_end1', (1519, 386, 147, 44)),
        ('bottom_play', (1375, 730, 147, 44)),
        ('bottom_start1', (1431, 730, 147, 44)),
    ]
    
    # Take a screenshot
    from calibration_utils import SCREEN_CAPTURE
    screenshot = SCREEN_CAPTURE.capture()
    
    if screenshot is None:
        print("❌ Could not capture screenshot")
        return
    
    print("Testing detected clock positions...")
    print("=" * 50)
    
    for name, (x, y, w, h) in detected_positions:
        print(f"\nTesting {name} at ({x}, {y}):")
        
        # Extract the region
        clock_region = screenshot[y:y+h, x:x+w]
        
        # Save the image for inspection
        debug_filename = f"debug_detected_{name.replace('_', '-')}.png"
        cv2.imwrite(debug_filename, clock_region)
        print(f"  Saved: {debug_filename}")
        
        # Try OCR
        try:
            ocr_result = read_clock(clock_region)
            if ocr_result is not None:
                print(f"  ✅ OCR SUCCESS: {ocr_result} seconds")
                minutes = ocr_result // 60
                seconds = ocr_result % 60
                print(f"      = {minutes:02d}:{seconds:02d}")
            else:
                print(f"  ❌ OCR FAILED: read_clock returned None")
        except Exception as e:
            print(f"  ❌ OCR ERROR: {e}")
        
        # Analyze the image
        print(f"  Image stats: shape={clock_region.shape}")
        if len(clock_region.shape) == 3:
            gray = cv2.cvtColor(clock_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = clock_region
            
        print(f"  Grayscale: min={gray.min()}, max={gray.max()}, mean={gray.mean():.1f}, std={gray.std():.1f}")
        
        # Check if it looks like it has text
        variance = np.var(gray)
        print(f"  Variance: {variance:.1f} {'(text-like)' if variance > 50 else '(uniform)'}")

def test_known_working_positions():
    """Test positions that should definitely work if there's a chess game open."""
    print("\n" + "=" * 50)
    print("Testing known chess game positions...")
    print("=" * 50)
    
    # Try the standard capture functions
    test_functions = [
        ('bottom_clock_play', lambda: capture_bottom_clock('play')),
        ('top_clock_play', lambda: capture_top_clock('play')),
        ('bottom_clock_start1', lambda: capture_bottom_clock('start1')),
        ('top_clock_start1', lambda: capture_top_clock('start1')),
    ]
    
    for name, capture_func in test_functions:
        print(f"\nTesting {name}:")
        try:
            clock_img = capture_func()
            if clock_img is not None:
                # Save for inspection
                debug_filename = f"debug_known_{name}.png"
                cv2.imwrite(debug_filename, clock_img)
                print(f"  Captured: {clock_img.shape} -> {debug_filename}")
                
                # Try OCR
                ocr_result = read_clock(clock_img)
                if ocr_result is not None:
                    print(f"  ✅ OCR SUCCESS: {ocr_result} seconds")
                    minutes = ocr_result // 60
                    seconds = ocr_result % 60
                    print(f"      = {minutes:02d}:{seconds:02d}")
                else:
                    print(f"  ❌ OCR FAILED: read_clock returned None")
            else:
                print(f"  ❌ CAPTURE FAILED: function returned None")
        except Exception as e:
            print(f"  ❌ ERROR: {e}")

def analyze_ocr_requirements():
    """Analyze what the read_clock function expects."""
    print("\n" + "=" * 50)
    print("OCR Requirements Analysis")
    print("=" * 50)
    
    print("The read_clock function expects:")
    print("• Image size: exactly 147x44 pixels")
    print("• Format: MM:SS (4 digits)")
    print("• Digit regions:")
    print("  - d1 (minute tens): columns 0:30")
    print("  - d2 (minute ones): columns 34:64") 
    print("  - d3 (second tens): columns 83:113")
    print("  - d4 (second ones): columns 117:147")
    print("• Template matching threshold: >= 0.5")
    print("• Valid time range: 0-3600 seconds")

if __name__ == "__main__":
    analyze_ocr_requirements()
    test_detected_positions()
    test_known_working_positions()
    
    print("\n" + "=" * 50)
    print("DEBUG SUMMARY")
    print("=" * 50)
    print("Check the saved images to see what the detection is actually finding:")
    print("• debug_detected_*.png - regions found by the detector")
    print("• debug_known_*.png - regions from standard capture functions")
    print("\nIf OCR is failing on all regions, the issue is likely:")
    print("1. Clock format has changed (different font/layout)")
    print("2. Template images are outdated") 
    print("3. Clock regions are showing different content (e.g., '--:--', game time, etc.)")
    print("4. Coordinate system has shifted")
