#!/usr/bin/env python3
"""
Test relaxed thresholds to see if current clocks can be detected.
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
    from chessimage.image_scrape_utils import (
        read_clock, remove_background_colours, TEMPLATES
    )
finally:
    os.chdir(original_cwd)

def multitemplate_match_with_threshold(img, templates, threshold=0.5):
    """Modified version that accepts custom threshold."""
    # assumes img is 2 dimensional WxH and templates is 3 dimensional i.e. NxWxH where N is the number of templates
    # assumes that img and template are the same shape
    T = templates.astype(float)
    I = img.astype(float)
    w, h = img.shape
    T_primes = T- np.expand_dims(1/(w*h)*T.sum(axis=(1,2)), (-1,-2))
    I_prime = I - np.expand_dims(1/(w*h)*I.sum(), (-1))
    
    T_denom = (T_primes**2).sum(axis=(1,2))
    I_denom = (I_prime**2).sum()
    denoms = np.sqrt(T_denom*I_denom) + 10**(-10)
    nums = (T_primes*np.expand_dims(I_prime,0)).sum(axis=(1,2))
    
    scores =  nums/denoms
    # if scores are all low, return None
    if scores.max() < threshold:
        return None, scores.max()
    return scores.argmax(), scores.max()

def read_clock_with_threshold(clock_image, threshold=0.5):
    """Modified read_clock that accepts custom threshold."""
    # assumes image is black and white
    if clock_image.ndim== 3:
        image = remove_background_colours(clock_image, thresh=1.6).astype(np.uint8)
    else:
        image = clock_image.copy()
        
    d1 = image[:, :30]
    d2 = image[:, 34:64]
    d3 = image[:, 83:113]
    d4 = image[:, 117:147]

    digit_1, score_1 = multitemplate_match_with_threshold(d1, TEMPLATES, threshold)
    digit_2, score_2 = multitemplate_match_with_threshold(d2, TEMPLATES, threshold)
    digit_3, score_3 = multitemplate_match_with_threshold(d3, TEMPLATES, threshold)
    digit_4, score_4 = multitemplate_match_with_threshold(d4, TEMPLATES, threshold)
    
    print(f"    Digit scores with threshold {threshold}: {score_1:.3f}, {score_2:.3f}, {score_3:.3f}, {score_4:.3f}")
    
    if digit_1 is not None and digit_2 is not None and digit_3 is not None and digit_4 is not None:
        total_seconds = digit_1 * 600 + digit_2*60 + digit_3*10 + digit_4
        print(f"    SUCCESS: {digit_1}{digit_2}:{digit_3}{digit_4} = {total_seconds} seconds")
        return total_seconds
    else:
        failed_digits = []
        if digit_1 is None: failed_digits.append("d1")
        if digit_2 is None: failed_digits.append("d2") 
        if digit_3 is None: failed_digits.append("d3")
        if digit_4 is None: failed_digits.append("d4")
        print(f"    FAILED: digits {failed_digits} below threshold")
        return None

def test_threshold_sensitivity():
    """Test different thresholds to see if current clocks can be detected."""
    
    from calibration_utils import SCREEN_CAPTURE
    screenshot = SCREEN_CAPTURE.capture()
    
    if screenshot is None:
        print("❌ Could not capture screenshot")
        return
    
    # Test known clock positions
    test_positions = [
        ("bottom_current", (1431, 730)),
        ("top_current", (1407, 386)),
        ("bottom_alt", (1391, 730)),
        ("top_alt", (1367, 386)),
    ]
    
    # Test different thresholds
    thresholds = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]
    
    print("Testing threshold sensitivity...")
    print("=" * 60)
    
    for name, (x, y) in test_positions:
        print(f"\nTesting {name} at ({x}, {y}):")
        
        # Extract region
        clock_region = screenshot[y:y+44, x:x+147]
        
        # Test each threshold
        for threshold in thresholds:
            print(f"  Threshold {threshold}:")
            result = read_clock_with_threshold(clock_region, threshold)
            
            if result is not None:
                minutes = result // 60
                seconds = result % 60
                print(f"    ✅ DETECTED: {minutes:02d}:{seconds:02d}")
                break  # Stop at first success
        else:
            print(f"    ❌ FAILED at all thresholds")

def scan_for_actual_clocks():
    """Scan broader area to find where actual readable clocks might be."""
    
    from calibration_utils import SCREEN_CAPTURE
    screenshot = SCREEN_CAPTURE.capture()
    
    if screenshot is None:
        print("❌ Could not capture screenshot")
        return
    
    print("\nScanning for actual clocks with low threshold...")
    print("=" * 60)
    
    # Scan around the detected board area
    board_x, board_y = 491, 149
    
    # Search right side of board
    search_x_start = board_x + 800  # Right of board
    search_x_end = search_x_start + 400
    search_y_start = board_y
    search_y_end = board_y + 880
    
    successes = []
    
    # Scan with larger steps to cover more ground
    for y in range(search_y_start, search_y_end - 44, 20):
        for x in range(search_x_start, search_x_end - 147, 20):
            clock_region = screenshot[y:y+44, x:x+147]
            
            # Try with very low threshold
            result = read_clock_with_threshold(clock_region, threshold=0.15)
            
            if result is not None and 0 <= result <= 3600:
                minutes = result // 60
                seconds = result % 60
                successes.append((x, y, result, f"{minutes:02d}:{seconds:02d}"))
                print(f"Found clock at ({x}, {y}): {minutes:02d}:{seconds:02d}")
    
    print(f"\nTotal clocks found: {len(successes)}")
    
    if successes:
        print("\nTop candidates:")
        for x, y, total_sec, time_str in successes[:10]:
            print(f"  ({x}, {y}): {time_str}")

if __name__ == "__main__":
    test_threshold_sensitivity()
    scan_for_actual_clocks()
