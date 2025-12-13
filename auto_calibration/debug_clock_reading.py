#!/usr/bin/env python3
"""
Debug script to understand why read_clock is failing.
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
        read_clock, remove_background_colours, capture_bottom_clock, 
        capture_top_clock, multitemplate_match_f, TEMPLATES
    )
finally:
    os.chdir(original_cwd)

def debug_read_clock(clock_image, save_debug=True):
    """
    Debug version of read_clock that shows what's happening at each step.
    """
    print(f"Input image shape: {clock_image.shape}")
    
    # Save original image
    if save_debug:
        cv2.imwrite("debug_original_clock.png", clock_image)
        print("Saved: debug_original_clock.png")
    
    # Step 1: Process image like read_clock does
    if clock_image.ndim == 3:
        processed_image = remove_background_colours(clock_image, thresh=1.6).astype(np.uint8)
        print("Applied remove_background_colours preprocessing")
    else:
        processed_image = clock_image.copy()
        print("Using image as-is (already grayscale)")
    
    if save_debug:
        cv2.imwrite("debug_processed_clock.png", processed_image)
        print("Saved: debug_processed_clock.png")
    
    print(f"Processed image shape: {processed_image.shape}")
    print(f"Processed image stats: min={processed_image.min()}, max={processed_image.max()}, mean={processed_image.mean():.1f}")
    
    # Step 2: Extract digit regions like read_clock does
    regions = {
        'd1': processed_image[:, :30],
        'd2': processed_image[:, 34:64], 
        'd3': processed_image[:, 83:113],
        'd4': processed_image[:, 117:147]
    }
    
    print(f"\nDigit region extraction:")
    for name, region in regions.items():
        print(f"  {name}: shape={region.shape}, stats: min={region.min()}, max={region.max()}, mean={region.mean():.1f}")
        
        if save_debug:
            # Resize for better visibility
            if region.size > 0:
                resized = cv2.resize(region, (60, 88), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(f"debug_{name}_region.png", resized)
    
    # Step 3: Try template matching for each region
    print(f"\nTemplate matching results:")
    digits = []
    
    for i, (name, region) in enumerate(regions.items(), 1):
        if region.size == 0:
            print(f"  {name}: EMPTY REGION")
            digits.append(None)
            continue
            
        try:
            digit = multitemplate_match_f(region, TEMPLATES)
            digits.append(digit)
            
            if digit is not None:
                print(f"  {name}: SUCCESS -> digit {digit}")
            else:
                print(f"  {name}: FAILED (no match above threshold)")
                
        except Exception as e:
            print(f"  {name}: ERROR -> {e}")
            digits.append(None)
    
    # Step 4: Calculate final result
    if all(d is not None for d in digits):
        total_seconds = digits[0] * 600 + digits[1] * 60 + digits[2] * 10 + digits[3]
        print(f"\nSUCCESS: {digits[0]}{digits[1]}:{digits[2]}{digits[3]} = {total_seconds} seconds")
        return total_seconds
    else:
        failed_regions = [f"d{i+1}" for i, d in enumerate(digits) if d is None]
        print(f"\nFAILED: Could not match digits in regions: {failed_regions}")
        return None

def test_clock_reading():
    """Test clock reading with current game state."""
    print("=" * 60)
    print("CLOCK READING DEBUG TEST")
    print("=" * 60)
    
    # Test both clock types
    clock_types = [
        ('bottom_clock', 'play'),
        ('top_clock', 'play'),
        ('bottom_clock', 'start1'),
        ('top_clock', 'start1')
    ]
    
    for clock_type, state in clock_types:
        print(f"\nTesting {clock_type} in {state} state:")
        print("-" * 40)
        
        try:
            if 'bottom' in clock_type:
                clock_img = capture_bottom_clock(state)
            else:
                clock_img = capture_top_clock(state)
            
            if clock_img is not None:
                print(f"Captured {clock_type} image: {clock_img.shape}")
                
                # Try original read_clock
                original_result = read_clock(clock_img)
                print(f"Original read_clock result: {original_result}")
                
                # Try debug version
                print(f"\nDetailed debug analysis:")
                debug_result = debug_read_clock(clock_img, save_debug=True)
                
                print(f"Debug read_clock result: {debug_result}")
                
                # Compare results
                if original_result == debug_result:
                    print("✅ Results match")
                else:
                    print("❌ Results differ!")
                
            else:
                print("❌ Failed to capture clock image")
                
        except Exception as e:
            print(f"❌ Error testing {clock_type}: {e}")
        
        print()

if __name__ == "__main__":
    test_clock_reading()
    
    print("\nDebug images saved in current directory:")
    print("- debug_original_clock.png (captured clock)")
    print("- debug_processed_clock.png (after preprocessing)")  
    print("- debug_d1_region.png through debug_d4_region.png (individual digits)")
    print("\nCheck these images to see what the clock detection is seeing!")
