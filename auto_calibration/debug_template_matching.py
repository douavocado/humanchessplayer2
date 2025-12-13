#!/usr/bin/env python3
"""
Deep debug of template matching to understand why OCR is failing.
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
        read_clock, remove_background_colours, multitemplate_match_f, TEMPLATES
    )
finally:
    os.chdir(original_cwd)

def debug_template_matching_detailed():
    """Debug template matching step by step."""
    
    print("Loading template images...")
    print(f"Templates shape: {TEMPLATES.shape}")
    print(f"Template range: {TEMPLATES.min()} to {TEMPLATES.max()}")
    
    # Save individual templates for inspection
    for i in range(10):
        template = TEMPLATES[i]
        print(f"Digit {i}: shape={template.shape}, range={template.min()}-{template.max()}")
        # Resize for better visibility
        resized = cv2.resize(template, (60, 88), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(f"template_digit_{i}.png", resized)
    
    print("\nTemplate images saved as template_digit_*.png")

def debug_specific_clock_region():
    """Debug a specific clock region step by step."""
    
    # Use one of the known regions
    from calibration_utils import SCREEN_CAPTURE
    screenshot = SCREEN_CAPTURE.capture()
    
    if screenshot is None:
        print("❌ Could not capture screenshot")
        return
    
    # Test known position from calibration
    x, y, w, h = 1431, 730, 147, 44  # bottom_start1 position
    clock_region = screenshot[y:y+h, x:x+w]
    
    print(f"Analyzing clock region at ({x}, {y})...")
    print(f"Original shape: {clock_region.shape}")
    
    # Save original
    cv2.imwrite("debug_step1_original.png", clock_region)
    
    # Step 1: Apply background removal like read_clock does
    if clock_region.ndim == 3:
        processed = remove_background_colours(clock_region, thresh=1.6).astype(np.uint8)
        print("Applied remove_background_colours preprocessing")
    else:
        processed = clock_region.copy()
        print("Using image as-is (already grayscale)")
    
    cv2.imwrite("debug_step2_processed.png", processed)
    print(f"Processed shape: {processed.shape}")
    print(f"Processed range: {processed.min()} to {processed.max()}")
    
    # Step 2: Extract digit regions
    regions = {
        'd1': processed[:, :30],
        'd2': processed[:, 34:64], 
        'd3': processed[:, 83:113],
        'd4': processed[:, 117:147]
    }
    
    print("\nDigit region analysis:")
    for name, region in regions.items():
        print(f"  {name}: shape={region.shape}, range={region.min()}-{region.max()}, mean={region.mean():.1f}")
        
        if region.size > 0:
            # Save enlarged for inspection
            enlarged = cv2.resize(region, (120, 176), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(f"debug_step3_{name}_region.png", enlarged)
            
            # Try template matching with detailed output
            print(f"    Template matching for {name}:")
            try:
                # Manual template matching with detailed scores
                T = TEMPLATES.astype(float)
                I = region.astype(float)
                w_img, h_img = region.shape
                
                if w_img == 0 or h_img == 0:
                    print(f"      Empty region")
                    continue
                
                # Calculate correlation scores manually
                T_primes = T - np.expand_dims(1/(w_img*h_img)*T.sum(axis=(1,2)), (-1,-2))
                I_prime = I - np.expand_dims(1/(w_img*h_img)*I.sum(), (-1))
                
                T_denom = (T_primes**2).sum(axis=(1,2))
                I_denom = (I_prime**2).sum()
                denoms = np.sqrt(T_denom*I_denom) + 1e-10
                nums = (T_primes*np.expand_dims(I_prime,0)).sum(axis=(1,2))
                
                scores = nums/denoms
                
                print(f"      Correlation scores:")
                for digit, score in enumerate(scores):
                    print(f"        Digit {digit}: {score:.4f}")
                
                max_score = scores.max()
                best_digit = scores.argmax()
                
                print(f"      Best match: digit {best_digit} (score: {max_score:.4f})")
                print(f"      Threshold: 0.5 -> {'PASS' if max_score >= 0.5 else 'FAIL'}")
                
                # Official function result
                official_result = multitemplate_match_f(region, TEMPLATES)
                print(f"      Official result: {official_result}")
                
            except Exception as e:
                print(f"      ERROR: {e}")

def check_clock_content():
    """Check what's actually in the clock regions - might not be time at all."""
    
    from calibration_utils import SCREEN_CAPTURE
    screenshot = SCREEN_CAPTURE.capture()
    
    if screenshot is None:
        print("❌ Could not capture screenshot")
        return
    
    # Check a few different positions
    test_positions = [
        ("detected_bottom", (1431, 730, 147, 44)),
        ("detected_top", (1407, 386, 147, 44)),
        ("manual_guess_bottom", (1400, 750, 147, 44)),
        ("manual_guess_top", (1400, 400, 147, 44)),
    ]
    
    print("\nChecking clock region content...")
    print("=" * 40)
    
    for name, (x, y, w, h) in test_positions:
        region = screenshot[y:y+h, x:x+w]
        
        # Convert to grayscale for analysis
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region
        
        # Check for common non-time patterns
        # Solid color (empty space)
        variance = np.var(gray)
        
        # Check for text-like patterns in expected clock positions
        colon_region = gray[:, 65:85]  # Where colon should be
        colon_darkness = colon_region.mean()
        
        print(f"\n{name} at ({x}, {y}):")
        print(f"  Variance: {variance:.1f} {'(has content)' if variance > 20 else '(mostly uniform)'}")
        print(f"  Colon region darkness: {colon_darkness:.1f}")
        print(f"  Overall brightness: {gray.mean():.1f}")
        
        # Save for visual inspection
        cv2.imwrite(f"debug_content_{name}.png", region)
        
        # Check if it might be showing "--:--" or similar
        dark_pixels = np.sum(gray < 128)
        total_pixels = gray.size
        dark_ratio = dark_pixels / total_pixels
        
        print(f"  Dark pixel ratio: {dark_ratio:.3f}")
        
        if dark_ratio < 0.1:
            print("  → Likely empty/blank region")
        elif variance < 50:
            print("  → Likely solid color or minimal content")
        elif dark_ratio > 0.3:
            print("  → Likely has significant dark content (text?)")
        else:
            print("  → Mixed content")

if __name__ == "__main__":
    print("Template Matching Debug")
    print("=" * 40)
    
    debug_template_matching_detailed()
    
    print("\n" + "=" * 40)
    print("Detailed Region Analysis") 
    print("=" * 40)
    
    debug_specific_clock_region()
    
    print("\n" + "=" * 40)
    print("Clock Content Analysis")
    print("=" * 40)
    
    check_clock_content()
    
    print("\n" + "=" * 40)
    print("DEBUG FILES CREATED:")
    print("=" * 40)
    print("• template_digit_*.png - Template images (0-9)")
    print("• debug_step1_original.png - Raw clock region")
    print("• debug_step2_processed.png - After background removal")
    print("• debug_step3_d*_region.png - Individual digit regions")
    print("• debug_content_*.png - Different clock positions")
    print("\nInspect these images to understand why template matching is failing!")
