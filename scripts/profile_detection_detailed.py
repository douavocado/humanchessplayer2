#!/usr/bin/env python3
"""
Detailed profiling to understand the bottleneck in remove_background_colours.
"""

import time
import numpy as np
import sys
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from chessimage.image_scrape_utils import (
    SCREEN_CAPTURE, capture_board, STEP, PIECE_STEP, START_X, START_Y,
    remove_background_colours
)

def profile(name, func, *args, runs=10, **kwargs):
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        times.append((time.perf_counter() - start) * 1000)
    print(f"  {name}: {np.mean(times):.2f}ms ± {np.std(times):.2f}ms")
    return result

def main():
    print("=" * 70)
    print("DETAILED BOTTLENECK ANALYSIS")
    print("=" * 70)
    
    # Get board image
    board_img = capture_board()
    
    print(f"\n1. IMAGE DIMENSIONS")
    print(f"  Board image shape: {board_img.shape}")
    print(f"  Total pixels: {board_img.shape[0] * board_img.shape[1]:,}")
    print(f"  STEP value: {STEP}")
    print(f"  PIECE_STEP value: {PIECE_STEP}")
    print(f"  START_X: {START_X}, START_Y: {START_Y}")
    
    print(f"\n2. NUMPY/OPENCV INFO")
    print(f"  NumPy version: {np.__version__}")
    print(f"  OpenCV version: {cv2.__version__}")
    print(f"  NumPy config:")
    np.show_config()
    
    print(f"\n3. PROFILING remove_background_colours() STEPS")
    print("-" * 50)
    
    img = board_img.astype(np.float64)  # Ensure float for division
    
    # Step by step breakdown
    def step1():
        return img[:,:,0]/(img[:,:,1]+10**(-10))
    
    def step2(ratio):
        return np.abs(ratio - 1)
    
    def step3(diff, thresh=1.04):
        return diff < (thresh - 1)
    
    def step4(mask, img):
        return img * np.expand_dims(mask, -1)
    
    # Profile each step
    print("\n  Individual operations on board image:")
    ratio1 = profile("Division (channel 0/1)", step1)
    diff1 = profile("Abs difference", step2, ratio1)
    mask1 = profile("Threshold comparison", step3, diff1)
    res1 = profile("Mask multiplication", step4, mask1, img)
    
    # Full function comparison
    print("\n  Full function timings:")
    profile("remove_background_colours (original)", remove_background_colours, board_img)
    
    # Optimized version using vectorized operations
    def remove_background_optimized(img, thresh=1.04):
        """Optimized version using fewer operations."""
        eps = 1e-10
        img_f = img.astype(np.float32)  # Use float32 instead of float64
        
        # Compute all ratios at once
        r0, r1, r2 = img_f[:,:,0], img_f[:,:,1], img_f[:,:,2]
        
        # Combined mask calculation
        t = thresh - 1
        mask = (
            (np.abs(r0/(r1+eps) - 1) < t) &
            (np.abs(r0/(r2+eps) - 1) < t) &
            (np.abs(r1/(r2+eps) - 1) < t)
        )
        
        # Apply mask and convert
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return (gray * mask).astype(np.uint8)
    
    profile("remove_background_colours (optimized)", remove_background_optimized, board_img)
    
    # OpenCV-based version
    def remove_background_opencv(img, thresh=1.04):
        """OpenCV-based version that should be faster."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Low saturation = grayscale (pieces), high saturation = colored (board)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Mask where saturation is low (grayscale-ish pixels)
        mask = hsv[:,:,1] < 30  # Low saturation threshold
        return (gray * mask).astype(np.uint8)
    
    profile("remove_background_colours (opencv hsv)", remove_background_opencv, board_img)
    
    # Simple grayscale comparison
    def remove_background_simple(img, thresh=15):
        """Simple version: keep pixels where R≈G≈B (grayscale)."""
        b, g, r = cv2.split(img)
        # Check if channels are similar
        mask = (
            (np.abs(r.astype(np.int16) - g.astype(np.int16)) < thresh) &
            (np.abs(r.astype(np.int16) - b.astype(np.int16)) < thresh) &
            (np.abs(g.astype(np.int16) - b.astype(np.int16)) < thresh)
        )
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return (gray * mask).astype(np.uint8)
    
    profile("remove_background_colours (simple int)", remove_background_simple, board_img)
    
    print("\n4. COMPARING FULL SCAN WITH DIFFERENT IMPLEMENTATIONS")
    print("-" * 50)
    
    from chessimage.image_scrape_utils import (
        get_fen_from_image, multitemplate_multimatch, PIECE_TEMPLATES
    )
    
    # Original full extraction
    def original_fen():
        return get_fen_from_image(board_img, bottom='w')
    
    profile("Original get_fen_from_image", original_fen)
    
    # With optimized background removal
    def optimized_fen():
        image = remove_background_optimized(board_img).astype(np.uint8)
        images = [image[x*STEP:x*STEP+PIECE_STEP, y*STEP:y*STEP+PIECE_STEP] 
                  for x in range(8) for y in range(8)]
        images = np.stack(images, axis=0)
        return multitemplate_multimatch(images, PIECE_TEMPLATES)
    
    profile("Optimized background removal", optimized_fen)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()

