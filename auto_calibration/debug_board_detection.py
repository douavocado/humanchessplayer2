#!/usr/bin/env python3
"""
Debug script for board detection.
Visualises each step of the colour-based detection process.
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from calibration_utils import SCREEN_CAPTURE
from board_detector import BoardDetector

# Output directory
DEBUG_DIR = Path(__file__).parent / "debug_outputs"
DEBUG_DIR.mkdir(exist_ok=True)


def debug_board_detection(search_region=None):
    """Run board detection with full debug output."""
    
    print("=" * 60)
    print("BOARD DETECTION DEBUG (Colour-based)")
    print("=" * 60)
    
    if search_region:
        print(f"Search region: {search_region}")
    
    # Step 1: Capture screenshot
    print("\nStep 1: Capturing screenshot...")
    screenshot = SCREEN_CAPTURE.capture()
    
    if screenshot is None:
        print("ERROR: Failed to capture screenshot!")
        return
    
    print(f"Screenshot shape: {screenshot.shape}")
    
    # Convert BGRA to BGR if needed
    if screenshot.shape[2] == 4:
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
    
    cv2.imwrite(str(DEBUG_DIR / "01_screenshot.png"), screenshot)
    print(f"Saved: {DEBUG_DIR / '01_screenshot.png'}")
    
    # Step 2: Apply search region if specified
    if search_region:
        sx, sy, sw, sh = search_region
        # Clamp to image bounds
        sx = max(0, sx)
        sy = max(0, sy)
        sw = min(sw, screenshot.shape[1] - sx)
        sh = min(sh, screenshot.shape[0] - sy)
        search_image = screenshot[sy:sy+sh, sx:sx+sw]
        offset_x, offset_y = sx, sy
        print(f"\nStep 2: Using search region ({sx}, {sy}) [{sw}x{sh}]")
        cv2.imwrite(str(DEBUG_DIR / "02_search_region.png"), search_image)
        print(f"Saved: {DEBUG_DIR / '02_search_region.png'}")
    else:
        search_image = screenshot
        offset_x, offset_y = 0, 0
        print("\nStep 2: Searching entire image")
    
    # Step 3: Create detector and find regions
    print("\nStep 3: Running board detector...")
    detector = BoardDetector(search_region=search_region)
    
    # Get candidates
    candidates = detector.find_checkerboard_regions(screenshot, search_region)
    
    print(f"\nFound {len(candidates)} candidate regions")
    
    # Visualise all candidates
    if candidates:
        candidates_img = screenshot.copy()
        
        for i, c in enumerate(candidates[:10]):  # Show top 10
            x, y, w, h = c['bbox']
            score = c['score']
            
            # Colour based on score (green = high, red = low)
            green = int(255 * score)
            red = int(255 * (1 - score))
            colour = (0, green, red)
            
            cv2.rectangle(candidates_img, (x, y), (x+w, y+h), colour, 2)
            
            # Add label
            label = f"{i+1}: {score:.2f}"
            cv2.putText(candidates_img, label, (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
            
            # Print details
            details = c.get('details', {})
            print(f"\n  Candidate {i+1}: ({x}, {y}) [{w}x{h}]")
            print(f"    Score: {score:.3f}")
            if details:
                print(f"    Light mean: {details.get('light_mean', 0):.1f}, Dark mean: {details.get('dark_mean', 0):.1f}")
                print(f"    Mean diff: {details.get('mean_diff', 0):.1f}")
                print(f"    Light std: {details.get('light_std', 0):.1f}, Dark std: {details.get('dark_std', 0):.1f}")
                print(f"    Separation: {details.get('separation_score', 0):.2f}, Uniformity: {details.get('uniformity_score', 0):.2f}")
        
        cv2.imwrite(str(DEBUG_DIR / "03_candidates.png"), candidates_img)
        print(f"\nSaved: {DEBUG_DIR / '03_candidates.png'}")
        
        # Extract and save best candidate region
        if candidates:
            best = candidates[0]
            x, y, w, h = best['bbox']
            board_region = screenshot[y:y+h, x:x+w]
            cv2.imwrite(str(DEBUG_DIR / "04_best_candidate.png"), board_region)
            print(f"Saved: {DEBUG_DIR / '04_best_candidate.png'}")
            
            # Visualise the 8x8 grid on best candidate
            grid_img = board_region.copy()
            square_size = w // 8
            for i in range(9):
                # Vertical lines
                cv2.line(grid_img, (i * square_size, 0), (i * square_size, h), (0, 0, 255), 1)
                # Horizontal lines
                cv2.line(grid_img, (0, i * square_size), (w, i * square_size), (0, 0, 255), 1)
            cv2.imwrite(str(DEBUG_DIR / "05_best_with_grid.png"), grid_img)
            print(f"Saved: {DEBUG_DIR / '05_best_with_grid.png'}")
            
            # Show cell sampling
            gray = cv2.cvtColor(board_region, cv2.COLOR_BGR2GRAY)
            sample_img = board_region.copy()
            
            for row in range(8):
                for col in range(8):
                    cy = row * square_size + square_size // 2
                    cx = col * square_size + square_size // 2
                    
                    # Expected colour based on chess pattern
                    if (row + col) % 2 == 0:
                        colour = (0, 255, 0)  # Green for light squares
                    else:
                        colour = (255, 0, 0)  # Blue for dark squares
                    
                    cv2.circle(sample_img, (cx, cy), 3, colour, -1)
            
            cv2.imwrite(str(DEBUG_DIR / "06_cell_sampling.png"), sample_img)
            print(f"Saved: {DEBUG_DIR / '06_cell_sampling.png'}")
    else:
        print("\n⚠️ No candidates found!")
        print("\nPossible issues:")
        print("  1. Board is not visible in the search region")
        print("  2. Board colours don't have enough contrast")
        print("  3. Board size is outside expected range (300-1200px)")
        
        # Save search region for manual inspection
        if search_region:
            print(f"\nCheck {DEBUG_DIR / '02_search_region.png'} to verify the board is visible")
    
    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print(f"Check images in: {DEBUG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug board detection")
    parser.add_argument("-l", "--left-monitor", action="store_true",
                        help="Search only the left monitor (first 1920 pixels)")
    parser.add_argument("-r", "--region", type=str, default=None,
                        help="Search region as 'x,y,width,height'")
    parser.add_argument("-w", "--monitor-width", type=int, default=1920,
                        help="Monitor width for --left-monitor")
    parser.add_argument("-s", "--scale", type=float, default=1.0,
                        help="Screenshot scale factor (e.g., 1.5 if HiDPI scaling affects capture)")
    
    args = parser.parse_args()
    
    search_region = None
    if args.region:
        parts = [int(x.strip()) for x in args.region.split(',')]
        if len(parts) == 4:
            search_region = tuple(parts)
    elif args.left_monitor:
        # Apply scale factor for HiDPI/mixed scaling setups
        scaled_width = int(args.monitor_width * args.scale)
        scaled_height = int(3000 * args.scale)
        search_region = (0, 0, scaled_width, scaled_height)
        if args.scale != 1.0:
            print(f"Using scale {args.scale}x: logical {args.monitor_width}px -> {scaled_width}px in screenshot")
    
    debug_board_detection(search_region)
