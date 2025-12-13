#!/usr/bin/env python3
"""
Debug script to test checkerboard detection at a specific position.
Use this to verify the algorithm works at the known board location.
"""

import cv2
import numpy as np
from pathlib import Path
from calibration_utils import SCREEN_CAPTURE
from board_detector import BoardDetector

DEBUG_DIR = Path(__file__).parent / "debug_outputs"
DEBUG_DIR.mkdir(exist_ok=True)


def test_at_position(x, y, board_size):
    """Test checkerboard detection at a specific position."""
    
    print(f"Testing position ({x}, {y}) with board_size={board_size}")
    
    # Capture screenshot
    screenshot = SCREEN_CAPTURE.capture()
    if screenshot is None:
        print("Failed to capture screenshot")
        return
    
    # Convert BGRA to BGR if needed
    if screenshot.shape[2] == 4:
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
    
    print(f"Screenshot size: {screenshot.shape[1]}x{screenshot.shape[0]}")
    
    # Extract the region
    region = screenshot[y:y+board_size, x:x+board_size]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    
    cv2.imwrite(str(DEBUG_DIR / "test_region.png"), region)
    print(f"Saved region to: {DEBUG_DIR / 'test_region.png'}")
    
    # Calculate square size
    square_size = board_size // 8
    print(f"Square size: {square_size}")
    
    # Sample each cell
    print("\nSampling 8x8 grid:")
    print("=" * 60)
    
    light_values = []
    dark_values = []
    
    # Create visualization
    viz = region.copy()
    
    for row in range(8):
        row_values = []
        for col in range(8):
            # Centre of this square
            cy = row * square_size + square_size // 2
            cx = col * square_size + square_size // 2
            
            # Sample region
            margin = min(5, square_size // 4)
            y1, y2 = cy - margin, cy + margin
            x1, x2 = cx - margin, cx + margin
            
            if y2 > board_size or x2 > board_size:
                row_values.append("---")
                continue
            
            cell_value = np.mean(gray[y1:y2, x1:x2])
            row_values.append(f"{cell_value:5.1f}")
            
            # Draw sampling point
            expected_light = (row + col) % 2 == 0
            if expected_light:
                light_values.append(cell_value)
                colour = (0, 255, 0)  # Green for expected light
            else:
                dark_values.append(cell_value)
                colour = (255, 0, 0)  # Blue for expected dark
            
            cv2.circle(viz, (cx, cy), 3, colour, -1)
        
        print(f"Row {row}: " + " | ".join(row_values))
    
    cv2.imwrite(str(DEBUG_DIR / "test_sampling.png"), viz)
    print(f"\nSaved sampling visualization to: {DEBUG_DIR / 'test_sampling.png'}")
    
    print("\n" + "=" * 60)
    print("STATISTICS:")
    print("=" * 60)
    
    if light_values and dark_values:
        light_mean = np.mean(light_values)
        dark_mean = np.mean(dark_values)
        light_std = np.std(light_values)
        dark_std = np.std(dark_values)
        mean_diff = abs(light_mean - dark_mean)
        
        print(f"Light squares (expected): mean={light_mean:.1f}, std={light_std:.1f}")
        print(f"Dark squares (expected):  mean={dark_mean:.1f}, std={dark_std:.1f}")
        print(f"Mean difference: {mean_diff:.1f}")
        
        # Calculate score components
        if mean_diff < 15:
            separation_score = 0.0
        elif mean_diff < 30:
            separation_score = 0.3
        elif mean_diff < 50:
            separation_score = 0.6
        else:
            separation_score = min(1.0, mean_diff / 80)
        
        avg_std = (light_std + dark_std) / 2
        if avg_std > 50:
            uniformity_score = 0.2
        elif avg_std > 30:
            uniformity_score = 0.5
        elif avg_std > 15:
            uniformity_score = 0.7
        else:
            uniformity_score = 0.9
        
        total_score = separation_score * 0.5 + uniformity_score * 0.3 + 0.2 * 0.9  # colour_score placeholder
        
        print(f"\nSeparation score: {separation_score:.2f}")
        print(f"Uniformity score: {uniformity_score:.2f}")
        print(f"Total score: {total_score:.2f}")
        
        # Check if light/dark assignment is correct
        if light_mean < dark_mean:
            print("\n⚠️  WARNING: Light squares are darker than dark squares!")
            print("   This suggests the (row+col)%2 assignment might be inverted.")
    else:
        print("Not enough values sampled")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test board detection at specific position")
    parser.add_argument("x", type=int, help="X coordinate")
    parser.add_argument("y", type=int, help="Y coordinate")
    parser.add_argument("size", type=int, help="Board size in pixels")
    
    args = parser.parse_args()
    
    test_at_position(args.x, args.y, args.size)
