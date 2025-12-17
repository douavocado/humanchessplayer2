#!/usr/bin/env python3
"""
Debug script to test game end detection on an offline screenshot.
This helps diagnose why game end wasn't detected.
"""
import sys
import os
import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auto_calibration.config import get_config

def test_game_end_detection(screenshot_path: str):
    """Test game end detection on a screenshot."""
    print(f"\n{'='*60}")
    print(f"Testing game end detection on: {screenshot_path}")
    print(f"{'='*60}\n")
    
    # Load the screenshot
    screenshot = cv2.imread(screenshot_path)
    if screenshot is None:
        print(f"ERROR: Could not load screenshot from {screenshot_path}")
        return
    
    print(f"Screenshot dimensions: {screenshot.shape[1]}x{screenshot.shape[0]}")
    
    # Get calibration config
    config = get_config()
    coords = config.get_coordinates()
    
    print(f"\n--- Calibration Info ---")
    print(f"Board: x={coords['board']['x']}, y={coords['board']['y']}, "
          f"w={coords['board']['width']}, h={coords['board']['height']}")
    
    # Result region
    result_region = coords.get('result_region', {})
    print(f"\nResult Region: {result_region}")
    
    if result_region:
        x, y = result_region['x'], result_region['y']
        w, h = result_region['width'], result_region['height']
        
        # Extract result region from screenshot
        result_img = screenshot[y:y+h, x:x+w]
        
        # Save for inspection
        output_dir = "scripts/debug_output"
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(f"{output_dir}/captured_result_region.png", result_img)
        print(f"\nCaptured result region saved to: {output_dir}/captured_result_region.png")
        print(f"Captured region dimensions: {result_img.shape[1]}x{result_img.shape[0]}")
        
        # Load reference images
        blackwin_ref = cv2.imread("chessimage/blackwin_result.png")
        whitewin_ref = cv2.imread("chessimage/whitewin_result.png")
        draw_ref = cv2.imread("chessimage/draw_result.png")
        
        print(f"\n--- Reference Images ---")
        if blackwin_ref is not None:
            print(f"blackwin_result.png: {blackwin_ref.shape[1]}x{blackwin_ref.shape[0]}")
        else:
            print("blackwin_result.png: NOT FOUND")
            
        if whitewin_ref is not None:
            print(f"whitewin_result.png: {whitewin_ref.shape[1]}x{whitewin_ref.shape[0]}")
        else:
            print("whitewin_result.png: NOT FOUND")
            
        if draw_ref is not None:
            print(f"draw_result.png: {draw_ref.shape[1]}x{draw_ref.shape[0]}")
        else:
            print("draw_result.png: NOT FOUND")
        
        # Test comparison
        print(f"\n--- Comparison Results ---")
        from chessimage.image_scrape_utils import compare_result_images
        
        if blackwin_ref is not None:
            score = compare_result_images(result_img, blackwin_ref)
            status = "MATCH" if score > 0.70 else "no match"
            print(f"vs blackwin (0-1): {score:.4f} - {status}")
            
        if whitewin_ref is not None:
            score = compare_result_images(result_img, whitewin_ref)
            status = "MATCH" if score > 0.70 else "no match"
            print(f"vs whitewin (1-0): {score:.4f} - {status}")
            
        if draw_ref is not None:
            score = compare_result_images(result_img, draw_ref)
            status = "MATCH" if score > 0.70 else "no match"
            print(f"vs draw (1/2-1/2): {score:.4f} - {status}")
    
    # Test clock regions
    print(f"\n--- Clock Regions ---")
    bottom_clock = coords.get('bottom_clock', {})
    
    for state in ['play', 'end1', 'end2', 'end3']:
        if state in bottom_clock:
            clock = bottom_clock[state]
            x, y, w, h = clock['x'], clock['y'], clock['width'], clock['height']
            
            if y + h <= screenshot.shape[0] and x + w <= screenshot.shape[1]:
                clock_img = screenshot[y:y+h, x:x+w]
                cv2.imwrite(f"{output_dir}/bottom_clock_{state}.png", clock_img)
                
                # Try to read the clock
                from chessimage.image_scrape_utils import read_clock
                time_val = read_clock(clock_img)
                print(f"bottom_clock[{state}]: y={y}, time={time_val}")
            else:
                print(f"bottom_clock[{state}]: y={y} - OUT OF BOUNDS")
    
    # Search for "0-1" text in the screenshot
    print(f"\n--- Searching for result text in screenshot ---")
    
    # The result "0-1" should be visible somewhere in the right panel
    # Let's check the notation area
    notation = coords.get('notation', {})
    if notation:
        x, y = notation['x'], notation['y']
        w, h = notation['width'], notation['height']
        if y + h <= screenshot.shape[0] and x + w <= screenshot.shape[1]:
            notation_img = screenshot[y:y+h, x:x+w]
            cv2.imwrite(f"{output_dir}/notation_region.png", notation_img)
            print(f"Notation region saved: x={x}, y={y}, w={w}, h={h}")
    
    # Also capture a larger area around where the result should be
    # Based on the screenshot, the result appears below the notation
    # Let's capture a wider area to find it
    board = coords['board']
    board_right = board['x'] + board['width']
    
    # Capture the entire right panel area
    panel_x = board_right + 10
    panel_y = 0
    panel_w = min(400, screenshot.shape[1] - panel_x)
    panel_h = screenshot.shape[0]
    
    if panel_x < screenshot.shape[1]:
        panel_img = screenshot[panel_y:panel_y+panel_h, panel_x:panel_x+panel_w]
        cv2.imwrite(f"{output_dir}/right_panel.png", panel_img)
        print(f"Right panel saved: x={panel_x}, y={panel_y}")
    
    print(f"\n{'='*60}")
    print("Debug output saved to scripts/debug_output/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        screenshot_path = sys.argv[1]
    else:
        screenshot_path = "auto_calibration/offline_screenshots/end_resigns.png"
    
    test_game_end_detection(screenshot_path)
