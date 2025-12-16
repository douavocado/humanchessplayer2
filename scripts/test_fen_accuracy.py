#!/usr/bin/env python3
"""
Test FEN extraction accuracy using saved screenshots.
"""

import time
import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from chessimage.image_scrape_utils import (
    get_fen_from_image, get_board_info, remove_background_colours
)
import chess

def extract_board_from_screenshot(screenshot_path):
    """Extract board region from a full screenshot."""
    # Load the screenshot
    full_img = cv2.imread(str(screenshot_path))
    if full_img is None:
        raise ValueError(f"Could not load image: {screenshot_path}")
    
    # Get board coordinates from calibration
    start_x, start_y, step = get_board_info()
    board_size = step * 8
    
    # Extract board region
    board_img = full_img[start_y:start_y+board_size, start_x:start_x+board_size]
    return board_img

def main():
    print("=" * 70)
    print("FEN EXTRACTION ACCURACY TEST")
    print("=" * 70)
    
    # Test with saved screenshots
    screenshot_dir = Path(__file__).parent.parent / "auto_calibration" / "offline_screenshots"
    
    # Test files - use ones that should have pieces on the board
    test_files = [
        "play.png",
        "start1.png",
        "start2.png",
    ]
    
    for filename in test_files:
        filepath = screenshot_dir / filename
        if not filepath.exists():
            print(f"\nSkipping {filename} - file not found")
            continue
        
        print(f"\n{'='*70}")
        print(f"Testing: {filename}")
        print(f"{'='*70}")
        
        try:
            board_img = extract_board_from_screenshot(filepath)
            print(f"Board image shape: {board_img.shape}")
            
            # Test fast mode
            start = time.perf_counter()
            fen_fast = get_fen_from_image(board_img, bottom='w', fast_mode=True)
            time_fast = (time.perf_counter() - start) * 1000
            
            # Test slow mode
            start = time.perf_counter()
            fen_slow = get_fen_from_image(board_img, bottom='w', fast_mode=False)
            time_slow = (time.perf_counter() - start) * 1000
            
            # Display results
            print(f"\nFast mode ({time_fast:.1f}ms):")
            print(f"  FEN: {fen_fast}")
            board_fast = chess.Board(fen_fast)
            print(board_fast)
            
            print(f"\nSlow mode ({time_slow:.1f}ms):")
            print(f"  FEN: {fen_slow}")
            board_slow = chess.Board(fen_slow)
            
            # Compare
            if fen_fast == fen_slow:
                print(f"\n✅ MATCH - Both modes produce identical FEN")
            else:
                print(f"\n⚠️  MISMATCH:")
                print(f"  Fast: {fen_fast}")
                print(f"  Slow: {fen_slow}")
                print("\nFast mode board:")
                print(board_fast)
                print("\nSlow mode board:")
                print(board_slow)
            
            print(f"\nSpeed improvement: {time_slow/time_fast:.2f}x faster")
            
            # Count pieces detected
            piece_count = sum(1 for sq in chess.SQUARES if board_fast.piece_at(sq))
            print(f"Pieces detected: {piece_count}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()

