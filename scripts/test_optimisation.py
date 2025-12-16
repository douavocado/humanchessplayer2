#!/usr/bin/env python3
"""
Test that the optimised FEN extraction still works correctly.
"""

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from chessimage.image_scrape_utils import (
    capture_board, get_fen_from_image, remove_background_colours
)
import numpy as np

def main():
    print("=" * 60)
    print("TESTING OPTIMISED FEN EXTRACTION")
    print("=" * 60)
    
    board_img = capture_board()
    print(f"\nBoard image shape: {board_img.shape}")
    
    # Test fast mode (default)
    print("\n1. Fast mode (downscaled processing):")
    start = time.perf_counter()
    fen_fast = get_fen_from_image(board_img, bottom='w', fast_mode=True)
    time_fast = (time.perf_counter() - start) * 1000
    print(f"   Time: {time_fast:.1f}ms")
    print(f"   FEN: {fen_fast}")
    
    # Test slow mode (original full resolution)
    print("\n2. Slow mode (full resolution):")
    start = time.perf_counter()
    fen_slow = get_fen_from_image(board_img, bottom='w', fast_mode=False)
    time_slow = (time.perf_counter() - start) * 1000
    print(f"   Time: {time_slow:.1f}ms")
    print(f"   FEN: {fen_slow}")
    
    # Compare results
    print("\n3. Comparison:")
    if fen_fast == fen_slow:
        print("   ✅ MATCH - Fast and slow modes produce identical FEN")
    else:
        print("   ⚠️  DIFFERENT - FENs differ between modes")
        print(f"   Fast: {fen_fast}")
        print(f"   Slow: {fen_slow}")
    
    print(f"\n   Speed improvement: {time_slow/time_fast:.1f}x faster")
    
    # Multiple runs to verify consistency
    print("\n4. Consistency test (5 fast extractions):")
    fens = []
    times = []
    for i in range(5):
        board_img = capture_board()
        start = time.perf_counter()
        fen = get_fen_from_image(board_img, bottom='w', fast_mode=True)
        times.append((time.perf_counter() - start) * 1000)
        fens.append(fen)
    
    print(f"   Times: {[f'{t:.1f}ms' for t in times]}")
    print(f"   Mean: {np.mean(times):.1f}ms ± {np.std(times):.1f}ms")
    
    if len(set(fens)) == 1:
        print("   ✅ All 5 extractions produced the same FEN")
    else:
        print("   ⚠️  FEN varied between extractions")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()

