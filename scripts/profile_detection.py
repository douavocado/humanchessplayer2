#!/usr/bin/env python3
"""
Profiling script to measure timing of board detection operations.
Run this while a chess game is visible on screen.
"""

import time
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chessimage.image_scrape_utils import (
    SCREEN_CAPTURE, capture_board, capture_top_clock, capture_bottom_clock,
    get_fen_from_image, read_clock, check_turn_from_last_moved,
    remove_background_colours, detect_last_move_from_img,
    multitemplate_multimatch, PIECE_TEMPLATES, STEP, PIECE_STEP
)
from common.utils import patch_fens

def profile_function(func, *args, runs=10, **kwargs):
    """Profile a function over multiple runs and return timing stats."""
    times = []
    result = None
    for _ in range(runs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'times': times,
        'result': result
    }

def main():
    print("=" * 70)
    print("BOARD DETECTION PROFILING")
    print("=" * 70)
    print("\nMake sure a chess game is visible on your screen!")
    print("Running 10 iterations of each operation...\n")
    
    # Profile screenshot capture
    print("-" * 50)
    print("1. SCREENSHOT CAPTURE")
    print("-" * 50)
    
    board_capture = profile_function(capture_board)
    print(f"  capture_board():       {board_capture['mean_ms']:7.2f}ms ± {board_capture['std_ms']:.2f}ms  (min: {board_capture['min_ms']:.2f}, max: {board_capture['max_ms']:.2f})")
    
    top_clock_capture = profile_function(capture_top_clock)
    print(f"  capture_top_clock():   {top_clock_capture['mean_ms']:7.2f}ms ± {top_clock_capture['std_ms']:.2f}ms  (min: {top_clock_capture['min_ms']:.2f}, max: {top_clock_capture['max_ms']:.2f})")
    
    bot_clock_capture = profile_function(capture_bottom_clock)
    print(f"  capture_bottom_clock():{bot_clock_capture['mean_ms']:7.2f}ms ± {bot_clock_capture['std_ms']:.2f}ms  (min: {bot_clock_capture['min_ms']:.2f}, max: {bot_clock_capture['max_ms']:.2f})")
    
    total_capture = board_capture['mean_ms'] + top_clock_capture['mean_ms'] + bot_clock_capture['mean_ms']
    print(f"\n  TOTAL CAPTURE TIME:    {total_capture:7.2f}ms")
    
    # Profile image processing
    print("\n" + "-" * 50)
    print("2. IMAGE PROCESSING")
    print("-" * 50)
    
    board_img = board_capture['result']
    top_clock_img = top_clock_capture['result']
    bot_clock_img = bot_clock_capture['result']
    
    # Profile remove_background_colours
    bg_removal = profile_function(remove_background_colours, board_img)
    print(f"  remove_background_colours(): {bg_removal['mean_ms']:7.2f}ms ± {bg_removal['std_ms']:.2f}ms")
    
    # Profile clock reading
    print("\n" + "-" * 50)
    print("3. CLOCK READING")
    print("-" * 50)
    
    top_clock_read = profile_function(read_clock, top_clock_img)
    print(f"  read_clock(top):       {top_clock_read['mean_ms']:7.2f}ms ± {top_clock_read['std_ms']:.2f}ms  (result: {top_clock_read['result']})")
    
    bot_clock_read = profile_function(read_clock, bot_clock_img)
    print(f"  read_clock(bottom):    {bot_clock_read['mean_ms']:7.2f}ms ± {bot_clock_read['std_ms']:.2f}ms  (result: {bot_clock_read['result']})")
    
    total_clock = top_clock_read['mean_ms'] + bot_clock_read['mean_ms']
    print(f"\n  TOTAL CLOCK READ TIME: {total_clock:7.2f}ms")
    
    # Profile FEN extraction
    print("\n" + "-" * 50)
    print("4. FEN EXTRACTION (get_fen_from_image)")
    print("-" * 50)
    
    fen_extraction = profile_function(get_fen_from_image, board_img, bottom='w')
    print(f"  get_fen_from_image():  {fen_extraction['mean_ms']:7.2f}ms ± {fen_extraction['std_ms']:.2f}ms")
    print(f"  Extracted FEN: {fen_extraction['result']}")
    
    # Profile individual steps of FEN extraction
    print("\n  Breaking down FEN extraction:")
    
    # Step 1: Background removal
    def step_bg_removal():
        return remove_background_colours(board_img).astype(np.uint8)
    
    bg_step = profile_function(step_bg_removal)
    print(f"    - Background removal: {bg_step['mean_ms']:7.2f}ms")
    
    # Step 2: Image slicing
    processed_img = bg_step['result']
    def step_slicing():
        images = [processed_img[x*STEP:x*STEP+PIECE_STEP, y*STEP:y*STEP+PIECE_STEP] for x in range(8) for y in range(8)]
        return np.stack(images, axis=0)
    
    slice_step = profile_function(step_slicing)
    print(f"    - Image slicing:      {slice_step['mean_ms']:7.2f}ms")
    
    # Step 3: Template matching
    images = slice_step['result']
    def step_template_match():
        return multitemplate_multimatch(images, PIECE_TEMPLATES)
    
    template_step = profile_function(step_template_match)
    print(f"    - Template matching:  {template_step['mean_ms']:7.2f}ms")
    
    # Profile turn detection
    print("\n" + "-" * 50)
    print("5. TURN DETECTION")
    print("-" * 50)
    
    fen = fen_extraction['result']
    turn_detection = profile_function(check_turn_from_last_moved, fen, board_img, 'w')
    print(f"  check_turn_from_last_moved(): {turn_detection['mean_ms']:7.2f}ms ± {turn_detection['std_ms']:.2f}ms")
    
    last_move_detect = profile_function(detect_last_move_from_img, board_img)
    print(f"  detect_last_move_from_img():  {last_move_detect['mean_ms']:7.2f}ms ± {last_move_detect['std_ms']:.2f}ms")
    
    # Profile patch_fens (if we have two positions)
    print("\n" + "-" * 50)
    print("6. FEN PATCHING")
    print("-" * 50)
    
    # Create dummy FENs for testing
    fen1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    fen2 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    
    patch_fen = profile_function(patch_fens, fen1, fen2)
    print(f"  patch_fens():          {patch_fen['mean_ms']:7.2f}ms ± {patch_fen['std_ms']:.2f}ms")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - ESTIMATED TOTAL TIME PER FULL SCAN")
    print("=" * 70)
    
    total_estimated = (
        total_capture +
        total_clock +
        fen_extraction['mean_ms'] +
        turn_detection['mean_ms'] +
        patch_fen['mean_ms']
    )
    
    print(f"  Screenshot captures:    {total_capture:7.2f}ms")
    print(f"  Clock reading:          {total_clock:7.2f}ms")
    print(f"  FEN extraction:         {fen_extraction['mean_ms']:7.2f}ms")
    print(f"  Turn detection:         {turn_detection['mean_ms']:7.2f}ms")
    print(f"  FEN patching:           {patch_fen['mean_ms']:7.2f}ms")
    print(f"  " + "-" * 35)
    print(f"  TOTAL:                  {total_estimated:7.2f}ms")
    print(f"  Scans possible per sec: {1000/total_estimated:7.1f}")
    
    # Run a full simulation
    print("\n" + "=" * 70)
    print("FULL SCAN SIMULATION (10 complete cycles)")
    print("=" * 70)
    
    def full_scan_cycle():
        # Capture
        board_img = capture_board()
        top_clock_img = capture_top_clock()
        bot_clock_img = capture_bottom_clock()
        
        # Read clocks
        our_time = read_clock(bot_clock_img)
        opp_time = read_clock(top_clock_img)
        
        # Get FEN
        fen = get_fen_from_image(board_img, bottom='w')
        
        # Check turn
        turn_res = check_turn_from_last_moved(fen, board_img, 'w')
        
        return fen, our_time, opp_time
    
    full_cycle = profile_function(full_scan_cycle)
    print(f"  Full scan cycle:       {full_cycle['mean_ms']:7.2f}ms ± {full_cycle['std_ms']:.2f}ms")
    print(f"  Min: {full_cycle['min_ms']:.2f}ms, Max: {full_cycle['max_ms']:.2f}ms")
    print(f"  Individual times: {[f'{t:.1f}' for t in full_cycle['times']]}")
    
    print("\n" + "=" * 70)
    print("PROFILING COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()

