#!/usr/bin/env python3
"""
Test if downscaling the board image before processing improves speed.
"""

import time
import numpy as np
import sys
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from chessimage.image_scrape_utils import (
    capture_board, STEP, PIECE_STEP, PIECE_TEMPLATES, TEMPLATES,
    remove_background_colours, multitemplate_multimatch, get_fen_from_image
)

def profile(name, func, *args, runs=10, **kwargs):
    times = []
    result = None
    for _ in range(runs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        times.append((time.perf_counter() - start) * 1000)
    print(f"  {name}: {np.mean(times):.2f}ms ± {np.std(times):.2f}ms")
    return result

def main():
    print("=" * 70)
    print("DOWNSCALING TEST")
    print("=" * 70)
    
    board_img = capture_board()
    print(f"\nOriginal board size: {board_img.shape}")
    print(f"Original STEP: {STEP}, PIECE_STEP: {PIECE_STEP}")
    print(f"PIECE_TEMPLATES shape: {PIECE_TEMPLATES.shape}")
    print(f"TEMPLATES (digits) shape: {TEMPLATES.shape}")
    
    # Original FEN extraction
    print("\n1. ORIGINAL (full resolution)")
    print("-" * 50)
    profile("get_fen_from_image (original)", get_fen_from_image, board_img, bottom='w')
    
    # Test at different scale factors
    for scale in [0.75, 0.5, 0.25]:
        print(f"\n2. DOWNSCALED TO {int(scale*100)}%")
        print("-" * 50)
        
        # Downscale board image
        new_size = (int(board_img.shape[1] * scale), int(board_img.shape[0] * scale))
        scaled_img = cv2.resize(board_img, new_size, interpolation=cv2.INTER_AREA)
        print(f"  Scaled board size: {scaled_img.shape}")
        
        scaled_step = int(STEP * scale)
        scaled_piece_step = int(PIECE_STEP * scale)
        print(f"  Scaled STEP: {scaled_step}, PIECE_STEP: {scaled_piece_step}")
        
        # Process at lower resolution
        def process_scaled():
            # Remove background
            gray = remove_background_colours(scaled_img).astype(np.uint8)
            
            # Extract pieces (need to resize each piece to match template)
            template_h, template_w = PIECE_TEMPLATES.shape[1:3]
            images = []
            for x in range(8):
                for y in range(8):
                    piece = gray[x*scaled_step:x*scaled_step+scaled_piece_step, 
                                y*scaled_step:y*scaled_step+scaled_piece_step]
                    # Resize to match template size
                    piece_resized = cv2.resize(piece, (template_w, template_h), 
                                               interpolation=cv2.INTER_AREA)
                    images.append(piece_resized)
            images = np.stack(images, axis=0)
            return multitemplate_multimatch(images, PIECE_TEMPLATES)
        
        profile("Scaled processing", process_scaled)
        
        # Breakdown
        def just_bg_removal():
            return remove_background_colours(scaled_img)
        profile("  - Background removal only", just_bg_removal)
    
    # Test just template matching time at original vs scaled piece sizes
    print("\n3. TEMPLATE MATCHING ANALYSIS")
    print("-" * 50)
    
    template_h, template_w = PIECE_TEMPLATES.shape[1:3]
    print(f"  Template size: {template_w}x{template_h}")
    
    # At original STEP
    gray = remove_background_colours(board_img).astype(np.uint8)
    images_orig = [gray[x*STEP:x*STEP+PIECE_STEP, y*STEP:y*STEP+PIECE_STEP] 
                   for x in range(8) for y in range(8)]
    images_orig = np.stack(images_orig, axis=0)
    print(f"  Original piece size: {images_orig.shape[1]}x{images_orig.shape[2]}")
    
    # Need to resize for template matching
    def resize_and_match_orig():
        resized = np.array([cv2.resize(img, (template_w, template_h)) for img in images_orig])
        return multitemplate_multimatch(resized, PIECE_TEMPLATES)
    
    profile("Template match (resized pieces)", resize_and_match_orig)
    
    # Compare: resize board first then extract
    def resize_board_first():
        # Resize to make STEP match template size
        target_size = template_w * 8
        scaled = cv2.resize(gray, (target_size, target_size), interpolation=cv2.INTER_AREA)
        step = template_w
        images = [scaled[x*step:x*step+step, y*step:y*step+step] 
                  for x in range(8) for y in range(8)]
        return multitemplate_multimatch(np.stack(images, axis=0), PIECE_TEMPLATES)
    
    profile("Template match (resize board)", resize_board_first)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
The 4K resolution (1656x1656 board) is ~4x larger than typical 1080p (828x828).
This directly impacts the remove_background_colours() function which processes
every pixel.

Possible solutions:
1. Downscale the board image before processing (50% scale = 4x faster bg removal)
2. Resize entire board to fixed size matching templates before extraction
3. Run at lower screen resolution
""")

if __name__ == "__main__":
    main()

