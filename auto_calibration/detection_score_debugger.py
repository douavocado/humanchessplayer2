#!/usr/bin/env python3
"""
Detailed detection score debugger.
Analyzes each square of a board and shows similarity scores for all piece templates.
"""

import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import chess
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def set_profile_env(profile: str) -> None:
    if profile:
        os.environ["HCP_CALIBRATION_PROFILE"] = profile

def main():
    parser = argparse.ArgumentParser(description="Debug detection scores square by square.")
    parser.add_argument("--screenshots", "-s", type=str, required=True, help="Directory of screenshots.")
    parser.add_argument("--profile", "-p", type=str, help="Calibration profile.")
    parser.add_argument("--limit", "-l", type=int, default=1, help="Limit number of screenshots.")
    parser.add_argument("--square", type=str, help="Debug only a specific square (e.g. e4).")
    parser.add_argument("--no-blur", action="store_true", help="Disable Gaussian blurring for speed comparison.")
    args = parser.parse_args()

    set_profile_env(args.profile)
    
    from auto_calibration.config import ChessConfig
    from auto_calibration.utils import load_image
    from chessimage.image_scrape_utils import (
        remove_background_colours, PIECE_TEMPLATES, INDEX_MAPPER,
        PIECE_STEP, template_match_f, w_rook
    )

    config = ChessConfig()
    coords = config.get_coordinates()
    board_coords = coords['board']
    bx, by, bw, bh = board_coords['x'], board_coords['y'], board_coords['width'], board_coords['height']
    
    # Floating point steps matching image_scrape_utils.py
    step_x = bw / 8.0
    step_y = bh / 8.0

    piece_names = [INDEX_MAPPER[i] for i in range(12)]
    
    screenshots_dir = Path(args.screenshots)
    shots = sorted([p for p in screenshots_dir.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    if args.limit:
        shots = shots[:args.limit]

    for shot in shots:
        print(f"\n{'='*80}")
        print(f"DEBUGGING: {shot.name}")
        print(f"{'='*80}")
        
        img = load_image(str(shot))
        if img is None:
            continue
            
        board_img = img[by:by+bh, bx:bx+bw]
        
        # Load ground truth if exists
        gt_fen = None
        fen_file = shot.parent / f"{shot.stem}_fen.txt"
        if fen_file.exists():
            gt_fen = fen_file.read_text().strip().split()[0]
            gt_board = chess.Board(gt_fen + " w - - 0 1")
        
        # Determine orientation using ground truth if available
        if gt_fen:
            from chessimage.image_scrape_utils import get_fen_from_image
            from auto_calibration.calibration_readback_test import compare_board_positions
            
            # Try both orientations and pick the one that matches ground truth best
            fen_w = get_fen_from_image(board_img, bottom='w', fast_mode=False)
            fen_b = get_fen_from_image(board_img, bottom='b', fast_mode=False)
            
            comp_w = compare_board_positions(fen_w, gt_fen)
            comp_b = compare_board_positions(fen_b, gt_fen)
            
            if comp_w['accuracy'] >= comp_b['accuracy']:
                bottom_color = 'w'
                print(f"Detected orientation: bottom={bottom_color} (w: {comp_w['accuracy']:.1f}%, b: {comp_b['accuracy']:.1f}%)")
            else:
                bottom_color = 'b'
                print(f"Detected orientation: bottom={bottom_color} (w: {comp_w['accuracy']:.1f}%, b: {comp_b['accuracy']:.1f}%)")
        else:
            # Fallback to simple method if no ground truth
            y1, y2 = int(7 * step_y), int(8 * step_y)
            x1, x2 = int(0 * step_x), int(1 * step_x)
            bl_sq = board_img[y1:y2, x1:x2]
            bl_proc = remove_background_colours(bl_sq).astype(np.uint8)
            bl_score = template_match_f(np.expand_dims(cv2.resize(bl_proc, (PIECE_STEP, PIECE_STEP)), 0), w_rook)
            bottom_color = 'w' if bl_score > 0.5 else 'b'
            print(f"Detected orientation: bottom={bottom_color} (fallback method)")

        # Pre-process board
        t0 = time.perf_counter()
        board_proc = remove_background_colours(board_img).astype(np.uint8)
        t_bg = (time.perf_counter() - t0) * 1000
        
        # Template statistics for NCC (same as multitemplate_multimatch)
        T = PIECE_TEMPLATES.astype(np.float32)
        T_means = T.mean(axis=(1, 2), keepdims=True)
        T_primes = T - T_means
        T_denoms = (T_primes ** 2).sum(axis=(1, 2))
        
        t_matching_total = 0
        
        for r in range(8):
            for c in range(8):
                t_sq_start = time.perf_counter()
                # Map row/col to square index and name
                if bottom_color == 'w':
                    idx = (7-r) * 8 + c
                else:
                    idx = r * 8 + (7-c)
                sq_name = chess.square_name(idx)
                
                if args.square and sq_name != args.square.lower():
                    continue
                
                # Extract square
                y1, y2 = int(r * step_y), int((r + 1) * step_y)
                x1, x2 = int(c * step_x), int((c + 1) * step_x)
                sq_img = board_proc[y1:y2, x1:x2]
                
                # Apply 5% inset to remove potential edge artifacts (same as get_fen_from_image)
                h, w = sq_img.shape[:2]
                inset = 0.05
                iy1, iy2 = int(h * inset), h - int(h * inset)
                ix1, ix2 = int(w * inset), w - int(w * inset)
                sq_img = sq_img[iy1:iy2, ix1:ix2]
                
                sq_img_resized = cv2.resize(sq_img, (PIECE_STEP, PIECE_STEP))
                
                I = sq_img_resized.astype(np.float32)
                I_mean = I.mean()
                I_prime = I - I_mean
                I_denom = (I_prime ** 2).sum()
                
                # Compute NCC scores
                num = (T_primes * I_prime).sum(axis=(1, 2))
                denom = np.sqrt(T_denoms * I_denom) + 1e-10
                scores = num / denom
                
                # Brightness/Empty detection metrics (matching production)
                w, h = sq_img_resized.shape
                inset = int(w * 0.1)
                I_crop = I[inset:-inset, inset:-inset]
                n_pixels_crop = I_crop.shape[0] * I_crop.shape[1]
                I_flat = I_crop.reshape(-1)
                I_nonzero_mask = I_flat > 1
                I_nonzero_counts = I_nonzero_mask.sum()
                
                is_empty = I_nonzero_counts < (n_pixels_crop * 0.015)
                I_nonzero_sums = (I_flat * I_nonzero_mask).sum()
                I_piece_brightness = I_nonzero_sums / max(1, I_nonzero_counts)
                
                # Robust Color Detection (matching production logic)
                fill_ratio = I_nonzero_counts / n_pixels_crop
                is_white = (I_piece_brightness > 160) and (fill_ratio > 0.28)
                is_white_pawn_candidate = (I_piece_brightness > 200) and (fill_ratio > 0.15) and (fill_ratio <= 0.28)
                
                # Output info
                gt_piece = gt_board.piece_at(idx).symbol() if gt_fen and gt_board.piece_at(idx) else "."
                
                best_idx = np.argmax(scores)
                best_score = scores[best_idx]
                
                # Robust Color Detection (matching production logic)
                fill_ratio = I_nonzero_counts / n_pixels_crop
                is_actually_white = (I_piece_brightness > 140) and (fill_ratio > 0.20)
                is_white_pawn_candidate = (I_piece_brightness > 140) and (fill_ratio > 0.12) and (fill_ratio <= 0.20)
                
                matched_white_idx = np.argmax(scores[0:6])
                matched_white_score = scores[matched_white_idx]
                
                matched_black_idx = np.argmax(scores[6:12]) + 6
                matched_black_score = scores[matched_black_idx]
                
                is_actually_white = is_actually_white or \
                                   (matched_white_idx == 5 and is_white_pawn_candidate) or \
                                   (matched_white_score > matched_black_score + 0.15)
                
                if is_actually_white:
                    detected_idx = matched_white_idx
                    best_score = matched_white_score
                else:
                    detected_idx = matched_black_idx
                    best_score = matched_black_score
                
                detected_piece = INDEX_MAPPER[detected_idx] if not is_empty and best_score > 0.4 else "."
                
                status = "✅" if detected_piece == gt_piece else "❌"
                
                print(f"[{sq_name}] GT:{gt_piece} DET:{detected_piece} {status}")
                print(f"      Metrics: Empty={is_empty} (dens={I_nonzero_counts/n_pixels_crop:.3f}), White={is_white} (bright={I_piece_brightness:.1f})")
                
                # Sort and show top 5 scores
                top_indices = np.argsort(scores)[::-1][:5]
                score_strs = [f"{INDEX_MAPPER[i]}:{scores[i]:.3f}" for i in top_indices]
                print(f"      Top NCC: {', '.join(score_strs)}")
                print(f"      Raw best by shape: {INDEX_MAPPER[best_idx]} ({scores[best_idx]:.3f})")
                
                t_matching_total += (time.perf_counter() - t_sq_start) * 1000

        print(f"\n{'-'*40}")
        print(f"PROFILING RESULTS (ms):")
        print(f"{'-'*40}")
        print(f"Background Removal: {t_bg:.2f} ms")
        print(f"Gaussian Blur:      {t_blur:.2f} ms")
        print(f"Template Matching:  {t_matching_total:.2f} ms (64 squares)")
        print(f"Total Processing:   {t_bg + t_blur + t_matching_total:.2f} ms")
        print(f"{'-'*40}")

if __name__ == "__main__":
    main()

