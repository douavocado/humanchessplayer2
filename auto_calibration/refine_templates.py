#!/usr/bin/env python3
"""
Refine templates by extracting and averaging pieces from screenshots using ground truth FENs.
This creates templates that are perfectly matched to the specific profile's rendering.
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import chess
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from auto_calibration.config import ChessConfig
from auto_calibration.utils import load_image
from chessimage.image_scrape_utils import remove_background_colours, PIECE_STEP

def main():
    parser = argparse.ArgumentParser(description="Refine piece templates using ground truth.")
    parser.add_argument("--screenshots", "-s", type=str, required=True, help="Directory of screenshots.")
    parser.add_argument("--profile", "-p", type=str, required=True, help="Profile name to update.")
    args = parser.parse_args()

    os.environ["HCP_CALIBRATION_PROFILE"] = args.profile
    config = ChessConfig()
    coords = config.get_coordinates()
    board_coords = coords['board']
    bx, by, bw, bh = board_coords['x'], board_coords['y'], board_coords['width'], board_coords['height']
    
    step_x = bw / 8.0
    step_y = bh / 8.0

    screenshots_dir = Path(args.screenshots)
    shots = sorted([p for p in screenshots_dir.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    
    # Dictionary to store accumulated pieces for averaging
    # Keys: piece symbols (K, Q, R, B, N, P, k, q, r, b, n, p)
    accumulators = {s: [] for s in "KQRBNPkqrbnp"}
    
    print(f"Processing {len(shots)} screenshots for profile '{args.profile}'...")

    for shot in shots:
        fen_file = shot.parent / f"{shot.stem}_fen.txt"
        if not fen_file.exists():
            continue
            
        # Determine bottom color from the test logic (best match to GT)
        # For simplicity in template extraction, we'll assume the user knows the bottom 
        # or we'll check common orientations. 
        # Most of your laptop screenshots seem to be bottom='w'
        bottom = 'w'
        
        gt_fen = fen_file.read_text().strip().split()[0]
        board = chess.Board(gt_fen + " w - - 0 1")
        
        img = load_image(str(shot))
        board_img = img[by:by+bh, bx:bx+bw]
        board_proc = remove_background_colours(board_img).astype(np.uint8)
        
        for r in range(8):
            for c in range(8):
                # Map image coords to square index
                if bottom == 'w':
                    idx = (7-r) * 8 + c
                else:
                    idx = r * 8 + (7-c)
                
                piece = board.piece_at(idx)
                if piece:
                    symbol = piece.symbol()
                    
                    # Extract square
                    y1, y2 = int(r * step_y), int((r + 1) * step_y)
                    x1, x2 = int(c * step_x), int((c + 1) * step_x)
                    sq_img = board_proc[y1:y2, x1:x2]
                    
                    # Apply 5% inset (same as production)
                    h, w = sq_img.shape[:2]
                    inset = 0.05
                    iy1, iy2 = int(h * inset), h - int(h * inset)
                    ix1, ix2 = int(w * inset), w - int(w * inset)
                    sq_crop = sq_img[iy1:iy2, ix1:ix2]
                    
                    # Resize to template size
                    sq_resized = cv2.resize(sq_crop, (PIECE_STEP, PIECE_STEP), interpolation=cv2.INTER_AREA)
                    accumulators[symbol].append(sq_resized)

    # Save directory
    output_dir = Path(f"auto_calibration/templates/{args.profile}/pieces")
    output_dir.mkdir(parents=True, exist_ok=True)

    symbol_to_name = {
        'K': 'w_king', 'Q': 'w_queen', 'R': 'w_rook', 'B': 'w_bishop', 'N': 'w_knight', 'P': 'w_pawn',
        'k': 'b_king', 'q': 'b_queen', 'r': 'b_rook', 'b': 'b_bishop', 'n': 'b_knight', 'p': 'b_pawn'
    }

    print("\nSaving refined templates:")
    for symbol, pieces in accumulators.items():
        if not pieces:
            print(f"  - {symbol_to_name[symbol]}: No instances found, skipping.")
            continue
            
        # Average all instances
        avg_piece = np.mean(pieces, axis=0).astype(np.uint8)
        
        # Save
        filename = f"{symbol_to_name[symbol]}.png"
        path = output_dir / filename
        cv2.imwrite(str(path), avg_piece)
        print(f"  - {symbol_to_name[symbol]}: Averaged {len(pieces)} instances -> {path}")

    print("\nRefinement complete! Run your tests again to see the improved accuracy.")

if __name__ == "__main__":
    main()

