#!/usr/bin/env python3
"""
Colour Extractor - Extract colour scheme from board screenshots.

Extracts the actual board colours from screenshots to enable accurate
detection on displays with different colour profiles.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter


def extract_colour_scheme(
    board_img: np.ndarray, 
    step: Optional[int] = None, 
    fen: Optional[str] = None, 
    bottom: str = 'w',
    highlighted_squares: Optional[List[int]] = None
) -> Dict:
    """
    Extract the colour scheme from a chess board image.
    
    Samples colours from known empty squares on a starting position to get
    accurate light and dark square colours. Also detects highlight colours
    by looking for deviations from the base colours, or by using provided
    ground truth highlighted squares.
    
    Args:
        board_img: BGR image of the chess board (should be 8x8 squares).
        step: Size of one square in pixels. If None, calculated from image.
        fen: Optional FEN to identify empty squares for more robust extraction.
        bottom: 'w' if white is at the bottom, 'b' if black. Used with fen.
        highlighted_squares: Optional list of square indices (0-63) that are 
                            known to be highlighted (ground truth).
    
    Returns:
        Dictionary with colour scheme in BGR format:
        - light_square: [B, G, R]
        - dark_square: [B, G, R]
        - highlight_light: [B, G, R] (extracted or estimated)
        - highlight_dark: [B, G, R] (extracted or estimated)
        - premove_light: [B, G, R] (estimated)
        - premove_dark: [B, G, R] (estimated)
    """
    if board_img is None or board_img.size == 0:
        return {}
    
    h, w = board_img.shape[:2]
    if step is None:
        step = w // 8
    
    # 1. Extract base square colours
    light_samples = []
    dark_samples = []
    
    # helper to convert square index to image (row, col)
    def sq_to_img_coords(sq):
        if bottom == 'w':
            row = 7 - (sq // 8)
            col = sq % 8
        else:
            row = sq // 8
            col = 7 - (sq % 8)
        return row, col

    # If FEN is provided, we can use all empty squares
    if fen:
        import chess
        try:
            board = chess.Board(fen)
            for sq in range(64):
                # Don't sample from squares we know are highlighted
                if highlighted_squares and sq in highlighted_squares:
                    continue
                    
                if board.piece_at(sq) is None:
                    row, col = sq_to_img_coords(sq)
                    
                    cx = col * step + step // 2
                    cy = row * step + step // 2
                    
                    margin = step // 6
                    region = board_img[cy - margin:cy + margin, cx - margin:cx + margin]
                    
                    if region.size > 0:
                        avg_colour = region.mean(axis=(0, 1)).astype(int).tolist()
                        is_light = (row + col) % 2 == 0
                        if is_light:
                            light_samples.append(avg_colour)
                        else:
                            dark_samples.append(avg_colour)
        except Exception:
            pass

    # If no FEN or FEN sampling failed, fall back to standard start position sampling
    if not light_samples or not dark_samples:
        for row in range(2, 6):  # Empty rows in start pos
            for col in range(8):
                cx = col * step + step // 2
                cy = row * step + step // 2
                margin = step // 6
                region = board_img[cy - margin:cy + margin, cx - margin:cx + margin]
                if region.size > 0:
                    avg_colour = region.mean(axis=(0, 1)).astype(int).tolist()
                    is_light = (row + col) % 2 == 0
                    if is_light:
                        light_samples.append(avg_colour)
                    else:
                        dark_samples.append(avg_colour)
    
    if not light_samples or not dark_samples:
        # Last resort: adaptive detection
        adaptive = detect_board_colours_adaptive(board_img)
        if adaptive:
            light_square = adaptive['light_square']
            dark_square = adaptive['dark_square']
        else:
            return {}
    else:
        # Average the samples
        light_square = np.mean(light_samples, axis=0).astype(int).tolist()
        dark_square = np.mean(dark_samples, axis=0).astype(int).tolist()
    
    # 2. Extract highlight colours
    highlight_light = None
    highlight_dark = None
    
    # A. If we have ground truth highlighted squares, use them directly
    if highlighted_squares:
        h_light_samples = []
        h_dark_samples = []
        
        for sq in highlighted_squares:
            row, col = sq_to_img_coords(sq)
            
            # Sample near corners to avoid pieces in the middle
            # (Top-left, top-right, bottom-left, bottom-right)
            margin = step // 8
            corner_offsets = [
                (margin, margin),
                (step - margin, margin),
                (margin, step - margin),
                (step - margin, step - margin)
            ]
            
            square_samples = []
            for ox, oy in corner_offsets:
                px = col * step + ox
                py = row * step + oy
                # Sample a small region around each corner
                r_margin = 2
                region = board_img[max(0, py-r_margin):min(board_img.shape[0], py+r_margin), 
                                   max(0, px-r_margin):min(board_img.shape[1], px+r_margin)]
                if region.size > 0:
                    square_samples.append(region.mean(axis=(0, 1)))
            
            if square_samples:
                # Use the sample that is MOST likely to be a highlight (not a piece)
                # For Lichess highlights, this is usually the most colorful/saturated or simply the brightest
                # for light squares, darkest for dark squares.
                # Let's just average them for now, but corners are much safer than center.
                avg_colour = np.mean(square_samples, axis=0).astype(int).tolist()
                is_light = (row + col) % 2 == 0
                if is_light:
                    h_light_samples.append(avg_colour)
                else:
                    h_dark_samples.append(avg_colour)
        
        if h_light_samples:
            highlight_light = np.mean(h_light_samples, axis=0).astype(int).tolist()
        if h_dark_samples:
            highlight_dark = np.mean(h_dark_samples, axis=0).astype(int).tolist()

    # B. If no ground truth, look for highlight candidates in empty squares (outliers)
    if highlight_light is None or highlight_dark is None:
        def find_best_highlight(samples, base_col):
            if not samples: return None
            base = np.array(base_col)
            dists = [np.sqrt(np.sum((np.array(s) - base)**2)) for s in samples]
            candidates = [(s, d) for s, d in zip(samples, dists) if d > 25]
            if not candidates: return None
            candidates.sort(key=lambda x: x[1], reverse=True)
            for s, d in candidates:
                if s[0] < base_col[0] - 5: # Blue should decrease
                    return s
            return candidates[0][0]

        if highlight_light is None:
            highlight_light = find_best_highlight(light_samples, light_square)
        if highlight_dark is None:
            highlight_dark = find_best_highlight(dark_samples, dark_square)

    # Estimate premove colours (greyish tint)
    premove_light = _estimate_premove_colour(light_square)
    premove_dark = _estimate_premove_colour(dark_square)
    
    res = {
        'light_square': light_square,
        'dark_square': dark_square,
        'premove_light': premove_light,
        'premove_dark': premove_dark,
    }
    
    if highlight_light is not None:
        res['highlight_light'] = highlight_light
    if highlight_dark is not None:
        res['highlight_dark'] = highlight_dark
        
    return res


def _estimate_highlight_colour(base_colour: List[int], is_light: bool) -> List[int]:
    """
    Estimate the highlight colour based on the base square colour.
    
    Lichess uses a yellowish-green overlay for last move highlights.
    """
    b, g, r = base_colour
    
    if is_light:
        # Light squares get a slight yellow-green tint
        # Reduce blue, boost green slightly
        h_b = max(0, int(b * 0.85))
        h_g = min(255, int(g * 1.02))
        h_r = min(255, int(r * 0.95))
    else:
        # Dark squares also get yellow-green shift
        h_b = max(0, int(b * 0.75))
        h_g = min(255, int(g * 1.05))
        h_r = min(255, int(r * 0.85))
    
    return [h_b, h_g, h_r]


def _estimate_premove_colour(base_colour: List[int]) -> List[int]:
    """
    Estimate the premove colour based on the base square colour.
    
    Premoves have a greyish/desaturated tint.
    """
    b, g, r = base_colour
    
    # Desaturate by moving towards grey
    avg = (b + g + r) // 3
    
    p_b = int(b * 0.7 + avg * 0.3)
    p_g = int(g * 0.7 + avg * 0.3)
    p_r = int(r * 0.7 + avg * 0.3)
    
    return [p_b, p_g, p_r]


def extract_highlight_colours_from_move(
    board_img: np.ndarray,
    highlighted_squares: List[int],
    step: Optional[int] = None
) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    """
    Extract actual highlight colours from squares where a move was made.
    
    Args:
        board_img: BGR image of the chess board.
        highlighted_squares: List of square indices (0-63) that are highlighted.
        step: Size of one square in pixels.
    
    Returns:
        Tuple of (highlight_light, highlight_dark) colours, or (None, None) if failed.
    """
    if len(highlighted_squares) != 2:
        return None, None
    
    h, w = board_img.shape[:2]
    if step is None:
        step = w // 8
    
    highlight_colours = []
    
    for sq in highlighted_squares:
        row = sq // 8
        col = sq % 8
        
        cx = col * step + step // 2
        cy = row * step + step // 2
        
        margin = step // 6
        region = board_img[cy - margin:cy + margin, cx - margin:cx + margin]
        
        if region.size > 0:
            avg_colour = region.mean(axis=(0, 1)).astype(int).tolist()
            is_light = (row + col) % 2 == 1
            highlight_colours.append((avg_colour, is_light))
    
    highlight_light = None
    highlight_dark = None
    
    for colour, is_light in highlight_colours:
        if is_light:
            highlight_light = colour
        else:
            highlight_dark = colour
    
    return highlight_light, highlight_dark


def detect_board_colours_adaptive(board_img: np.ndarray) -> Dict:
    """
    Adaptively detect board colours using k-means clustering.
    
    This works even when we don't know the board orientation or
    which squares are empty.
    
    Args:
        board_img: BGR image of the chess board.
    
    Returns:
        Dictionary with light_square and dark_square colours.
    """
    if board_img is None or board_img.size == 0:
        return {}
    
    h, w = board_img.shape[:2]
    step = w // 8
    
    # Sample centre pixels from all squares
    all_colours = []
    
    for row in range(8):
        for col in range(8):
            cx = col * step + step // 2
            cy = row * step + step // 2
            
            margin = step // 8
            region = board_img[cy - margin:cy + margin, cx - margin:cx + margin]
            
            if region.size > 0:
                avg = region.mean(axis=(0, 1))
                all_colours.append(avg)
    
    if len(all_colours) < 16:
        return {}
    
    # Use k-means to find the two dominant colours
    colours_array = np.array(all_colours, dtype=np.float32)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centres = cv2.kmeans(colours_array, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Sort by brightness to determine which is light/dark
    brightness = [sum(c) for c in centres]
    if brightness[0] > brightness[1]:
        light_idx, dark_idx = 0, 1
    else:
        light_idx, dark_idx = 1, 0
    
    return {
        'light_square': centres[light_idx].astype(int).tolist(),
        'dark_square': centres[dark_idx].astype(int).tolist(),
    }


def generate_highlight_colour_array(colour_scheme: Dict, tolerance: int = 15) -> np.ndarray:
    """
    Generate the highlight colour array for move detection.
    
    Creates an array compatible with the vectorised detection in image_scrape_utils.
    
    Args:
        colour_scheme: Colour scheme dictionary with highlight colours.
        tolerance: Tolerance value to include colour variants.
    
    Returns:
        Nx3 numpy array of highlight colours in BGR format.
    """
    colours = []
    
    for key in ['highlight_light', 'highlight_dark', 'premove_light', 'premove_dark']:
        if key in colour_scheme:
            base = colour_scheme[key]
            colours.append(base)
            
            # Add variants with slight shifts for robustness
            for shift in [-5, 5]:
                variant = [max(0, min(255, c + shift)) for c in base]
                colours.append(variant)
    
    if not colours:
        # Return default colours
        return np.array([
            [144, 151, 100],
            [138, 147, 94],
            [205, 209, 177],
            [189, 207, 174],
            [147, 140, 135],
            [170, 165, 160],
        ], dtype=np.int16)
    
    return np.array(colours, dtype=np.int16)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python colour_extractor.py <board_image.png>")
        sys.exit(1)
    
    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Failed to load: {sys.argv[1]}")
        sys.exit(1)
    
    scheme = extract_colour_scheme(img)
    
    if scheme:
        print("Extracted colour scheme (BGR):")
        for key, value in scheme.items():
            print(f"  {key}: {value}")
    else:
        print("Failed to extract colour scheme")
        
        # Try adaptive method
        print("\nTrying adaptive detection...")
        adaptive = detect_board_colours_adaptive(img)
        if adaptive:
            print("Adaptive colour scheme (BGR):")
            for key, value in adaptive.items():
                print(f"  {key}: {value}")

