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


def extract_colour_scheme(board_img: np.ndarray, step: Optional[int] = None) -> Dict:
    """
    Extract the colour scheme from a chess board image.
    
    Samples colours from known empty squares on a starting position to get
    accurate light and dark square colours. Also detects highlight colours
    by looking for deviations from the base colours.
    
    Args:
        board_img: BGR image of the chess board (should be 8x8 squares).
        step: Size of one square in pixels. If None, calculated from image.
    
    Returns:
        Dictionary with colour scheme in BGR format:
        - light_square: [B, G, R]
        - dark_square: [B, G, R]
        - highlight_light: [B, G, R] (estimated)
        - highlight_dark: [B, G, R] (estimated)
        - premove_light: [B, G, R] (estimated)
        - premove_dark: [B, G, R] (estimated)
    """
    if board_img is None or board_img.size == 0:
        return {}
    
    h, w = board_img.shape[:2]
    if step is None:
        step = w // 8
    
    # Define which squares are empty in starting position
    # Rows 2-5 (indices 2, 3, 4, 5) are empty
    # Light squares: (row + col) % 2 == 1 when viewing from white's perspective
    # Dark squares: (row + col) % 2 == 0
    
    light_samples = []
    dark_samples = []
    
    for row in range(2, 6):  # Empty rows
        for col in range(8):
            # Sample from centre of square to avoid edges
            cx = col * step + step // 2
            cy = row * step + step // 2
            
            # Sample a small region around the centre
            margin = step // 6
            region = board_img[cy - margin:cy + margin, cx - margin:cx + margin]
            
            if region.size > 0:
                avg_colour = region.mean(axis=(0, 1)).astype(int).tolist()
                
                # Determine if light or dark square
                # In image coordinates: row 0 is at top (rank 8), col 0 is left (a-file)
                # a8 (row=0, col=0) is a light square
                # So (row + col) % 2 == 0 means LIGHT square
                is_light = (row + col) % 2 == 0
                
                if is_light:
                    light_samples.append(avg_colour)
                else:
                    dark_samples.append(avg_colour)
    
    if not light_samples or not dark_samples:
        return {}
    
    # Average the samples
    light_square = np.mean(light_samples, axis=0).astype(int).tolist()
    dark_square = np.mean(dark_samples, axis=0).astype(int).tolist()
    
    # Estimate highlight colours
    # Lichess highlights shift colours towards yellow/green
    # We approximate by adjusting the base colours
    highlight_light = _estimate_highlight_colour(light_square, is_light=True)
    highlight_dark = _estimate_highlight_colour(dark_square, is_light=False)
    
    # Estimate premove colours (greyish tint)
    premove_light = _estimate_premove_colour(light_square)
    premove_dark = _estimate_premove_colour(dark_square)
    
    return {
        'light_square': light_square,
        'dark_square': dark_square,
        'highlight_light': highlight_light,
        'highlight_dark': highlight_dark,
        'premove_light': premove_light,
        'premove_dark': premove_dark,
    }


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

