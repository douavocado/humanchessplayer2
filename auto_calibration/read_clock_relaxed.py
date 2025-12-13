#!/usr/bin/env python3
"""
Relaxed version of read_clock with adjustable threshold.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

# Change to parent directory for relative path imports
original_cwd = os.getcwd()
try:
    os.chdir(parent_dir)
    from chessimage.image_scrape_utils import remove_background_colours, TEMPLATES
finally:
    os.chdir(original_cwd)

def multitemplate_match_relaxed(img, templates, threshold=0.45):
    """Relaxed template matching with lower threshold."""
    T = templates.astype(float)
    I = img.astype(float)
    w, h = img.shape
    T_primes = T- np.expand_dims(1/(w*h)*T.sum(axis=(1,2)), (-1,-2))
    I_prime = I - np.expand_dims(1/(w*h)*I.sum(), (-1))
    
    T_denom = (T_primes**2).sum(axis=(1,2))
    I_denom = (I_prime**2).sum()
    denoms = np.sqrt(T_denom*I_denom) + 10**(-10)
    nums = (T_primes*np.expand_dims(I_prime,0)).sum(axis=(1,2))
    
    scores =  nums/denoms
    # Relaxed threshold
    if scores.max() < threshold:
        return None
    return scores.argmax()

def read_clock_relaxed(clock_image, threshold=0.45):
    """Relaxed clock reading with lower template matching threshold."""
    # assumes image is black and white
    if clock_image.ndim== 3:
        image = remove_background_colours(clock_image, thresh=1.6).astype(np.uint8)
    else:
        image = clock_image.copy()
        
    d1 = image[:, :30]
    d2 = image[:, 34:64]
    d3 = image[:, 83:113]
    d4 = image[:, 117:147]

    digit_1 = multitemplate_match_relaxed(d1, TEMPLATES, threshold)
    digit_2 = multitemplate_match_relaxed(d2, TEMPLATES, threshold)
    digit_3 = multitemplate_match_relaxed(d3, TEMPLATES, threshold)
    digit_4 = multitemplate_match_relaxed(d4, TEMPLATES, threshold)
    
    if digit_1 is not None and digit_2 is not None and digit_3 is not None and digit_4 is not None:
        total_seconds = digit_1 * 600 + digit_2*60 + digit_3*10 + digit_4
        # Sanity check - reasonable time values
        if 0 <= total_seconds <= 7200:  # 0 to 2 hours
            return total_seconds
    
    return None
