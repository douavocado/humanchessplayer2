#!/usr/bin/env python3
"""
Utility functions for chess board auto-calibration.
Simplified and optimized for standalone operation.
"""

import cv2
import numpy as np
from fastgrab import screenshot
import sys
from pathlib import Path

# Add parent directory to path to access chessimage
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

def remove_background_colours(img, thresh=1.04):
    """Remove background colours from image."""
    res = img * np.expand_dims((np.abs(img[:,:,0]/(img[:,:,1]+10**(-10))-1) < thresh-1), -1)
    res = res * np.expand_dims((np.abs(img[:,:,0]/(img[:,:,2]+10**(-10))-1) < thresh-1), -1)
    res = res * np.expand_dims((np.abs(img[:,:,1]/(img[:,:,2]+10**(-10))-1) < thresh-1), -1)
    res = res.astype(np.uint8)
    
    # Turn image grey scale
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    return res

def analyze_clock_region_quality(clock_region: np.ndarray) -> float:
    """
    Analyze the quality of a clock region to determine if it likely contains a clock.
    Returns confidence score 0.0-1.0.
    """
    try:
        if clock_region.size == 0:
            return 0.0
        
        # Convert to grayscale if needed
        if len(clock_region.shape) == 3:
            gray = cv2.cvtColor(clock_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = clock_region
        
        # Check for typical clock characteristics
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        
        # Good clocks should have reasonable contrast
        contrast_score = min(std_val / 50.0, 1.0)
        
        # Check for horizontal structures (darker text regions)
        dark_pixels = np.sum(gray < mean_val * 0.8) / gray.size
        text_score = min(dark_pixels / 0.3, 1.0)
        
        # Edge detection to find text-like patterns
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        edge_score = min(edge_density / 0.1, 1.0)
        
        # Combine scores (weighted average)
        total_score = (contrast_score * 0.4 + text_score * 0.3 + edge_score * 0.3)
        
        return total_score
        
    except Exception:
        return 0.0

def simple_clock_test(clock_image):
    """
    Simple test to see if a region looks like it contains a clock.
    Returns True if the region looks like it contains readable time.
    """
    try:
        if clock_image is None or clock_image.size == 0:
            return False
            
        # Convert to grayscale if needed
        if clock_image.ndim == 3:
            image = remove_background_colours(clock_image, thresh=1.6).astype(np.uint8)
        else:
            image = clock_image.copy()
        
        # Simple check: look for clock-like patterns
        mean_val = np.mean(image)
        std_val = np.std(image)
        
        # Check if there's reasonable contrast (indicating text)
        if std_val > 20 and mean_val > 50:
            height, width = image.shape
            if width > 80 and height > 20:  # Reasonable clock dimensions
                return True
        
        return False
        
    except Exception:
        return False

# Create global screen capture instance
SCREEN_CAPTURE = screenshot.Screenshot()
