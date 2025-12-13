#!/usr/bin/env python3
"""
Utility functions for auto-calibration that don't depend on loading all piece images.
"""

import cv2
import numpy as np
from fastgrab import screenshot
import sys
from pathlib import Path

# Add parent directories to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.append(str(parent_dir))

def remove_background_colours(img, thresh=1.04):
    """Remove background colours from image (copied from image_scrape_utils)."""
    res = img * np.expand_dims((np.abs(img[:,:,0]/(img[:,:,1]+10**(-10))-1) < thresh-1), -1)
    res = res * np.expand_dims((np.abs(img[:,:,0]/(img[:,:,2]+10**(-10))-1) < thresh-1), -1)
    res = res * np.expand_dims((np.abs(img[:,:,1]/(img[:,:,2]+10**(-10))-1) < thresh-1), -1)
    res = res.astype(np.uint8)
    
    # Turn image grey scale
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    return res

def load_digit_templates():
    """Load digit templates for clock reading."""
    digit_templates = []
    
    # Try to load digit images
    for digit in range(10):
        digit_path = parent_dir / "chessimage" / f"{digit}.png"
        if digit_path.exists():
            img = cv2.imread(str(digit_path))
            if img is not None:
                processed = remove_background_colours(img, thresh=1.6).astype(np.uint8)
                digit_templates.append(processed)
            else:
                # Create a placeholder if image can't be loaded
                digit_templates.append(np.zeros((44, 30), dtype=np.uint8))
        else:
            # Create a placeholder if image doesn't exist
            digit_templates.append(np.zeros((44, 30), dtype=np.uint8))
    
    return np.stack(digit_templates, axis=0)

def simple_read_clock(clock_image):
    """
    Simplified clock reading function that doesn't require the full image_scrape_utils.
    Returns True if the region looks like it contains a clock, False otherwise.
    """
    try:
        if clock_image is None or clock_image.size == 0:
            return None
            
        # Convert to grayscale if needed
        if clock_image.ndim == 3:
            image = remove_background_colours(clock_image, thresh=1.6).astype(np.uint8)
        else:
            image = clock_image.copy()
        
        # Simple check: look for clock-like patterns
        # Clocks should have some dark regions (text) and reasonable contrast
        mean_val = np.mean(image)
        std_val = np.std(image)
        
        # Check if there's reasonable contrast (indicating text)
        if std_val > 20 and mean_val > 50:  # Some contrast and not too dark
            # Look for horizontal structures that might be digits
            height, width = image.shape
            if width > 80 and height > 20:  # Reasonable clock dimensions
                return True
        
        return None
        
    except Exception:
        return None

# Create screen capture instance
SCREEN_CAPTURE = screenshot.Screenshot()
