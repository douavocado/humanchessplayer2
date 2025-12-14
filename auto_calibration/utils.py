#!/usr/bin/env python3
"""
Shared utilities for auto-calibration.

Provides screen capture, image processing, and helper functions used
across the calibration module.
"""

import cv2
import numpy as np
import sys
from pathlib import Path
from typing import Tuple, Optional, List
from datetime import datetime

# Add parent directory to path for imports
PARENT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PARENT_DIR))

# Screen capture singleton
_screen_capture = None


def get_screen_capture():
    """Get the global screen capture instance."""
    global _screen_capture
    if _screen_capture is None:
        from fastgrab import screenshot
        _screen_capture = screenshot.Screenshot()
    return _screen_capture


def capture_screenshot(region: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
    """
    Capture a screenshot of the screen or a specific region.
    
    Args:
        region: Optional (x, y, width, height) tuple to capture specific region.
                If None, captures the entire screen.
    
    Returns:
        BGR image as numpy array, or None if capture failed.
    """
    try:
        sc = get_screen_capture()
        if region:
            img = sc.capture(region)
        else:
            img = sc.capture()
        
        if img is None:
            return None
        
        # Convert BGRA to BGR if needed
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        return img
    except Exception as e:
        print(f"Screenshot capture failed: {e}")
        return None


def load_image(path: str) -> Optional[np.ndarray]:
    """
    Load an image from file.
    
    Args:
        path: Path to image file.
    
    Returns:
        BGR image as numpy array, or None if loading failed.
    """
    try:
        img = cv2.imread(str(path))
        if img is None:
            print(f"Failed to load image: {path}")
            return None
        return img
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None


def save_image(path: str, image: np.ndarray) -> bool:
    """
    Save an image to file.
    
    Args:
        path: Path to save image.
        image: Image as numpy array.
    
    Returns:
        True if successful, False otherwise.
    """
    try:
        # Ensure parent directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image)
        return True
    except Exception as e:
        print(f"Error saving image {path}: {e}")
        return False


def remove_background_colours(img: np.ndarray, thresh: float = 1.04) -> np.ndarray:
    """
    Remove background colours from image, keeping only near-grey pixels.
    Used for processing clock digits and piece templates.
    
    Args:
        img: BGR image.
        thresh: Threshold for colour similarity (1.0 = exact grey).
    
    Returns:
        Greyscale image with background removed.
    """
    res = img * np.expand_dims((np.abs(img[:, :, 0] / (img[:, :, 1] + 1e-10) - 1) < thresh - 1), -1)
    res = res * np.expand_dims((np.abs(img[:, :, 0] / (img[:, :, 2] + 1e-10) - 1) < thresh - 1), -1)
    res = res * np.expand_dims((np.abs(img[:, :, 1] / (img[:, :, 2] + 1e-10) - 1) < thresh - 1), -1)
    res = res.astype(np.uint8)
    
    # Convert to greyscale
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    return res


def get_timestamp() -> str:
    """Get current timestamp string for filenames."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def ensure_directory(path: str) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_output_directory() -> Path:
    """Get the calibration output directory for this session."""
    base = Path(__file__).parent / "calibration_outputs"
    timestamp = get_timestamp()
    output_dir = base / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_screenshots_directory() -> Path:
    """Get the directory for saving calibration screenshots."""
    screenshots_dir = Path(__file__).parent / "calibration_screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    return screenshots_dir


def extract_region(image: np.ndarray, x: int, y: int, width: int, height: int) -> Optional[np.ndarray]:
    """
    Extract a region from an image with bounds checking.
    
    Args:
        image: Source image.
        x, y: Top-left corner of region.
        width, height: Size of region.
    
    Returns:
        Extracted region, or None if out of bounds.
    """
    img_h, img_w = image.shape[:2]
    
    # Bounds checking
    if x < 0 or y < 0:
        return None
    if x + width > img_w or y + height > img_h:
        return None
    
    return image[y:y+height, x:x+width].copy()


def downscale_image(image: np.ndarray, factor: int = 8) -> np.ndarray:
    """
    Downscale an image by a factor for faster processing.
    
    Args:
        image: Source image.
        factor: Downscale factor (e.g., 8 = 1/8th size).
    
    Returns:
        Downscaled image.
    """
    h, w = image.shape[:2]
    new_w = max(1, w // factor)
    new_h = max(1, h // factor)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def upscale_coordinates(coords: Tuple[int, int, int, int], factor: int = 8) -> Tuple[int, int, int, int]:
    """
    Scale coordinates back up after downscaled detection.
    
    Args:
        coords: (x, y, width, height) in downscaled space.
        factor: The downscale factor that was used.
    
    Returns:
        Coordinates in original image space.
    """
    x, y, w, h = coords
    return (x * factor, y * factor, w * factor, h * factor)


def cluster_values(values: List[int], threshold: int = 10) -> List[List[int]]:
    """
    Cluster nearby integer values together.
    
    Args:
        values: List of integer values.
        threshold: Maximum distance to consider values as same cluster.
    
    Returns:
        List of clusters, where each cluster is a list of values.
    """
    if not values:
        return []
    
    sorted_values = sorted(values)
    clusters = [[sorted_values[0]]]
    
    for val in sorted_values[1:]:
        if val - clusters[-1][-1] <= threshold:
            clusters[-1].append(val)
        else:
            clusters.append([val])
    
    return clusters


def cluster_mean(cluster: List[int]) -> int:
    """Get the mean value of a cluster, rounded to int."""
    return int(round(sum(cluster) / len(cluster)))


# Lichess colour definitions (HSV ranges)
# These are STRICT ranges to avoid picking up move highlights
# Light squares: cream/beige - NOT the yellow/teal highlights
LICHESS_LIGHT_HSV = {
    'lower': np.array([20, 15, 210]),   # Cream/beige - tighter saturation
    'upper': np.array([40, 70, 255])
}

# Dark squares: green - NOT the teal/blue highlights
LICHESS_DARK_HSV = {
    'lower': np.array([40, 50, 100]),   # Green - avoid teal highlights (H < 40)
    'upper': np.array([75, 160, 170])   # Avoid bright highlights
}


def create_board_colour_mask(image: np.ndarray) -> np.ndarray:
    """
    Create a binary mask for Lichess board colours.
    Excludes move highlight colours (yellow/teal).
    
    Args:
        image: BGR image.
    
    Returns:
        Binary mask where board pixels are white.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create masks for light and dark squares
    light_mask = cv2.inRange(hsv, LICHESS_LIGHT_HSV['lower'], LICHESS_LIGHT_HSV['upper'])
    dark_mask = cv2.inRange(hsv, LICHESS_DARK_HSV['lower'], LICHESS_DARK_HSV['upper'])
    
    # Combine masks
    combined = cv2.bitwise_or(light_mask, dark_mask)
    
    # Clean up with morphological operations - use smaller kernel to preserve edges
    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    
    return combined


def create_board_colour_mask_strict(image: np.ndarray) -> np.ndarray:
    """
    Create a strict binary mask using only the most common board colours.
    Uses BGR colour matching for more precision.
    
    Args:
        image: BGR image.
    
    Returns:
        Binary mask where board pixels are white.
    """
    # Lichess standard colours (BGR)
    # Light square: approximately (214, 235, 238) - cream
    # Dark square: approximately (118, 150, 86) - green
    
    # Create masks using colour distance
    light_target = np.array([214, 235, 238])  # BGR
    dark_target = np.array([118, 150, 86])    # BGR
    
    # Calculate distance from target colours
    light_diff = np.sqrt(np.sum((image.astype(float) - light_target) ** 2, axis=2))
    dark_diff = np.sqrt(np.sum((image.astype(float) - dark_target) ** 2, axis=2))
    
    # Threshold - pixels within this distance are considered matches
    threshold = 50
    
    light_mask = (light_diff < threshold).astype(np.uint8) * 255
    dark_mask = (dark_diff < threshold).astype(np.uint8) * 255
    
    # Combine
    combined = cv2.bitwise_or(light_mask, dark_mask)
    
    # Clean up
    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    
    return combined


def find_largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Find the largest contour in a binary mask.
    
    Args:
        mask: Binary mask image.
    
    Returns:
        Largest contour, or None if no contours found.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    return max(contours, key=cv2.contourArea)


def get_bounding_rect(contour: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Get the bounding rectangle of a contour.
    
    Args:
        contour: OpenCV contour.
    
    Returns:
        (x, y, width, height) tuple.
    """
    return cv2.boundingRect(contour)


def is_approximately_square(width: int, height: int, tolerance: float = 0.15) -> bool:
    """
    Check if dimensions are approximately square.
    
    Args:
        width: Width dimension.
        height: Height dimension.
        tolerance: Allowed deviation from 1:1 ratio (0.15 = 15%).
    
    Returns:
        True if approximately square.
    """
    if width == 0 or height == 0:
        return False
    
    ratio = width / height
    return (1 - tolerance) <= ratio <= (1 + tolerance)


def draw_rectangle(image: np.ndarray, x: int, y: int, w: int, h: int,
                   colour: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    """
    Draw a rectangle on an image.
    
    Args:
        image: Image to draw on.
        x, y, w, h: Rectangle coordinates.
        colour: BGR colour tuple.
        thickness: Line thickness.
    
    Returns:
        Image with rectangle drawn.
    """
    result = image.copy()
    cv2.rectangle(result, (x, y), (x + w, y + h), colour, thickness)
    return result


def draw_grid(image: np.ndarray, x: int, y: int, size: int, divisions: int = 8,
              colour: Tuple[int, int, int] = (0, 200, 0), thickness: int = 1) -> np.ndarray:
    """
    Draw a grid overlay on an image.
    
    Args:
        image: Image to draw on.
        x, y: Top-left corner.
        size: Size of grid (assumes square).
        divisions: Number of divisions (8 for chess board).
        colour: BGR colour tuple.
        thickness: Line thickness.
    
    Returns:
        Image with grid drawn.
    """
    result = image.copy()
    step = size // divisions
    
    # Draw vertical lines
    for i in range(divisions + 1):
        x_pos = x + i * step
        cv2.line(result, (x_pos, y), (x_pos, y + size), colour, thickness)
    
    # Draw horizontal lines
    for i in range(divisions + 1):
        y_pos = y + i * step
        cv2.line(result, (x, y_pos), (x + size, y_pos), colour, thickness)
    
    return result


def put_text(image: np.ndarray, text: str, x: int, y: int,
             colour: Tuple[int, int, int] = (255, 255, 255),
             font_scale: float = 0.6, thickness: int = 2,
             background: bool = True) -> np.ndarray:
    """
    Put text on an image with optional background.
    
    Args:
        image: Image to draw on.
        text: Text to draw.
        x, y: Position.
        colour: Text colour.
        font_scale: Font scale.
        thickness: Text thickness.
        background: Whether to draw a background rectangle.
    
    Returns:
        Image with text drawn.
    """
    result = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    if background:
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(result, (x - 2, y - text_h - 2), (x + text_w + 2, y + baseline + 2), (0, 0, 0), -1)
    
    cv2.putText(result, text, (x, y), font, font_scale, colour, thickness)
    return result
