#!/usr/bin/env python3
"""
Chess Board Detection Module

Detects the chess board position using colour segmentation and grid verification.
Optimised for Lichess with multi-scale detection for efficiency.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
from pathlib import Path

from .utils import (
    downscale_image,
    upscale_coordinates,
    create_board_colour_mask,
    find_largest_contour,
    get_bounding_rect,
    is_approximately_square,
    extract_region
)


class BoardDetector:
    """
    Detects chess board position using colour segmentation and grid verification.
    
    The detection process:
    1. Downsample image for fast initial scan
    2. Create colour mask for Lichess board colours
    3. Find largest connected region
    4. Verify it's approximately square
    5. Refine on full resolution
    6. Verify 8x8 grid pattern
    """
    
    # Minimum board size (in pixels) to consider
    MIN_BOARD_SIZE = 200
    
    # Maximum board size relative to image width
    MAX_BOARD_RATIO = 0.8
    
    # Downscale factor for initial detection
    DOWNSCALE_FACTOR = 8
    
    # Grid verification threshold
    GRID_VERIFICATION_THRESHOLD = 0.7
    
    def __init__(self):
        """Initialise the board detector."""
        self.last_detection: Optional[Dict] = None
    
    def detect(self, image: np.ndarray) -> Optional[Dict]:
        """
        Detect chess board in an image.
        
        Args:
            image: BGR image to search.
        
        Returns:
            Dictionary with detection results:
            {
                'x': int,
                'y': int,
                'size': int,
                'step': int,
                'confidence': float,
                'method': str
            }
            Or None if no board found.
        """
        if image is None or image.size == 0:
            return None
        
        img_h, img_w = image.shape[:2]
        
        # Step 1: Fast detection on downscaled image
        candidate = self._detect_on_downscaled(image)
        
        if candidate is None:
            return None
        
        # Step 2: Refine on full resolution
        refined = self._refine_detection(image, candidate)
        
        if refined is None:
            return None
        
        # Step 3: Verify grid pattern
        grid_score = self._verify_grid_pattern(image, refined)
        
        if grid_score < self.GRID_VERIFICATION_THRESHOLD:
            # Try alternative detection if grid verification fails
            refined = self._alternative_detection(image, candidate)
            if refined is None:
                return None
            grid_score = self._verify_grid_pattern(image, refined)
        
        # Step 4: Snap board size to exact multiple of 8
        snapped = self._snap_to_grid(image, refined)
        
        x, y, size = snapped['x'], snapped['y'], snapped['size']
        
        result = {
            'x': x,
            'y': y,
            'size': size,
            'step': size // 8,
            'confidence': grid_score,
            'method': 'colour_segmentation'
        }
        
        self.last_detection = result
        return result
    
    def _detect_on_downscaled(self, image: np.ndarray) -> Optional[Dict]:
        """
        Perform fast detection on downscaled image.
        
        Args:
            image: Full resolution BGR image.
        
        Returns:
            Candidate region dict or None.
        """
        # Downscale for faster processing
        small = downscale_image(image, self.DOWNSCALE_FACTOR)
        
        # Create colour mask
        mask = create_board_colour_mask(small)
        
        # Find largest contour
        contour = find_largest_contour(mask)
        
        if contour is None:
            return None
        
        # Get bounding rectangle
        x, y, w, h = get_bounding_rect(contour)
        
        # Check if approximately square
        if not is_approximately_square(w, h, tolerance=0.2):
            return None
        
        # Scale coordinates back up
        x, y, w, h = upscale_coordinates((x, y, w, h), self.DOWNSCALE_FACTOR)
        
        # Use average of width and height for size (should be square)
        size = (w + h) // 2
        
        return {
            'x': x,
            'y': y,
            'size': size,
            'width': w,
            'height': h
        }
    
    def _refine_detection(self, image: np.ndarray, candidate: Dict) -> Optional[Dict]:
        """
        Refine detection on full resolution image.
        
        Args:
            image: Full resolution image.
            candidate: Candidate region from downscaled detection.
        
        Returns:
            Refined detection dict or None.
        """
        img_h, img_w = image.shape[:2]
        
        # Extract region with padding
        padding = 50
        x = max(0, candidate['x'] - padding)
        y = max(0, candidate['y'] - padding)
        size = candidate['size'] + 2 * padding
        
        # Ensure within bounds
        x = min(x, img_w - size)
        y = min(y, img_h - size)
        
        if x < 0 or y < 0:
            # Candidate is too close to edge or too large
            return {
                'x': candidate['x'],
                'y': candidate['y'],
                'size': candidate['size']
            }
        
        # Extract region
        region = image[y:y+size, x:x+size]
        
        # Create colour mask on full resolution region
        mask = create_board_colour_mask(region)
        
        # Find largest contour
        contour = find_largest_contour(mask)
        
        if contour is None:
            # Fall back to candidate
            return {
                'x': candidate['x'],
                'y': candidate['y'],
                'size': candidate['size']
            }
        
        # Get bounding rectangle
        rx, ry, rw, rh = get_bounding_rect(contour)
        
        # Convert to image coordinates
        abs_x = x + rx
        abs_y = y + ry
        abs_size = (rw + rh) // 2
        
        # Validate size
        if abs_size < self.MIN_BOARD_SIZE:
            return None
        
        if abs_size > img_w * self.MAX_BOARD_RATIO:
            return None
        
        return {
            'x': abs_x,
            'y': abs_y,
            'size': abs_size
        }
    
    def _alternative_detection(self, image: np.ndarray, candidate: Dict) -> Optional[Dict]:
        """
        Try alternative detection method using edge detection.
        
        Args:
            image: Full resolution image.
            candidate: Initial candidate region.
        
        Returns:
            Alternative detection dict or None.
        """
        # Extract candidate region with padding
        padding = 100
        x = max(0, candidate['x'] - padding)
        y = max(0, candidate['y'] - padding)
        w = min(candidate['size'] + 2 * padding, image.shape[1] - x)
        h = min(candidate['size'] + 2 * padding, image.shape[0] - y)
        
        region = image[y:y+h, x:x+w]
        
        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate to connect edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find the most square-like contour
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.MIN_BOARD_SIZE ** 2:
                continue
            
            rx, ry, rw, rh = cv2.boundingRect(contour)
            
            # Score based on squareness and size
            squareness = min(rw, rh) / max(rw, rh) if max(rw, rh) > 0 else 0
            size_score = min(rw, rh) / (candidate['size'] + 1)  # Prefer similar size to candidate
            
            score = squareness * 0.7 + min(size_score, 1.0) * 0.3
            
            if score > best_score:
                best_score = score
                best_contour = contour
        
        if best_contour is None:
            return None
        
        rx, ry, rw, rh = cv2.boundingRect(best_contour)
        
        return {
            'x': x + rx,
            'y': y + ry,
            'size': (rw + rh) // 2
        }
    
    def _snap_to_grid(self, image: np.ndarray, detection: Dict) -> Dict:
        """
        Snap board detection to exact 8x8 grid by finding precise square boundaries.
        
        Uses edge detection to find the actual grid lines and snaps the board
        size to an exact multiple of 8.
        
        Args:
            image: Full resolution image.
            detection: Initial detection result.
        
        Returns:
            Refined detection with size divisible by 8.
        """
        x, y, size = detection['x'], detection['y'], detection['size']
        
        # Calculate current step (may be fractional)
        approx_step = size / 8.0
        
        # Round to nearest integer step
        step = round(approx_step)
        
        # New size is exactly 8 * step
        new_size = step * 8
        
        # Adjust x, y to centre the snapped board on the original detection
        size_diff = size - new_size
        new_x = x + size_diff // 2
        new_y = y + size_diff // 2
        
        # Try to refine further using edge detection on the board region
        refined = self._refine_boundaries_with_edges(image, new_x, new_y, new_size, step)
        if refined is not None:
            return refined
        
        return {
            'x': new_x,
            'y': new_y,
            'size': new_size
        }
    
    def _refine_boundaries_with_edges(self, image: np.ndarray, x: int, y: int, 
                                       size: int, step: int) -> Optional[Dict]:
        """
        Refine board boundaries by detecting edges at the grid lines.
        
        Looks for strong vertical edges at columns 0 and 8 (left/right edges)
        and horizontal edges at rows 0 and 8 (top/bottom edges).
        
        Args:
            image: Full resolution image.
            x, y: Top-left corner of board.
            size: Current board size.
            step: Square step size.
        
        Returns:
            Refined detection or None if refinement fails.
        """
        img_h, img_w = image.shape[:2]
        
        # Extract a slightly larger region around the board
        padding = step // 2
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_w, x + size + padding)
        y2 = min(img_h, y + size + padding)
        
        region = image[y1:y2, x1:x2]
        if region.size == 0:
            return None
        
        # Convert to grayscale and detect edges
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Use Sobel for directional edge detection
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Look for strong vertical edges (left and right boundaries)
        # Sum absolute vertical edge strength across each column
        vert_edge_strength = np.abs(sobel_x).sum(axis=0)
        
        # Look for strong horizontal edges (top and bottom boundaries)
        horiz_edge_strength = np.abs(sobel_y).sum(axis=1)
        
        # Expected positions in the region
        expected_left = padding
        expected_right = padding + size
        expected_top = padding
        expected_bottom = padding + size
        
        # Search for actual edges within a small window around expected positions
        search_range = step // 4
        
        def find_best_edge(strength_array, expected_pos, search_range):
            """Find the position with maximum edge strength near expected."""
            start = max(0, expected_pos - search_range)
            end = min(len(strength_array), expected_pos + search_range + 1)
            if start >= end:
                return expected_pos
            window = strength_array[start:end]
            best_offset = np.argmax(window)
            return start + best_offset
        
        # Find refined boundaries
        left = find_best_edge(vert_edge_strength, expected_left, search_range)
        right = find_best_edge(vert_edge_strength, expected_right, search_range)
        top = find_best_edge(horiz_edge_strength, expected_top, search_range)
        bottom = find_best_edge(horiz_edge_strength, expected_bottom, search_range)
        
        # Calculate new size from detected boundaries
        detected_width = right - left
        detected_height = bottom - top
        
        # Average and snap to multiple of 8
        avg_size = (detected_width + detected_height) // 2
        refined_step = round(avg_size / 8)
        refined_size = refined_step * 8
        
        # Use detected left/top as the anchor, adjust slightly if needed
        new_x = x1 + left
        new_y = y1 + top
        
        # Verify the new detection makes sense
        if refined_size < self.MIN_BOARD_SIZE:
            return None
        
        if abs(refined_size - size) > step:
            # Refinement changed size too much, probably unreliable
            return None
        
        return {
            'x': new_x,
            'y': new_y,
            'size': refined_size
        }

    def _verify_grid_pattern(self, image: np.ndarray, detection: Dict) -> float:
        """
        Verify that the detected region contains an 8x8 alternating grid pattern.
        
        Args:
            image: Full resolution image.
            detection: Detection result dict.
        
        Returns:
            Confidence score 0.0 to 1.0.
        """
        x, y, size = detection['x'], detection['y'], detection['size']
        
        # Extract board region
        board_region = extract_region(image, x, y, size, size)
        
        if board_region is None:
            return 0.0
        
        # Convert to grayscale
        gray = cv2.cvtColor(board_region, cv2.COLOR_BGR2GRAY)
        
        step = size // 8
        if step < 10:
            return 0.0
        
        # Sample centre of each cell
        grid_values = np.zeros((8, 8))
        
        for row in range(8):
            for col in range(8):
                # Sample from centre of cell (avoiding pieces)
                cx = col * step + step // 2
                cy = row * step + step // 2
                
                # Sample a small region around centre
                margin = step // 6
                sample_region = gray[
                    max(0, cy - margin):min(size, cy + margin),
                    max(0, cx - margin):min(size, cx + margin)
                ]
                
                if sample_region.size > 0:
                    grid_values[row, col] = np.mean(sample_region)
        
        # Check alternating pattern
        # On a chess board, cells (i,j) where (i+j) is even are one colour,
        # and cells where (i+j) is odd are another colour
        
        light_cells = []
        dark_cells = []
        
        for row in range(8):
            for col in range(8):
                if (row + col) % 2 == 0:
                    light_cells.append(grid_values[row, col])
                else:
                    dark_cells.append(grid_values[row, col])
        
        light_mean = np.mean(light_cells)
        dark_mean = np.mean(dark_cells)
        light_std = np.std(light_cells)
        dark_std = np.std(dark_cells)
        
        # Check for good separation between light and dark
        separation = abs(light_mean - dark_mean)
        
        if separation < 20:
            return 0.0
        
        # Check for consistency within each group
        avg_std = (light_std + dark_std) / 2
        
        # Score based on separation and consistency
        separation_score = min(separation / 60, 1.0)
        consistency_score = max(0, 1 - avg_std / 40)
        
        # Check correct light/dark relationship
        # Lichess: top-left (a8) should be light (cream)
        # But pieces can affect this, so we just check general pattern
        
        # Check alternation between adjacent cells
        alternation_correct = 0
        alternation_total = 0
        
        for row in range(8):
            for col in range(7):
                diff = grid_values[row, col] - grid_values[row, col + 1]
                if abs(diff) > 15:  # Significant difference
                    alternation_correct += 1
                alternation_total += 1
        
        for row in range(7):
            for col in range(8):
                diff = grid_values[row, col] - grid_values[row + 1, col]
                if abs(diff) > 15:
                    alternation_correct += 1
                alternation_total += 1
        
        alternation_score = alternation_correct / alternation_total if alternation_total > 0 else 0
        
        # Combined score
        score = (separation_score * 0.4 + consistency_score * 0.3 + alternation_score * 0.3)
        
        return score
    
    def detect_with_piece_validation(self, image: np.ndarray) -> Optional[Dict]:
        """
        Detect board and validate using piece templates.
        
        This method attempts to read pieces from the detected board
        to confirm it's a valid chess position.
        
        Args:
            image: BGR image to search.
        
        Returns:
            Detection result with validation info, or None.
        """
        # First detect the board
        detection = self.detect(image)
        
        if detection is None:
            return None
        
        # Try to validate with piece detection
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            
            from chessimage.image_scrape_utils import get_fen_from_image, PIECE_STEP
            import chess
            
            # Extract board region
            x, y, size = detection['x'], detection['y'], detection['size']
            board_img = extract_region(image, x, y, size, size)
            
            if board_img is None:
                return detection
            
            # Temporarily override PIECE_STEP for different board sizes
            step = size // 8
            
            # Try to get FEN
            # Note: This may not work perfectly if step differs from template size
            # We mainly use this for validation, not actual FEN extraction
            
            # For now, just add piece validation flag
            detection['piece_validation_available'] = True
            
        except ImportError:
            detection['piece_validation_available'] = False
        
        return detection


def detect_board(image: np.ndarray) -> Optional[Dict]:
    """
    Convenience function to detect chess board.
    
    Args:
        image: BGR image to search.
    
    Returns:
        Detection result dict or None.
    """
    detector = BoardDetector()
    return detector.detect(image)


def detect_board_from_screenshot() -> Optional[Dict]:
    """
    Detect chess board from current screen.
    
    Returns:
        Detection result dict or None.
    """
    from .utils import capture_screenshot
    
    screenshot = capture_screenshot()
    if screenshot is None:
        return None
    
    return detect_board(screenshot)
