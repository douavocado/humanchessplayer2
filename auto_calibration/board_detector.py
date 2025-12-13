#!/usr/bin/env python3
"""
Chess Board Detection Module

Automatically detects chess board position using colour-based detection.
Finds the board by detecting the checkerboard pattern of alternating coloured squares.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from calibration_utils import SCREEN_CAPTURE


class BoardDetector:
    """Detects chess board position using colour pattern analysis."""
    
    # Minimum confidence threshold
    MIN_CONFIDENCE = 0.5
    
    # Common chess board colour ranges (HSV)
    # These cover most common board themes
    BOARD_COLOURS = [
        # Lichess green theme
        {'light': ([35, 30, 150], [85, 120, 255]), 'dark': ([35, 50, 80], [85, 180, 180])},
        # Brown/beige theme
        {'light': ([15, 20, 180], [35, 80, 255]), 'dark': ([10, 50, 80], [30, 150, 160])},
        # Blue theme
        {'light': ([90, 20, 180], [130, 80, 255]), 'dark': ([90, 50, 80], [130, 150, 180])},
        # Grey theme  
        {'light': ([0, 0, 180], [180, 30, 255]), 'dark': ([0, 0, 80], [180, 30, 160])},
    ]
    
    def __init__(self, search_region: Optional[Tuple[int, int, int, int]] = None):
        """
        Initialise board detector.
        
        Args:
            search_region: Optional (x, y, width, height) to limit search area.
                          If None, searches entire screen.
                          Use (0, 0, 1920, 1080) to search only left monitor.
        """
        self.screen_capture = SCREEN_CAPTURE
        self.search_region = search_region
        
        # Expected board size - on Lichess/Chess.com, board is typically ~half screen width
        # For 1920px monitor at 1.5x scale = 2880px screenshot, expect ~1350px board
        # We'll calculate expected size based on search region width
        if search_region:
            self.expected_size = int(search_region[2] * 0.47)  # ~47% of search region width
        else:
            self.expected_size = 900  # Default for 1920px monitor
        
        # Test sizes within ±20% of expected (tighter range)
        self.expected_board_size_min = int(self.expected_size * 0.8)
        self.expected_board_size_max = int(self.expected_size * 1.2)
    
    def find_checkerboard_regions(self, image: np.ndarray, 
                                    search_region: Optional[Tuple[int, int, int, int]] = None) -> List[Dict]:
        """
        Find regions that have checkerboard-like colour patterns.
        
        This works by:
        1. Scanning across the image at multiple scales
        2. For each position, checking if it contains an alternating colour pattern
        3. Using smaller steps and more size variations to find the board
        
        Args:
            image: BGR image to search
            search_region: Optional (x, y, width, height) to limit search.
                          Coordinates are in full image space.
        """
        if image is None or image.size == 0:
            return []
        
        full_height, full_width = image.shape[:2]
        
        # Apply search region if specified
        if search_region:
            sx, sy, sw, sh = search_region
            # Clamp to image bounds
            sx = max(0, sx)
            sy = max(0, sy)
            sw = min(sw, full_width - sx)
            sh = min(sh, full_height - sy)
            search_image = image[sy:sy+sh, sx:sx+sw]
            offset_x, offset_y = sx, sy
            print(f"Searching within region: ({sx}, {sy}) [{sw}x{sh}]")
        else:
            search_image = image
            offset_x, offset_y = 0, 0
        
        height, width = search_image.shape[:2]
        
        # Get grayscale for intensity analysis
        gray = cv2.cvtColor(search_image, cv2.COLOR_BGR2GRAY)
        
        candidates = []
        
        # Use smaller step for finer scanning
        scan_step = 50  # Fixed small step for better coverage
        
        # Try different potential board sizes - use ~6 sizes for speed
        size_range = self.expected_board_size_max - self.expected_board_size_min
        step = max(100, size_range // 5)  # Approximately 6 sizes
        board_sizes = list(range(self.expected_board_size_min,
                                 self.expected_board_size_max + 1, step))

        print(f"Testing {len(board_sizes)} board sizes from {self.expected_board_size_min} to {self.expected_board_size_max}")
        
        for board_size in board_sizes:
            square_size = board_size // 8
            
            if square_size < 30:
                continue
            
            # Scan across image
            for y in range(0, height - board_size, scan_step):
                for x in range(0, width - board_size, scan_step):
                    # Extract potential board region
                    region = gray[y:y+board_size, x:x+board_size]
                    region_colour = search_image[y:y+board_size, x:x+board_size]
                    
                    # Check if this region has checkerboard pattern
                    score, details = self._check_checkerboard_pattern(region, region_colour, square_size)
                    
                    if score > 0.4:  # Promising candidate
                        # Verify grid pattern for additional confidence
                        grid_score = self.verify_grid_pattern(region)
                        
                        # Add size bonus - prefer larger boards (main board vs thumbnails)
                        # Normalise: 300px = 0 bonus, 650px = 0.15 bonus, 1000px = 0.3 bonus
                        size_bonus = min(0.3, (board_size - 300) / 2333)
                        
                        # Add grid verification bonus
                        grid_bonus = grid_score * 0.3
                        
                        adjusted_score = score + size_bonus + grid_bonus
                        
                        # Convert back to full image coordinates
                        candidates.append({
                            'bbox': (x + offset_x, y + offset_y, board_size, board_size),
                            'score': adjusted_score,
                            'base_score': score,
                            'size_bonus': size_bonus,
                            'grid_score': grid_score,
                            'grid_bonus': grid_bonus,
                            'square_size': square_size,
                            'details': details
                        })
        
        print(f"Found {len(candidates)} raw candidates before deduplication")
        
        # Remove overlapping candidates, keep best
        candidates = self._remove_overlapping(candidates)
        
        # Sort by score, then by size (prefer larger boards when scores are similar)
        # We want the main chess board, not thumbnail boards
        candidates.sort(key=lambda c: (c['score'], c['bbox'][2]), reverse=True)
        
        return candidates
    
    def _check_checkerboard_pattern(self, gray_region: np.ndarray, 
                                     colour_region: np.ndarray,
                                     square_size: int) -> Tuple[float, Dict]:
        """
        Check if a region contains a checkerboard pattern.
        
        Uses bimodal distribution detection - a chess board should have
        two distinct colour clusters (light and dark squares).
        
        Returns:
            (score, details) - score from 0-1, details dict
        """
        h, w = gray_region.shape[:2]
        
        if h < 100 or w < 100:
            return 0.0, {}
        
        # Method 1: Check for bimodal distribution (two colour clusters)
        # Sample many points across the region
        sample_points = []
        sample_step = max(5, min(h, w) // 30)  # Sample about 30x30 = 900 points
        
        for y in range(sample_step // 2, h - sample_step // 2, sample_step):
            for x in range(sample_step // 2, w - sample_step // 2, sample_step):
                # Sample small region around point
                y1, y2 = max(0, y - 2), min(h, y + 3)
                x1, x2 = max(0, x - 2), min(w, x + 3)
                val = np.mean(gray_region[y1:y2, x1:x2])
                sample_points.append(val)
        
        if len(sample_points) < 50:
            return 0.0, {}
        
        sample_points = np.array(sample_points)
        
        # Find the histogram and look for bimodal distribution
        hist, bin_edges = np.histogram(sample_points, bins=20)
        
        # Look for two peaks (bimodal distribution)
        # A chess board should have ~50% light and ~50% dark pixels
        total = np.sum(hist)
        
        # Find the threshold that best separates the two modes
        best_separation = 0
        best_threshold_idx = 10
        
        for i in range(5, 15):  # Try different split points
            lower_sum = np.sum(hist[:i])
            upper_sum = np.sum(hist[i:])
            
            # Both sides should have significant mass (30-70% each)
            lower_frac = lower_sum / total
            upper_frac = upper_sum / total
            
            if 0.25 < lower_frac < 0.75:
                # Calculate the separation between the two groups
                lower_vals = sample_points[sample_points < bin_edges[i]]
                upper_vals = sample_points[sample_points >= bin_edges[i]]
                
                if len(lower_vals) > 10 and len(upper_vals) > 10:
                    lower_mean = np.mean(lower_vals)
                    upper_mean = np.mean(upper_vals)
                    separation = abs(upper_mean - lower_mean)
                    
                    if separation > best_separation:
                        best_separation = separation
                        best_threshold_idx = i
        
        # Calculate final statistics using the best threshold
        threshold = bin_edges[best_threshold_idx]
        dark_vals = sample_points[sample_points < threshold]
        light_vals = sample_points[sample_points >= threshold]
        
        if len(dark_vals) < 20 or len(light_vals) < 20:
            return 0.0, {}
        
        dark_mean = np.mean(dark_vals)
        light_mean = np.mean(light_vals)
        dark_std = np.std(dark_vals)
        light_std = np.std(light_vals)
        mean_diff = abs(light_mean - dark_mean)
        
        # Chess board specific constraints:
        # - Dark squares should be in range 80-180 (not black/very dark)
        # - Light squares should be in range 150-245 (not pure white)
        # - The difference should be reasonable (30-120)
        
        if dark_mean < 50:  # Too dark - likely detecting black UI elements
            return 0.0, {'rejected': 'dark_too_black', 'dark_mean': dark_mean}
        
        if dark_mean > 200:  # "Dark" cluster is actually light - no real contrast
            return 0.0, {'rejected': 'dark_too_light', 'dark_mean': dark_mean}
        
        if light_mean < 120:  # "Light" cluster too dark
            return 0.0, {'rejected': 'light_too_dark', 'light_mean': light_mean}
        
        if mean_diff < 25:  # Not enough contrast for chess board
            return 0.0, {'rejected': 'insufficient_contrast', 'mean_diff': mean_diff}
        
        if mean_diff > 150:  # Too much contrast - likely black text on white
            return 0.0, {'rejected': 'excessive_contrast', 'mean_diff': mean_diff}
        
        # Score components:
        
        # 1. Colour separation - should have at least 30+ difference for chess board
        if mean_diff < 20:
            separation_score = 0.0
        elif mean_diff < 40:
            separation_score = 0.4
        elif mean_diff < 60:
            separation_score = 0.7
        else:
            separation_score = min(1.0, mean_diff / 80)
        
        # 2. Balance - should be roughly 50/50 split
        dark_frac = len(dark_vals) / len(sample_points)
        balance_deviation = abs(dark_frac - 0.5)
        if balance_deviation < 0.1:
            balance_score = 1.0
        elif balance_deviation < 0.2:
            balance_score = 0.7
        elif balance_deviation < 0.3:
            balance_score = 0.4
        else:
            balance_score = 0.1
        
        # 3. Uniformity within each cluster
        avg_std = (light_std + dark_std) / 2
        if avg_std > 40:
            uniformity_score = 0.2
        elif avg_std > 25:
            uniformity_score = 0.5
        elif avg_std > 15:
            uniformity_score = 0.7
        else:
            uniformity_score = 0.9
        
        # Combined score
        total_score = (separation_score * 0.5 + balance_score * 0.25 + uniformity_score * 0.25)
        
        details = {
            'light_mean': light_mean,
            'dark_mean': dark_mean,
            'mean_diff': mean_diff,
            'light_std': light_std,
            'dark_std': dark_std,
            'dark_fraction': dark_frac,
            'separation_score': separation_score,
            'balance_score': balance_score,
            'uniformity_score': uniformity_score,
            'region_size': h  # Add size for sorting
        }
        
        return total_score, details
    
    def _remove_overlapping(self, candidates: List[Dict], 
                            overlap_threshold: float = 0.5) -> List[Dict]:
        """Remove candidates that overlap significantly, keeping highest scoring."""
        if not candidates:
            return []
        
        # Sort by score descending
        candidates.sort(key=lambda c: c['score'], reverse=True)
        
        kept = []
        for candidate in candidates:
            x1, y1, w1, h1 = candidate['bbox']
            
            overlaps = False
            for existing in kept:
                x2, y2, w2, h2 = existing['bbox']
                
                # Calculate overlap
                x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = x_overlap * y_overlap
                
                min_area = min(w1 * h1, w2 * h2)
                if min_area > 0 and overlap_area / min_area > overlap_threshold:
                    overlaps = True
                    break
            
            if not overlaps:
                kept.append(candidate)
        
        return kept
    
    def verify_grid_pattern(self, gray_region: np.ndarray) -> float:
        """
        Verify that a region contains an 8x8 alternating grid pattern.
        
        Returns a confidence score 0-1 based on how well the region
        matches the expected chess board pattern.
        """
        h, w = gray_region.shape[:2]
        if h < 100 or w < 100:
            return 0.0
        
        square_h = h // 8
        square_w = w // 8
        
        if square_h < 10 or square_w < 10:
            return 0.0
        
        # Calculate average brightness of each of the 64 squares
        grid_values = np.zeros((8, 8))
        for row in range(8):
            for col in range(8):
                y1 = row * square_h + square_h // 4
                y2 = (row + 1) * square_h - square_h // 4
                x1 = col * square_w + square_w // 4
                x2 = (col + 1) * square_w - square_w // 4
                grid_values[row, col] = np.mean(gray_region[y1:y2, x1:x2])
        
        # Check alternating pattern: adjacent cells should have different brightness
        correct_alternations = 0
        total_comparisons = 0
        
        for row in range(8):
            for col in range(8):
                current = grid_values[row, col]
                
                # Check horizontal neighbor
                if col < 7:
                    neighbor = grid_values[row, col + 1]
                    diff = abs(current - neighbor)
                    if diff > 20:  # Significant difference
                        correct_alternations += 1
                    total_comparisons += 1
                
                # Check vertical neighbor
                if row < 7:
                    neighbor = grid_values[row + 1, col]
                    diff = abs(current - neighbor)
                    if diff > 20:  # Significant difference
                        correct_alternations += 1
                    total_comparisons += 1
        
        if total_comparisons == 0:
            return 0.0
        
        alternation_score = correct_alternations / total_comparisons
        
        # Also check that same-color squares have similar values
        light_squares = [grid_values[r, c] for r in range(8) for c in range(8) if (r + c) % 2 == 0]
        dark_squares = [grid_values[r, c] for r in range(8) for c in range(8) if (r + c) % 2 == 1]
        
        light_std = np.std(light_squares) if light_squares else 100
        dark_std = np.std(dark_squares) if dark_squares else 100
        light_mean = np.mean(light_squares) if light_squares else 0
        dark_mean = np.mean(dark_squares) if dark_squares else 0
        
        # Uniformity score - same color squares should be similar
        uniformity_score = max(0, 1 - (light_std + dark_std) / 100)
        
        # Separation score - light and dark should be different
        mean_diff = abs(light_mean - dark_mean)
        separation_score = min(1.0, mean_diff / 60)
        
        # Combined score
        total_score = (alternation_score * 0.4 + uniformity_score * 0.3 + separation_score * 0.3)
        
        return total_score

    def refine_board_position(self, image: np.ndarray,
                               bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Refine the board position using expected board size and centering.
        
        Since we know the expected board size from monitor width, we use that
        and center the board on the detected region.
        """
        x, y, w, h = bbox
        
        # Use the expected board size (calculated from monitor width)
        target_size = self.expected_size
        
        # Center the target size on the detected region
        center_x = x + w // 2
        center_y = y + h // 2
        
        refined_x = center_x - target_size // 2
        refined_y = center_y - target_size // 2
        
        # Ensure within image bounds
        refined_x = max(0, refined_x)
        refined_y = max(0, refined_y)
        
        return (refined_x, refined_y, target_size, target_size)
    
    def find_chess_board(self, max_attempts: int = 3) -> Optional[Dict]:
        """
        Main function to find chess board on screen.
        
        Returns:
            Dictionary with board position and metadata or None
        """
        print("Searching for chess board on screen...")
        
        best_detection = None
        best_confidence = 0
        
        for attempt in range(max_attempts):
            print(f"\nAttempt {attempt + 1}/{max_attempts}")
            
            # Capture screenshot
            screenshot_img = self.screen_capture.capture()
            if screenshot_img is None:
                print(f"Failed to capture screenshot on attempt {attempt + 1}")
                continue
            
            # Convert BGRA to BGR if needed
            if screenshot_img.shape[2] == 4:
                screenshot_img = cv2.cvtColor(screenshot_img, cv2.COLOR_BGRA2BGR)
            
            print(f"Screenshot size: {screenshot_img.shape[1]}x{screenshot_img.shape[0]}")
            
            # Find checkerboard regions (using search_region if set)
            print("Scanning for checkerboard patterns...")
            candidates = self.find_checkerboard_regions(screenshot_img, self.search_region)
            
            print(f"Found {len(candidates)} candidate regions")
            
            if candidates:
                # Show top candidates
                for i, c in enumerate(candidates[:5]):
                    x, y, w, h = c['bbox']
                    print(f"  {i+1}. ({x}, {y}) [{w}x{h}] score={c['score']:.3f}")
                
                # Take best candidate
                best = candidates[0]
                
                if best['score'] > best_confidence:
                    # Refine position
                    refined_bbox = self.refine_board_position(screenshot_img, best['bbox'])
                    
                    best_confidence = best['score']
                    best_detection = {
                        'method': 'colour_pattern',
                        'position': refined_bbox,
                        'confidence': best['score'],
                        'details': best.get('details', {})
                    }
                    
                    if best_confidence >= 0.8:
                        print(f"High confidence detection ({best_confidence:.3f}), stopping early")
                        break
            
            # Wait before next attempt
            if attempt < max_attempts - 1:
                time.sleep(0.5)
        
        if best_detection and best_confidence >= self.MIN_CONFIDENCE:
            x, y, w, h = best_detection['position']
            print(f"\n✅ Board detected using {best_detection['method']} with confidence {best_confidence:.3f}")
            print(f"   Position: ({x}, {y}) [{w}x{h}]")
            return best_detection
        else:
            if best_detection:
                print(f"\n❌ Best match had confidence {best_confidence:.3f}, below threshold {self.MIN_CONFIDENCE}")
            else:
                print("\n❌ No checkerboard patterns found")
            print("Failed to detect chess board")
            return None
