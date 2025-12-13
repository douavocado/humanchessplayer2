#!/usr/bin/env python3
"""
Chess Clock Detection Module

Automatically detects chess clock positions using OCR-based validation.
Unlike the board detector, this uses the read_clock function to validate
potential clock regions by attempting to read the time display.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from calibration_utils import SCREEN_CAPTURE
import sys

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

# Import with context manager to handle path issues
import os
original_cwd = os.getcwd()
try:
    # Temporarily change to parent directory for imports that use relative paths
    os.chdir(parent_dir)
    from chessimage.image_scrape_utils import read_clock, remove_background_colours
finally:
    # Always restore original directory
    os.chdir(original_cwd)

# Import relaxed OCR function
from read_clock_relaxed import read_clock_relaxed

class ClockDetector:
    """Detects chess clock positions using OCR validation."""
    
    def __init__(self, board_position: Optional[Dict] = None):
        self.screen_capture = SCREEN_CAPTURE
        self.board_position = board_position
        
        # Standard clock dimensions based on existing coordinates
        self.clock_width = 147
        self.clock_height = 44
        
        # Search constraints
        self.min_search_step = 3  # Minimum pixel step for search grid
        self.max_search_step = 8  # Maximum pixel step for search grid
        
        # Game states to detect
        self.game_states = ['play', 'start1', 'start2', 'end1', 'end2', 'end3']
        
    def set_board_position(self, board_data: Dict):
        """Set the detected board position to constrain search area."""
        self.board_position = board_data
        
    def get_search_region(self, screenshot_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Calculate search region for clocks based on board position or screen dimensions.
        Returns (x, y, width, height) of search area.
        """
        screen_height, screen_width = screenshot_shape
        
        if self.board_position:
            # Search to the right of the detected board
            board_x, board_y, board_width, board_height = self.board_position['position']
            
            # Start search with small overlap to ensure we catch clocks at board edge
            search_x = board_x + board_width - 20  # Small overlap with board edge
            search_y = max(0, board_y - 50)  # Start slightly above board
            search_width = min(520, screen_width - search_x)  # Slightly wider search
            search_height = min(board_height + 100, screen_height - search_y)  # Extend below board
            
            print(f"Using board-constrained search region: ({search_x}, {search_y}) [{search_width}x{search_height}]")
            print(f"Board position was: ({board_x}, {board_y}) [{board_width}x{board_height}]")
        else:
            # Fallback: search right half of screen
            search_x = screen_width // 2
            search_y = 0
            search_width = screen_width // 2
            search_height = screen_height
            
            print(f"Using fallback search region: ({search_x}, {search_y}) [{search_width}x{search_height}]")
        
        return search_x, search_y, search_width, search_height
    
    def extract_clock_region(self, screenshot_img: np.ndarray, x: int, y: int) -> np.ndarray:
        """Extract clock-sized region from screenshot at given position."""
        h, w = screenshot_img.shape[:2]
        
        # Ensure region is within bounds
        x = max(0, min(x, w - self.clock_width))
        y = max(0, min(y, h - self.clock_height))
        
        return screenshot_img[y:y+self.clock_height, x:x+self.clock_width]
    
    def validate_clock_region(self, clock_region: np.ndarray, x: int, y: int) -> Tuple[bool, Optional[int], float]:
        """
        Validate if a region contains a readable clock using strict OCR-only validation.
        
        Returns:
            (is_valid, time_value, confidence)
        """
        if clock_region.shape[0] != self.clock_height or clock_region.shape[1] != self.clock_width:
            return False, None, 0.0
        
        # Try relaxed OCR first (for current chess sites)
        try:
            time_value = read_clock_relaxed(clock_region, threshold=0.2)
            if time_value is not None:
                if 0 <= time_value <= 7200:  # 0 to 2 hours
                    # High confidence for successful relaxed OCR
                    if 30 <= time_value <= 1800:  # 30 seconds to 30 minutes
                        confidence = 0.90
                    elif 10 <= time_value <= 3600:  # 10 seconds to 1 hour
                        confidence = 0.80
                    else:
                        confidence = 0.70
                    
                    return True, time_value, confidence
        except:
            pass
        
        # Fallback to strict OCR (for legacy compatibility)
        try:
            time_value = read_clock(clock_region)
            if time_value is not None:
                if 0 <= time_value <= 3600:
                    # High confidence for successful strict OCR
                    confidence = 0.95
                    return True, time_value, confidence
        except:
            pass
        
        # NO FALLBACK - only return True if OCR actually succeeds
        # This prevents false positives from blank/empty regions
        return False, None, 0.0
    
    def _calculate_clock_likelihood(self, clock_region: np.ndarray, x: int, y: int) -> float:
        """
        Calculate likelihood that this region contains a clock based on 
        multiple factors including position and visual characteristics.
        """
        likelihood = 0.0
        
        # Factor 1: Position-based likelihood
        # Clocks are typically at specific Y ranges relative to board
        if self.board_position:
            board_x, board_y, board_width, board_height = self.board_position['position']
            
            # Expected clock Y positions (approximate)
            expected_top_clock_y = board_y + 150
            expected_bottom_clock_y = board_y + board_height - 150
            
            # Distance from expected positions
            dist_to_top = abs(y - expected_top_clock_y)
            dist_to_bottom = abs(y - expected_bottom_clock_y)
            min_dist = min(dist_to_top, dist_to_bottom)
            
            if min_dist < 50:
                position_score = 0.4
            elif min_dist < 100:
                position_score = 0.3
            elif min_dist < 150:
                position_score = 0.2
            else:
                position_score = 0.0
                
            likelihood += position_score
        
        # Factor 2: Visual characteristics
        try:
            # Convert to grayscale
            if len(clock_region.shape) == 3:
                gray = cv2.cvtColor(clock_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = clock_region.copy()
            
            # Check for digital clock characteristics
            
            # 1. Text-like variance (not uniform)
            variance = np.var(gray)
            if variance > 20:
                likelihood += 0.2
            
            # 2. Horizontal structure (typical of digital clocks)
            horizontal_grad = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            vertical_grad = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            h_energy = np.sum(np.abs(horizontal_grad))
            v_energy = np.sum(np.abs(vertical_grad))
            
            if v_energy > h_energy and v_energy > 1000:
                likelihood += 0.2
            
            # 3. Check for colon-like structure in middle
            middle_region = gray[:, 60:90]  # Approximate middle area
            if np.var(middle_region) > 15:
                likelihood += 0.1
                
        except:
            pass
        
        return min(1.0, likelihood)
    
    def _validate_visual_clock_pattern(self, clock_region: np.ndarray) -> float:
        """
        Check if region has visual characteristics of a clock display.
        Returns confidence score 0.0 to 1.0.
        """
        try:
            # Convert to grayscale
            if len(clock_region.shape) == 3:
                gray = cv2.cvtColor(clock_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = clock_region.copy()
            
            # Check for digital-clock-like characteristics
            
            # 1. Check for text/number-like regions (not uniform color)
            mean_intensity = np.mean(gray)
            intensity_variance = np.var(gray)
            
            # Clocks should have some variation (text vs background)
            if intensity_variance < 10:  # Too uniform
                return 0.0
            
            # 2. Check for typical clock layout patterns
            # Digital clocks often have text in specific regions
            
            # Check if there are darker regions (text) against lighter background
            # or lighter regions (text) against darker background
            binary_threshold = mean_intensity
            _, binary = cv2.threshold(gray, binary_threshold, 255, cv2.THRESH_BINARY)
            
            # Count connected components (should have some for digits/text)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
            
            # Good clocks typically have 2-6 distinct regions (digits, colons, etc.)
            if 2 <= num_labels <= 8:
                component_confidence = 0.4
            elif num_labels > 1:
                component_confidence = 0.2
            else:
                component_confidence = 0.0
            
            # 3. Check for horizontal text-like patterns
            # Clocks typically have horizontally arranged digits
            horizontal_edges = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            horizontal_energy = np.sum(np.abs(horizontal_edges))
            
            vertical_edges = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            vertical_energy = np.sum(np.abs(vertical_edges))
            
            # Clock text typically has more vertical edges than horizontal
            if vertical_energy > 0 and horizontal_energy > 0:
                edge_ratio = vertical_energy / (horizontal_energy + vertical_energy)
                if 0.3 <= edge_ratio <= 0.8:  # Balanced edges typical of text
                    edge_confidence = 0.3
                else:
                    edge_confidence = 0.1
            else:
                edge_confidence = 0.0
            
            # Combine confidence factors
            total_confidence = min(1.0, component_confidence + edge_confidence)
            
            return total_confidence
            
        except Exception as e:
            return 0.0
    
    def search_clock_positions(self, screenshot_img: np.ndarray, 
                             clock_type: str = "bottom") -> List[Dict]:
        """
        Search for clock positions in the screenshot.
        
        Args:
            screenshot_img: Screenshot to search in
            clock_type: "top" or "bottom" to guide search priority
            
        Returns:
            List of detected clock positions with metadata
        """
        search_x, search_y, search_width, search_height = self.get_search_region(screenshot_img.shape[:2])
        
        detections = []
        
        # Adaptive step size based on search area
        area = search_width * search_height
        if area > 100000:  # Large area
            step = self.max_search_step
        elif area > 50000:  # Medium area
            step = 5
        else:  # Small area
            step = self.min_search_step
        
        print(f"Searching for {clock_type} clock with step size {step}...")
        
        # Search grid
        positions_tested = 0
        valid_detections = 0
        ocr_attempts = 0
        ocr_successes = 0
        
        # Sample some positions for debugging
        debug_samples = []
        sample_count = 0
        max_samples = 10
        
        for y in range(search_y, search_y + search_height - self.clock_height, step):
            for x in range(search_x, search_x + search_width - self.clock_width, step):
                positions_tested += 1
                
                # Extract potential clock region
                clock_region = self.extract_clock_region(screenshot_img, x, y)
                
                # Validate the region
                is_valid, time_value, confidence = self.validate_clock_region(clock_region, x, y)
                
                # Track OCR statistics
                ocr_attempts += 1
                if time_value is not None:
                    ocr_successes += 1
                
                # Sample some regions for debugging (every 500th position)
                if sample_count < max_samples and positions_tested % 500 == 0:
                    debug_samples.append({
                        'position': (x, y),
                        'time_value': time_value,
                        'is_valid': is_valid,
                        'confidence': confidence
                    })
                    sample_count += 1
                
                if is_valid and confidence > 0.7:
                    valid_detections += 1
                    
                    detection = {
                        'position': (x, y, self.clock_width, self.clock_height),
                        'time_value': time_value,
                        'confidence': confidence,
                        'clock_type': clock_type,
                        'state': 'unknown'  # Will be determined later
                    }
                    
                    detections.append(detection)
                    
                    print(f"  Found clock at ({x}, {y}): {time_value}s (confidence: {confidence:.3f})")
        
        # Print debug statistics
        ocr_success_rate = (ocr_successes / ocr_attempts) * 100 if ocr_attempts > 0 else 0
        print(f"Tested {positions_tested} positions, found {valid_detections} valid clocks")
        print(f"OCR success rate: {ocr_successes}/{ocr_attempts} ({ocr_success_rate:.1f}%)")
        
        # Print some sample results for debugging
        if debug_samples:
            print(f"Sample validation results:")
            for i, sample in enumerate(debug_samples[:3]):  # Show first 3 samples
                pos = sample['position']
                print(f"  Position ({pos[0]}, {pos[1]}): time={sample['time_value']}, "
                      f"valid={sample['is_valid']}, conf={sample['confidence']:.3f}")
        
        # Sort by confidence
        detections.sort(key=lambda d: d['confidence'], reverse=True)
        
        return detections
    
    def cluster_detections(self, detections: List[Dict], 
                          cluster_distance: int = 50) -> List[Dict]:
        """
        Cluster nearby detections and keep the best one from each cluster.
        This handles cases where the same clock is detected at slightly different positions.
        """
        if not detections:
            return []
        
        clusters = []
        
        for detection in detections:
            x, y, w, h = detection['position']
            
            # Find if this detection belongs to an existing cluster
            assigned = False
            for cluster in clusters:
                cluster_x, cluster_y, _, _ = cluster['position']
                
                # Check if within cluster distance
                if (abs(x - cluster_x) <= cluster_distance and 
                    abs(y - cluster_y) <= cluster_distance):
                    
                    # Keep the detection with higher confidence
                    if detection['confidence'] > cluster['confidence']:
                        clusters.remove(cluster)
                        clusters.append(detection)
                    assigned = True
                    break
            
            if not assigned:
                clusters.append(detection)
        
        print(f"Clustered {len(detections)} detections into {len(clusters)} unique positions")
        return clusters
    
    def assign_clock_states(self, detections: List[Dict]) -> List[Dict]:
        """
        Attempt to assign game states to detected clocks based on their positions.
        This is heuristic-based since we can't know the actual game state.
        """
        if not detections:
            return []
        
        # Sort by Y coordinate to separate top and bottom clocks
        sorted_detections = sorted(detections, key=lambda d: d['position'][1])
        
        # If we have multiple detections, try to assign top/bottom and states
        if len(sorted_detections) >= 2:
            # Assume roughly equal Y spacing between different states
            y_positions = [d['position'][1] for d in sorted_detections]
            y_range = max(y_positions) - min(y_positions)
            
            # If Y range is significant, we likely have different states
            if y_range > 50:
                # Group by approximate Y regions
                mid_y = (min(y_positions) + max(y_positions)) / 2
                
                top_clocks = [d for d in sorted_detections if d['position'][1] < mid_y]
                bottom_clocks = [d for d in sorted_detections if d['position'][1] >= mid_y]
                
                # Assign clock types
                for clock in top_clocks:
                    clock['clock_type'] = 'top'
                for clock in bottom_clocks:
                    clock['clock_type'] = 'bottom'
                
                # Assign states based on Y ordering within each group
                self._assign_states_to_group(top_clocks, 'top')
                self._assign_states_to_group(bottom_clocks, 'bottom')
            else:
                # All detections are at similar Y levels - might be same state
                for i, detection in enumerate(sorted_detections):
                    detection['state'] = 'play' if i == 0 else f'variant_{i}'
        
        return sorted_detections
    
    def _assign_states_to_group(self, clocks: List[Dict], clock_type: str):
        """Assign states to a group of clocks based on Y position."""
        if not clocks:
            return
        
        # Sort by Y position
        clocks.sort(key=lambda c: c['position'][1])
        
        # Assign states based on expected patterns
        state_order = ['play', 'start1', 'start2', 'end1', 'end2', 'end3']
        
        for i, clock in enumerate(clocks):
            if i < len(state_order):
                clock['state'] = state_order[i]
            else:
                clock['state'] = f'extra_{i}'
    
    def find_clocks(self, max_attempts: int = 2) -> Optional[Dict]:
        """
        Main function to find both top and bottom clocks on screen.
        
        Returns:
            Dictionary with detected clock positions or None
        """
        print("Searching for chess clocks on screen...")
        
        all_detections = []
        
        for attempt in range(max_attempts):
            print(f"Attempt {attempt + 1}/{max_attempts}")
            
            # Capture screenshot
            screenshot_img = self.screen_capture.capture()
            if screenshot_img is None:
                print(f"Failed to capture screenshot on attempt {attempt + 1}")
                continue
            
            # Search for clocks (both top and bottom in one pass)
            detections = self.search_clock_positions(screenshot_img, "both")
            
            if detections:
                all_detections.extend(detections)
                break
        
        if not all_detections:
            print("No clocks detected")
            return None
        
        # Filter for higher confidence detections first
        high_confidence = [d for d in all_detections if d['confidence'] >= 0.8]
        medium_confidence = [d for d in all_detections if 0.65 <= d['confidence'] < 0.8]
        
        print(f"High confidence detections: {len(high_confidence)}")
        print(f"Medium confidence detections: {len(medium_confidence)}")
        
        # Use high confidence if available, otherwise medium, but limit the total
        detections_to_use = high_confidence if high_confidence else medium_confidence
        
        # Limit detections to prevent too many false positives
        detections_to_use = detections_to_use[:50]
        
        if not detections_to_use:
            print("No suitable clock detections found")
            return None
        
        # Cluster nearby detections
        clustered = self.cluster_detections(detections_to_use)
        
        # Limit to most promising candidates
        clustered = clustered[:10]  # Top 10 candidates
        
        # Assign clock types and states
        final_detections = self.assign_clock_states(clustered)
        
        # Organize results
        result = {
            'detection_method': 'ocr_validation',
            'total_detections': len(final_detections),
            'clocks': final_detections,
            'timestamp': time.time()
        }
        
        print(f"Successfully detected {len(final_detections)} clock positions")
        for detection in final_detections:
            pos = detection['position']
            print(f"  {detection['clock_type']} clock ({detection['state']}): "
                  f"({pos[0]}, {pos[1]}) - {detection['time_value']}s "
                  f"(confidence: {detection['confidence']:.3f})")
        
        return result
    
    def validate_existing_coordinates(self, coordinates: Dict, 
                                    screenshot_img: Optional[np.ndarray] = None) -> Dict:
        """
        Validate existing clock coordinates by attempting to read them.
        
        Args:
            coordinates: Dictionary of clock coordinates to validate
            screenshot_img: Optional screenshot, will capture if not provided
            
        Returns:
            Dictionary with validation results
        """
        if screenshot_img is None:
            screenshot_img = self.screen_capture.capture()
            if screenshot_img is None:
                return {'error': 'Could not capture screenshot'}
        
        validation_results = {
            'total_tested': 0,
            'total_successful': 0,
            'clock_results': {},
            'overall_success_rate': 0.0
        }
        
        # Test each clock type and state
        for clock_type in ['bottom_clock', 'top_clock']:
            if clock_type in coordinates:
                validation_results['clock_results'][clock_type] = {}
                
                for state, coords in coordinates[clock_type].items():
                    validation_results['total_tested'] += 1
                    
                    # Extract clock region
                    x, y = coords['x'], coords['y']
                    w, h = coords['width'], coords['height']
                    
                    # Ensure coordinates are within bounds
                    if (x >= 0 and y >= 0 and 
                        x + w <= screenshot_img.shape[1] and 
                        y + h <= screenshot_img.shape[0]):
                        
                        clock_region = screenshot_img[y:y+h, x:x+w]
                        
                        # Validate using read_clock
                        is_valid, time_value, confidence = self.validate_clock_region(clock_region, x, y)
                        
                        validation_results['clock_results'][clock_type][state] = {
                            'success': is_valid,
                            'time_value': time_value,
                            'confidence': confidence,
                            'coordinates': coords
                        }
                        
                        if is_valid:
                            validation_results['total_successful'] += 1
                        
                        status = '✅' if is_valid else '❌'
                        time_str = f"{time_value}s" if time_value is not None else "None"
                        print(f"  {clock_type}.{state}: {status} {time_str} (confidence: {confidence:.3f})")
                    else:
                        validation_results['clock_results'][clock_type][state] = {
                            'success': False,
                            'error': 'Coordinates out of bounds',
                            'coordinates': coords
                        }
                        print(f"  {clock_type}.{state}: ❌ (out of bounds)")
        
        # Calculate overall success rate
        if validation_results['total_tested'] > 0:
            validation_results['overall_success_rate'] = (
                validation_results['total_successful'] / validation_results['total_tested']
            )
        
        print(f"Validation complete: {validation_results['total_successful']}/{validation_results['total_tested']} "
              f"({validation_results['overall_success_rate']:.1%}) success rate")
        
        return validation_results

if __name__ == "__main__":
    # Test the clock detector
    detector = ClockDetector()
    
    print("Testing clock detection...")
    result = detector.find_clocks()
    
    if result:
        print(f"\nDetected {result['total_detections']} clocks:")
        for clock in result['clocks']:
            pos = clock['position']
            print(f"  {clock['clock_type']} ({clock['state']}): ({pos[0]}, {pos[1]}) "
                  f"- {clock['time_value']}s (confidence: {clock['confidence']:.3f})")
    else:
        print("No clocks detected")
