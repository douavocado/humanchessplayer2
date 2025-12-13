#!/usr/bin/env python3
"""
Improved clock detector that handles modern chess site layouts.
This version relaxes the OCR requirements and focuses on detecting regions 
that look like they could contain clocks.
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

# Change to parent directory for relative path imports
import os
original_cwd = os.getcwd()
try:
    os.chdir(parent_dir)
    from chessimage.image_scrape_utils import read_clock, remove_background_colours
finally:
    os.chdir(original_cwd)

class ImprovedClockDetector:
    """Improved clock detector with relaxed OCR requirements."""
    
    def __init__(self, board_position: Optional[Dict] = None):
        self.screen_capture = SCREEN_CAPTURE
        self.board_position = board_position
        
        # Standard clock dimensions
        self.clock_width = 147
        self.clock_height = 44
        
        # More relaxed search constraints
        self.min_search_step = 5
        self.max_search_step = 12
        
    def set_board_position(self, board_data: Dict):
        """Set the detected board position to constrain search area."""
        self.board_position = board_data
        
    def get_search_region(self, screenshot_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Calculate search region for clocks."""
        screen_height, screen_width = screenshot_shape
        
        if self.board_position:
            board_x, board_y, board_width, board_height = self.board_position['position']
            
            # Search to the right of the board with some overlap
            search_x = board_x + board_width - 30
            search_y = max(0, board_y - 50)
            search_width = min(400, screen_width - search_x)
            search_height = min(board_height + 100, screen_height - search_y)
            
            print(f"Search region: ({search_x}, {search_y}) [{search_width}x{search_height}]")
            print(f"Board: ({board_x}, {board_y}) [{board_width}x{board_height}]")
        else:
            # Fallback
            search_x = screen_width // 2
            search_y = 0
            search_width = screen_width // 2
            search_height = screen_height
            
        return search_x, search_y, search_width, search_height
    
    def validate_clock_region_relaxed(self, clock_region: np.ndarray, x: int, y: int) -> Tuple[bool, Optional[int], float]:
        """
        More relaxed validation that looks for clock-like characteristics.
        """
        if clock_region.shape[0] != self.clock_height or clock_region.shape[1] != self.clock_width:
            return False, None, 0.0
        
        # First, try strict OCR
        try:
            time_value = read_clock(clock_region)
            if time_value is not None:
                if 0 <= time_value <= 3600:
                    confidence = 0.95  # High confidence for OCR success
                    return True, time_value, confidence
        except:
            pass
        
        # Relaxed validation: Look for regions that could be clocks
        confidence = self._calculate_clock_likelihood(clock_region, x, y)
        
        if confidence > 0.4:  # Lower threshold
            estimated_time = 300  # Default estimate
            return True, estimated_time, confidence
        
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
    
    def find_clocks(self, max_attempts: int = 1) -> Optional[Dict]:
        """Find clocks using relaxed detection."""
        print("Searching for chess clocks (relaxed mode)...")
        
        for attempt in range(max_attempts):
            print(f"Attempt {attempt + 1}/{max_attempts}")
            
            screenshot_img = self.screen_capture.capture()
            if screenshot_img is None:
                continue
            
            search_x, search_y, search_width, search_height = self.get_search_region(screenshot_img.shape[:2])
            
            detections = []
            positions_tested = 0
            step = 8  # Fixed step size
            
            print(f"Searching with step size {step}...")
            
            for y in range(search_y, search_y + search_height - self.clock_height, step):
                for x in range(search_x, search_x + search_width - self.clock_width, step):
                    positions_tested += 1
                    
                    # Extract region
                    clock_region = screenshot_img[y:y+self.clock_height, x:x+self.clock_width]
                    
                    # Validate
                    is_valid, time_value, confidence = self.validate_clock_region_relaxed(clock_region, x, y)
                    
                    if is_valid and confidence > 0.4:
                        detection = {
                            'position': (x, y, self.clock_width, self.clock_height),
                            'time_value': time_value,
                            'confidence': confidence,
                            'clock_type': 'unknown',
                            'state': 'unknown'
                        }
                        detections.append(detection)
                        
                        print(f"  Found potential clock at ({x}, {y}): confidence={confidence:.3f}")
            
            print(f"Tested {positions_tested} positions, found {len(detections)} candidates")
            
            if detections:
                # Sort by confidence and position
                detections.sort(key=lambda d: d['confidence'], reverse=True)
                
                # Take top candidates and assign types
                final_detections = self._assign_clock_types(detections[:20])  # Top 20 candidates
                
                return {
                    'detection_method': 'relaxed_ocr_validation',
                    'total_detections': len(final_detections),
                    'clocks': final_detections,
                    'timestamp': time.time()
                }
        
        return None
    
    def _assign_clock_types(self, detections: List[Dict]) -> List[Dict]:
        """Assign top/bottom clock types based on Y positions."""
        if not detections:
            return []
        
        # Sort by Y position
        sorted_by_y = sorted(detections, key=lambda d: d['position'][1])
        
        # Split into top and bottom halves
        mid_idx = len(sorted_by_y) // 2
        
        top_clocks = sorted_by_y[:mid_idx]
        bottom_clocks = sorted_by_y[mid_idx:]
        
        # Assign types
        for clock in top_clocks:
            clock['clock_type'] = 'top'
            
        for clock in bottom_clocks:
            clock['clock_type'] = 'bottom'
        
        # Assign states (just use simple numbering for now)
        states = ['play', 'start1', 'start2', 'end1', 'end2', 'end3']
        
        for clock_list in [top_clocks, bottom_clocks]:
            for i, clock in enumerate(clock_list):
                if i < len(states):
                    clock['state'] = states[i]
                else:
                    clock['state'] = f'extra_{i}'
        
        return top_clocks + bottom_clocks

def test_improved_detector():
    """Test the improved detector."""
    print("Testing Improved Clock Detector")
    print("=" * 40)
    
    # First detect board
    from board_detector import BoardDetector
    board_detector = BoardDetector()
    board_data = board_detector.find_chess_board()
    
    if not board_data:
        print("❌ Could not detect board")
        return
    
    print(f"✅ Board detected: {board_data['position']}")
    
    # Test improved clock detector
    clock_detector = ImprovedClockDetector(board_data)
    clock_result = clock_detector.find_clocks()
    
    if clock_result:
        print(f"✅ Found {clock_result['total_detections']} potential clocks")
        
        for clock in clock_result['clocks'][:5]:  # Show top 5
            pos = clock['position']
            print(f"  {clock['clock_type']} ({clock['state']}): ({pos[0]}, {pos[1]}) "
                  f"confidence={clock['confidence']:.3f}")
    else:
        print("❌ No clocks detected")

if __name__ == "__main__":
    test_improved_detector()
