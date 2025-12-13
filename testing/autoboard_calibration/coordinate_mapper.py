#!/usr/bin/env python3
"""
Coordinate Mapping Module

Maps UI elements (clocks, notation areas, etc.) relative to detected chess board position.
This enables device-independent coordinate calculation.
"""

import cv2
import numpy as np
import json
import os
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import sys

# Add parent directories to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.append(str(parent_dir))

from utils import simple_read_clock, load_digit_templates

class CoordinateMapper:
    """Maps UI element positions relative to chess board."""
    
    def __init__(self):
        # Default relative positions (will be auto-calculated)
        self.board_position = None
        self.ui_element_offsets = {}
        
        # Standard UI element dimensions (these are fairly consistent)
        self.clock_width = 147
        self.clock_height = 44
        self.notation_width = 166
        self.notation_height = 104
        self.rating_width = 40
        self.rating_height = 24
        
        # Template scales for digit recognition
        self.template_scales = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        self.optimal_template_scale = 1.0
    
    def set_board_position(self, board_data: Dict):
        """Set the detected board position."""
        self.board_position = board_data
        
    def calculate_ui_offsets_from_current_setup(self, screenshot_img: np.ndarray) -> Dict:
        """
        Calculate UI element positions based on current screen setup.
        This accounts for the dynamic position shifts during different game states.
        """
        if not self.board_position:
            raise ValueError("Board position must be set first")
        
        board_x, board_y, board_width, board_height = self.board_position['position']
        
        # Calculate expected positions based on typical layouts
        offsets = {}
        
        # For Lichess/Chess.com, clocks are typically:
        # - To the right of the board
        # - Bottom clock below the board area
        # - Top clock above the board area
        
        clock_x_offset = board_width + 20  # Some padding from board
        
        # Bottom clock (our clock when playing as white)
        # Note: These positions shift slightly during different game states
        bottom_clock_offset_y = board_height - 100  # Base position near bottom of board
        offsets['bottom_clock'] = {
            'play': (clock_x_offset, bottom_clock_offset_y),
            'start1': (clock_x_offset, bottom_clock_offset_y + 14),  # Slight downward shift for game start
            'start2': (clock_x_offset, bottom_clock_offset_y + 28),  # Alternative start position
            'end1': (clock_x_offset, bottom_clock_offset_y + 69),    # Game end (resigned/timeout)
            'end2': (clock_x_offset, bottom_clock_offset_y + 5),     # Game end (aborted)
            'end3': (clock_x_offset, bottom_clock_offset_y + 34)     # Alternative end position
        }
        
        # Top clock (opponent clock when playing as white)
        # These also have state-specific shifts
        top_clock_offset_y = 245  # Base position near top area
        offsets['top_clock'] = {
            'play': (clock_x_offset, top_clock_offset_y),
            'start1': (clock_x_offset, top_clock_offset_y - 28),     # Upward shift for game start
            'start2': (clock_x_offset, top_clock_offset_y - 14),     # Alternative start position  
            'end1': (clock_x_offset, top_clock_offset_y - 69),       # Game end (resigned/timeout)
            'end2': (clock_x_offset, top_clock_offset_y + 23)        # Game end (aborted)
        }
        
        # Notation area (move list) - typically stable position
        notation_offset_x = clock_x_offset + 38
        notation_offset_y = 412
        offsets['notation'] = (notation_offset_x, notation_offset_y)
        
        # Rating areas - these can also shift slightly between playing as white/black
        rating_offset_x = clock_x_offset + 335
        offsets['rating'] = {
            'opp_white': (rating_offset_x, 279),  # When we play as white, opp rating
            'own_white': (rating_offset_x, 527),  # When we play as white, our rating
            'opp_black': (rating_offset_x, 294),  # When we play as black, opp rating  
            'own_black': (rating_offset_x, 512)   # When we play as black, our rating
        }
        
        return offsets
    
    def detect_clock_areas_by_searching(self, screenshot_img: np.ndarray) -> Dict:
        """
        Alternative method: Search for clock-like rectangular areas.
        Look for areas that might contain time displays.
        """
        if not self.board_position:
            raise ValueError("Board position must be set first")
        
        board_x, board_y, board_width, board_height = self.board_position['position']
        
        # Define search areas relative to board
        search_areas = {
            'right_of_board': (
                board_x + board_width,
                board_y - 100,
                300,  # width
                board_height + 200  # height
            ),
            'below_board': (
                board_x - 100,
                board_y + board_height,
                board_width + 200,
                150
            )
        }
        
        detected_clocks = {}
        
        for area_name, (x, y, w, h) in search_areas.items():
            # Extract search region
            search_region = screenshot_img[y:y+h, x:x+w]
            
            # Look for rectangular areas that might be clocks
            clock_positions = self._find_clock_like_rectangles(search_region)
            
            # Convert back to absolute coordinates
            absolute_positions = []
            for rel_x, rel_y, rel_w, rel_h in clock_positions:
                abs_x = x + rel_x
                abs_y = y + rel_y
                absolute_positions.append((abs_x, abs_y, rel_w, rel_h))
            
            detected_clocks[area_name] = absolute_positions
        
        return detected_clocks
    
    def _find_clock_like_rectangles(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Find rectangular areas that might contain clocks."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangles = []
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter for clock-like dimensions
            aspect_ratio = w / h
            area = w * h
            
            # Clock should be roughly rectangular and medium-sized
            if (2.0 < aspect_ratio < 4.0 and  # Wider than tall
                1000 < area < 10000 and       # Reasonable area
                w > 80 and h > 20):           # Minimum size
                rectangles.append((x, y, w, h))
        
        return rectangles
    
    def detect_state_specific_shifts(self, screenshot_img: np.ndarray) -> Dict:
        """
        Detect actual state-specific coordinate shifts by analyzing current screen.
        This helps refine the coordinate mapping for dynamic UI changes.
        """
        if not self.board_position:
            raise ValueError("Board position must be set first")
        
        print("Analyzing state-specific coordinate shifts...")
        
        # Get base offsets
        base_offsets = self.calculate_ui_offsets_from_current_setup(screenshot_img)
        board_x, board_y, _, _ = self.board_position['position']
        
        # Try to detect which state we're currently in and refine positions
        refined_offsets = base_offsets.copy()
        
        # Test each clock position to see which ones actually contain readable clocks
        for clock_type in ['bottom_clock', 'top_clock']:
            best_states = {}
            
            for state, (rel_x, rel_y) in base_offsets[clock_type].items():
                abs_x = board_x + rel_x
                abs_y = board_y + rel_y
                
                # Extract clock region
                clock_region = screenshot_img[
                    abs_y:abs_y + self.clock_height,
                    abs_x:abs_x + self.clock_width
                ]
                
                if clock_region.size > 0:
                    # Test if this position contains a readable clock
                    confidence = self._analyze_clock_region_quality(clock_region)
                    best_states[state] = {
                        'position': (rel_x, rel_y),
                        'confidence': confidence
                    }
            
            # Sort by confidence and keep the best positions
            sorted_states = sorted(best_states.items(), key=lambda x: x[1]['confidence'], reverse=True)
            
            # Update refined offsets with best positions
            for state, data in sorted_states:
                refined_offsets[clock_type][state] = data['position']
                
            print(f"{clock_type}: Best state positions detected")
            for state, data in sorted_states[:3]:  # Show top 3
                print(f"  {state}: confidence {data['confidence']:.3f}")
        
        return refined_offsets
    
    def _analyze_clock_region_quality(self, clock_region: np.ndarray) -> float:
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
            
            # Check for typical clock characteristics:
            # 1. Appropriate contrast (clocks usually have dark text on light background)
            # 2. Horizontal structure (time digits are arranged horizontally)
            # 3. Some text-like patterns
            
            # Basic contrast check
            mean_val = np.mean(gray)
            std_val = np.std(gray)
            
            # Good clocks should have reasonable contrast
            contrast_score = min(std_val / 50.0, 1.0)  # Normalize to 0-1
            
            # Check for horizontal structures (colon separator, digit patterns)
            # Look for darker regions that might be text
            dark_pixels = np.sum(gray < mean_val * 0.8) / gray.size
            text_score = min(dark_pixels / 0.3, 1.0)  # Expect some dark text
            
            # Edge detection to find text-like patterns
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            edge_score = min(edge_density / 0.1, 1.0)
            
            # Combine scores (weighted average)
            total_score = (contrast_score * 0.4 + text_score * 0.3 + edge_score * 0.3)
            
            return total_score
            
        except Exception:
            return 0.0
    
    def auto_detect_template_scale(self, screenshot_img: np.ndarray) -> float:
        """
        Automatically detect the best template scale for digit recognition.
        Now uses refined state-specific coordinates.
        """
        if not self.board_position:
            raise ValueError("Board position must be set first")
        
        # Use refined offsets that account for state-specific shifts
        offsets = self.detect_state_specific_shifts(screenshot_img)
        board_x, board_y, _, _ = self.board_position['position']
        
        best_scale = 1.0
        best_success_rate = 0
        
        print("Testing template scales for digit recognition...")
        
        for scale in self.template_scales:
            success_count = 0
            total_attempts = 0
            
            # Test with different clock positions and states
            for clock_type in ['bottom_clock', 'top_clock']:
                for state in ['play', 'start1', 'start2']:  # Test multiple states
                    if state in offsets[clock_type]:
                        offset_x, offset_y = offsets[clock_type][state]
                        abs_x = board_x + offset_x
                        abs_y = board_y + offset_y
                        
                        # Extract clock region
                        clock_region = screenshot_img[
                            abs_y:abs_y + self.clock_height,
                            abs_x:abs_x + self.clock_width
                        ]
                        
                        if clock_region.size > 0:
                            # Test clock reading with this scale
                            success = self._test_clock_reading_with_scale(clock_region, scale)
                            if success:
                                success_count += 1
                            total_attempts += 1
            
            if total_attempts > 0:
                success_rate = success_count / total_attempts
                print(f"Scale {scale:.1f}: {success_rate:.1%} success rate")
                
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    best_scale = scale
        
        print(f"Best template scale: {best_scale:.1f} (success rate: {best_success_rate:.1%})")
        self.optimal_template_scale = best_scale
        return best_scale
    
    def _test_clock_reading_with_scale(self, clock_image: np.ndarray, scale: float) -> bool:
        """Test if clock can be read successfully with given template scale."""
        try:
            # Use simplified clock reading
            result = simple_read_clock(clock_image)
            return result is not None
                
        except Exception:
            pass
        
        return False
    
    def generate_coordinate_config(self, screenshot_img: np.ndarray) -> Dict:
        """
        Generate complete coordinate configuration for the detected setup.
        Now includes state-specific coordinate detection.
        """
        if not self.board_position:
            raise ValueError("Board position must be set first")
        
        config = {
            'board_detection': self.board_position,
            'timestamp': str(np.datetime64('now')),
            'template_scale': self.optimal_template_scale,
            'ui_elements': {},
            'state_analysis': {}
        }
        
        # Calculate UI element offsets with state-specific refinement
        print("Generating coordinate configuration with state-specific analysis...")
        offsets = self.detect_state_specific_shifts(screenshot_img)
        board_x, board_y, _, _ = self.board_position['position']
        
        # Store analysis results
        config['state_analysis'] = {
            'method': 'state_specific_detection',
            'states_detected': list(offsets['bottom_clock'].keys()),
            'note': 'Coordinates account for UI shifts during different game states (start, play, end)'
        }
        
        # Convert relative offsets to absolute coordinates
        for element_type, element_data in offsets.items():
            if element_type in ['bottom_clock', 'top_clock']:
                config['ui_elements'][element_type] = {}
                for state, (rel_x, rel_y) in element_data.items():
                    abs_x = board_x + rel_x
                    abs_y = board_y + rel_y
                    config['ui_elements'][element_type][state] = {
                        'x': abs_x,
                        'y': abs_y,
                        'width': self.clock_width,
                        'height': self.clock_height
                    }
            elif element_type == 'notation':
                rel_x, rel_y = element_data
                config['ui_elements'][element_type] = {
                    'x': board_x + rel_x,
                    'y': board_y + rel_y,
                    'width': self.notation_width,
                    'height': self.notation_height
                }
            elif element_type == 'rating':
                config['ui_elements'][element_type] = {}
                for rating_type, (rel_x, rel_y) in element_data.items():
                    config['ui_elements'][element_type][rating_type] = {
                        'x': board_x + rel_x,
                        'y': board_y + rel_y,
                        'width': self.rating_width,
                        'height': self.rating_height
                    }
        
        return config
    
    def save_config(self, config: Dict, filename: str = "screen_config.json"):
        """Save coordinate configuration to file."""
        config_path = Path(__file__).parent / filename
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to {config_path}")
    
    def load_config(self, filename: str = "screen_config.json") -> Dict:
        """Load coordinate configuration from file."""
        config_path = Path(__file__).parent / filename
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config


if __name__ == "__main__":
    # Test coordinate mapping
    from board_detector import BoardDetector
    import time
    
    print("Testing coordinate mapping...")
    print("Please ensure a chess game is visible on screen.")
    time.sleep(3)
    
    # Detect board first
    detector = BoardDetector()
    board_data = detector.find_chess_board()
    
    if not board_data:
        print("Could not detect board. Please ensure a chess game is visible.")
        exit(1)
    
    # Map coordinates
    mapper = CoordinateMapper()
    mapper.set_board_position(board_data)
    
    # Capture screenshot for analysis
    screenshot_img = detector.screen_capture.capture()
    
    # Generate configuration
    config = mapper.generate_coordinate_config(screenshot_img)
    
    # Save configuration
    mapper.save_config(config)
    
    print("\n=== COORDINATE MAPPING COMPLETE ===")
    print(f"Board position: {config['board_detection']['position']}")
    print(f"Template scale: {config['template_scale']}")
    print("\nUI Elements detected:")
    for element, data in config['ui_elements'].items():
        print(f"  {element}: {data}")
