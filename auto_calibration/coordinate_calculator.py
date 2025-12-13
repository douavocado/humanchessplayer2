#!/usr/bin/env python3
"""
Coordinate Calculator Module

Calculates UI element coordinates relative to detected board position.
Handles state-specific coordinate variations for different game phases.
"""

import cv2
import numpy as np
from typing import Dict, Tuple
from calibration_utils import analyze_clock_region_quality, simple_clock_test

class CoordinateCalculator:
    """Calculates UI element positions relative to chess board."""
    
    def __init__(self):
        self.board_position = None
        
        # Standard UI element dimensions (consistent across sites)
        self.clock_width = 147
        self.clock_height = 44
        self.notation_width = 166
        self.notation_height = 104
        self.rating_width = 40
        self.rating_height = 24
        
    def set_board_position(self, board_data: Dict):
        """Set the detected board position."""
        self.board_position = board_data
        
    def calculate_ui_coordinates(self, screenshot_img: np.ndarray) -> Dict:
        """
        Calculate all UI element coordinates based on detected board position.
        Returns absolute coordinates for all game states.
        """
        if not self.board_position:
            raise ValueError("Board position must be set first")
        
        board_x, board_y, board_width, board_height = self.board_position['position']
        
        print(f"Calculating UI coordinates relative to board at ({board_x}, {board_y})")
        
        # Calculate base offsets (these work for most chess sites)
        clock_x_offset = board_width + 20  # Clocks typically to the right of board
        
        # Bottom clock base position (our clock when playing as white)
        bottom_clock_base_y = board_height - 100
        
        # Top clock base position (opponent clock when playing as white)  
        top_clock_base_y = 245
        
        # Define state-specific Y offsets (relative to base position)
        bottom_clock_offsets = {
            'play': 0,       # Base position during normal gameplay
            'start1': 14,    # New game start (primary)
            'start2': 28,    # New game start (alternative) 
            'end1': 69,      # Game over (resigned/timeout)
            'end2': 5,       # Game over (aborted)
            'end3': 34       # Game over (alternative)
        }
        
        top_clock_offsets = {
            'play': 0,       # Base position during normal gameplay
            'start1': -28,   # New game start (moves up)
            'start2': -14,   # New game start (alternative)
            'end1': -69,     # Game over (moves up significantly)
            'end2': 23       # Game over (moves down slightly)
        }
        
        # Calculate absolute coordinates for all states
        coordinates = {
            'board': {
                'x': board_x,
                'y': board_y,
                'width': board_width,
                'height': board_height
            },
            'bottom_clock': {},
            'top_clock': {},
            'notation': {
                'x': board_x + clock_x_offset + 38,
                'y': board_y + 412,
                'width': self.notation_width,
                'height': self.notation_height
            },
            'rating': {
                'opp_white': {
                    'x': board_x + clock_x_offset + 335,
                    'y': board_y + 279,
                    'width': self.rating_width,
                    'height': self.rating_height
                },
                'own_white': {
                    'x': board_x + clock_x_offset + 335,
                    'y': board_y + 527,
                    'width': self.rating_width,
                    'height': self.rating_height
                },
                'opp_black': {
                    'x': board_x + clock_x_offset + 335,
                    'y': board_y + 294,
                    'width': self.rating_width,
                    'height': self.rating_height
                },
                'own_black': {
                    'x': board_x + clock_x_offset + 335,
                    'y': board_y + 512,
                    'width': self.rating_width,
                    'height': self.rating_height
                }
            }
        }
        
        # Calculate bottom clock coordinates for all states
        for state, y_offset in bottom_clock_offsets.items():
            coordinates['bottom_clock'][state] = {
                'x': board_x + clock_x_offset,
                'y': board_y + bottom_clock_base_y + y_offset,
                'width': self.clock_width,
                'height': self.clock_height
            }
        
        # Calculate top clock coordinates for all states
        for state, y_offset in top_clock_offsets.items():
            coordinates['top_clock'][state] = {
                'x': board_x + clock_x_offset,
                'y': board_y + top_clock_base_y + y_offset,
                'width': self.clock_width,
                'height': self.clock_height
            }
        
        return coordinates
    
    def validate_coordinates(self, screenshot_img: np.ndarray, coordinates: Dict) -> Dict:
        """
        Validate calculated coordinates by testing clock detection.
        Returns validation results with success rates for each position.
        """
        print("Validating calculated coordinates...")
        
        validation_results = {
            'total_tested': 0,
            'total_successful': 0,
            'clock_results': {},
            'overall_success_rate': 0.0
        }
        
        # Test bottom and top clock positions
        for clock_type in ['bottom_clock', 'top_clock']:
            if clock_type in coordinates:
                validation_results['clock_results'][clock_type] = {}
                
                for state, coords in coordinates[clock_type].items():
                    validation_results['total_tested'] += 1
                    
                    # Extract clock region
                    x, y = coords['x'], coords['y']
                    w, h = coords['width'], coords['height']
                    
                    # Ensure coordinates are within screenshot bounds
                    if (x >= 0 and y >= 0 and 
                        x + w <= screenshot_img.shape[1] and 
                        y + h <= screenshot_img.shape[0]):
                        
                        clock_region = screenshot_img[y:y+h, x:x+w]
                        
                        # Test clock detection
                        success = simple_clock_test(clock_region)
                        confidence = analyze_clock_region_quality(clock_region)
                        
                        validation_results['clock_results'][clock_type][state] = {
                            'success': success,
                            'confidence': confidence,
                            'coordinates': coords
                        }
                        
                        if success:
                            validation_results['total_successful'] += 1
                            
                        print(f"  {clock_type}.{state}: {'✅' if success else '❌'} (confidence: {confidence:.3f})")
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
