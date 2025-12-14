#!/usr/bin/env python3
"""
Coordinate Calculator Module

Derives notation and rating positions from board and clock detections.
All positions are calculated relative to the detected board position.
"""

from typing import Dict, Optional, Tuple

from .config import ChessConfig


class CoordinateCalculator:
    """
    Calculates UI element positions relative to board and clock detections.
    
    Once we have:
    - Board position (x, y, size)
    - Clock X position
    
    We can derive:
    - Notation panel position
    - Rating positions (4 variants)
    
    All dimensions and offsets are scaled based on detected board size.
    """
    
    # Reference board size (all dimensions are relative to this)
    REFERENCE_BOARD_SIZE = 848
    
    # Standard dimensions at reference size
    REF_NOTATION_WIDTH = 166
    REF_NOTATION_HEIGHT = 104
    REF_RATING_WIDTH = 40
    REF_RATING_HEIGHT = 24
    REF_CLOCK_WIDTH = 147
    REF_CLOCK_HEIGHT = 44
    
    # Offsets relative to clock X position at reference size (848px board)
    REF_NOTATION_X_OFFSET = 38  # From clock X
    REF_RATING_X_OFFSET = 180   # From clock X (rating is just right of player name)
    REF_CLOCK_GAP = 29          # Gap between board and clock
    
    def __init__(self, board_detection: Optional[Dict] = None,
                 clock_detection: Optional[Dict] = None):
        """
        Initialise calculator.
        
        Args:
            board_detection: Board detection result.
            clock_detection: Clock detection result.
        """
        self.board = board_detection
        self.clocks = clock_detection
        self.scale = 1.0
        
        if board_detection:
            self.scale = board_detection['size'] / self.REFERENCE_BOARD_SIZE
    
    def set_board(self, board_detection: Dict):
        """Set board detection result and update scale."""
        self.board = board_detection
        self.scale = board_detection['size'] / self.REFERENCE_BOARD_SIZE
    
    def set_clocks(self, clock_detection: Dict):
        """Set clock detection result."""
        self.clocks = clock_detection
    
    def calculate_all(self) -> Dict:
        """
        Calculate all UI element coordinates.
        
        All dimensions and positions are scaled based on detected board size.
        
        Returns:
            Complete coordinates dictionary.
        """
        if self.board is None:
            raise ValueError("Board detection required")
        
        board_x = self.board['x']
        board_y = self.board['y']
        board_size = self.board['size']
        step = board_size // 8
        
        # Calculate scaled dimensions
        clock_width = int(self.REF_CLOCK_WIDTH * self.scale)
        clock_height = int(self.REF_CLOCK_HEIGHT * self.scale)
        notation_width = int(self.REF_NOTATION_WIDTH * self.scale)
        notation_height = int(self.REF_NOTATION_HEIGHT * self.scale)
        rating_width = int(self.REF_RATING_WIDTH * self.scale)
        rating_height = int(self.REF_RATING_HEIGHT * self.scale)
        
        # Get clock X position
        if self.clocks and 'clock_x' in self.clocks:
            clock_x = self.clocks['clock_x']
        else:
            # Estimate from board position (scaled gap)
            clock_x = board_x + board_size + int(self.REF_CLOCK_GAP * self.scale)
        
        coordinates = {
            'board': {
                'x': board_x,
                'y': board_y,
                'width': board_size,
                'height': board_size
            }
        }
        
        # Add clock coordinates if available
        if self.clocks:
            if 'bottom_clock' in self.clocks:
                coordinates['bottom_clock'] = self._format_clock_coords(self.clocks['bottom_clock'])
            if 'top_clock' in self.clocks:
                coordinates['top_clock'] = self._format_clock_coords(self.clocks['top_clock'])
        
        # Calculate notation position
        coordinates['notation'] = self._calculate_notation(clock_x, board_y, board_size)
        
        # Calculate rating positions
        coordinates['rating'] = self._calculate_ratings(clock_x, self.clocks)
        
        return coordinates
    
    def _format_clock_coords(self, clock_states: Dict) -> Dict:
        """Format clock coordinates for output."""
        result = {}
        for state, coords in clock_states.items():
            result[state] = {
                'x': int(coords['x']),
                'y': int(coords['y']),
                'width': int(coords.get('width', 147)),
                'height': int(coords.get('height', 44))
            }
        return result
    
    def _calculate_notation(self, clock_x: int, board_y: int, board_size: int) -> Dict:
        """
        Calculate notation panel position (scaled).
        
        The notation panel is between the two clocks, roughly centred vertically.
        """
        notation_x_offset = int(self.REF_NOTATION_X_OFFSET * self.scale)
        notation_width = int(self.REF_NOTATION_WIDTH * self.scale)
        notation_height = int(self.REF_NOTATION_HEIGHT * self.scale)
        
        return {
            'x': clock_x + notation_x_offset,
            'y': board_y + board_size // 2 - notation_height // 2 + int(10 * self.scale),
            'width': notation_width,
            'height': notation_height
        }
    
    def _calculate_ratings(self, clock_x: int, clock_detection: Optional[Dict]) -> Dict:
        """
        Calculate rating positions (scaled).
        
        Ratings appear in the player info lines:
        - Top player info: just BELOW the top clock
        - Bottom player info: just ABOVE the bottom clock
        
        There are 4 positions:
        - opp_white: Opponent rating when we play white (below top clock)
        - own_white: Our rating when we play white (above bottom clock)
        - opp_black: Opponent rating when we play black
        - own_black: Our rating when we play black
        """
        rating_x_offset = int(self.REF_RATING_X_OFFSET * self.scale)
        rating_x = clock_x + rating_x_offset
        rating_width = int(self.REF_RATING_WIDTH * self.scale)
        rating_height = int(self.REF_RATING_HEIGHT * self.scale)
        
        # Get clock positions for reference
        if clock_detection:
            top_clock = clock_detection.get('top_clock', {})
            bottom_clock = clock_detection.get('bottom_clock', {})
            
            if 'play' in top_clock:
                top_clock_y = top_clock['play']['y']
                top_clock_h = top_clock['play'].get('height', int(44 * self.scale))
            else:
                top_clock_y = int(424 * self.scale)
                top_clock_h = int(44 * self.scale)
            
            if 'play' in bottom_clock:
                bottom_clock_y = bottom_clock['play']['y']
            else:
                bottom_clock_y = int(742 * self.scale)
        else:
            top_clock_y = int(424 * self.scale)
            top_clock_h = int(44 * self.scale)
            bottom_clock_y = int(742 * self.scale)
        
        # Player info lines are:
        # - Top player (opponent when white): starts at top_clock_y + clock_height + small gap
        # - Bottom player (our own when white): ends at bottom_clock_y - small gap
        
        gap = int(5 * self.scale)  # Small gap between clock and player info
        player_line_height = int(25 * self.scale)  # Height of player info line
        
        # Top player info starts below top clock
        top_player_y = top_clock_y + top_clock_h + gap
        
        # Bottom player info ends above bottom clock
        bottom_player_y = bottom_clock_y - gap - player_line_height
        
        return {
            'opp_white': {
                'x': rating_x,
                'y': top_player_y,
                'width': rating_width,
                'height': rating_height
            },
            'own_white': {
                'x': rating_x,
                'y': bottom_player_y,
                'width': rating_width,
                'height': rating_height
            },
            'opp_black': {
                'x': rating_x,
                'y': bottom_player_y,  # Same as own_white (positions swap)
                'width': rating_width,
                'height': rating_height
            },
            'own_black': {
                'x': rating_x,
                'y': top_player_y,  # Same as opp_white (positions swap)
                'width': rating_width,
                'height': rating_height
            }
        }
    
    def _get_clock_y(self, clock_states: Dict) -> int:
        """Get Y position from clock states, preferring 'play' state."""
        if 'play' in clock_states:
            return clock_states['play']['y']
        elif clock_states:
            # Use first available state
            first_state = next(iter(clock_states.values()))
            return first_state['y']
        else:
            return 0
    
    def estimate_missing_clock_states(self, detected_clocks: Dict) -> Dict:
        """
        Estimate missing clock states based on detected ones.
        
        Uses known Y offsets between states to fill in gaps.
        All offsets are scaled based on detected board size.
        
        Args:
            detected_clocks: Partially detected clock positions.
        
        Returns:
            Complete clock positions with estimates.
        """
        # Known Y offsets between states (relative to 'play') at reference size
        # These are approximate and may need tuning
        ref_bottom_offsets = {
            'play': 0,
            'start1': 14,
            'start2': 28,
            'end1': 69,
            'end2': 5,
            'end3': 34
        }
        
        ref_top_offsets = {
            'play': 0,
            'start1': -28,
            'start2': -14,
            'end1': -69,
            'end2': 23
        }
        
        # Scale offsets
        bottom_offsets = {k: int(v * self.scale) for k, v in ref_bottom_offsets.items()}
        top_offsets = {k: int(v * self.scale) for k, v in ref_top_offsets.items()}
        
        # Default clock dimensions (scaled)
        default_width = int(self.REF_CLOCK_WIDTH * self.scale)
        default_height = int(self.REF_CLOCK_HEIGHT * self.scale)
        
        result = {'bottom_clock': {}, 'top_clock': {}}
        
        for clock_type, offsets in [('bottom_clock', bottom_offsets), ('top_clock', top_offsets)]:
            if clock_type not in detected_clocks or not detected_clocks[clock_type]:
                continue
            
            states = detected_clocks[clock_type]
            
            # Find reference state (prefer 'play')
            if 'play' in states:
                ref_state = 'play'
            else:
                ref_state = next(iter(states.keys()))
            
            ref_coords = states[ref_state]
            ref_offset = offsets.get(ref_state, 0)
            
            # Calculate base Y (what 'play' Y would be)
            base_y = ref_coords['y'] - ref_offset
            
            # Fill in all states
            for state, offset in offsets.items():
                if state in states:
                    # Use detected value
                    result[clock_type][state] = states[state].copy()
                    result[clock_type][state]['estimated'] = False
                else:
                    # Estimate from offset
                    result[clock_type][state] = {
                        'x': ref_coords['x'],
                        'y': base_y + offset,
                        'width': ref_coords.get('width', default_width),
                        'height': ref_coords.get('height', default_height),
                        'estimated': True
                    }
        
        return result


def calculate_coordinates(board_detection: Dict,
                         clock_detection: Optional[Dict] = None) -> Dict:
    """
    Convenience function to calculate all coordinates.
    
    Args:
        board_detection: Board detection result.
        clock_detection: Optional clock detection result.
    
    Returns:
        Complete coordinates dictionary.
    """
    calculator = CoordinateCalculator(board_detection, clock_detection)
    return calculator.calculate_all()
