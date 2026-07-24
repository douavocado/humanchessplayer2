#!/usr/bin/env python3
"""
Panel Detector Module

Detects clocks and info elements to the right of the chess board.

The detection strategy:
1. Look for dark text blocks (clock digits) on the light Lichess background
2. Identify clock regions by their characteristic shape (wide, short, aspect ~6:1)
3. Use the largest matching blocks at top and bottom as clocks
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

from .utils import extract_region


class ClockTextDetector:
    """
    Detects clock regions by finding dark text blocks on light background.
    
    Lichess uses dark text for clock digits on a light grey background.
    Clock displays have a characteristic wide aspect ratio (~6:1).
    """
    
    # Text detection threshold (pixels darker than this are considered text)
    TEXT_THRESHOLD = 100
    
    # Clock aspect ratio range (width/height)
    CLOCK_ASPECT_MIN = 3.0
    CLOCK_ASPECT_MAX = 10.0
    
    # Minimum clock dimensions (relative to board size)
    MIN_CLOCK_WIDTH_RATIO = 0.10  # At least 10% of board size
    MIN_CLOCK_HEIGHT_RATIO = 0.015  # At least 1.5% of board size

    # Maximum clock width (relative to board size). A real clock chip is a
    # compact element; a block this wide is almost always several merged UI
    # elements (nav bar, ad text) that happen to land in the clock aspect
    # range, not an actual clock.
    MAX_CLOCK_WIDTH_RATIO = 0.25
    
    def __init__(self, board_detection: Optional[Dict] = None):
        """
        Initialise clock text detector.
        
        Args:
            board_detection: Board detection result.
        """
        self.board = board_detection
        self.read_clock = None
        self._load_read_clock()
    
    def _load_read_clock(self):
        """Load the read_clock function from image_scrape_utils."""
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from chessimage.image_scrape_utils import read_clock
            self.read_clock = read_clock
        except ImportError:
            print("Warning: Could not import read_clock from image_scrape_utils")
            self.read_clock = None
    
    def set_board(self, board_detection: Dict):
        """Set the board detection result."""
        self.board = board_detection
    
    def detect(self, image: np.ndarray) -> Optional[Dict]:
        """
        Detect clock regions in the image.
        
        Args:
            image: BGR image.
        
        Returns:
            Dictionary with clock detections:
            {
                'top_clock': {'x', 'y', 'width', 'height'},
                'bottom_clock': {'x', 'y', 'width', 'height'},
                'all_text_blocks': [...]  # All detected text blocks
            }
            Or None if not found.
        """
        if image is None or image.size == 0:
            return None
        
        if self.board is None:
            print("Error: Board detection required for clock detection")
            return None
        
        img_h, img_w = image.shape[:2]
        board_x = self.board['x']
        board_y = self.board['y']
        board_size = self.board['size']
        
        # Define search region. Lichess puts the clock in a side panel to the
        # right of the board; chess.com puts it inside the board's own right
        # edge, in the header/footer bar above/below the board (~75-100% of
        # board width from board_x). Start the search early enough to cover
        # both without shrinking the original rightward (Lichess) reach.
        search_x_start = board_x + int(board_size * 0.75)
        search_x_end = min(img_w, board_x + board_size + int(board_size * 0.4))
        search_y_start = max(0, board_y - int(board_size * 0.1))
        search_y_end = min(img_h, board_y + board_size + int(board_size * 0.1))
        
        # Extract search region
        search_region = image[search_y_start:search_y_end, 
                              search_x_start:search_x_end]
        
        if search_region.size == 0:
            return None
        
        print(f"Clock search region: ({search_x_start}, {search_y_start}) to "
              f"({search_x_end}, {search_y_end})")
        
        # Find text blocks. Lichess renders the clock as dark digits on a
        # light panel, so a dark-pixel mask isolates the digit strokes.
        # chess.com's dark theme's "active" clock is a mid-grey chip (~150
        # gray) that stands out against the near-black page (~40-46 gray);
        # a narrow mid-grey band mask isolates that whole chip as one blob
        # (the embedded digit strokes just become holes in it, which
        # findContours' outer-boundary mode ignores) without also picking up
        # pure-white nav/ad text elsewhere in the region, since that text
        # renders much brighter (~210+) than the chip. Run both passes
        # unconditionally and combine candidates.
        text_blocks = self._find_text_blocks(search_region, board_size, mode='dark')
        text_blocks += self._find_text_blocks(search_region, board_size, mode='chip')

        print(f"Found {len(text_blocks)} text blocks")

        if not text_blocks:
            return None

        # Adjust coordinates to full image space
        for block in text_blocks:
            block['x'] += search_x_start
            block['y'] += search_y_start

        # Identify top and bottom clocks
        top_clock, bottom_clock = self._identify_clocks(
            text_blocks, board_y, board_size
        )

        # A whole-region 'light' pass (for the "inactive" white-on-dark
        # clock) is unsafe here: chess.com's header/footer bars are packed
        # with other white UI text (nav tabs, ads) only ~30px from the
        # clock, well within the dilation kernel's reach, so it merges into
        # one wide non-clock-shaped blob and the real digits get lost in it.
        # Once one clock is found by the 'dark' pass, though, we know
        # clock_x precisely - re-run 'light' mode restricted to a narrow
        # column around that x, which excludes the unrelated UI text.
        if top_clock and not bottom_clock:
            bottom_clock = self._find_missing_clock_light(
                image, top_clock, board_y, board_size, half='bottom'
            )
        elif bottom_clock and not top_clock:
            top_clock = self._find_missing_clock_light(
                image, bottom_clock, board_y, board_size, half='top'
            )

        if top_clock:
            print(f"Top clock: ({top_clock['x']}, {top_clock['y']}) "
                  f"{top_clock['width']}x{top_clock['height']}")
        if bottom_clock:
            print(f"Bottom clock: ({bottom_clock['x']}, {bottom_clock['y']}) "
                  f"{bottom_clock['width']}x{bottom_clock['height']}")
        
        # Validate with OCR if available
        if self.read_clock and top_clock:
            is_valid, time_val = self._validate_clock(image, top_clock)
            top_clock['validated'] = is_valid
            top_clock['time_value'] = time_val
            print(f"  Top clock OCR: valid={is_valid}, value={time_val}")
        
        if self.read_clock and bottom_clock:
            is_valid, time_val = self._validate_clock(image, bottom_clock)
            bottom_clock['validated'] = is_valid
            bottom_clock['time_value'] = time_val
            print(f"  Bottom clock OCR: valid={is_valid}, value={time_val}")
        
        return {
            'top_clock': top_clock,
            'bottom_clock': bottom_clock,
            'all_text_blocks': text_blocks
        }

    def _find_missing_clock_light(self, image: np.ndarray, found_clock: Dict,
                                  board_y: int, board_size: int, half: str) -> Optional[Dict]:
        """
        Look for the other clock using a 'light' (white-on-dark) pass,
        restricted to a narrow column around an already-found clock's x
        position and a thin band hugging the board's top/bottom edge.

        Args:
            image: Full BGR image.
            found_clock: The already-identified clock (gives us clock_x).
            board_y: Board Y position.
            board_size: Board size.
            half: 'top' or 'bottom' - which edge band to search.

        Returns:
            Clock block dict (full-image coordinates) or None.
        """
        img_h, img_w = image.shape[:2]

        margin = int(found_clock['width'] * 0.5)
        x_start = max(0, found_clock['x'] - margin)
        x_end = min(img_w, found_clock['x'] + found_clock['width'] + margin)

        edge_band = int(board_size * 0.15)
        if half == 'top':
            y_start = max(0, board_y - edge_band)
            y_end = board_y
        else:
            y_start = board_y + board_size
            y_end = min(img_h, board_y + board_size + edge_band)

        if x_end <= x_start or y_end <= y_start:
            return None

        region = image[y_start:y_end, x_start:x_end]
        if region.size == 0:
            return None

        blocks = self._find_text_blocks(region, board_size, mode='light')
        clock_like = [b for b in blocks if b['is_clock_like']]
        if not clock_like:
            return None

        best = max(clock_like, key=lambda b: b['area'])
        best['x'] += x_start
        best['y'] += y_start
        return best

    # Mid-grey band for the chess.com-style "active" clock chip: bright
    # enough to stand out from the near-black page (~40-46), but capped
    # below pure-white nav/ad text (~210+) so a full-region scan doesn't
    # merge the chip with unrelated bright UI text.
    CHIP_LOWER = 90
    CHIP_UPPER = 210

    # Brightness floor for plain white-on-dark clock digits (chess.com's
    # "inactive" clock, which has no chip - the digits themselves are the
    # light thing here). Only safe to scan for over a narrow column already
    # anchored to a known clock_x (see _find_missing_clock_light) - over a
    # full region this also matches every other white UI label.
    LIGHT_TEXT_THRESHOLD = 180

    def _find_text_blocks(self, region: np.ndarray,
                          board_size: int, mode: str = 'dark') -> List[Dict]:
        """
        Find text blocks / clock chips in a region.

        mode='dark': detect dark pixels on light background (Lichess: dark
        digit strokes are themselves the distinguishing feature).
        mode='chip': detect a mid-grey blob (chess.com's "active" clock: a
        light-grey chip that stands out against a near-black surround,
        without also catching pure-white text elsewhere in the region).
        mode='light': detect bright (near-white) pixels (chess.com's
        "inactive" clock: plain white digits directly on the dark page).

        Args:
            region: BGR image region to search.
            board_size: Board size for scaling thresholds.
            mode: 'dark', 'chip', or 'light'.

        Returns:
            List of text block dictionaries.
        """
        h, w = region.shape[:2]

        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        if mode == 'chip':
            dark_mask = cv2.inRange(gray, self.CHIP_LOWER, self.CHIP_UPPER)
        elif mode == 'light':
            _, dark_mask = cv2.threshold(gray, self.LIGHT_TEXT_THRESHOLD, 255,
                                          cv2.THRESH_BINARY)
        else:
            # Threshold to find dark text
            _, dark_mask = cv2.threshold(gray, self.TEXT_THRESHOLD, 255,
                                          cv2.THRESH_BINARY_INV)

        # Dilate horizontally to connect characters in same text block
        kernel_h = np.ones((1, 20), np.uint8)
        dilated = cv2.dilate(dark_mask, kernel_h, iterations=2)
        
        # Also dilate slightly vertically to connect multi-line elements
        kernel_v = np.ones((3, 1), np.uint8)
        dilated = cv2.dilate(dilated, kernel_v, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and collect text blocks
        min_width = int(board_size * self.MIN_CLOCK_WIDTH_RATIO)
        min_height = int(board_size * self.MIN_CLOCK_HEIGHT_RATIO)
        max_width = int(board_size * self.MAX_CLOCK_WIDTH_RATIO)

        text_blocks = []

        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)

            # Skip tiny or implausibly wide (merged-UI) blocks
            if bw < min_width or bh < min_height or bw > max_width:
                continue
            
            # Add vertical padding to text blocks to ensure full digit coverage.
            # Tight bounding boxes often chop the very top or bottom of digits.
            # We add 15% of the height as padding top and bottom.
            v_padding = int(bh * 0.15)
            y = max(0, y - v_padding)
            bh = bh + v_padding * 2
            
            aspect = bw / bh if bh > 0 else 0
            area = bw * bh
            
            text_blocks.append({
                'x': x,
                'y': y,
                'width': bw,
                'height': bh,
                'aspect': aspect,
                'area': area,
                'is_clock_like': self.CLOCK_ASPECT_MIN <= aspect <= self.CLOCK_ASPECT_MAX
            })
        
        # Sort by area (largest first)
        text_blocks.sort(key=lambda b: b['area'], reverse=True)
        
        return text_blocks
    
    def _identify_clocks(self, text_blocks: List[Dict],
                         board_y: int, board_size: int
                         ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Identify top and bottom clocks from text blocks.
        
        Clocks are the largest clock-like blocks in the top and bottom
        portions of the board area.
        
        Args:
            text_blocks: List of detected text blocks.
            board_y: Board Y position.
            board_size: Board size.
        
        Returns:
            (top_clock, bottom_clock) tuple.
        """
        # Filter to clock-like blocks
        clock_like = [b for b in text_blocks if b['is_clock_like']]
        
        if not clock_like:
            # Fallback: use largest blocks with reasonable aspect
            clock_like = [b for b in text_blocks 
                         if b['aspect'] > 2.0 and b['aspect'] < 15.0]
        
        if not clock_like:
            return None, None
        
        # Board vertical midpoint
        board_mid = board_y + board_size // 2
        
        # Split into top and bottom candidates
        top_candidates = []
        bottom_candidates = []
        
        for block in clock_like:
            block_mid_y = block['y'] + block['height'] // 2
            
            if block_mid_y < board_mid:
                top_candidates.append(block)
            else:
                bottom_candidates.append(block)
        
        # Get the best (largest) from each region
        top_clock = top_candidates[0] if top_candidates else None
        bottom_clock = bottom_candidates[0] if bottom_candidates else None
        
        return top_clock, bottom_clock
    
    def _validate_clock(self, image: np.ndarray, 
                        clock_region: Dict) -> Tuple[bool, Optional[int]]:
        """
        Validate a clock region by attempting to read the time.
        
        Handles both dark-on-light and light-on-dark text by trying
        both orientations.
        
        Args:
            image: Full BGR image.
            clock_region: Clock region dict with x, y, width, height.
        
        Returns:
            (is_valid, time_value) tuple.
        """
        if self.read_clock is None:
            return False, None
        
        region = extract_region(image, clock_region['x'], clock_region['y'],
                               clock_region['width'], clock_region['height'])
        
        if region is None:
            return False, None
        
        # Resize to standard size for read_clock (147x44)
        resized = cv2.resize(region, (147, 44), interpolation=cv2.INTER_AREA)
        
        # Try original (works for white-on-dark)
        time_value = self.read_clock(resized)
        
        if time_value is not None:
            return True, time_value
        
        # Try inverted (works for dark-on-light)
        # This is needed for Lichess which uses dark text on light background
        inverted = cv2.bitwise_not(resized)
        time_value = self.read_clock(inverted)
        
        return time_value is not None, time_value


def detect_clocks(image: np.ndarray, 
                  board_detection: Dict) -> Optional[Dict]:
    """
    Convenience function to detect clocks.
    
    Args:
        image: BGR image.
        board_detection: Board detection result.
    
    Returns:
        Dictionary with clock detections or None.
    """
    detector = ClockTextDetector(board_detection)
    return detector.detect(image)
