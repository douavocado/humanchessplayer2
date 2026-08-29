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
from .config import DEFAULT_SITE


class ClockTextDetector:
    """
    Detects clock regions by finding dark text blocks on light background.
    
    Lichess uses dark text for clock digits on a light grey background.
    Clock displays have a characteristic wide aspect ratio (~6:1).

    chess.com needs a different strategy entirely (see _detect_chess_com),
    selected by the ``site`` argument.
    """

    # ---- chess.com chip detection -------------------------------------
    # chess.com renders each clock as a rounded rectangular "chip" sitting
    # in the player row above/below the board, right-aligned to the board's
    # own right edge. The chip's *fill* carries the state: near-white when
    # that side is to move, near-black (a shade off the page) when it is
    # not, red under time pressure. No single brightness threshold spans
    # those, so the chip is found by difference from the band's background
    # colour instead - which is what every one of them has in common.
    CC_SEARCH_X_START_RATIO = 0.50   # of board size, right of board_x
    CC_SEARCH_X_END_GAP_RATIO = 0.06  # past the board's right edge
    CC_TOP_BAND_RATIO = 0.20         # height of the band above the board
    CC_BOTTOM_BAND_RATIO = 0.25      # ...and below it (taller: rank labels)
    # The inactive chip differs from the page by only ~10 grey levels, so
    # this has to stay low; h264 compression noise in a screen recording
    # sits well under it.
    CC_BG_DELTA = 6
    CC_CLOSE_KERNEL = (5, 9)
    CC_MIN_WIDTH_RATIO = 0.15
    CC_MAX_WIDTH_RATIO = 0.45
    CC_MIN_HEIGHT_RATIO = 0.05
    CC_ASPECT_MIN = 2.0
    CC_ASPECT_MAX = 8.0

    # ---- chess.com player row (name + inline rating) ------------------
    CC_ROW_CHIP_GAP = 4          # stop this far short of the clock chip
    CC_ROW_TEXT_LEVEL = 150      # light text against the dark player row
    CC_ROW_AVATAR_FILL = 0.5     # column lit over this much of the row height
    CC_ROW_AVATAR_GAP = 12       # clear columns that end the avatar block
    CC_ROW_MIN_LIT = 12          # lit pixels making a row "text", not icons
    CC_ROW_MIN_HEIGHT = 6        # ignore runs thinner than this
    # The lit-pixel test finds the bulk of the text but trims descenders and
    # cap tops, which cost real OCR accuracy ("gold71" read as "aold71").
    # Pad back out, clipped to the chip band so the captured-piece row below
    # still cannot leak in.
    CC_ROW_PAD_RATIO = 0.35
    
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
    
    def __init__(self, board_detection: Optional[Dict] = None,
                 site: Optional[str] = None):
        """
        Initialise clock text detector.
        
        Args:
            board_detection: Board detection result.
            site: Which site the screenshots came from. chess.com puts its
                  clocks in the player rows above/below the board rather
                  than in a side panel, so it takes a separate detector.
        """
        self.board = board_detection
        self.site = site or DEFAULT_SITE
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

        if self.site == "chess_com":
            return self._detect_chess_com(image)

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

    def _detect_chess_com(self, image: np.ndarray) -> Optional[Dict]:
        """
        Detect chess.com's clock chips above and below the board.

        The Lichess detector keys off the digits themselves, which does not
        transfer: chess.com's inactive clock draws mid-grey digits on a
        near-black chip, and that blob is both too square (aspect ~1.8) and
        too low-contrast to survive the shared filters. The chip around the
        digits is the stable thing - always the same size and x position
        whatever state it is in - so that is what this finds.

        Two bands are searched, one above the board and one below, each
        clipped on the left so the player name, avatar and captured-piece
        icons stay out of it, and on the right just past the board edge to
        exclude the side panel (whose move list and nav tabs are what the
        Lichess detector kept locking onto). Within a band the chip is the
        right-most blob of the right shape, since chess.com right-aligns it
        to the board.

        Args:
            image: BGR image.

        Returns:
            Same shape as detect(): top_clock/bottom_clock/all_text_blocks.
        """
        img_h, img_w = image.shape[:2]
        board_x = self.board['x']
        board_y = self.board['y']
        board_size = self.board['size']

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        x_start = board_x + int(board_size * self.CC_SEARCH_X_START_RATIO)
        x_end = min(img_w, board_x + board_size
                    + int(board_size * self.CC_SEARCH_X_END_GAP_RATIO))

        bands = {
            'top_clock': (max(0, board_y - int(board_size * self.CC_TOP_BAND_RATIO)),
                          board_y),
            'bottom_clock': (board_y + board_size,
                             min(img_h, board_y + board_size
                                 + int(board_size * self.CC_BOTTOM_BAND_RATIO))),
        }

        print(f"Clock search region (chess.com): x {x_start}..{x_end}, "
              f"bands {bands['top_clock']} / {bands['bottom_clock']}")

        found = {}
        all_blocks = []

        for name, (y_start, y_end) in bands.items():
            if x_end <= x_start or y_end <= y_start:
                found[name] = None
                continue

            band = gray[y_start:y_end, x_start:x_end]
            if band.size == 0:
                found[name] = None
                continue

            blocks = self._find_chips(band, board_size)
            for b in blocks:
                b['x'] += x_start
                b['y'] += y_start
            all_blocks.extend(blocks)

            # Right-most chip: chess.com right-aligns the clock to the board.
            found[name] = max(blocks, key=lambda b: b['x'] + b['width']) if blocks else None

        top_clock = found.get('top_clock')
        bottom_clock = found.get('bottom_clock')

        if top_clock:
            print(f"Top clock: ({top_clock['x']}, {top_clock['y']}) "
                  f"{top_clock['width']}x{top_clock['height']}")
        if bottom_clock:
            print(f"Bottom clock: ({bottom_clock['x']}, {bottom_clock['y']}) "
                  f"{bottom_clock['width']}x{bottom_clock['height']}")

        if not top_clock and not bottom_clock:
            return None

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
            'all_text_blocks': all_blocks,
        }

    def detect_player_row(self, image: np.ndarray, chip: Dict) -> Optional[Dict]:
        """
        Locate the name-and-rating text row beside one chess.com clock chip.

        chess.com writes the rating inline after a username of unpredictable
        length ("AlbertXu2010 (1905)"), so the crop has to be the whole text
        row; a fixed-width box lands on whatever characters happen to fall at
        that x. Deriving the row from board-relative constants does not
        travel either - the row's furniture (avatar, flag, membership icons)
        is not a fixed fraction of the board - which is what left the fitted
        box sitting on the avatar on this layout.

        Detecting it is straightforward once you notice the row and the clock
        chip are the same flex row: the chip gives the vertical band and the
        right-hand limit, and within that band the name is the first dense
        line of light text. Two things have to be excluded:

        - The avatar, which is a solid block lit over the full row height
          (the captured-piece icons below the name are sparse by comparison,
          so a density test separates name from pieces only after the avatar
          is gone). It is square and fills the row, so its width is the band
          height - which catches a dark photo avatar that the density test
          alone would miss.
        - Everything right of the chip.

        Args:
            image: Full BGR frame.
            chip: A detected clock chip (x/y/width/height).

        Returns:
            Row dict with x/y/width/height, or None if no text row was found.
        """
        if image is None or image.size == 0 or self.board is None or not chip:
            return None

        board_x = self.board['x']
        x_start, x_end = board_x, chip['x'] - self.CC_ROW_CHIP_GAP
        height = chip['height']
        if x_end <= x_start or height <= 0:
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        band = gray[chip['y']:chip['y'] + height, x_start:x_end] > self.CC_ROW_TEXT_LEVEL
        if band.size == 0:
            return None

        # Skip the avatar: leading columns lit over most of the row height.
        column_fill = band.mean(axis=0)
        offset = 0
        for index, fill in enumerate(column_fill):
            if fill > self.CC_ROW_AVATAR_FILL:
                offset = index + 1
            elif index - offset > self.CC_ROW_AVATAR_GAP:
                break
        offset = max(offset, height)
        band = band[:, offset:]
        if band.size == 0:
            return None

        rows = band.sum(axis=1) >= self.CC_ROW_MIN_LIT
        runs, start = [], None
        for index, lit in enumerate(list(rows) + [False]):
            if lit and start is None:
                start = index
            elif not lit and start is not None:
                if index - start >= self.CC_ROW_MIN_HEIGHT:
                    runs.append((start, index))
                start = None

        if not runs:
            return None

        # The name sits above the captured-piece row, so the first run is it.
        top, bottom = runs[0]
        columns = np.where(band[top:bottom].sum(axis=0) > 0)[0]
        if not columns.size:
            return None

        pad = max(1, int((bottom - top) * self.CC_ROW_PAD_RATIO))
        top = max(0, top - pad)
        bottom = min(height, bottom + pad)

        return {
            'x': x_start + offset + int(columns.min()),
            'y': chip['y'] + top,
            'width': int(columns.max() - columns.min() + 1),
            'height': bottom - top,
        }

    def _find_chips(self, band: np.ndarray, board_size: int) -> List[Dict]:
        """
        Find chip-shaped blobs in a band, by difference from its background.

        Args:
            band: Grayscale band to search.
            board_size: Board size, for scaling the shape filters.

        Returns:
            List of block dictionaries in band-local coordinates.
        """
        # The band is mostly empty page, so its median is the background.
        background = int(np.median(band))
        mask = (cv2.absdiff(band, np.full_like(band, background))
                > self.CC_BG_DELTA).astype(np.uint8) * 255

        # Close over the digit gaps so a chip reads as one solid blob.
        kernel = np.ones(self.CC_CLOSE_KERNEL, np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        min_width = board_size * self.CC_MIN_WIDTH_RATIO
        max_width = board_size * self.CC_MAX_WIDTH_RATIO
        min_height = board_size * self.CC_MIN_HEIGHT_RATIO

        blocks = []
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            if not (min_width <= bw <= max_width) or bh < min_height:
                continue
            aspect = bw / bh if bh > 0 else 0
            if not (self.CC_ASPECT_MIN <= aspect <= self.CC_ASPECT_MAX):
                continue
            blocks.append({
                'x': x, 'y': y, 'width': bw, 'height': bh,
                'aspect': aspect, 'area': bw * bh,
                'is_clock_like': True,
            })

        blocks.sort(key=lambda b: b['area'], reverse=True)
        return blocks

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


# --- move-navigation bar ("is this a game page?") --------------------------
# The bar of |< < > > >| buttons under the moves list. sites/chess_com.py uses
# it as the one signal that separates a game page from the lobby, whose
# preview board and full clock are otherwise indistinguishable from a game
# that has just begun.
#
# It has to be *measured* rather than derived from the board, because the side
# panel is laid out by the viewport and not by the board: across two devices
# the bar kept the same absolute height (62px vs 65px) while the board changed
# by a factor of 3.5. Board-relative offsets therefore only hold at the board
# size they were fitted on, which is exactly the trap the rating rows fell
# into.
CC_NAV_PANEL_DARK_MAX = 80        # a "dark panel" pixel
CC_NAV_PANEL_MIN_FRACTION = 0.7   # fraction of a column that must be dark
CC_NAV_BUTTON_MIN_DELTA = 8       # button fill sits just above the panel
CC_NAV_BUTTON_MAX_DELTA = 40      # ...but well below its own glyph
CC_NAV_MIN_SIZE = 20              # px, smallest credible button
CC_NAV_ASPECT_MIN = 0.6
CC_NAV_ASPECT_MAX = 2.0
CC_NAV_MIN_BUTTONS = 3            # of five; the disabled two can wash out
CC_NAV_HEIGHT_TOLERANCE = 1.4     # max/min button height within a row
CC_NAV_CLOSE_KERNEL = (7, 7)


def _find_moves_panel(gray: np.ndarray, board: Dict) -> Optional[Tuple[int, int]]:
    """
    Horizontal extent of the dark moves panel to the right of the board.

    Bounding the search this way is not cosmetic. The first attempt searched
    everything right of the board and failed: a bright advertisement further
    right pulled the band's median background above the button fill, so the
    buttons fell outside the threshold and no bar was found at all.

    Args:
        gray: Full-screen grayscale image.
        board: Board detection with 'x', 'y', 'size'.

    Returns:
        (x_start, x_end) of the panel, or None if no dark run was found.
    """
    height, width = gray.shape
    top = max(0, board['y'])
    bottom = min(height, board['y'] + board['size'])
    if bottom <= top:
        return None

    dark = (gray[top:bottom, :] < CC_NAV_PANEL_DARK_MAX).mean(axis=0)

    x = board['x'] + board['size']
    while x < width and dark[x] < CC_NAV_PANEL_MIN_FRACTION:
        x += 1
    start = x
    while x < width and dark[x] >= CC_NAV_PANEL_MIN_FRACTION:
        x += 1

    if x - start < CC_NAV_MIN_SIZE:
        return None
    return start, x


def detect_game_controls(image: np.ndarray,
                         board_detection: Dict) -> Optional[Dict]:
    """
    Locate the chess.com move-navigation bar.

    Args:
        image: Full-screen BGR image of a game page.
        board_detection: Board detection with 'x', 'y', 'size'.

    Returns:
        {'x', 'y', 'width', 'height'} bounding the buttons, or None when the
        bar was not found - which is the correct answer for a lobby screenshot
        as well as for a failure, so callers must not read it as "no bar on
        screen" outside a frame already known to be a game page.
    """
    if image is None or image.size == 0:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    panel = _find_moves_panel(gray, board_detection)
    if panel is None:
        return None
    px0, px1 = panel

    # The bar sits below the moves list, so search the lower part of the panel.
    y0 = board_detection['y'] + board_detection['size'] // 2
    y1 = min(gray.shape[0], board_detection['y'] + 2 * board_detection['size'])
    if y1 <= y0:
        return None

    sub = gray[y0:y1, px0:px1].astype(np.int16)
    background = int(np.median(sub))
    mask = ((sub >= background + CC_NAV_BUTTON_MIN_DELTA)
            & (sub <= background + CC_NAV_BUTTON_MAX_DELTA))
    mask = mask.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones(CC_NAV_CLOSE_KERNEL, np.uint8))

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    buttons = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if w < CC_NAV_MIN_SIZE or h < CC_NAV_MIN_SIZE:
            continue
        # A button is a filled rounded rectangle; text and icons are not.
        if area < 0.5 * w * h:
            continue
        if not (CC_NAV_ASPECT_MIN <= w / h <= CC_NAV_ASPECT_MAX):
            continue
        buttons.append((x + px0, y + y0, w, h))

    # Group by row: the bar is the widest run of same-height buttons sharing a
    # baseline. Anything else in the panel is scattered or differently sized.
    rows: Dict[float, List[Tuple[int, int, int, int]]] = {}
    for button in buttons:
        centre = button[1] + button[3] / 2
        key = next((k for k in rows if abs(k - centre) < 0.5 * button[3]),
                   centre)
        rows.setdefault(key, []).append(button)

    best = None
    for row in rows.values():
        if len(row) < CC_NAV_MIN_BUTTONS:
            continue
        heights = [b[3] for b in row]
        if max(heights) > CC_NAV_HEIGHT_TOLERANCE * min(heights):
            continue
        if best is None or len(row) > len(best):
            best = row

    if best is None:
        return None

    x0 = min(b[0] for b in best)
    x1 = max(b[0] + b[2] for b in best)
    ry0 = min(b[1] for b in best)
    ry1 = max(b[1] + b[3] for b in best)
    return {'x': int(x0), 'y': int(ry0),
            'width': int(x1 - x0), 'height': int(ry1 - ry0)}
