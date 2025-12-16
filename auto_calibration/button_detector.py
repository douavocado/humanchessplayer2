#!/usr/bin/env python3
"""
Button Detection Module for Lichess UI

Dynamically detects UI buttons on Lichess for:
- "PLAY" button in the navigation bar
- Time control buttons (1+0, 2+1, 3+0, etc.) in the play menu

Uses OCR and colour detection to find buttons regardless of screen resolution.
"""

import cv2
import numpy as np
import pytesseract
from typing import Tuple, Optional, Dict, List
from pathlib import Path

from .utils import capture_screenshot, load_image


class ButtonDetector:
    """
    Detects UI buttons on Lichess using OCR and colour matching.
    """
    
    # Lichess navigation bar colour (dark grey) - BGR
    NAV_BAR_COLOUR_BGR = np.array([51, 51, 51])  # #333333
    NAV_BAR_TOLERANCE = 20
    
    # Time control button colours (Lichess uses a light background for buttons)
    BUTTON_BG_LIGHT = np.array([250, 250, 250])  # Light grey/white
    BUTTON_BG_TOLERANCE = 30
    
    # Common time controls to look for
    TIME_CONTROLS = {
        "1+0": "Bullet",
        "2+1": "Bullet", 
        "3+0": "Blitz",
        "3+2": "Blitz",
        "5+0": "Blitz",
        "5+3": "Blitz",
        "10+0": "Rapid",
        "10+5": "Rapid",
        "15+10": "Rapid",
        "30+0": "Classical",
        "30+20": "Classical",
    }
    
    def __init__(self):
        """Initialise the button detector."""
        self._play_button_cache = None
        self._time_control_cache = {}
    
    def find_play_button(self, image: np.ndarray = None) -> Optional[Tuple[int, int]]:
        """
        Find the "PLAY" button in the Lichess navigation bar.
        
        Args:
            image: BGR image to search, or None to capture screenshot.
        
        Returns:
            (x, y) centre coordinates of the PLAY button, or None if not found.
        """
        if image is None:
            image = capture_screenshot()
            if image is None:
                return None
        
        img_h, img_w = image.shape[:2]
        
        # The Lichess nav bar is typically in the top portion, after the browser chrome
        # Search from y=100 to y=250 to skip browser tabs/address bar
        nav_top = min(80, img_h // 20)
        nav_bottom = min(250, img_h // 8)
        nav_region = image[nav_top:nav_bottom, :]
        
        # Use precise matching for "PLAY" - exact match only
        result = self._find_exact_play_button(nav_region, offset_y=nav_top)
        if result is not None:
            return result
        
        # Try the full top region if not found in expected location
        top_region = image[0:min(300, img_h), :]
        result = self._find_exact_play_button(top_region, offset_y=0)
        if result is not None:
            return result
        
        return None
    
    def _find_exact_play_button(
        self,
        region: np.ndarray,
        offset_y: int = 0
    ) -> Optional[Tuple[int, int]]:
        """
        Find the exact "PLAY" text (not "Plays" or other variants).
        
        Args:
            region: BGR image region to search.
            offset_y: Y offset to add to result coordinates.
        
        Returns:
            (x, y) centre coordinates, or None if not found.
        """
        if region is None or region.size == 0:
            return None
        
        # Convert to grayscale
        if region.ndim == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region.copy()
        
        try:
            # Get OCR data with bounding boxes
            data = pytesseract.image_to_data(
                gray,
                output_type=pytesseract.Output.DICT,
                config='--oem 3 --psm 11'
            )
            
            candidates = []
            
            for i, text in enumerate(data['text']):
                if not text.strip():
                    continue
                
                # Look for exact "PLAY" match (case-insensitive but must be the whole word)
                text_clean = text.strip().upper()
                
                # Must be exactly "PLAY" - not "PLAYS", "PLAYED", etc.
                if text_clean == "PLAY":
                    x = data['left'][i]
                    y = data['top'][i]
                    w = data['width'][i]
                    h = data['height'][i]
                    conf = int(data['conf'][i]) if data['conf'][i] != '-1' else 0
                    
                    centre_x = x + w // 2
                    centre_y = offset_y + y + h // 2
                    
                    candidates.append({
                        'pos': (centre_x, centre_y),
                        'conf': conf,
                        'y': y
                    })
            
            if not candidates:
                return None
            
            # If multiple candidates, prefer the one with highest confidence
            # and that's in a reasonable position (not at the very top which would be browser chrome)
            best = max(candidates, key=lambda c: c['conf'])
            return best['pos']
        
        except Exception as e:
            return None
    
    def find_time_control_button(self, time_control: str, image: np.ndarray = None) -> Optional[Tuple[int, int]]:
        """
        Find a specific time control button (e.g., "1+0", "3+0") in the play menu.
        
        Args:
            time_control: Time control string like "1+0", "3+0", etc.
            image: BGR image to search, or None to capture screenshot.
        
        Returns:
            (x, y) centre coordinates of the button, or None if not found.
        """
        if image is None:
            image = capture_screenshot()
            if image is None:
                return None
        
        img_h, img_w = image.shape[:2]
        
        # The time control buttons appear in the centre/upper portion of the screen
        # when the play menu is open
        search_region_top = 0
        search_region_bottom = int(img_h * 0.6)  # Top 60% of screen
        search_region = image[search_region_top:search_region_bottom, :]
        
        # Use OCR to find the time control text
        result = self._find_text_in_region(
            search_region, 
            time_control, 
            offset_y=search_region_top
        )
        
        if result is not None:
            return result
        
        # Try with different formatting (some OCR might add spaces)
        alt_format = time_control.replace("+", " + ")
        result = self._find_text_in_region(
            search_region,
            alt_format,
            offset_y=search_region_top
        )
        
        return result
    
    def find_all_time_controls(self, image: np.ndarray = None) -> Dict[str, Tuple[int, int]]:
        """
        Find all visible time control buttons.
        
        Args:
            image: BGR image to search, or None to capture screenshot.
        
        Returns:
            Dictionary mapping time control strings to (x, y) coordinates.
        """
        if image is None:
            image = capture_screenshot()
            if image is None:
                return {}
        
        results = {}
        
        # Run OCR once and find all time controls
        img_h, img_w = image.shape[:2]
        search_region_top = 0
        search_region_bottom = int(img_h * 0.6)
        search_region = image[search_region_top:search_region_bottom, :]
        
        # Get all text with positions from OCR
        ocr_results = self._get_all_text_positions(search_region)
        
        for time_control in self.TIME_CONTROLS.keys():
            for text, (x, y, w, h) in ocr_results:
                # Check if this text matches the time control
                if self._text_matches_time_control(text, time_control):
                    # Calculate centre position
                    centre_x = x + w // 2
                    centre_y = search_region_top + y + h // 2
                    results[time_control] = (centre_x, centre_y)
                    break
        
        return results
    
    def find_new_opponent_button(self, image: np.ndarray = None) -> Optional[Tuple[int, int]]:
        """
        Find the "NEW OPPONENT" button that appears after a game ends.
        
        Args:
            image: BGR image to search, or None to capture screenshot.
        
        Returns:
            (x, y) centre coordinates of the button, or None if not found.
        """
        if image is None:
            image = capture_screenshot()
            if image is None:
                return None
        
        # Search in the right portion of the screen (where the panel is)
        img_h, img_w = image.shape[:2]
        search_region_left = int(img_w * 0.5)
        search_region = image[:, search_region_left:]
        
        # Try to find "NEW OPPONENT" text
        result = self._find_text_in_region(
            search_region,
            "NEW OPPONENT",
            offset_x=search_region_left
        )
        if result is not None:
            return result
        
        # Also try "OPPONENT" alone
        result = self._find_text_in_region(
            search_region,
            "OPPONENT",
            offset_x=search_region_left
        )
        
        return result
    
    def _find_text_in_region(
        self,
        region: np.ndarray,
        target_text: str,
        offset_x: int = 0,
        offset_y: int = 0,
        case_insensitive: bool = False
    ) -> Optional[Tuple[int, int]]:
        """
        Find text in a region using OCR.
        
        Args:
            region: BGR image region to search.
            target_text: Text to find.
            offset_x: X offset to add to result coordinates.
            offset_y: Y offset to add to result coordinates.
            case_insensitive: Whether to do case-insensitive matching.
        
        Returns:
            (x, y) centre coordinates, or None if not found.
        """
        if region is None or region.size == 0:
            return None
        
        # Convert to grayscale
        if region.ndim == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region.copy()
        
        # Apply threshold to improve OCR
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Also try inverted for dark backgrounds
        thresh_inv = cv2.bitwise_not(thresh)
        
        for img in [gray, thresh, thresh_inv]:
            try:
                # Get OCR data with bounding boxes
                data = pytesseract.image_to_data(
                    img,
                    output_type=pytesseract.Output.DICT,
                    config='--oem 3 --psm 11'
                )
                
                for i, text in enumerate(data['text']):
                    if not text.strip():
                        continue
                    
                    # Check for match
                    if case_insensitive:
                        match = target_text.lower() in text.lower()
                    else:
                        match = target_text in text
                    
                    if match:
                        x = data['left'][i]
                        y = data['top'][i]
                        w = data['width'][i]
                        h = data['height'][i]
                        
                        centre_x = offset_x + x + w // 2
                        centre_y = offset_y + y + h // 2
                        
                        return (centre_x, centre_y)
            
            except Exception as e:
                continue
        
        return None
    
    def _get_all_text_positions(self, region: np.ndarray) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        """
        Get all detected text and their positions from OCR.
        
        Args:
            region: BGR image region to search.
        
        Returns:
            List of (text, (x, y, width, height)) tuples.
        """
        results = []
        
        if region is None or region.size == 0:
            return results
        
        # Convert to grayscale
        if region.ndim == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region.copy()
        
        try:
            data = pytesseract.image_to_data(
                gray,
                output_type=pytesseract.Output.DICT,
                config='--oem 3 --psm 11'
            )
            
            for i, text in enumerate(data['text']):
                if not text.strip():
                    continue
                
                x = data['left'][i]
                y = data['top'][i]
                w = data['width'][i]
                h = data['height'][i]
                
                results.append((text, (x, y, w, h)))
        
        except Exception:
            pass
        
        return results
    
    def _text_matches_time_control(self, text: str, time_control: str) -> bool:
        """
        Check if OCR text matches a time control.
        
        Args:
            text: OCR detected text.
            time_control: Target time control string (e.g., "1+0").
        
        Returns:
            True if they match.
        """
        # Clean up the text
        text = text.strip()
        
        # Exact match
        if text == time_control:
            return True
        
        # Handle common OCR mistakes
        # Remove spaces
        text_clean = text.replace(" ", "")
        tc_clean = time_control.replace(" ", "")
        
        if text_clean == tc_clean:
            return True
        
        # Handle 'l' being read as '1' or vice versa
        text_normalised = text_clean.replace('l', '1').replace('I', '1').replace('O', '0')
        
        if text_normalised == tc_clean:
            return True
        
        return False


class QuickPairingDetector:
    """
    Detects the Quick Pairing buttons on the Lichess home page.
    These are the 9 time control buttons arranged in a 3x3 grid.
    """
    
    # Grid layout for time controls (row, col) -> time_control
    GRID_LAYOUT = {
        (0, 0): "1+0",
        (0, 1): "2+1", 
        (0, 2): "3+0",
        (1, 0): "3+2",
        (1, 1): "5+0",
        (1, 2): "5+3",
        (2, 0): "10+0",
        (2, 1): "10+5",
        (2, 2): "15+10",
    }
    
    # Reverse mapping
    TIME_CONTROL_TO_GRID = {v: k for k, v in GRID_LAYOUT.items()}
    
    def __init__(self):
        self._grid_cache = None
    
    def detect_grid(self, image: np.ndarray = None) -> Optional[Dict]:
        """
        Detect the time control grid on the home page.
        
        Args:
            image: BGR image to search, or None to capture screenshot.
        
        Returns:
            Dictionary with grid information:
            {
                'top_left': (x, y),
                'cell_width': int,
                'cell_height': int,
                'time_controls': {time_control: (x, y), ...}
            }
        """
        if image is None:
            image = capture_screenshot()
            if image is None:
                return None
        
        img_h, img_w = image.shape[:2]
        
        # Use OCR to find at least one time control to anchor the grid
        button_detector = ButtonDetector()
        
        # Find "1+0" as anchor point (top-left of grid)
        anchor_pos = button_detector.find_time_control_button("1+0", image)
        
        if anchor_pos is None:
            # Try other time controls
            for tc in ["2+1", "3+0", "5+0"]:
                anchor_pos = button_detector.find_time_control_button(tc, image)
                if anchor_pos is not None:
                    break
        
        if anchor_pos is None:
            return None
        
        # Try to find more buttons to estimate grid size
        all_buttons = button_detector.find_all_time_controls(image)
        
        if len(all_buttons) < 2:
            return None
        
        # Estimate cell size from button positions
        x_coords = [pos[0] for pos in all_buttons.values()]
        y_coords = [pos[1] for pos in all_buttons.values()]
        
        # Find unique x and y positions (cluster similar values)
        x_unique = self._cluster_coordinates(x_coords)
        y_unique = self._cluster_coordinates(y_coords)
        
        if len(x_unique) < 2 or len(y_unique) < 2:
            return None
        
        # Calculate cell dimensions
        x_sorted = sorted(x_unique)
        y_sorted = sorted(y_unique)
        
        cell_width = int(np.mean(np.diff(x_sorted))) if len(x_sorted) > 1 else 100
        cell_height = int(np.mean(np.diff(y_sorted))) if len(y_sorted) > 1 else 60
        
        # Estimate top-left of grid
        grid_left = min(x_sorted) - cell_width // 2
        grid_top = min(y_sorted) - cell_height // 2
        
        result = {
            'top_left': (grid_left, grid_top),
            'cell_width': cell_width,
            'cell_height': cell_height,
            'time_controls': all_buttons
        }
        
        self._grid_cache = result
        return result
    
    def get_button_position(
        self,
        time_control: str,
        image: np.ndarray = None,
        use_cache: bool = True
    ) -> Optional[Tuple[int, int]]:
        """
        Get the position of a specific time control button.
        
        Args:
            time_control: Time control string (e.g., "1+0", "3+0").
            image: BGR image to search, or None to capture screenshot.
            use_cache: Whether to use cached grid information.
        
        Returns:
            (x, y) centre coordinates of the button, or None if not found.
        """
        # First try direct OCR detection
        button_detector = ButtonDetector()
        result = button_detector.find_time_control_button(time_control, image)
        
        if result is not None:
            return result
        
        # Fall back to grid-based estimation
        if use_cache and self._grid_cache is not None:
            grid_info = self._grid_cache
        else:
            grid_info = self.detect_grid(image)
        
        if grid_info is None:
            return None
        
        # Check if this time control was directly detected
        if time_control in grid_info['time_controls']:
            return grid_info['time_controls'][time_control]
        
        # Estimate position based on grid layout
        if time_control not in self.TIME_CONTROL_TO_GRID:
            return None
        
        row, col = self.TIME_CONTROL_TO_GRID[time_control]
        
        x = grid_info['top_left'][0] + col * grid_info['cell_width'] + grid_info['cell_width'] // 2
        y = grid_info['top_left'][1] + row * grid_info['cell_height'] + grid_info['cell_height'] // 2
        
        return (x, y)
    
    def _cluster_coordinates(self, coords: List[int], threshold: int = 30) -> List[int]:
        """
        Cluster nearby coordinates together.
        
        Args:
            coords: List of coordinates.
            threshold: Maximum distance to consider coordinates as same.
        
        Returns:
            List of cluster centres.
        """
        if not coords:
            return []
        
        sorted_coords = sorted(coords)
        clusters = [[sorted_coords[0]]]
        
        for coord in sorted_coords[1:]:
            if coord - clusters[-1][-1] <= threshold:
                clusters[-1].append(coord)
            else:
                clusters.append([coord])
        
        # Return cluster centres
        return [int(np.mean(cluster)) for cluster in clusters]


# Convenience functions

def find_play_button(image: np.ndarray = None) -> Optional[Tuple[int, int]]:
    """Find the PLAY button position."""
    detector = ButtonDetector()
    return detector.find_play_button(image)


def find_time_control_button(time_control: str, image: np.ndarray = None) -> Optional[Tuple[int, int]]:
    """Find a specific time control button position."""
    detector = QuickPairingDetector()
    return detector.get_button_position(time_control, image)


def find_new_opponent_button(image: np.ndarray = None) -> Optional[Tuple[int, int]]:
    """Find the NEW OPPONENT button position."""
    detector = ButtonDetector()
    return detector.find_new_opponent_button(image)


# Test function
def test_detection(image_path: str = None):
    """
    Test button detection on an image or live screenshot.
    
    Args:
        image_path: Path to image file, or None to use live screenshot.
    """
    if image_path:
        image = load_image(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return
        print(f"Testing on image: {image_path}")
    else:
        image = capture_screenshot()
        if image is None:
            print("Failed to capture screenshot")
            return
        print("Testing on live screenshot")
    
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Test PLAY button detection
    print("\n--- PLAY Button Detection ---")
    play_pos = find_play_button(image)
    if play_pos:
        print(f"✓ Found PLAY button at: {play_pos}")
    else:
        print("✗ PLAY button not found")
    
    # Test NEW OPPONENT button detection
    print("\n--- NEW OPPONENT Button Detection ---")
    new_opp_pos = find_new_opponent_button(image)
    if new_opp_pos:
        print(f"✓ Found NEW OPPONENT button at: {new_opp_pos}")
    else:
        print("✗ NEW OPPONENT button not found")
    
    # Test time control detection
    print("\n--- Time Control Detection ---")
    detector = ButtonDetector()
    all_tc = detector.find_all_time_controls(image)
    
    if all_tc:
        print(f"✓ Found {len(all_tc)} time controls:")
        for tc, pos in all_tc.items():
            print(f"  - {tc}: {pos}")
    else:
        print("✗ No time controls found")
    
    # Test grid detection
    print("\n--- Grid Detection ---")
    grid_detector = QuickPairingDetector()
    grid_info = grid_detector.detect_grid(image)
    
    if grid_info:
        print(f"✓ Grid detected:")
        print(f"  - Top-left: {grid_info['top_left']}")
        print(f"  - Cell size: {grid_info['cell_width']}x{grid_info['cell_height']}")
    else:
        print("✗ Grid not detected")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_detection(sys.argv[1])
    else:
        test_detection()
