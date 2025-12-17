#!/usr/bin/env python3
"""
Interactive Calibrator - Manual template extraction with GUI.

Provides a visual interface for manually selecting and capturing templates.
Supports both live capture and offline screenshots.

Usage:
    # Live mode (captures from screen)
    python -m auto_calibration.interactive_calibrator --guided
    python -m auto_calibration.interactive_calibrator --digit 7
    
    # Offline mode (from saved screenshots)
    python -m auto_calibration.interactive_calibrator --offline ./screenshots/
    python -m auto_calibration.interactive_calibrator --offline ./screenshots/ --digit 7
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from datetime import datetime

from .template_extractor import TemplateExtractor, remove_background_colours, DIGIT_TEMPLATE_SIZE
from .config import get_config, ChessConfig


class InteractiveCalibrator:
    """
    Interactive GUI tool for manual template extraction.
    Supports both live capture and offline screenshots.
    """
    
    WINDOW_NAME = "Interactive Calibrator"
    
    def __init__(self, config: Optional[ChessConfig] = None, offline_dir: Optional[str] = None):
        """
        Initialise the interactive calibrator.
        
        Args:
            config: Chess configuration. Uses global config if None.
            offline_dir: Directory containing offline screenshots. If provided,
                        works in offline mode instead of live capture.
        """
        self.config = config or get_config()
        self.extractor = TemplateExtractor()
        
        # Offline mode
        self.offline_mode = offline_dir is not None
        self.offline_images: List[Path] = []
        self.current_image_idx = 0
        
        if self.offline_mode:
            self._load_offline_images(offline_dir)
        else:
            # Live capture mode
            from fastgrab import screenshot
            self._screen = screenshot.Screenshot()
        
        # Mouse state
        self._mouse_pos = (0, 0)
        self._selection_start = None
        self._selection_end = None
        self._selecting = False
        
        # Current screenshot
        self._current_screenshot = None
        self._display_image = None
    
    def _load_offline_images(self, offline_dir: str):
        """Load list of offline screenshots."""
        offline_path = Path(offline_dir)
        if not offline_path.exists():
            print(f"Warning: Offline directory not found: {offline_dir}")
            return
        
        self.offline_images = sorted(offline_path.glob("*.png"))
        if not self.offline_images:
            print(f"Warning: No PNG images found in {offline_dir}")
        else:
            print(f"Loaded {len(self.offline_images)} offline screenshots")
    
    def _get_current_offline_image(self) -> Optional[np.ndarray]:
        """Get the current offline image."""
        if not self.offline_images:
            return None
        
        path = self.offline_images[self.current_image_idx]
        img = cv2.imread(str(path))
        return img
    
    def _next_offline_image(self):
        """Switch to next offline image."""
        if self.offline_images:
            self.current_image_idx = (self.current_image_idx + 1) % len(self.offline_images)
            self._selection_start = None
            self._selection_end = None
    
    def _prev_offline_image(self):
        """Switch to previous offline image."""
        if self.offline_images:
            self.current_image_idx = (self.current_image_idx - 1) % len(self.offline_images)
            self._selection_start = None
            self._selection_end = None
    
    def _capture_screen(self) -> np.ndarray:
        """Capture the entire screen."""
        if self.offline_mode:
            return self._get_current_offline_image()
        else:
            img = self._screen.capture((0, 0, 3840, 2160)).copy()
            return img[:, :, :3]
    
    def _capture_region(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Capture a specific screen region (or crop from offline image)."""
        if self.offline_mode:
            full_img = self._get_current_offline_image()
            if full_img is None:
                return np.zeros((h, w, 3), dtype=np.uint8)
            # Crop the region
            y2 = min(y + h, full_img.shape[0])
            x2 = min(x + w, full_img.shape[1])
            return full_img[y:y2, x:x2]
        else:
            img = self._screen.capture((x, y, w, h)).copy()
            return img[:, :, :3]
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events in the OpenCV window."""
        # Scale coordinates back to original image size
        if self._display_image is not None and self._current_screenshot is not None:
            scale_x = self._current_screenshot.shape[1] / self._display_image.shape[1]
            scale_y = self._current_screenshot.shape[0] / self._display_image.shape[0]
            x = int(x * scale_x)
            y = int(y * scale_y)
        
        self._mouse_pos = (x, y)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self._selection_start = (x, y)
            self._selecting = True
        elif event == cv2.EVENT_MOUSEMOVE and self._selecting:
            self._selection_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._selection_end = (x, y)
            self._selecting = False
    
    def _get_selection_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """Get the current selection as (x, y, w, h)."""
        if self._selection_start is None or self._selection_end is None:
            return None
        
        x1 = min(self._selection_start[0], self._selection_end[0])
        y1 = min(self._selection_start[1], self._selection_end[1])
        x2 = max(self._selection_start[0], self._selection_end[0])
        y2 = max(self._selection_start[1], self._selection_end[1])
        
        w = x2 - x1
        h = y2 - y1
        
        if w < 5 or h < 5:
            return None
        
        return (x1, y1, w, h)
    
    def _draw_overlay(self, img: np.ndarray, extra_text: str = "") -> np.ndarray:
        """Draw selection overlay on the image."""
        display = img.copy()
        
        # Draw current selection
        rect = self._get_selection_rect()
        if rect is not None:
            x, y, w, h = rect
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        elif self._selection_start is not None and self._selecting:
            # Draw selection in progress
            cv2.rectangle(display, self._selection_start, self._mouse_pos, (0, 255, 0), 1)
        
        # Show image info in offline mode
        if self.offline_mode and self.offline_images:
            filename = self.offline_images[self.current_image_idx].name
            info_text = f"[{self.current_image_idx + 1}/{len(self.offline_images)}] {filename}"
            cv2.putText(display, info_text, (10, display.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return display
    
    def _show_scaled(self, img: np.ndarray, max_width: int = 1600, max_height: int = 900) -> np.ndarray:
        """Show image scaled to fit screen."""
        h, w = img.shape[:2]
        scale = min(max_width / w, max_height / h, 1.0)
        
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img
    
    def capture_digit_interactive(self, digit: int) -> bool:
        """
        Interactively capture a specific digit template.
        
        Args:
            digit: The digit to capture (0-9)
        
        Returns:
            True if captured successfully
        """
        print(f"\n📷 Capturing digit '{digit}'")
        print("=" * 50)
        if self.offline_mode:
            print("OFFLINE MODE - Using saved screenshots")
            print("Instructions:")
            print(f"  1. Find a screenshot where the clock shows digit '{digit}'")
            print("  2. Use LEFT/RIGHT arrow keys to browse screenshots")
            print("  3. Click and drag to select the digit region")
            print("  4. Press 's' to save, 'r' to retry, 'q' to quit")
        else:
            print("LIVE MODE - Capturing from screen")
            print("Instructions:")
            print(f"  1. Make sure the clock shows the digit '{digit}'")
            print("  2. Click and drag to select the digit region")
            print("  3. Press 's' to save, 'r' to retry, 'q' to quit")
        print()
        
        # Use calibrated clock position as starting point
        clock_x, clock_y, clock_w, clock_h = self.config.get_clock_position('bottom_clock', 'play')
        
        # Capture region around the clock (or full image in offline mode for context)
        if self.offline_mode:
            # Show more context in offline mode
            margin = 200
        else:
            margin = 50
        
        region_x = max(0, clock_x - margin)
        region_y = max(0, clock_y - margin)
        region_w = clock_w + 2 * margin
        region_h = clock_h + 2 * margin
        
        cv2.namedWindow(self.WINDOW_NAME)
        cv2.setMouseCallback(self.WINDOW_NAME, self._mouse_callback)
        
        saved = False
        
        while True:
            # Capture/load screenshot
            self._current_screenshot = self._capture_region(region_x, region_y, region_w, region_h)
            
            # Draw overlay
            display = self._draw_overlay(self._current_screenshot)
            self._display_image = self._show_scaled(display)
            
            # Add instructions
            if self.offline_mode:
                help_text = f"Select digit '{digit}' | LEFT/RIGHT=browse | s=save | r=reset | q=quit"
            else:
                help_text = f"Select digit '{digit}' | s=save | r=reset | q=quit"
            cv2.putText(self._display_image, help_text,
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow(self.WINDOW_NAME, self._display_image)
            
            key = cv2.waitKey(50 if not self.offline_mode else 100) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset selection
                self._selection_start = None
                self._selection_end = None
            elif key == 81 or key == 2:  # LEFT arrow
                self._prev_offline_image()
            elif key == 83 or key == 3:  # RIGHT arrow
                self._next_offline_image()
            elif key == ord('s'):
                # Save selection
                rect = self._get_selection_rect()
                if rect is not None:
                    x, y, w, h = rect
                    region = self._current_screenshot[y:y+h, x:x+w]
                    
                    # Process and save
                    processed = remove_background_colours(region, thresh=1.6)
                    processed = cv2.resize(processed, DIGIT_TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)
                    
                    if self.extractor.save_digit_template(digit, processed, overwrite=True):
                        print(f"✓ Saved digit '{digit}' template")
                        saved = True
                        break
                    else:
                        print("✗ Failed to save template")
                else:
                    print("No region selected!")
        
        cv2.destroyAllWindows()
        return saved
    
    def capture_digit_from_full_image(self, digit: int) -> bool:
        """
        Capture a digit by showing the FULL screenshot (not cropped to clock region).
        Useful when clock position isn't perfectly calibrated.
        
        Args:
            digit: The digit to capture (0-9)
        
        Returns:
            True if captured successfully
        """
        print(f"\n📷 Capturing digit '{digit}' (Full Image Mode)")
        print("=" * 50)
        if self.offline_mode:
            print("OFFLINE MODE - Showing full screenshots")
            print("Instructions:")
            print(f"  1. Find a screenshot where the clock shows digit '{digit}'")
            print("  2. Use LEFT/RIGHT arrow keys to browse screenshots")
            print("  3. Click and drag to select the digit region")
            print("  4. Press 's' to save, 'r' to retry, 'q' to quit")
        else:
            print("This mode is designed for offline screenshots.")
            return False
        print()
        
        cv2.namedWindow(self.WINDOW_NAME)
        cv2.setMouseCallback(self.WINDOW_NAME, self._mouse_callback)
        
        saved = False
        
        while True:
            # Load full screenshot
            self._current_screenshot = self._get_current_offline_image()
            if self._current_screenshot is None:
                print("No images available!")
                break
            
            # Draw overlay
            display = self._draw_overlay(self._current_screenshot)
            self._display_image = self._show_scaled(display)
            
            help_text = f"Select digit '{digit}' | LEFT/RIGHT=browse | s=save | r=reset | q=quit"
            cv2.putText(self._display_image, help_text,
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow(self.WINDOW_NAME, self._display_image)
            
            key = cv2.waitKey(100) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                self._selection_start = None
                self._selection_end = None
            elif key == 81 or key == 2:  # LEFT arrow
                self._prev_offline_image()
            elif key == 83 or key == 3:  # RIGHT arrow
                self._next_offline_image()
            elif key == ord('s'):
                rect = self._get_selection_rect()
                if rect is not None:
                    x, y, w, h = rect
                    region = self._current_screenshot[y:y+h, x:x+w]
                    
                    processed = remove_background_colours(region, thresh=1.6)
                    processed = cv2.resize(processed, DIGIT_TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)
                    
                    if self.extractor.save_digit_template(digit, processed, overwrite=True):
                        print(f"✓ Saved digit '{digit}' template")
                        saved = True
                        break
                    else:
                        print("✗ Failed to save template")
                else:
                    print("No region selected!")
        
        cv2.destroyAllWindows()
        return saved
    
    def capture_result_interactive(self, result_type: str) -> bool:
        """
        Interactively capture a game result template.
        
        Args:
            result_type: 'white_win', 'black_win', or 'draw'
        
        Returns:
            True if captured successfully
        """
        result_display = {
            'white_win': '1-0 (White wins)',
            'black_win': '0-1 (Black wins)',
            'draw': '½-½ (Draw)'
        }
        
        print(f"\n📷 Capturing result template: {result_display[result_type]}")
        print("=" * 50)
        if self.offline_mode:
            print("OFFLINE MODE - Using saved screenshots")
            print("Instructions:")
            print(f"  1. Find a screenshot showing '{result_display[result_type]}'")
            print("  2. Use LEFT/RIGHT arrow keys to browse screenshots")
            print("  3. Click and drag to select the result text region")
            print("  4. Press 's' to save, 'r' to retry, 'q' to quit")
        else:
            print("LIVE MODE - Capturing from screen")
            print("Instructions:")
            print(f"  1. Make sure the screen shows '{result_display[result_type]}'")
            print("  2. Click and drag to select the result text region")
            print("  3. Press 's' to save, 'r' to retry, 'q' to quit")
        print()
        
        cv2.namedWindow(self.WINDOW_NAME)
        cv2.setMouseCallback(self.WINDOW_NAME, self._mouse_callback)
        
        saved = False
        
        while True:
            if self.offline_mode:
                self._current_screenshot = self._get_current_offline_image()
                if self._current_screenshot is None:
                    print("No images available!")
                    break
            else:
                # Use calibrated result region as starting point
                result_x, result_y, result_w, result_h = self.config.get_result_region_position()
                margin = 100
                region_x = max(0, result_x - margin)
                region_y = max(0, result_y - margin)
                region_w = result_w + 2 * margin
                region_h = result_h + 2 * margin
                self._current_screenshot = self._capture_region(region_x, region_y, region_w, region_h)
            
            display = self._draw_overlay(self._current_screenshot)
            self._display_image = self._show_scaled(display)
            
            if self.offline_mode:
                help_text = f"Select '{result_display[result_type]}' | LEFT/RIGHT=browse | s=save | q=quit"
            else:
                help_text = f"Select '{result_display[result_type]}' | s=save | q=quit"
            cv2.putText(self._display_image, help_text,
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow(self.WINDOW_NAME, self._display_image)
            
            key = cv2.waitKey(50 if not self.offline_mode else 100) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                self._selection_start = None
                self._selection_end = None
            elif key == 81 or key == 2:  # LEFT arrow
                self._prev_offline_image()
            elif key == 83 or key == 3:  # RIGHT arrow
                self._next_offline_image()
            elif key == ord('s'):
                rect = self._get_selection_rect()
                if rect is not None:
                    x, y, w, h = rect
                    region = self._current_screenshot[y:y+h, x:x+w]
                    
                    if self.extractor.save_result_template(result_type, region, overwrite=True):
                        print(f"✓ Saved {result_type} template")
                        saved = True
                        break
                    else:
                        print("✗ Failed to save template")
                else:
                    print("No region selected!")
        
        cv2.destroyAllWindows()
        return saved
    
    def capture_pieces_from_board(self) -> int:
        """
        Capture piece templates from the current board position.
        Assumes the board is in starting position.
        
        Returns:
            Number of pieces captured
        """
        print("\n📷 Capturing piece templates from starting position")
        print("=" * 50)
        if self.offline_mode:
            print("OFFLINE MODE - Using saved screenshots")
            print("Instructions:")
            print("  1. Find a screenshot showing the starting position")
            print("  2. Use LEFT/RIGHT arrow keys to browse screenshots")
            print("  3. Press 'c' for white at bottom, 'b' for black at bottom")
            print("  4. Press 'q' to quit without capturing")
        else:
            print("LIVE MODE - Capturing from screen")
            print("Instructions:")
            print("  1. Make sure a new game is showing the starting position")
            print("  2. Press 'c' for white at bottom, 'b' for black at bottom")
            print("  3. Press 'q' to quit without capturing")
        print()
        
        board_x, board_y, board_w, board_h = self.config.get_board_position()
        
        cv2.namedWindow(self.WINDOW_NAME)
        
        count = 0
        
        while True:
            if self.offline_mode:
                full_img = self._get_current_offline_image()
                if full_img is None:
                    print("No images available!")
                    break
                # Crop to board region
                y2 = min(board_y + board_h, full_img.shape[0])
                x2 = min(board_x + board_w, full_img.shape[1])
                board_img = full_img[board_y:y2, board_x:x2]
            else:
                board_img = self._capture_region(board_x, board_y, board_w, board_h)
            
            self._current_screenshot = board_img
            
            display = board_img.copy()
            
            # Draw grid overlay
            if board_img.shape[0] > 0 and board_img.shape[1] > 0:
                step = board_img.shape[1] // 8
                for i in range(9):
                    cv2.line(display, (i * step, 0), (i * step, board_img.shape[0]), (0, 255, 0), 1)
                    cv2.line(display, (0, i * step), (board_img.shape[1], i * step), (0, 255, 0), 1)
            
            self._display_image = self._show_scaled(display)
            
            if self.offline_mode:
                help_text = "LEFT/RIGHT=browse | c=capture(white) | b=capture(black) | q=quit"
                # Show image info
                filename = self.offline_images[self.current_image_idx].name
                cv2.putText(self._display_image, f"[{self.current_image_idx + 1}/{len(self.offline_images)}] {filename}",
                           (10, self._display_image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            else:
                help_text = "c=capture(white at bottom) | b=capture(black at bottom) | q=quit"
            
            cv2.putText(self._display_image, help_text,
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow(self.WINDOW_NAME, self._display_image)
            
            key = cv2.waitKey(50 if not self.offline_mode else 100) & 0xFF
            
            if key == ord('q'):
                break
            elif key == 81 or key == 2:  # LEFT arrow
                self._prev_offline_image()
            elif key == 83 or key == 3:  # RIGHT arrow
                self._next_offline_image()
            elif key == ord('c'):
                # Capture with white at bottom
                count = self.extractor.extract_all_from_starting_position(board_img, 'w', overwrite=True)
                print(f"✓ Captured {count} piece templates (white at bottom)")
                break
            elif key == ord('b'):
                # Capture with black at bottom
                count = self.extractor.extract_all_from_starting_position(board_img, 'b', overwrite=True)
                print(f"✓ Captured {count} piece templates (black at bottom)")
                break
        
        cv2.destroyAllWindows()
        return count
    
    def run_guided_calibration(self):
        """
        Run a guided calibration session to capture all missing templates.
        """
        print("\n" + "=" * 60)
        print("  Interactive Template Calibration")
        if self.offline_mode:
            print("  (OFFLINE MODE)")
        print("=" * 60)
        
        missing = self.extractor.get_missing_items()
        print(self.extractor.get_completion_summary())
        
        total_missing = sum(len(v) for v in missing.values())
        
        if total_missing == 0:
            print("\n✓ All templates already captured!")
            return
        
        print(f"\nMissing {total_missing} template(s). Starting guided capture...\n")
        
        # Capture pieces first (if missing)
        if len(missing["pieces"]) > 0:
            print("\n--- Step 1: Piece Templates ---")
            if self.offline_mode:
                print("Browse to find a starting position screenshot.")
            else:
                print("Please show a game at the starting position.")
            input("Press Enter when ready...")
            self.capture_pieces_from_board()
        
        # Capture digits
        missing_digits = missing["digits"]
        if len(missing_digits) > 0:
            print(f"\n--- Step 2: Digit Templates (missing: {', '.join(missing_digits)}) ---")
            for digit in missing_digits:
                if self.offline_mode:
                    print(f"\nBrowse to find a screenshot with digit '{digit}' visible.")
                else:
                    print(f"\nPlease show the clock with digit '{digit}' visible.")
                input(f"Press Enter when ready to capture digit '{digit}'...")
                
                if self.offline_mode:
                    if not self.capture_digit_from_full_image(int(digit)):
                        print(f"Skipped digit '{digit}'")
                else:
                    if not self.capture_digit_interactive(int(digit)):
                        print(f"Skipped digit '{digit}'")
        
        # Capture results
        missing_results = missing["results"]
        if len(missing_results) > 0:
            print(f"\n--- Step 3: Result Templates (missing: {', '.join(missing_results)}) ---")
            result_display = {
                'white_win': '1-0',
                'black_win': '0-1', 
                'draw': '½-½'
            }
            
            for result in missing_results:
                if self.offline_mode:
                    print(f"\nBrowse to find a screenshot showing '{result_display[result]}'")
                else:
                    print(f"\nPlease show a game result: {result_display[result]}")
                input(f"Press Enter when ready to capture '{result_display[result]}'...")
                
                if not self.capture_result_interactive(result):
                    print(f"Skipped {result} template")
        
        print("\n" + "=" * 60)
        print("  Calibration Complete!")
        print("=" * 60)
        print(self.extractor.get_completion_summary())


def main():
    """Command-line interface for interactive calibrator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive Template Calibrator")
    parser.add_argument("--digit", type=int, choices=range(10), metavar="N",
                       help="Capture a specific digit (0-9)")
    parser.add_argument("--result", choices=["white_win", "black_win", "draw"],
                       help="Capture a specific result template")
    parser.add_argument("--pieces", action="store_true",
                       help="Capture piece templates from starting position")
    parser.add_argument("--guided", action="store_true",
                       help="Run guided calibration for all missing templates")
    parser.add_argument("--status", action="store_true",
                       help="Show current template status")
    parser.add_argument("--offline", type=str, metavar="DIR",
                       help="Use offline screenshots from directory instead of live capture")
    
    args = parser.parse_args()
    
    # Check status before creating calibrator (doesn't need screenshots)
    if args.status:
        extractor = TemplateExtractor()
        print(extractor.get_completion_summary())
        return
    
    # Create calibrator with offline mode if specified
    calibrator = InteractiveCalibrator(offline_dir=args.offline)
    
    if args.digit is not None:
        if args.offline:
            calibrator.capture_digit_from_full_image(args.digit)
        else:
            calibrator.capture_digit_interactive(args.digit)
    elif args.result is not None:
        calibrator.capture_result_interactive(args.result)
    elif args.pieces:
        calibrator.capture_pieces_from_board()
    elif args.guided:
        calibrator.run_guided_calibration()
    else:
        # Default: run guided calibration
        calibrator.run_guided_calibration()


if __name__ == "__main__":
    main()
