#!/usr/bin/env python3
"""
Shadow Calibrator - Passive template extraction during gameplay.

Runs alongside normal gameplay, watching for opportunities to extract templates:
- Extracts piece templates from the starting position
- Collects digit templates as the clock counts down
- Captures result templates when games end

Usage:
    # Run alongside your normal chess client
    python -m auto_calibration.shadow_calibrator --duration 1
    
    # Or integrate into your existing workflow
    from auto_calibration.shadow_calibrator import ShadowCalibrator
    calibrator = ShadowCalibrator()
    calibrator.start()  # Runs in background
    # ... play your games ...
    calibrator.stop()
"""

import cv2
import numpy as np
import time
import threading
from pathlib import Path
from typing import Dict, Optional, Callable, Any
from datetime import datetime
import json

from fastgrab import screenshot

from .template_extractor import TemplateExtractor, remove_background_colours
from .config import get_config, ChessConfig


class ShadowCalibrator:
    """
    Passively extracts calibration templates by watching gameplay.
    
    Designed to run in the background during normal gameplay, collecting
    templates as opportunities arise (clock changes, game starts/ends).
    """
    
    def __init__(
        self,
        config: Optional[ChessConfig] = None,
        on_progress: Optional[Callable[[str], None]] = None,
        verbose: bool = True
    ):
        """
        Initialise the shadow calibrator.
        
        Args:
            config: Chess configuration with coordinates. Uses global config if None.
            on_progress: Optional callback for progress updates
            verbose: If True, print progress messages
        """
        self.config = config or get_config()
        self.extractor = TemplateExtractor()
        self.on_progress = on_progress
        self.verbose = verbose
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._screen = screenshot.Screenshot()
        
        # Tracking state
        self._last_clock_time: Optional[int] = None
        self._game_started = False
        self._pieces_extracted = False
        self._seen_digits = set()
        
        # Session stats
        self.session_stats = {
            "digits_extracted": 0,
            "pieces_extracted": 0,
            "results_extracted": 0,
            "games_observed": 0,
            "start_time": None
        }
    
    def _log(self, message: str):
        """Log a message."""
        if self.verbose:
            print(f"[Shadow] {message}")
        if self.on_progress:
            self.on_progress(message)
    
    def _capture_region(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Capture a screen region."""
        try:
            img = self._screen.capture((x, y, w, h)).copy()
            return img[:, :, :3]  # Remove alpha channel
        except Exception as e:
            return None
    
    def _capture_board(self) -> Optional[np.ndarray]:
        """Capture the chess board."""
        x, y, w, h = self.config.get_board_position()
        return self._capture_region(x, y, w, h)
    
    def _capture_clock(self, clock_type: str, state: str = "play") -> Optional[np.ndarray]:
        """Capture a clock region."""
        x, y, w, h = self.config.get_clock_position(clock_type, state)
        return self._capture_region(x, y, w, h)
    
    def _capture_result(self) -> Optional[np.ndarray]:
        """Capture the result region."""
        x, y, w, h = self.config.get_result_region_position()
        return self._capture_region(x, y, w, h)
    
    def _read_clock_with_templates(self, clock_img: np.ndarray) -> Optional[int]:
        """
        Try to read a clock using existing templates.
        Returns time in seconds, or None if unreadable.
        """
        # Use the existing read_clock function if available
        try:
            from chessimage.image_scrape_utils import read_clock
            return read_clock(clock_img)
        except ImportError:
            return None
    
    def _is_starting_position(self, board_img: np.ndarray) -> bool:
        """
        Check if the board shows the starting position.
        Uses simple heuristics rather than full FEN parsing for speed.
        """
        if board_img is None or board_img.size == 0:
            return False
        
        h, w = board_img.shape[:2]
        step = w // 8
        
        # Check a few key squares that should have pieces in starting position
        # Sample from corners where we expect rooks
        
        # Process to grayscale for piece detection
        gray = remove_background_colours(board_img)
        
        # Check if there's significant content in the corner squares
        # (rooks should create distinct patterns)
        corner_tl = gray[0:step, 0:step]
        corner_br = gray[7*step:8*step, 7*step:8*step]
        
        # If corners have significant non-zero pixels, likely has pieces
        tl_density = np.count_nonzero(corner_tl) / corner_tl.size
        br_density = np.count_nonzero(corner_br) / corner_br.size
        
        # Starting position should have pieces in corners
        return tl_density > 0.1 and br_density > 0.1
    
    def _detect_bottom_colour(self, board_img: np.ndarray) -> str:
        """
        Detect which colour is at the bottom of the board.
        Returns 'w' or 'b'.
        """
        # In starting position, check bottom-left square for white rook pattern
        # This is a simplified check - could be made more robust
        try:
            from chessimage.image_scrape_utils import find_initial_side
            # This function returns True if white is at bottom
            # We'll capture the board ourselves though
            h, w = board_img.shape[:2]
            step = w // 8
            
            # Bottom-left square
            a1_region = board_img[7*step:8*step, 0:step]
            a1_gray = remove_background_colours(a1_region)
            
            # Check if this looks like a white piece (higher average intensity)
            # White pieces tend to be lighter than black pieces
            intensity = np.mean(a1_gray[a1_gray > 0]) if np.any(a1_gray > 0) else 0
            
            # White rook has higher intensity than black rook
            return 'w' if intensity > 100 else 'b'
        except Exception:
            return 'w'  # Default assumption
    
    def _check_game_result(self, result_img: np.ndarray) -> Optional[str]:
        """
        Check if the result region shows a game result.
        Returns 'white_win', 'black_win', 'draw', or None.
        """
        if result_img is None or result_img.size == 0:
            return None
        
        # Convert to grayscale and check for text patterns
        gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
        
        # Result text should have high contrast (white text on darker background)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Count white pixels - result text should have significant amount
        white_ratio = np.count_nonzero(thresh) / thresh.size
        
        if white_ratio < 0.05:
            return None  # Probably no result shown
        
        # Try to determine result type by analyzing the pattern
        # "1-0" vs "0-1" vs "½-½" have different pixel distributions
        
        h, w = thresh.shape
        left_half = thresh[:, :w//2]
        right_half = thresh[:, w//2:]
        
        left_density = np.count_nonzero(left_half) / left_half.size
        right_density = np.count_nonzero(right_half) / right_half.size
        
        # "1-0" has more content on left, "0-1" has more on right
        # "½-½" is roughly symmetric
        if abs(left_density - right_density) < 0.05:
            return 'draw'  # Symmetric = draw
        elif left_density > right_density:
            return 'white_win'  # "1-0"
        else:
            return 'black_win'  # "0-1"
    
    def extract_pieces_now(self, bottom: str = 'w') -> int:
        """
        Extract piece templates from current board state.
        Assumes board is in starting position.
        
        Args:
            bottom: 'w' if white at bottom, 'b' if black
        
        Returns:
            Number of pieces extracted
        """
        board_img = self._capture_board()
        if board_img is None:
            return 0
        
        count = self.extractor.extract_all_from_starting_position(board_img, bottom, overwrite=False)
        
        if count > 0:
            self._log(f"Extracted {count} piece templates")
            self.session_stats["pieces_extracted"] += count
        
        return count
    
    def extract_digits_from_current_clock(self, known_time: int) -> int:
        """
        Extract digit templates from the current clock display.
        
        Args:
            known_time: The time currently displayed (in seconds)
        
        Returns:
            Number of NEW digits extracted
        """
        clock_img = self._capture_clock('bottom_clock', 'play')
        if clock_img is None:
            return 0
        
        digit_positions = self.config.get_digit_positions()
        if digit_positions is None:
            self._log("No digit positions calibrated - skipping digit extraction")
            return 0
        
        count = self.extractor.extract_digits_from_known_time(
            clock_img, digit_positions, known_time, overwrite=False
        )
        
        if count > 0:
            # Figure out which digits we extracted
            minutes = known_time // 60
            seconds = known_time % 60
            digits = {minutes // 10, minutes % 10, seconds // 10, seconds % 10}
            new_digits = digits - self._seen_digits
            self._seen_digits.update(digits)
            
            self._log(f"Extracted {count} new digit(s): {sorted(new_digits)}")
            self.session_stats["digits_extracted"] += count
        
        return count
    
    def capture_result_template(self, result_type: str) -> bool:
        """
        Capture and save a result template.
        
        Args:
            result_type: 'white_win', 'black_win', or 'draw'
        
        Returns:
            True if saved successfully
        """
        result_img = self._capture_result()
        if result_img is None:
            return False
        
        if self.extractor.save_result_template(result_type, result_img, overwrite=False):
            self._log(f"Captured {result_type} result template")
            self.session_stats["results_extracted"] += 1
            return True
        
        return False
    
    def _shadow_loop(self, poll_interval: float = 0.5):
        """
        Main shadow calibration loop.
        Runs in background, watching for extraction opportunities.
        """
        last_time = None
        consecutive_no_time = 0
        game_in_progress = False
        
        while self._running:
            try:
                # Try to read the clock
                clock_img = self._capture_clock('bottom_clock', 'play')
                current_time = self._read_clock_with_templates(clock_img) if clock_img is not None else None
                
                if current_time is not None:
                    consecutive_no_time = 0
                    
                    # Check for game start (starting position with full time)
                    if not game_in_progress:
                        # Might be a new game
                        board_img = self._capture_board()
                        if board_img is not None and self._is_starting_position(board_img):
                            game_in_progress = True
                            self.session_stats["games_observed"] += 1
                            self._log(f"Game {self.session_stats['games_observed']} detected")
                            
                            # Extract pieces if we don't have them all
                            if not self.extractor.is_complete():
                                missing = self.extractor.get_missing_items()
                                if len(missing["pieces"]) > 0:
                                    bottom = self._detect_bottom_colour(board_img)
                                    self.extract_pieces_now(bottom)
                    
                    # Extract digits when time changes
                    if current_time != last_time:
                        # Don't extract if time is too high (might be starting time with --:--)
                        if current_time < 36000:  # Less than 10 hours
                            self.extract_digits_from_current_clock(current_time)
                        last_time = current_time
                    
                    game_in_progress = True
                
                else:
                    consecutive_no_time += 1
                    
                    # If we can't read the clock for a while, game might have ended
                    if consecutive_no_time > 5 and game_in_progress:
                        # Check for result
                        result_img = self._capture_result()
                        result_type = self._check_game_result(result_img)
                        
                        if result_type is not None:
                            # Try to capture the result template
                            if not self.extractor.progress["results"][result_type]:
                                self.capture_result_template(result_type)
                        
                        game_in_progress = False
                        last_time = None
                
                # Check if we're done
                if self.extractor.is_complete():
                    self._log("All templates extracted! Shadow calibration complete.")
                    self._running = False
                    break
                
            except Exception as e:
                self._log(f"Error in shadow loop: {e}")
            
            time.sleep(poll_interval)
    
    def start(self, poll_interval: float = 0.5):
        """
        Start shadow calibration in background thread.
        
        Args:
            poll_interval: How often to check for extraction opportunities (seconds)
        """
        if self._running:
            self._log("Shadow calibrator already running")
            return
        
        self._running = True
        self.session_stats["start_time"] = datetime.now().isoformat()
        
        self._log("Starting shadow calibration...")
        self._log(self.extractor.get_completion_summary())
        
        self._thread = threading.Thread(target=self._shadow_loop, args=(poll_interval,), daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop shadow calibration."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        
        self._log("Shadow calibration stopped")
        self._log(self.get_session_summary())
    
    def is_running(self) -> bool:
        """Check if shadow calibration is running."""
        return self._running
    
    def get_session_summary(self) -> str:
        """Get a summary of this calibration session."""
        lines = [
            "📊 Shadow Calibration Session Summary:",
            f"   Games observed: {self.session_stats['games_observed']}",
            f"   Digits extracted: {self.session_stats['digits_extracted']}",
            f"   Pieces extracted: {self.session_stats['pieces_extracted']}",
            f"   Results extracted: {self.session_stats['results_extracted']}",
            "",
            self.extractor.get_completion_summary()
        ]
        return "\n".join(lines)
    
    def run_for_duration(self, minutes: float = 1.0, poll_interval: float = 0.5):
        """
        Run shadow calibration for a specified duration.
        Blocks until duration elapsed or all templates extracted.
        
        Args:
            minutes: How long to run (in minutes)
            poll_interval: How often to check (seconds)
        """
        self.start(poll_interval)
        
        end_time = time.time() + (minutes * 60)
        
        try:
            while time.time() < end_time and self._running:
                time.sleep(1.0)
                
                # Show periodic progress
                elapsed = time.time() - (end_time - minutes * 60)
                if int(elapsed) % 30 == 0:  # Every 30 seconds
                    missing = self.extractor.get_missing_items()
                    total_missing = sum(len(v) for v in missing.values())
                    if total_missing > 0:
                        self._log(f"Still missing {total_missing} templates...")
        except KeyboardInterrupt:
            self._log("Interrupted by user")
        finally:
            self.stop()


def main():
    """Command-line interface for shadow calibrator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Shadow Calibrator - Passive template extraction")
    parser.add_argument("--duration", type=float, default=5.0,
                       help="Duration to run (in minutes)")
    parser.add_argument("--poll-interval", type=float, default=0.5,
                       help="Poll interval (seconds)")
    parser.add_argument("--status", action="store_true",
                       help="Just show current extraction status")
    parser.add_argument("--reset", choices=["all", "digits", "pieces", "results"],
                       help="Reset extraction progress")
    
    args = parser.parse_args()
    
    extractor = TemplateExtractor()
    
    if args.status:
        print(extractor.get_completion_summary())
        return
    
    if args.reset:
        category = None if args.reset == "all" else args.reset
        extractor.reset_progress(category)
        print(f"Reset {args.reset} extraction progress")
        return
    
    print("=" * 60)
    print("Shadow Calibrator")
    print("=" * 60)
    print()
    print("This tool passively extracts calibration templates while you play.")
    print("Start a chess game on Lichess and the calibrator will:")
    print("  • Extract piece templates from the starting position")
    print("  • Capture digit templates as the clock counts down")
    print("  • Save result templates when games end")
    print()
    print(f"Running for {args.duration} minutes...")
    print("Press Ctrl+C to stop early.")
    print()
    
    calibrator = ShadowCalibrator(verbose=True)
    calibrator.run_for_duration(args.duration, args.poll_interval)


if __name__ == "__main__":
    main()
