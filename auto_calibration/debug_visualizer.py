#!/usr/bin/env python3
"""
Debug Visualizer for Chess Board Auto-Calibration

Shows detected coordinates as colored overlays on the screen.
This helps debug and verify that calibration detected the right positions.
"""

import cv2
import numpy as np
import json
import time
import shutil
from pathlib import Path
from typing import Dict
from calibration_utils import SCREEN_CAPTURE

class CalibrationDebugger:
    """Visual debugger for calibration results."""
    
    def __init__(self, config_file: str = "chess_config.json"):
        self.config_file = config_file
        self.config = None
        self.screen_capture = SCREEN_CAPTURE
        self.debug_dir = Path(__file__).parent / "debug_outputs"
        self._prepare_debug_dir()
        self.load_config()
        
        # Colors for different UI elements (BGR format for OpenCV)
        self.colors = {
            'board': (0, 255, 0),        # Green
            'bottom_clock': (0, 0, 255), # Red
            'top_clock': (255, 0, 0),    # Blue
            'notation': (0, 255, 255),   # Yellow
            'rating': (255, 0, 255),     # Magenta
        }
        
    def _prepare_debug_dir(self):
        """Reset debug directory before generating new artefacts."""
        if self.debug_dir.exists():
            for path in self.debug_dir.iterdir():
                if path.is_file() or path.is_symlink():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
        else:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        # Ensure directory exists after cleanup
        self.debug_dir.mkdir(parents=True, exist_ok=True)
    
    def load_config(self):
        """Load calibration configuration."""
        config_path = Path(__file__).parent / self.config_file
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            print(f"✅ Loaded config from {config_path}")
        else:
            print(f"❌ Config file not found: {config_path}")
            print("Run 'python calibrator.py' first to generate configuration.")
            
    def draw_rectangle_with_label(self, img, coords, color, label, thickness=2):
        """Draw a rectangle with label on the image."""
        x, y, w, h = coords['x'], coords['y'], coords['width'], coords['height']
        
        # Draw rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.rectangle(img, (x, y - label_size[1] - 10), 
                     (x + label_size[0] + 10, y), color, -1)
        
        # Draw label text
        cv2.putText(img, label, (x + 5, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
    def create_debug_overlay(self, show_all_states=False):
        """Create an overlay showing all detected coordinates."""
        if not self.config:
            print("No configuration loaded!")
            return None
            
        # Capture current screen
        screenshot = self.screen_capture.capture()
        if screenshot is None:
            print("Failed to capture screenshot!")
            return None
            
        # Create overlay image
        overlay = screenshot.copy()
        
        coordinates = self.config['coordinates']
        
        # Draw board
        if 'board' in coordinates:
            board = coordinates['board']
            self.draw_rectangle_with_label(
                overlay, board, self.colors['board'], 
                "BOARD", thickness=3
            )
            print(f"Board: ({board['x']}, {board['y']}) [{board['width']}x{board['height']}]")
        
        # Draw clocks
        for clock_type in ['bottom_clock', 'top_clock']:
            if clock_type in coordinates:
                color = self.colors[clock_type]
                
                if show_all_states:
                    # Show all states
                    for state, coords in coordinates[clock_type].items():
                        label = f"{clock_type.replace('_', ' ').title()} ({state})"
                        self.draw_rectangle_with_label(overlay, coords, color, label, thickness=1)
                        print(f"{label}: ({coords['x']}, {coords['y']})")
                else:
                    # Show only 'play' state for cleaner view
                    if 'play' in coordinates[clock_type]:
                        coords = coordinates[clock_type]['play']
                        label = f"{clock_type.replace('_', ' ').title()}"
                        self.draw_rectangle_with_label(overlay, coords, color, label)
                        print(f"{label}: ({coords['x']}, {coords['y']})")
        
        # Draw notation area
        if 'notation' in coordinates:
            notation = coordinates['notation']
            self.draw_rectangle_with_label(
                overlay, notation, self.colors['notation'], "NOTATION"
            )
            print(f"Notation: ({notation['x']}, {notation['y']})")
        
        # Draw rating areas (just show one for cleaner view unless showing all)
        if 'rating' in coordinates and show_all_states:
            for rating_type, coords in coordinates['rating'].items():
                label = f"Rating ({rating_type})"
                self.draw_rectangle_with_label(overlay, coords, self.colors['rating'], label)
        elif 'rating' in coordinates and 'opp_white' in coordinates['rating']:
            coords = coordinates['rating']['opp_white']
            self.draw_rectangle_with_label(overlay, coords, self.colors['rating'], "RATING")
        
        return overlay
    
    def show_debug_overlay(self, show_all_states=False, save_image=True):
        """Create and save the debug overlay image."""
        overlay = self.create_debug_overlay(show_all_states)
        
        if overlay is None:
            return
        
        # Add calibration info
        if 'calibration_info' in self.config:
            info = self.config['calibration_info']
            success_rate = info.get('validation_success_rate', 0)
            method = info.get('board_detection', {}).get('method', 'Unknown')
            
            info_text = f"Method: {method} | Success: {success_rate:.1%}"
            cv2.putText(overlay, info_text, (10, overlay.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Always save image (since we can't display)
        save_path = self.debug_dir / "debug_overlay.png"
        cv2.imwrite(str(save_path), overlay)
        print(f"✅ Debug overlay saved to: {save_path}")
        
        # Also save a version with all states if not already showing all
        if not show_all_states:
            overlay_all = self.create_debug_overlay(show_all_states=True)
            if overlay_all is not None:
                # Add info text
                if 'calibration_info' in self.config:
                    info = self.config['calibration_info']
                    success_rate = info.get('validation_success_rate', 0)
                    method = info.get('board_detection', {}).get('method', 'Unknown')
                    
                    info_text = f"Method: {method} | Success: {success_rate:.1%} | ALL STATES"
                    cv2.putText(overlay_all, info_text, (10, overlay_all.shape[0] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                save_path_all = self.debug_dir / "debug_overlay_all_states.png"
                cv2.imwrite(str(save_path_all), overlay_all)
                print(f"✅ All states overlay saved to: {save_path_all}")
        
        return str(save_path)
    
    def live_debug_mode(self):
        """Live debug mode with keyboard controls."""
        if not self.config:
            print("No configuration loaded!")
            return
            
        print("Live Debug Mode")
        print("Controls:")
        print("  ESC/Q: Exit")
        print("  S: Save current overlay")
        print("  A: Toggle show all states")
        print("  R: Refresh overlay")
        print("  SPACE: Pause/Resume")
        
        show_all_states = False
        paused = False
        
        while True:
            if not paused:
                overlay = self.create_debug_overlay(show_all_states)
                
                if overlay is not None:
                    # Add status info
                    status_text = f"All States: {'ON' if show_all_states else 'OFF'} | {'PAUSED' if paused else 'LIVE'}"
                    cv2.putText(overlay, status_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    cv2.imshow('Chess Calibration Live Debug', overlay)
            
            # Handle keyboard input
            key = cv2.waitKey(100) & 0xFF  # 100ms refresh rate
            
            if key == 27 or key == ord('q'):  # ESC or Q
                break
            elif key == ord('s'):  # Save
                if overlay is not None:
                    save_path = self.debug_dir / f"debug_overlay_{int(time.time())}.png"
                    cv2.imwrite(str(save_path), overlay)
                    print(f"Saved overlay to: {save_path}")
            elif key == ord('a'):  # Toggle all states
                show_all_states = not show_all_states
                print(f"Show all states: {'ON' if show_all_states else 'OFF'}")
            elif key == ord('r'):  # Refresh
                print("Refreshing overlay...")
            elif key == ord(' '):  # Space - pause/resume
                paused = not paused
                print(f"{'PAUSED' if paused else 'RESUMED'}")
        
        cv2.destroyAllWindows()
    
    def test_specific_coordinates(self):
        """Test specific coordinate positions by showing clock regions."""
        if not self.config:
            print("No configuration loaded!")
            return
            
        coordinates = self.config['coordinates']
        
        print("\nTesting specific coordinate regions:")
        print("=" * 50)
        
        # Test each clock position
        for clock_type in ['bottom_clock', 'top_clock']:
            if clock_type in coordinates:
                print(f"\n{clock_type.upper()}:")
                
                for state, coords in coordinates[clock_type].items():
                    x, y, w, h = coords['x'], coords['y'], coords['width'], coords['height']
                    
                    # Capture the specific region
                    try:
                        region = self.screen_capture.capture((x, y, w, h))
                        if region is not None:
                            # Save the region for inspection
                            filename = f"debug_region_{clock_type}_{state}.png"
                            save_path = self.debug_dir / filename
                            cv2.imwrite(str(save_path), region)
                            
                            print(f"  {state}: ({x}, {y}) -> saved as {filename}")
                        else:
                            print(f"  {state}: ({x}, {y}) -> FAILED to capture")
                    except Exception as e:
                        print(f"  {state}: ({x}, {y}) -> ERROR: {e}")


def main():
    """Main function with command line interface."""
    import sys
    
    print("Chess Calibration Debug Visualizer")
    print("=" * 40)
    
    debugger = CalibrationDebugger()
    
    if not debugger.config:
        return
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        print("\nAvailable modes:")
        print("1. Static overlay (default)")
        print("2. Live debug mode")
        print("3. Test coordinate regions")
        print("4. Show all states")
        
        choice = input("\nEnter choice (1-4) or press Enter for default: ").strip()
        
        if choice == "2":
            mode = "live"
        elif choice == "3":
            mode = "test"
        elif choice == "4":
            mode = "all"
        else:
            mode = "static"
    
    if mode == "live":
        debugger.live_debug_mode()
    elif mode == "test":
        debugger.test_specific_coordinates()
    elif mode == "all":
        print("\nShowing overlay with all states...")
        debugger.show_debug_overlay(show_all_states=True)
    else:
        print("\nShowing basic overlay...")
        debugger.show_debug_overlay(show_all_states=False)


if __name__ == "__main__":
    main()
