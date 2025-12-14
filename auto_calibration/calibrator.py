#!/usr/bin/env python3
"""
Chess Board Auto-Calibration Tool

Main entry point for calibration. Supports:
- Live calibration from current screen
- Offline calibration from saved screenshots
- Verification of existing configuration

Usage:
    # Live calibration (default)
    python calibrator.py --live
    
    # Offline calibration from screenshots
    python calibrator.py --offline ./calibration_screenshots/
    
    # Offline from single screenshot
    python calibrator.py --screenshot ./my_screenshot.png
    
    # Verify existing configuration
    python calibrator.py --verify
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Dict

from .board_detector import BoardDetector
from .clock_detector import ClockDetector
from .coordinate_calculator import CoordinateCalculator
from .visualiser import CalibrationVisualiser
from .config import save_config, ChessConfig
from .utils import capture_screenshot, load_image


class Calibrator:
    """
    Main calibration orchestrator.
    
    Coordinates board detection, clock detection, coordinate calculation,
    and visualisation into a complete calibration workflow.
    """
    
    def __init__(self):
        """Initialise calibrator."""
        self.board_detector = BoardDetector()
        self.clock_detector = ClockDetector()
        self.calculator = CoordinateCalculator()
        self.visualiser: Optional[CalibrationVisualiser] = None
    
    def run_live(self, countdown: int = 5, visualise: bool = True) -> Optional[Dict]:
        """
        Run live calibration from current screen.
        
        Args:
            countdown: Seconds to wait before capturing.
            visualise: Whether to create debug visualisations.
        
        Returns:
            Calibration configuration, or None if failed.
        """
        print("=" * 60)
        print("CHESS BOARD AUTO-CALIBRATION")
        print("=" * 60)
        print()
        print("This tool will automatically detect your chess board and UI elements.")
        print()
        print("Instructions:")
        print("1. Open a chess game in your browser (Lichess)")
        print("2. Ensure the board and clocks are visible")
        print("3. The calibration will start after the countdown")
        print()
        
        # Countdown
        if countdown > 0:
            for i in range(countdown, 0, -1):
                print(f"Starting in {i}...")
                time.sleep(1)
        
        print("\n" + "-" * 40)
        print("STEP 1: Capturing screenshot...")
        print("-" * 40)
        
        # Capture screenshot
        image = capture_screenshot()
        if image is None:
            print("❌ Failed to capture screenshot")
            return None
        
        print(f"✅ Screenshot captured: {image.shape[1]}x{image.shape[0]}")
        
        # Mark as live calibration
        self._is_offline = False
        
        # Run calibration on image
        return self._calibrate_image(image, visualise)
    
    def run_offline(self, screenshot_path: str, 
                   state_hint: Optional[str] = None,
                   visualise: bool = True) -> Optional[Dict]:
        """
        Run calibration from a saved screenshot.
        
        Args:
            screenshot_path: Path to screenshot file.
            state_hint: Optional hint about the game state.
            visualise: Whether to create debug visualisations.
        
        Returns:
            Calibration configuration, or None if failed.
        """
        print("=" * 60)
        print("OFFLINE CALIBRATION")
        print("=" * 60)
        print()
        print(f"Source: {screenshot_path}")
        if state_hint:
            print(f"State hint: {state_hint}")
        print()
        
        # Load image
        image = load_image(screenshot_path)
        if image is None:
            print(f"❌ Failed to load image: {screenshot_path}")
            return None
        
        print(f"✅ Image loaded: {image.shape[1]}x{image.shape[0]}")
        
        # Mark as offline calibration
        self._is_offline = True
        
        # Run calibration
        config = self._calibrate_image(image, visualise)
        
        if config:
            # Add source info
            config['calibration_info']['source'] = screenshot_path
            if state_hint:
                config['calibration_info']['state_hint'] = state_hint
        
        return config
    
    def _calibrate_image(self, image, visualise: bool = True) -> Optional[Dict]:
        """
        Run calibration on an image.
        
        Args:
            image: BGR image to calibrate from.
            visualise: Whether to create debug visualisations.
        
        Returns:
            Calibration configuration, or None if failed.
        """
        # Step 2: Detect board
        print("\n" + "-" * 40)
        print("STEP 2: Detecting chess board...")
        print("-" * 40)
        
        board_detection = self.board_detector.detect(image)
        
        if board_detection is None:
            print("❌ Failed to detect chess board")
            print()
            print("Troubleshooting:")
            print("• Ensure a chess game is visible on screen")
            print("• The board should be clearly visible with standard Lichess colours")
            print("• Try adjusting browser zoom to 100%")
            return None
        
        print(f"✅ Board detected:")
        print(f"   Position: ({board_detection['x']}, {board_detection['y']})")
        print(f"   Size: {board_detection['size']}x{board_detection['size']}")
        print(f"   Step: {board_detection['step']}px")
        print(f"   Confidence: {board_detection['confidence']:.3f}")
        
        # Step 3: Detect clocks
        print("\n" + "-" * 40)
        print("STEP 3: Detecting clocks...")
        print("-" * 40)
        
        self.clock_detector.set_board(board_detection)
        clock_detection = self.clock_detector.detect(image)
        
        if clock_detection:
            print(f"✅ Clocks detected: {clock_detection['detection_count']} positions")
            print(f"   Clock X: {clock_detection['clock_x']}")
            
            for clock_type in ['bottom_clock', 'top_clock']:
                if clock_type in clock_detection and clock_detection[clock_type]:
                    states = list(clock_detection[clock_type].keys())
                    print(f"   {clock_type}: {', '.join(states)}")
        else:
            print("⚠️  Clock detection failed - will estimate positions")
            clock_detection = None
        
        # Step 4: Calculate coordinates
        print("\n" + "-" * 40)
        print("STEP 4: Calculating coordinates...")
        print("-" * 40)
        
        self.calculator.set_board(board_detection)
        self.calculator.set_clocks(clock_detection)
        coordinates = self.calculator.calculate_all()
        
        # Estimate missing clock states
        if clock_detection:
            estimated = self.calculator.estimate_missing_clock_states(clock_detection)
            for clock_type in ['bottom_clock', 'top_clock']:
                if clock_type in estimated:
                    for state, coords in estimated[clock_type].items():
                        if state not in coordinates.get(clock_type, {}):
                            if clock_type not in coordinates:
                                coordinates[clock_type] = {}
                            coordinates[clock_type][state] = coords
        
        print("✅ Coordinates calculated")
        
        # Step 5: Visualise
        if visualise:
            print("\n" + "-" * 40)
            print("STEP 5: Creating visualisations...")
            print("-" * 40)
            
            self.visualiser = CalibrationVisualiser()
            outputs = self.visualiser.visualise_all(
                image, board_detection, clock_detection, coordinates
            )
            
            print(f"✅ Debug output saved to: {self.visualiser.get_output_dir()}")
        
        # Build final config
        config = {
            'calibration_info': {
                'method': 'offline' if getattr(self, '_is_offline', False) else 'live',
                'board_confidence': board_detection['confidence'],
                'clock_detection_count': clock_detection['detection_count'] if clock_detection else 0,
                'clock_states_detected': []
            },
            'coordinates': coordinates
        }
        
        # List detected clock states
        for clock_type in ['bottom_clock', 'top_clock']:
            if clock_detection and clock_type in clock_detection:
                for state in clock_detection[clock_type]:
                    config['calibration_info']['clock_states_detected'].append(
                        f"{clock_type}.{state}"
                    )
        
        return config
    
    def verify(self, verbose: bool = True) -> Dict:
        """
        Verify existing configuration by testing clock detection.
        
        Args:
            verbose: Whether to print detailed results.
        
        Returns:
            Verification results dictionary.
        """
        print("=" * 60)
        print("CONFIGURATION VERIFICATION")
        print("=" * 60)
        print()
        
        # Load current config
        config = ChessConfig()
        config.print_status()
        print()
        
        if config.is_using_fallback():
            print("⚠️  No calibration file found - using fallback coordinates")
            print("   Run calibration first: python -m auto_calibration.calibrator --live")
            return {'success': False, 'reason': 'no_config'}
        
        # Capture screenshot
        print("Capturing screenshot for verification...")
        image = capture_screenshot()
        
        if image is None:
            print("❌ Failed to capture screenshot")
            return {'success': False, 'reason': 'capture_failed'}
        
        # Test clock positions
        print("\nTesting clock positions...")
        
        coords = config.get_coordinates()
        results = {}
        total_tested = 0
        total_success = 0
        
        self.clock_detector.set_board({
            'x': coords['board']['x'],
            'y': coords['board']['y'],
            'size': coords['board']['width']
        })
        
        for clock_type in ['bottom_clock', 'top_clock']:
            if clock_type not in coords:
                continue
            
            results[clock_type] = {}
            
            for state, pos in coords[clock_type].items():
                total_tested += 1
                
                is_valid, time_value = self.clock_detector.validate_position(
                    image, pos['x'], pos['y']
                )
                
                results[clock_type][state] = {
                    'valid': is_valid,
                    'time_value': time_value,
                    'position': (pos['x'], pos['y'])
                }
                
                if is_valid:
                    total_success += 1
                    status = f"✅ {time_value}s"
                else:
                    status = "❌"
                
                if verbose:
                    print(f"  {clock_type}.{state}: {status}")
        
        success_rate = total_success / total_tested if total_tested > 0 else 0
        
        print()
        print("-" * 40)
        print(f"Success rate: {total_success}/{total_tested} ({success_rate:.1%})")
        
        if success_rate >= 0.5:
            print("✅ Configuration appears valid")
        elif success_rate > 0:
            print("⚠️  Configuration partially valid - some states may need recalibration")
        else:
            print("❌ Configuration invalid - recalibration recommended")
        
        return {
            'success': success_rate >= 0.5,
            'success_rate': success_rate,
            'total_tested': total_tested,
            'total_success': total_success,
            'results': results
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Chess Board Auto-Calibration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live calibration (default)
  python -m auto_calibration.calibrator --live
  
  # Offline from screenshot
  python -m auto_calibration.calibrator --screenshot ./screenshot.png
  
  # Offline from directory
  python -m auto_calibration.calibrator --offline ./screenshots/
  
  # Verify existing config
  python -m auto_calibration.calibrator --verify
        """
    )
    
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--live", "-l", action="store_true",
                           help="Live calibration from current screen (default)")
    mode_group.add_argument("--screenshot", "-s", type=str,
                           help="Offline calibration from single screenshot")
    mode_group.add_argument("--offline", "-o", type=str, nargs='?', const='.',
                           help="Offline calibration from screenshots directory")
    mode_group.add_argument("--verify", "-v", action="store_true",
                           help="Verify existing configuration")
    
    parser.add_argument("--countdown", "-c", type=int, default=5,
                       help="Countdown seconds before live capture (default: 5)")
    parser.add_argument("--state", type=str,
                       help="State hint for offline calibration")
    parser.add_argument("--no-visualise", action="store_true",
                       help="Skip debug visualisation")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save configuration to file")
    
    args = parser.parse_args()
    
    calibrator = Calibrator()
    config = None
    
    try:
        if args.verify:
            calibrator.verify()
            return 0
        
        elif args.screenshot:
            config = calibrator.run_offline(
                args.screenshot,
                args.state,
                not args.no_visualise
            )
        
        elif args.offline:
            # Use offline fitter for directory
            from .offline_fitter import fit_from_screenshots
            config = fit_from_screenshots(args.offline, not args.no_save)
            return 0 if config else 1
        
        else:
            # Default: live calibration
            config = calibrator.run_live(
                args.countdown,
                not args.no_visualise
            )
        
        if config:
            print("\n" + "=" * 60)
            print("CALIBRATION COMPLETE")
            print("=" * 60)
            
            # Print summary
            info = config['calibration_info']
            print(f"\nBoard confidence: {info['board_confidence']:.3f}")
            print(f"Clock states detected: {len(info.get('clock_states_detected', []))}")
            
            if not args.no_save:
                output_path = save_config(config)
                print(f"\n✅ Configuration saved to: {output_path}")
                print("\nYour system will now use these calibrated coordinates.")
            
            return 0
        else:
            print("\n❌ Calibration failed")
            return 1
    
    except KeyboardInterrupt:
        print("\n\nCalibration cancelled by user.")
        return 1
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
