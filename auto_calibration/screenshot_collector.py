#!/usr/bin/env python3
"""
Screenshot Collector

Helper tool to save screenshots for offline calibration.
Saves screenshots with optional state labels for later fitting.
"""

import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from .utils import capture_screenshot, save_image, get_screenshots_directory


def save_calibration_screenshot(state_hint: Optional[str] = None,
                                delay: float = 0,
                                output_dir: Optional[str] = None) -> Optional[str]:
    """
    Save a screenshot for calibration.
    
    Args:
        state_hint: Optional state hint (e.g., 'play', 'start1', 'end_resign').
        delay: Delay in seconds before capturing.
        output_dir: Output directory. If None, uses default.
    
    Returns:
        Path to saved screenshot, or None if failed.
    """
    if output_dir is None:
        output_dir = get_screenshots_directory()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Wait if delay specified
    if delay > 0:
        print(f"Capturing in {delay} seconds...")
        for i in range(int(delay), 0, -1):
            print(f"  {i}...")
            time.sleep(1)
        # Handle fractional delay
        remaining = delay - int(delay)
        if remaining > 0:
            time.sleep(remaining)
    
    # Capture screenshot
    print("Capturing screenshot...")
    screenshot = capture_screenshot()
    
    if screenshot is None:
        print("Failed to capture screenshot")
        return None
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"calibration_{timestamp}"
    if state_hint:
        # Sanitise state hint
        safe_hint = "".join(c if c.isalnum() or c == '_' else '_' for c in state_hint)
        filename += f"_{safe_hint}"
    filename += ".png"
    
    # Save
    output_path = output_dir / filename
    if save_image(str(output_path), screenshot):
        print(f"Saved: {output_path}")
        return str(output_path)
    else:
        print("Failed to save screenshot")
        return None


def collect_game_states(delay_between: float = 2.0):
    """
    Interactive collection of multiple game states.
    
    Prompts user to prepare each state and captures screenshots.
    """
    states = [
        ('play', 'Normal gameplay (after several moves)'),
        ('start1', 'Game start (before first move)'),
        ('start2', 'After first move'),
        ('end1', 'Game over (by resignation)'),
        ('end2', 'Game over (by timeout)'),
        ('end3', 'Game over (by checkmate/stalemate)')
    ]
    
    print("=" * 60)
    print("CALIBRATION SCREENSHOT COLLECTOR")
    print("=" * 60)
    print()
    print("This tool will help you capture screenshots for each game state.")
    print("For each state, prepare the screen and press Enter to capture.")
    print("Press Ctrl+C at any time to stop.")
    print()
    
    collected = []
    
    for state, description in states:
        print("-" * 40)
        print(f"State: {state}")
        print(f"Description: {description}")
        print()
        
        try:
            input(f"Press Enter when ready to capture '{state}'...")
        except KeyboardInterrupt:
            print("\n\nCollection stopped by user.")
            break
        
        path = save_calibration_screenshot(state, delay=delay_between)
        
        if path:
            collected.append((state, path))
            print(f"✅ Captured {state}")
        else:
            print(f"❌ Failed to capture {state}")
        
        print()
    
    # Summary
    print("=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)
    print(f"Captured {len(collected)} screenshots:")
    for state, path in collected:
        print(f"  {state}: {Path(path).name}")
    print()
    print("To fit calibration from these screenshots, run:")
    print("  python -m auto_calibration.calibrator --offline")
    
    return collected


def quick_capture(count: int = 1, delay: float = 3.0):
    """
    Quick capture of multiple screenshots.
    
    Args:
        count: Number of screenshots to capture.
        delay: Delay before first capture.
    """
    print(f"Capturing {count} screenshot(s) in {delay} seconds...")
    
    for i in range(int(delay), 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    
    for i in range(count):
        if i > 0:
            print("Next capture in 1 second...")
            time.sleep(1)
        
        path = save_calibration_screenshot()
        if path:
            print(f"✅ Captured {i+1}/{count}: {Path(path).name}")
        else:
            print(f"❌ Failed to capture {i+1}/{count}")


def main():
    """Main entry point for screenshot collector."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Screenshot collector for calibration")
    parser.add_argument("--state", "-s", type=str, 
                       help="State hint (e.g., play, start1, end1)")
    parser.add_argument("--delay", "-d", type=float, default=3.0,
                       help="Delay before capture (default: 3s)")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Interactive mode: collect all game states")
    parser.add_argument("--count", "-c", type=int, default=1,
                       help="Number of screenshots to capture")
    parser.add_argument("--output", "-o", type=str,
                       help="Output directory")
    
    args = parser.parse_args()
    
    if args.interactive:
        collect_game_states(delay_between=args.delay)
    elif args.count > 1:
        quick_capture(args.count, args.delay)
    else:
        save_calibration_screenshot(args.state, args.delay, args.output)


if __name__ == "__main__":
    main()
