#!/usr/bin/env python3
"""
Log every gate of new-game detection, live, while a game starts.

`detect_new_game` is an AND of several independent checks, and when it
returns None it says only "not a new game" - not which check said so. That is
fine in production and useless when a profile is being brought up on a new
machine, where the answer is usually one gate failing for a few seconds
around the start and no screenshot of that window ever being saved (the
client only saves a frame once the whole 60s wait has timed out, by which
point the game is well underway and *every* gate legitimately says no).

This polls the same functions the client polls, prints a line whenever any
gate changes value, and saves the frame at each change. Start it, then start
a game the way you normally would; the transition is in the log.

Usage:
    python -m scripts.utilities.diagnose_new_game --seconds 90 --expected-time 180
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2

# Must run before any capture backend is built or any coordinate is read.
from common.platform_compat import init as init_platform

init_platform()


def gate_states(site, expected_time):
    """
    Read every gate of detect_new_game once, tolerating failures.

    Args:
        site: The Site instance.
        expected_time: Expected starting clock in seconds.

    Returns:
        Ordered dict-like list of (name, value) pairs.
    """
    from chessimage.image_scrape_utils import (
        capture_board,
        capture_bottom_clock,
        get_fen_from_image,
        read_clock,
    )

    def safe(fn, default="ERR"):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - a diagnostic must not die
            return f"{default}({type(exc).__name__})"

    clock = safe(lambda: read_clock(capture_bottom_clock(state="play")))
    matches = safe(lambda: site.clock_matches_new_game(clock, expected_time)
                   if isinstance(clock, int) else False)

    def start_like():
        board_img = capture_board()
        allowed = site.start_like_board_fens()
        import chess
        for bottom in ("w", "b"):
            fen = get_fen_from_image(board_img, bottom=bottom, fast_mode=True)
            if chess.Board(fen).board_fen() in allowed:
                return bottom
        return False

    return [
        ("clock", clock),
        ("clock_ok", matches),
        ("controls", safe(site._game_controls_visible)),
        ("start_btn", safe(lambda: site._find_start_game_button() is not None)),
        ("game_over", safe(site.game_over_screen_visible)),
        ("start_like", safe(start_like)),
        ("DETECTED", safe(lambda: site.detect_new_game(
            expected_time=expected_time))),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Live per-gate log of new-game detection.")
    parser.add_argument("--seconds", type=float, default=90.0,
                        help="How long to watch (default 90).")
    parser.add_argument("--expected-time", type=int, default=180,
                        help="Expected starting clock, seconds (default 180).")
    parser.add_argument("--out", default="logs/new_game_diagnosis",
                        help="Directory for the saved frames.")
    parser.add_argument("--calibration-profile",
                        help="Named calibration profile to use.")
    args = parser.parse_args()

    if args.calibration_profile:
        os.environ["HCP_CALIBRATION_PROFILE"] = args.calibration_profile

    from chessimage.image_scrape_utils import SCREEN_CAPTURE
    from sites import get_site_for_config

    site = get_site_for_config()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Watching for {args.seconds:.0f}s, expecting a {args.expected_time}s "
          f"clock. Start your game now.")
    print("Frames are saved whenever any gate changes.\n")

    previous = None
    started = time.time()
    saved = 0
    while time.time() - started < args.seconds:
        states = gate_states(site, args.expected_time)
        current = [(name, str(value)) for name, value in states]
        if current != previous:
            now = time.time()
            stamp = (time.strftime("%H:%M:%S", time.localtime(now))
                     + f".{int(now % 1 * 1000):03d}")
            elapsed = time.time() - started
            print(f"[{stamp}  t+{elapsed:5.1f}s] " +
                  "  ".join(f"{name}={value}" for name, value in current))
            frame = SCREEN_CAPTURE.capture(None)
            if frame is not None:
                path = out / f"gate_{saved:03d}_{stamp.replace(':', '-')}.png"
                cv2.imwrite(str(path), frame[:, :, :3])
                saved += 1
            previous = current

        if any(name == "DETECTED" and value not in ("None", "False")
               for name, value in current):
            print("\nDETECTED - stopping.")
            break

    print(f"\nSaved {saved} frames to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
