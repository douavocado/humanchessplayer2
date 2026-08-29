#!/usr/bin/env python3
"""
Frame extraction and ground-truth generation from a screen recording.

Calibration normally starts from a directory of screenshots, each optionally
carrying a hand-written ``<name>_fen.txt`` sidecar so the fitter knows what
position and clock times a frame shows (it needs those to label the piece and
digit templates it cuts out). Writing those by hand is the tedious part, and
it is why calibrating from a recording of a real game is attractive: a video
of one game holds every state the fitter wants, sampled as densely as you
like.

This module closes that gap. Given the recording and the PGN of the game that
was played, it samples frames, scrapes each one's position with the *live*
scraper, and matches that position against the game's plies to recover the
ground truth automatically. The PGN supplies the clock times via its ``%clk``
comments, so the sidecars come out complete.

Two properties of the alignment are worth stating, because they are what make
it trustworthy:

- **Matching is exact, never nearest.** A frame is used only when its scraped
  placement equals some ply's placement character for character. Piece
  animation means a fair share of sampled frames catch a piece mid-flight, in
  a position that never existed; those must not become training crops, and
  refusing anything inexact drops them for free. (An outright misread lands in
  the same bucket and is likewise dropped, which is the conservative outcome.)
- **Only the stationary clock is recorded.** After a move the mover's clock
  stops and reads exactly what the PGN says; the opponent's is ticking and its
  displayed value is unknowable from the PGN. So each frame contributes the one
  clock that can be trusted, and the other is left unset.

This is a bootstrapping tool: it needs a profile good enough to scrape a
position, and produces the ground truth that makes a better one. Fit a rough
profile from a couple of frames first (see offline_fitter), run this, then
re-fit against the full labelled set.

Usage:
    python -m auto_calibration.video_frames \\
        --video game.mp4 --pgn game.pgn --out ./frames/ \\
        --site chess_com
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

from .board_detector import BoardDetector
from .config import DEFAULT_SITE

# How much of the video to skip between sampled frames, in seconds. Piece
# animation is ~0.2s, so this is not trying to avoid it (exact matching does
# that); it is just a density that keeps the frame count manageable.
DEFAULT_STEP_SECONDS = 2.0

# Board detections below this confidence are treated as "no board on screen"
# (lobby, result modal, a scrolled page).
MIN_BOARD_CONFIDENCE = 0.4

# chess.com switches from "M:SS" to "M:SS.T" below this many seconds.
CHESS_COM_TENTHS_BELOW = 20.0

CLOCK_COMMENT_RE = re.compile(r"\[%clk\s+(\d+):(\d+):([\d.]+)\]", re.DOTALL)


def parse_pgn_plies(pgn_path: str) -> List[Dict]:
    """
    Read a PGN into one record per ply.

    Args:
        pgn_path: Path to a PGN file. Only the first game is read.

    Returns:
        List of dicts with 'ply', 'placement' (the FEN's board field),
        'fen' (the full FEN), 'mover' ('w'/'b'), 'clock' (seconds left for
        the mover after the move, or None) and 'san'.
    """
    import chess.pgn

    with open(pgn_path, encoding="utf-8-sig") as handle:
        game = chess.pgn.read_game(handle)

    if game is None:
        raise ValueError(f"No game found in {pgn_path}")

    plies = []
    for index, node in enumerate(game.mainline(), start=1):
        match = CLOCK_COMMENT_RE.search(node.comment or "")
        clock = None
        if match:
            clock = (int(match.group(1)) * 3600
                     + int(match.group(2)) * 60
                     + float(match.group(3)))

        # The side that just moved is the side whose clock is now stopped.
        mover = "w" if node.parent.board().turn else "b"

        board = node.board()
        plies.append({
            "ply": index,
            # Placement alone is what a scraped frame can be matched on; the
            # full FEN additionally carries the side to move, which is ground
            # truth for turn detection.
            "placement": board.board_fen(),
            "fen": board.fen(),
            "mover": mover,
            "clock": clock,
            "san": node.san(),
        })

    return plies


def index_by_placement(plies: List[Dict]) -> Dict[str, Dict]:
    """
    Index plies by board placement, dropping placements that repeat.

    A repeated position (a shuffle, a threefold) cannot be pinned to one ply
    from the picture alone, and guessing between them would attach the wrong
    clock time to a frame. Dropping them costs a handful of frames out of a
    whole game.

    Args:
        plies: Ply records from parse_pgn_plies.

    Returns:
        Mapping of placement string to its unique ply record.
    """
    counts: Dict[str, int] = {}
    for ply in plies:
        counts[ply["placement"]] = counts.get(ply["placement"], 0) + 1

    return {ply["placement"]: ply for ply in plies
            if counts[ply["placement"]] == 1}


def detect_board_in_video(capture, step_frames: int,
                          max_probes: int = 40) -> Optional[Dict]:
    """
    Find the board by probing frames until one detects confidently.

    The board does not move during a game, so one detection serves every
    frame - and re-detecting per frame would let a mid-animation frame shift
    the crop under the scraper.

    Args:
        capture: An open cv2.VideoCapture.
        step_frames: Frames to advance between probes.
        max_probes: Give up after this many probes.

    Returns:
        Board detection dict, or None if no frame showed a board.
    """
    detector = BoardDetector()

    for probe in range(max_probes):
        capture.set(cv2.CAP_PROP_POS_FRAMES, probe * step_frames)
        ok, frame = capture.read()
        if not ok:
            break

        detection = detector.detect(frame)
        if detection and detection.get("confidence", 0) >= MIN_BOARD_CONFIDENCE:
            return detection

    return None


def displayed_clock_text(seconds: float, site: str) -> str:
    """
    Render a clock the way the site draws it.

    The digits a template extractor cuts out are the digits actually on
    screen, so their labels have to come from the *rendered* form, not from
    the raw seconds. chess.com drops the leading zero on minutes and switches
    to tenths of a second below twenty; Lichess's zero-padded MM:SS is the
    fallback for everything else.

    Args:
        seconds: Time remaining.
        site: Site identifier.

    Returns:
        The clock string, e.g. "2:58", "0:04.9" or "02:58".
    """
    minutes = int(seconds) // 60
    remainder = seconds - minutes * 60

    if site == "chess_com":
        if seconds < CHESS_COM_TENTHS_BELOW:
            return f"{minutes}:{remainder:04.1f}"
        return f"{minutes}:{int(remainder):02d}"

    return f"{minutes:02d}:{int(remainder):02d}"


def write_sidecar(path: Path, ply: Dict, bottom: str) -> None:
    """
    Write the ``_fen.txt`` ground truth for one matched frame.

    Args:
        path: The frame's image path (the sidecar is named from its stem).
        ply: The matched ply record.
        bottom: Which colour is at the bottom of the board ('w' or 'b').
    """
    lines = [ply.get("fen") or ply["placement"], f"side:{bottom}"]

    if ply["clock"] is not None:
        # Only the mover's clock has stopped; the other is still ticking and
        # its displayed value cannot be recovered from the PGN.
        which = "bottom" if ply["mover"] == bottom else "top"
        lines.append(f"{which}:{ply['clock']:g}")

    sidecar = path.parent / f"{path.stem}_fen.txt"
    sidecar.write_text("\n".join(lines) + "\n", encoding="utf-8")


def extract(video: str, pgn: str, out_dir: str,
            step_seconds: float = DEFAULT_STEP_SECONDS,
            bottom: str = "w",
            site: str = DEFAULT_SITE,
            state: str = "play",
            board: Optional[Dict] = None) -> Tuple[int, int]:
    """
    Sample a recording into labelled calibration frames.

    Args:
        video: Path to the screen recording.
        pgn: Path to the PGN of the game in the recording.
        out_dir: Directory to write frames and sidecars into.
        step_seconds: Seconds between sampled frames.
        bottom: Colour at the bottom of the board in the recording.
        site: Site identifier, for the clock rendering rules.
        state: State prefix for the written filenames (the fitter reads the
               game state out of the filename).
        board: Board detection to reuse; detected from the video if omitted.

    Returns:
        (written, skipped) frame counts.
    """
    from chessimage.image_scrape_utils import get_fen_from_image

    plies = parse_pgn_plies(pgn)
    by_placement = index_by_placement(plies)
    dropped = len(plies) - len(by_placement)
    print(f"PGN: {len(plies)} plies, {len(by_placement)} uniquely placed"
          + (f" ({dropped} repeated, unusable)" if dropped else ""))

    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    step_frames = max(1, round(fps * step_seconds))
    print(f"Video: {total} frames @ {fps:.1f}fps, sampling every {step_frames}")

    if board is None:
        board = detect_board_in_video(capture, step_frames)
        if board is None:
            raise ValueError("No board detected anywhere in the video")
    print(f"Board: ({board['x']}, {board['y']}) size={board['size']} "
          f"conf={board.get('confidence', 0):.2f}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    bx, by, bs = board["x"], board["y"], board["size"]
    written = skipped = 0
    seen_plies = set()

    for index in range(0, total, step_frames):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            break

        board_img = frame[by:by + bs, bx:bx + bs]
        if board_img.size == 0:
            skipped += 1
            continue

        try:
            fen = get_fen_from_image(board_img, bottom=bottom, fast_mode=True)
        except Exception:  # noqa: BLE001 - one bad frame must not end the run
            skipped += 1
            continue

        ply = by_placement.get(fen.split()[0])
        if ply is None:
            # Mid-animation, off-game (lobby/modal), or a misread. All three
            # are things we would rather not cut templates out of.
            skipped += 1
            continue

        # One frame per ply is enough, and the first is the least likely to
        # have the next move's animation starting on top of it.
        if ply["ply"] in seen_plies:
            continue
        seen_plies.add(ply["ply"])

        name = f"{state}_{index:06d}_ply{ply['ply']:03d}.png"
        frame_path = out_path / name
        cv2.imwrite(str(frame_path), frame)
        write_sidecar(frame_path, ply, bottom)
        written += 1

    capture.release()

    print(f"\nWrote {written} labelled frames to {out_path}")
    print(f"Skipped {skipped} (animation, off-game, or unmatched)")
    return written, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Extract labelled calibration frames from a screen recording.")
    parser.add_argument("--video", required=True,
                        help="Screen recording of a game.")
    parser.add_argument("--pgn", required=True,
                        help="PGN of that game (with %%clk comments).")
    parser.add_argument("--out", required=True,
                        help="Output directory for frames.")
    parser.add_argument("--step-secs", type=float, default=DEFAULT_STEP_SECONDS,
                        help="Seconds between sampled frames "
                             f"(default {DEFAULT_STEP_SECONDS}).")
    parser.add_argument("--bottom", choices=["w", "b"], default="w",
                        help="Colour at the bottom of the board (default w).")
    parser.add_argument("--site", default=DEFAULT_SITE,
                        help="Site the recording is from (lichess, chess_com).")
    parser.add_argument("--state", default="play",
                        help="Game-state prefix for output filenames.")
    args = parser.parse_args()

    written, _ = extract(
        video=args.video, pgn=args.pgn, out_dir=args.out,
        step_seconds=args.step_secs, bottom=args.bottom,
        site=args.site, state=args.state,
    )

    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
