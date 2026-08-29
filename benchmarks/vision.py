"""Screen capture and board-reading cost.

Two halves, measured separately because only one of them is portable:

  capture    Real grabs off the real screen, at the region sizes the client
             actually uses. Needs a display and measures the whole stack
             (GPU/compositor/backend), so it is the part that varies most
             wildly between devices -- and the part that cannot be faked.

  recognition
             FEN extraction and background removal, run on board images
             synthesised in-process from the piece templates tracked in
             chessimage/. Nothing is read off the screen, so this half runs
             headless and, more importantly, gives every device *identical
             input*. That is what makes the numbers comparable: the
             existing offline path (auto_calibration/offline_screenshots,
             which simulation.calibrate detection drives) is gitignored, so
             two machines would otherwise be timing different pictures.

Clock OCR is not covered here -- the digit templates alone do not
reconstruct a realistic clock crop, and `python -m simulation.calibrate
detection` already measures it wherever real screenshots exist.

Two caveats when comparing devices:

  remove_background_colours is timed at full board resolution. Production's
  get_fen_from_image downscales first (fast_mode), so the standalone figure
  is an upper bound on that operation, not the cost production pays.

  Piece-template size depends on whether a calibration profile is present,
  and an uncalibrated machine falls back to chessimage/'s own templates at
  a different scale. That changes the matching workload, so the result
  records `using_profile_templates` and the compare command refuses to read
  a recognition difference as hardware when the two sides disagree on it.
"""

from __future__ import annotations

import time

import chess
import numpy as np

from benchmarks import positions as _positions
from benchmarks.stats import summarise

# Lichess board square colours, close enough that remove_background_colours
# treats them as coloured background exactly as it does on a real board --
# which is the property being timed. Not a rendering of any real client.
LIGHT_BGR = (181, 217, 240)
DARK_BGR = (99, 136, 181)

REPEATS = 20


def synth_board(fen, step, templates):
    """A board image built from the tracked piece templates.

    Deterministic and display-free, so it is the same workload on every
    machine. Pieces are pasted as grayscale over coloured squares, which is
    the structure remove_background_colours keys off (it keeps pixels whose
    channels are near-equal and discards the coloured squares).
    """
    size = int(8 * step)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    board = chess.Board(fen)
    for rank in range(8):
        for file in range(8):
            y0, x0 = int(rank * step), int(file * step)
            y1, x1 = int((rank + 1) * step), int((file + 1) * step)
            # rank 0 of the image is rank 8 of the board (white at bottom)
            square = chess.square(file, 7 - rank)
            light = (file + rank) % 2 == 0
            img[y0:y1, x0:x1] = LIGHT_BGR if light else DARK_BGR
            piece = board.piece_at(square)
            if piece is None:
                continue
            tpl = templates.get(piece.symbol())
            if tpl is None:
                continue
            h, w = min(y1 - y0, tpl.shape[0]), min(x1 - x0, tpl.shape[1])
            patch = tpl[:h, :w]
            # Only the glyph is stamped; near-black template background is
            # left as the square colour, so the square keeps its tint.
            mask = patch > 40
            img[y0:y0 + h, x0:x0 + w][mask] = np.stack([patch] * 3, axis=-1)[mask]
    return img


def _time(fn, repeats=REPEATS):
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return samples


def run(repeats=REPEATS, progress=True):
    import chessimage.image_scrape_utils as isu

    step = isu.STEP
    result = {"board_step_px": step, "repeats": repeats,
              "using_profile_templates": isu._using_profile_templates}

    # ---------------------------------------------------------- recognition
    boards = [synth_board(fen, step, isu.ALL_PIECES)
              for fen, _ in _positions.POSITIONS]
    if progress:
        print(f"  synthesised {len(boards)} board images at {int(8 * step)}x{int(8 * step)}px", flush=True)

    bg_samples, fen_samples = [], []
    for img in boards:
        bg_samples += _time(lambda im=img: isu.remove_background_colours(im),
                            repeats)
        fen_samples += _time(
            lambda im=img: isu.get_fen_from_image(im, bottom="w"), repeats)
    result["recognition"] = {
        "remove_background_colours": summarise(bg_samples),
        "get_fen_from_image": summarise(fen_samples),
    }

    # Whether the synthesised boards are actually readable. Not an accuracy
    # claim about production (these are not screenshots) -- it is a guard
    # that the timings above come from the real matching path rather than
    # from an early-out on an image the extractor cannot parse at all.
    placements = 0
    for (fen, _), img in zip(_positions.POSITIONS, boards):
        try:
            got = isu.get_fen_from_image(img, bottom="w")
        except Exception:  # noqa: BLE001, S112 - an unreadable board just
            continue   # fails the self-check; the timings above still stand
        if got and got.split()[0] == fen.split()[0]:
            placements += 1
    result["recognition"]["exact_placement_reads"] = placements
    result["recognition"]["boards"] = len(boards)

    # -------------------------------------------------------------- capture
    result["capture"] = _capture_suite(isu, step, repeats, progress)
    return result


def _capture_suite(isu, step, repeats, progress):
    """Real screen grabs, or a recorded reason why not."""
    try:
        cap = isu.SCREEN_CAPTURE
        board_px = int(8 * step)
        regions = {
            "board_region": (0, 0, board_px, board_px),
            "clock_region": (0, 0, 160, 60),
            "full_screen": None,
        }
        out = {}
        for name, box in regions.items():
            try:
                if box is None:
                    samples = _time(lambda: cap.capture(), repeats)
                    shape = cap.capture().shape
                else:
                    samples = _time(lambda b=box: cap.capture(b), repeats)
                    shape = cap.capture(box).shape
                out[name] = summarise(samples)
                out[name + "_shape"] = list(shape)
            except Exception as exc:  # noqa: BLE001 - one unavailable
                # region must not lose the others
                out[name] = {"n": 0, "error": f"{type(exc).__name__}: {exc}"}
        if progress:
            print("  captured live screen regions", flush=True)
        return out
    except Exception as exc:  # noqa: BLE001
        # Headless CI, no X11, locked session: not a failure, just a
        # measurement this machine cannot make.
        return {"unavailable": f"{type(exc).__name__}: {exc}"}


def report(result):
    lines = []
    rec = result.get("recognition", {})
    lines.append("  recognition (synthesised boards, identical on every device"
                 "{})".format("" if result.get("using_profile_templates")
                              else "; FALLBACK templates, uncalibrated"))
    lines.append("    stage                     mean_ms    p50_ms    p90_ms")
    for name in ("remove_background_colours", "get_fen_from_image"):
        s = rec.get(name) or {}
        if s.get("n"):
            lines.append("    {:<24s} {:9.2f} {:9.2f} {:9.2f}".format(
                name, s["mean"] * 1000, s["p50"] * 1000, s["p90"] * 1000))
    if "exact_placement_reads" in rec:
        lines.append("    placement read back exactly on {}/{} boards".format(
            rec["exact_placement_reads"], rec["boards"]))

    cap = result.get("capture", {})
    lines.append("  capture (live screen)")
    if "unavailable" in cap:
        lines.append("    unavailable: {}".format(cap["unavailable"]))
    else:
        for name in ("board_region", "clock_region", "full_screen"):
            s = cap.get(name) or {}
            if s.get("n"):
                shape = cap.get(name + "_shape")
                lines.append("    {:<24s} {:9.2f} {:9.2f} {:9.2f}   {}".format(
                    name, s["mean"] * 1000, s["p50"] * 1000, s["p90"] * 1000,
                    "x".join(str(d) for d in shape[:2]) if shape else ""))
            elif s.get("error"):
                lines.append("    {:<24s} error: {}".format(name, s["error"]))
    return "\n".join(lines)
