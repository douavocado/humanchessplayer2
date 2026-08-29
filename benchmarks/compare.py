"""Compare benchmark results across devices.

Reads two or more result JSONs and lines them up against the first, which
is treated as the baseline. Two things it tries hard to do:

  Refuse to blame hardware for a configuration difference. A machine
  running a different Stockfish build, a different torch thread count, or
  uncalibrated piece templates will produce different numbers for reasons
  that have nothing to do with how fast it is. Those are reported as
  comparability warnings rather than silently folded into a ratio.

  Say what a difference implies for the pacing knobs, which is the reason
  to run this at all. The short version is that a slow device is mostly
  NOT a QUICKNESS problem -- see `interpret` below.
"""

from __future__ import annotations

import json

from benchmarks.compute import MACHINE_VARIABLE, TIME_CAPPED

# cheat_detection/config.py: emt below this counts as an "instant" move,
# and it is deliberately absolute (a human motor floor) rather than a share
# of the clock, so it is the right yardstick for a compute floor.
INSTANT_MOVE_SECS = 1.0

# Fields where a mismatch makes the timing comparison unsound rather than
# merely interesting.
HARDWARE_KEYS = ["cpu", "cpu_count_logical", "ram_gb", "platform", "python",
                 "torch", "torch_num_threads", "numpy", "opencv",
                 "stockfish_play", "stockfish_threads", "stockfish_hash_mb",
                 "resolution_scale", "cpu_reference_secs"]
INVALIDATING_KEYS = ["stockfish_play", "stockfish_threads",
                     "stockfish_hash_mb", "torch_num_threads"]

# Widest stage label is "remove_background_colours" (25); leave a space.
LABEL_COL = 26

HEADLINE_STAGES = ["sf_scan_fullwidth", "sf_sharpness", "nn_move_scorer",
                   "nn_probabilities", "nn_alter", "make_move_total",
                   "compute_floor"]


def load(paths):
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
        data["_path"] = p
        out.append(data)
    return out


def _label(res, idx):
    return res.get("label") or res.get("hardware", {}).get("cpu") or \
        f"device{idx}"


def _p50(res, stage):
    s = (res.get("compute", {}).get("stages", {}).get(stage) or {})
    return s.get("p50")


def warnings(results):
    """Configuration differences that would corrupt a hardware reading."""
    out = []
    base = results[0]
    for idx, res in enumerate(results[1:], start=1):
        for key in INVALIDATING_KEYS:
            a = base.get("hardware", {}).get(key)
            b = res.get("hardware", {}).get(key)
            if a != b:
                out.append(f"{_label(base, 0)} vs {_label(res, idx)}: {key} differs ({a} vs {b}) -- engine timings "
                           "are not comparable")
        a_tpl = base.get("vision", {}).get("using_profile_templates")
        b_tpl = res.get("vision", {}).get("using_profile_templates")
        if a_tpl is not None and b_tpl is not None and a_tpl != b_tpl:
            out.append(f"{_label(base, 0)} vs {_label(res, idx)}: one is running calibrated piece templates "
                       "and the other the chessimage/ fallback -- recognition "
                       "timings are not comparable")
    return out


def interpret(results):
    """What the differences mean for the tuning knobs.

    The reasoning, so it can be argued with rather than trusted:

    QUICKNESS scales the whole move-time distribution, so it can make the
    bot slower or faster on average, but it cannot lift a move above the
    compute floor -- the client sleeps only the *remainder* of the intended
    think time and skips the sleep when compute already overran it. So a
    device whose floor sits above the 1s instant-move threshold cannot
    produce instant moves at any QUICKNESS, and the instant-move rate is a
    tracked human-likeness feature that rises with rating. That is a
    compute problem (ponder width, SHARPNESS_SCAN_DEPTH, fewer re-evals),
    not a pacing one.

    The reverse case is the one QUICKNESS does own: if two devices have
    comparable floors and differ only in how long they *choose* to think,
    that is pacing and the knob is the right lever.
    """
    lines = []
    for idx, res in enumerate(results):
        label = _label(res, idx)
        floor = _p50(res, "compute_floor")
        if floor is None:
            continue
        if floor > INSTANT_MOVE_SECS:
            lines.append(
                f"{label}: compute floor p50 {floor:.2f}s is ABOVE the {INSTANT_MOVE_SECS:.0f}s "
                "instant-move threshold. Instant moves are unreachable here "
                "at any QUICKNESS; reduce compute rather than pacing."
                )
        elif floor > INSTANT_MOVE_SECS * 0.5:
            lines.append(
                f"{label}: compute floor p50 {floor:.2f}s leaves little room under the "
                f"{INSTANT_MOVE_SECS:.0f}s instant-move threshold -- the fast tail will be thin."
                )
        else:
            lines.append(
                f"{label}: compute floor p50 {floor:.2f}s clears the {INSTANT_MOVE_SECS:.0f}s "
                "instant-move threshold; pacing knobs are free to act."
                )

    if len(results) > 1:
        base = results[0]
        for idx, res in enumerate(results[1:], start=1):
            a, b = _p50(base, "compute_floor"), _p50(res, "compute_floor")
            if a and b:
                lines.append(
                    f"{_label(res, idx)} runs the move pipeline x{b / a:.2f} relative to {_label(base, 0)} "
                    f"(floor p50 {b:.2f}s vs {a:.2f}s)")
            da = (base.get("compute", {}).get("sf_scan_depth") or {}).get("mean")
            db = (res.get("compute", {}).get("sf_scan_depth") or {}).get("mean")
            if da and db and abs(da - db) >= 0.5:
                lines.append(
                    f"{_label(res, idx)} reaches depth {db:.1f} on the time-capped scan vs {da:.1f} "
                    f"on {_label(base, 0)} -- that is a strength difference, not a speed one, "
                    "and no pacing knob addresses it.")

    for idx, res in enumerate(results):
        live = res.get("mouse", {}).get("live", {})
        if live.get("moveTo_fits_step_budget") is False:
            g = live.get("gesture", {}).get("overrun_ratio", {})
            lines.append(
                "{}: pyautogui.moveTo is too slow for quick_move_to's 10ms "
                "step budget, so gestures overrun{}. This lands directly on "
                "the game clock.".format(
                    _label(res, idx),
                    " (x{:.2f} realised/planned)".format(g["p50"])
                    if g.get("p50") else ""))
    return lines


def report(results):
    labels = [_label(r, i) for i, r in enumerate(results)]
    width = max(14, max(len(x) for x in labels) + 2)

    lines = ["Hardware", "  " + "field".ljust(LABEL_COL)
             + "".join(x.ljust(width) for x in labels)]
    for key in HARDWARE_KEYS:
        vals = [str(r.get("hardware", {}).get(key, "-")) for r in results]
        if all(v == "-" for v in vals):
            continue
        lines.append("  " + key.ljust(LABEL_COL)
                     + "".join(v[:width - 1].ljust(width) for v in vals))

    lines.append("")
    lines.append(f"Compute stages (p50 ms; ratio vs {labels[0]})")
    lines.append("  " + "stage".ljust(LABEL_COL)
                 + "".join(x.ljust(width) for x in labels))
    for stage in HEADLINE_STAGES:
        cells = []
        base = _p50(results[0], stage)
        for i, res in enumerate(results):
            v = _p50(res, stage)
            if v is None:
                cells.append("-".ljust(width))
                continue
            cell = f"{v * 1000:.0f}"
            if i > 0 and base:
                cell += f" (x{v / base:.2f})"
            cells.append(cell.ljust(width))
        tag = ""
        if stage in TIME_CAPPED:
            tag = "  [time-capped: read depth, not time]"
        elif stage in MACHINE_VARIABLE:
            tag = "  [tracks CPU speed]"
        lines.append("  " + stage.ljust(LABEL_COL) + "".join(cells) + tag)

    depth_cells = []
    for res in results:
        d = (res.get("compute", {}).get("sf_scan_depth") or {}).get("mean")
        depth_cells.append((f"{d:.2f}" if d else "-").ljust(width))
    lines.append("  " + "capped-scan depth".ljust(LABEL_COL) + "".join(depth_cells))

    nps_cells = []
    for res in results:
        n = (res.get("compute", {}).get("nps_probe") or {}).get("nps")
        nps_cells.append((f"{n / 1e6:.2f} Mnps" if n else "-")
                         .ljust(width))
    lines.append("  " + "stockfish throughput".ljust(LABEL_COL) + "".join(nps_cells))

    # --- vision / mouse ---------------------------------------------------
    lines.append("")
    lines.append("Vision (p50 ms)")
    lines.append("  " + "stage".ljust(LABEL_COL)
                 + "".join(x.ljust(width) for x in labels))
    for group, stage in (("recognition", "get_fen_from_image"),
                         ("recognition", "remove_background_colours"),
                         ("capture", "board_region"),
                         ("capture", "full_screen")):
        cells = []
        for res in results:
            s = (res.get("vision", {}).get(group, {}).get(stage) or {})
            v = s.get("p50")
            cells.append((f"{v * 1000:.1f}" if v else "-").ljust(width))
        lines.append("  " + stage.ljust(LABEL_COL) + "".join(cells))

    lines.append("")
    lines.append("Mouse (p50 ms)")
    for name in ("moveTo_call", "position_call"):
        cells = []
        for res in results:
            s = (res.get("mouse", {}).get("live", {}).get(name) or {})
            v = s.get("p50")
            cells.append((f"{v * 1000:.2f}" if v else "-").ljust(width))
        lines.append("  " + name.ljust(LABEL_COL) + "".join(cells))

    warns = warnings(results)
    if warns:
        lines.append("")
        lines.append("Comparability warnings")
        for w in warns:
            lines.append("  ! " + w)

    notes = interpret(results)
    if notes:
        lines.append("")
        lines.append("What this means for tuning")
        for n in notes:
            lines.append("  - " + n)
    return "\n".join(lines)
