"""Mouse-gesture cost, planned versus realised.

The planned half is free of device variance by construction: every gesture
duration the bot uses is *decided* by a formula in common/move_timing.py
(movement_duration, clamped to [0.04, 0.12]s, plus the settle sleeps), not
measured from the hardware. Those numbers are reported anyway, because they
are the target the realised half is judged against.

The realised half is where a device can betray the bot.
CustomCursor.quick_move_to walks a Bezier curve at ~100Hz --

    steps = max(3, int(duration / 0.01))
    for each step: pyautogui.moveTo(..., _pause=False); sleep the remainder

-- so the loop can only hold its schedule if one pyautogui.moveTo costs
appreciably less than the 10ms it has per step. On a machine where that
call is slow (remote desktop, heavy DPI virtualisation, a busy compositor)
the sleep is already negative on arrival and the gesture silently overruns.
A 0.04s planned flick that takes 0.2s is 160ms of extra time on the clock
per move, which no amount of pacing tuning will hand back.

The realised measurements move the real cursor, so they are opt-in
(`--mouse-live`). The cursor is returned to where it started.
"""

from __future__ import annotations

import math
import random
import time

from benchmarks.stats import summarise
from common.move_timing import (
    click_settle_sleep,
    drag_settle_sleep,
    movement_duration,
)

# Board-square pixel size at 1080p, matching simulation/latency_model.py's
# BASE_STEP_PX so the two agree on what a "one square" move costs.
BASE_STEP_PX = 105.0

# Distances a real gesture covers: adjacent square up to a corner-to-corner
# reach across the board.
DISTANCE_STEPS = [1, 2, 3, 5, 7]

SAMPLES = 400
LIVE_REPEATS = 15

# Keep every live target this far from any screen edge: PyAutoGUI aborts
# with FailSafeException on a move into a corner.
MARGIN_PX = 120


def planned(mouse_quickness, resolution_scale, samples=SAMPLES):
    """The durations the formulas decide. Device-independent by design;
    measured here so a comparison can *demonstrate* that rather than assume
    it, and so the realised half has something to be judged against."""
    rng = random.Random(4242)
    step_px = BASE_STEP_PX * resolution_scale
    by_distance = {}
    for n in DISTANCE_STEPS:
        dist = step_px * n
        vals = [movement_duration(dist, mouse_quickness, resolution_scale,
                                  rng=rng) for _ in range(samples)]
        by_distance[f"{n}_squares"] = summarise(vals)
    return {
        "mouse_quickness": mouse_quickness,
        "resolution_scale": resolution_scale,
        "movement_duration": by_distance,
        "click_settle_sleep": summarise(
            [click_settle_sleep(rng) for _ in range(samples)]),
        "drag_settle_sleep": summarise(
            [drag_settle_sleep(rng) for _ in range(samples)]),
    }


def live(resolution_scale, repeats=LIVE_REPEATS, progress=True):
    """Realised cursor cost. Moves the real mouse."""
    import pyautogui

    from common.custom_cursor import CustomCursor

    origin = pyautogui.position()
    out = {}

    # Every target is clamped well inside the screen. PyAutoGUI's fail-safe
    # aborts on any move into a corner, and the live client never trips it
    # because it only ever moves to board squares -- so the benchmark stays
    # inside a safe box too rather than switching the safety off.
    screen_w, screen_h = pyautogui.size()
    margin = MARGIN_PX
    cx, cy = screen_w // 2, screen_h // 2
    max_reach = max(40, min(cx, cy) - margin)

    def clamp(x, y):
        return (int(min(max(x, margin), screen_w - margin)),
                int(min(max(y, margin), screen_h - margin)))

    def move_ignoring_failsafe(x, y):
        """Move even when the cursor currently sits in a fail-safe corner.

        pyautogui runs failSafeCheck() on entry to every call, so a cursor
        resting at a corner (a very common idle position) makes even the
        move that would rescue it raise. The check is suspended for this
        one repositioning and restored immediately, so it still guards the
        measurement loop below -- which is what the fail-safe is actually
        for: letting a human abort automation mid-run.
        """
        previous = pyautogui.FAILSAFE
        pyautogui.FAILSAFE = False
        try:
            pyautogui.moveTo(int(x), int(y), _pause=False)
        finally:
            pyautogui.FAILSAFE = previous

    try:
        # --- primitive costs ---------------------------------------------
        # position() is called once per gesture; moveTo() once per ~10ms
        # step, so it is the one that has to stay cheap.
        out["position_call"] = summarise(
            _time(lambda: pyautogui.position(), repeats * 4))

        x0, y0 = cx, cy
        move_ignoring_failsafe(x0, y0)
        targets = [clamp(x0 + 40, y0), clamp(x0, y0 + 40),
                   clamp(x0 - 40, y0), clamp(x0, y0 - 40)]
        samples = []
        for i in range(repeats * 4):
            tx, ty = targets[i % len(targets)]
            t0 = time.perf_counter()
            pyautogui.moveTo(tx, ty, _pause=False)
            samples.append(time.perf_counter() - t0)
        out["moveTo_call"] = summarise(samples)

        step_ms = 10.0
        s = out["moveTo_call"]
        out["step_budget_ms"] = step_ms
        out["moveTo_fits_step_budget"] = s["p90"] * 1000 < step_ms

        # --- the real gesture --------------------------------------------
        step_px = BASE_STEP_PX * resolution_scale
        gestures = []
        for n in DISTANCE_STEPS:
            for i in range(repeats):
                dist = min(step_px * n, max_reach)
                # Duration comes from the true distance, so the planned
                # figure stays the one production would use even where the
                # screen is too small to travel that far.
                duration = movement_duration(step_px * n, 1.0,
                                             resolution_scale)
                angle = (i / repeats) * 2 * math.pi
                tx, ty = clamp(x0 + dist * math.cos(angle),
                               y0 + dist * math.sin(angle))
                t0 = time.perf_counter()
                CustomCursor.quick_move_to([tx, ty], duration=duration,
                                           resolution_scale=resolution_scale)
                realised = time.perf_counter() - t0
                gestures.append({"squares": n, "planned": duration,
                                 "realised": realised,
                                 "overrun": realised - duration})
        out["gesture"] = {
            "planned": summarise([g["planned"] for g in gestures]),
            "realised": summarise([g["realised"] for g in gestures]),
            "overrun": summarise([g["overrun"] for g in gestures]),
            "overrun_ratio": summarise(
                [g["realised"] / g["planned"] for g in gestures
                 if g["planned"] > 0]),
        }
        out["gestures"] = gestures
        if progress:
            print(f"  measured {len(gestures)} live gestures",
                  flush=True)
    except Exception as exc:  # noqa: BLE001 - a device that cannot be
        # measured is a result, not a crash; record why and carry on.
        out["unavailable"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            move_ignoring_failsafe(origin[0], origin[1])
        except Exception:  # noqa: BLE001, S110 - best-effort cursor restore;
            pass       # never mask the real error from the block above
    return out


def _time(fn, repeats):
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return samples


def run(include_live=False, progress=True):
    from common.constants import MOUSE_QUICKNESS, RESOLUTION_SCALE

    result = {"planned": planned(MOUSE_QUICKNESS, RESOLUTION_SCALE)}
    if include_live:
        result["live"] = live(RESOLUTION_SCALE, progress=progress)
    else:
        result["live"] = {"skipped": "pass --mouse-live to move the real cursor"}
    return result


def report(result):
    p = result.get("planned", {})
    lines = ["  planned durations (formula; no device variance by design)",
             "    distance          mean_ms    p50_ms    p90_ms"]
    for name, s in (p.get("movement_duration") or {}).items():
        if s.get("n"):
            lines.append("    {:<16s} {:9.1f} {:9.1f} {:9.1f}".format(
                name, s["mean"] * 1000, s["p50"] * 1000, s["p90"] * 1000))
    for name in ("click_settle_sleep", "drag_settle_sleep"):
        s = p.get(name) or {}
        if s.get("n"):
            lines.append("    {:<16s} {:9.1f} {:9.1f} {:9.1f}".format(
                name.replace("_sleep", ""), s["mean"] * 1000,
                s["p50"] * 1000, s["p90"] * 1000))

    live_res = result.get("live", {})
    lines.append("  realised (live cursor)")
    if "skipped" in live_res:
        lines.append("    skipped: {}".format(live_res["skipped"]))
        return "\n".join(lines)
    if "unavailable" in live_res:
        lines.append("    unavailable: {}".format(live_res["unavailable"]))
        return "\n".join(lines)

    for name in ("position_call", "moveTo_call"):
        s = live_res.get(name) or {}
        if s.get("n"):
            lines.append("    {:<16s} {:9.2f} {:9.2f} {:9.2f}".format(
                name, s["mean"] * 1000, s["p50"] * 1000, s["p90"] * 1000))
    if "moveTo_fits_step_budget" in live_res:
        ok = live_res["moveTo_fits_step_budget"]
        lines.append("    moveTo p90 {} the {}ms step budget quick_move_to "
                     "assumes".format("fits inside" if ok else "EXCEEDS",
                                      live_res["step_budget_ms"]))
    g = live_res.get("gesture") or {}
    if g.get("realised", {}).get("n"):
        lines.append("    gesture planned p50 {:.0f}ms -> realised p50 "
                     "{:.0f}ms (x{:.2f})".format(
                         g["planned"]["p50"] * 1000,
                         g["realised"]["p50"] * 1000,
                         g["overrun_ratio"]["p50"]))
        lines.append("    worst overrun +{:.0f}ms".format(
            g["overrun"]["max"] * 1000))
    return "\n".join(lines)
