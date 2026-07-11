"""Samples the latency components a live move would incur on the clock.

Server-side, a player's clock runs from the moment the opponent's move
registers until their own move registers. For the bot that window contains:

  detection   seeing the opponent's move: screen capture + FEN extraction +
              turn detection + clock OCR (+ an occasional 0.15s confirmation
              re-capture, skipped under time pressure).
  think       the client sleeps up to (time_take - MOVE_DELAY); real engine
              compute overruns that budget on heavy positions, so realised
              think = max(time_take - MOVE_DELAY, compute).
  execution   the mouse gesture up to the point the move registers
              (mouseUp / second click): two movement legs + the intra-gesture
              settle pause. The post-move settle and verification polling
              happen on the opponent's clock and are not charged.

Detection and compute distributions come from Phase-0 calibration JSONs
(simulation/calibration/*.json) when present, with documented fallbacks so
the simulator runs before calibration has ever been done. All mouse-gesture
formulas are the exact code the live client uses (common/move_timing.py).
"""

from __future__ import annotations

import json
import math
import os
import random
from typing import Optional

import chess

from common.move_timing import (
    MOUSE_SLIP_PROB,
    click_settle_sleep,
    drag_probability,
    drag_settle_sleep,
    movement_duration,
)

CALIB_DIR = os.path.join(os.path.dirname(__file__), "calibration")

# Pixel size of one board square at 1080p; scaled by resolution below.
BASE_STEP_PX = 105.0

# Confirmation re-captures fire only on unlinkable/ambiguous frames; rare in
# a clean game. Tunable in Phase 3 against live [PERF] scan logs.
CONFIRM_RECAPTURE_PROB = 0.05
CONFIRM_RECAPTURE_SLEEP = 0.15

# From clients/mp_original.py: below this clock the confirmation re-capture
# is skipped entirely.
RESYNC_CONFIRM_MIN_TIME = 15.0

# Between scans the client idles with hover()/wander() cursor gestures
# (mp_original await_move): if the opponent's move lands mid-gesture, the
# next scan waits for it to finish. Gestures run only above 15s own time.
# Fitted to live [PERF] "Detection/overhead" data (2026-07-11 session,
# n=87: mean 166ms, p50 158ms, p90 344ms).
IDLE_GESTURE_PROB = 0.9          # chance a gesture is in flight when the move lands
IDLE_GESTURE_MAX = 0.25          # hover duration ~ U(0, 0.25)s

# Rare detection stalls: mis-linked scans, move-not-registered polling,
# animation confusion. Produces the realised-time tail seen in live games.
DETECTION_STALL_PROB = 0.03
DETECTION_STALL_RANGE = (0.3, 0.8)

# Server round-trip residual charged by Lichess on a normal move. Lichess
# lag compensation refunds most of the round trip, so this is small; kept
# distinct from detection (which live [PERF] data measures directly).
NETWORK_LAG_MEDIAN = 0.04
NETWORK_LAG_SIGMA = 0.6


def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


class LatencyModel:
    def __init__(self, rng: random.Random,
                 mouse_quickness: float,
                 resolution_scale: float,
                 calib_dir: str = CALIB_DIR):
        self.rng = rng
        self.mouse_quickness = mouse_quickness
        self.resolution_scale = resolution_scale
        self.step_px = BASE_STEP_PX * resolution_scale

        det = _load_json(os.path.join(calib_dir, "detection_latency.json")) or {}
        vision = det.get("vision_ms", {})
        self._vision_avg_ms = sum(
            v.get("avg_ms", 0.0) for v in vision.values()
        ) or 13.0  # fallback: measured ~12+0.6+0.4ms on the dev machine
        self._capture_ms = det.get("capture_ms_estimate", 8.0)

        comp = _load_json(os.path.join(calib_dir, "compute_time.json")) or {}
        self._compute_by_phase: dict[str, list[float]] = {}
        for rec in comp.get("records", []):
            self._compute_by_phase.setdefault(rec["phase"], []).append(
                rec["total_secs"])

    # ------------------------------------------------------------ detection
    def detection_latency(self, own_time: float) -> float:
        """Seconds between the opponent's move registering and the client
        adopting a scan of it."""
        # One full scan (capture + vision), lognormal-ish jitter around the
        # measured mean, plus up to one lightweight-scan period of phase lag
        # before the change is even looked at.
        scan_s = (self._capture_ms + self._vision_avg_ms) / 1000.0
        scan_s *= 0.8 + 0.5 * self.rng.random()
        phase_lag = self.rng.random() * 0.02
        total = scan_s + phase_lag
        if own_time >= RESYNC_CONFIRM_MIN_TIME:
            # Idle hover/wander gestures run between scans; a move landing
            # mid-gesture waits for it to finish before the next scan.
            if self.rng.random() < IDLE_GESTURE_PROB:
                total += self.rng.random() * IDLE_GESTURE_MAX
            if self.rng.random() < CONFIRM_RECAPTURE_PROB:
                total += CONFIRM_RECAPTURE_SLEEP + scan_s
        if self.rng.random() < DETECTION_STALL_PROB:
            lo, hi = DETECTION_STALL_RANGE
            total += lo + (hi - lo) * self.rng.random()
        return total

    # -------------------------------------------------------------- network
    def network_lag(self) -> float:
        """Server round-trip residual Lichess charges on a normal move."""
        return NETWORK_LAG_MEDIAN * math.exp(
            NETWORK_LAG_SIGMA * self.rng.gauss(0.0, 1.0))

    # -------------------------------------------------------------- compute
    def compute_time(self, phase: str) -> float:
        """Wall time the real client spends inside update_info + make_move,
        resampled from Phase-0 measurements on this machine."""
        vals = self._compute_by_phase.get(phase)
        if not vals:
            vals = [v for vs in self._compute_by_phase.values() for v in vs]
        if vals:
            return self.rng.choice(vals)
        # No calibration yet: rough constant so the simulator still runs.
        return 0.4 + 0.3 * self.rng.random()

    # ------------------------------------------------------------ execution
    def _square_px(self, square: int) -> tuple[float, float]:
        return (chess.square_file(square) * self.step_px,
                chess.square_rank(square) * self.step_px)

    def _leg_distance(self, a: Optional[int], b: int) -> float:
        """Pixel distance between two squares; when the cursor's square is
        unknown (game start / after idle wander) sample a plausible reach."""
        if a is None:
            return self.step_px * (2 + 3 * self.rng.random())
        ax, ay = self._square_px(a)
        bx, by = self._square_px(b)
        wobble = self.step_px * 0.3 * self.rng.random()
        return math.hypot(bx - ax, by - ay) + wobble

    def gesture_time(self, move: chess.Move, own_time: float,
                     cursor_square: Optional[int]) -> tuple[float, bool]:
        """Charged duration of the move gesture up to move registration,
        and whether it ended as a drag (affects settle constants only).

        Includes the live client's mouse-slip retry: a slipped gesture is
        re-attempted once, roughly doubling execution time for that move.
        """
        total = 0.0
        attempts = 1
        if self.rng.random() < MOUSE_SLIP_PROB:
            attempts = 2
        for _ in range(attempts):
            dragged = self.rng.random() < drag_probability(max(own_time, 1))
            leg_from = movement_duration(
                self._leg_distance(cursor_square, move.from_square),
                self.mouse_quickness, self.resolution_scale, rng=self.rng)
            leg_to = movement_duration(
                self._leg_distance(move.from_square, move.to_square),
                self.mouse_quickness, self.resolution_scale, rng=self.rng)
            settle = (drag_settle_sleep(self.rng) if dragged
                      else click_settle_sleep(self.rng))
            total += leg_from + settle + leg_to
        return total, dragged
