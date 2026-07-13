"""Timing formulas shared by the live client and the offline simulator.

Every duration in a live game that is *decided* (rather than measured) lives
here, so `clients/mp_original.py` and `simulation/` sample identical
distributions. If you tune how long a gesture or wait takes, change it here
and both stay in sync — the simulator's realism depends on it.

All functions take an optional ``rng`` (a ``random.Random``) so the simulator
can seed its draws; the live client omits it and uses the module-level RNG.
"""

from __future__ import annotations

import math
import random as _random
from typing import Optional

# The amount of time we take per move minus the engine calc time, covering
# other aspects (time scrape, position updates, moving pieces etc.). The
# client sleeps up to (time_take - MOVE_DELAY) before executing the move.
MOVE_DELAY = 0.25
# Time to allow a move to snap onto the board before taking a screenshot.
DRAG_MOVE_DELAY = 0.07
CLICK_MOVE_DELAY = 0.03

# Chance a gesture is deliberately fumbled to look human.
MOUSE_SLIP_PROB = 0.03

# _verify_move_registered: poll interval and give-up timeout.
VERIFY_POLL_INTERVAL = 0.04
VERIFY_TIMEOUT = 0.5


def _rng(rng: Optional[_random.Random]) -> _random.Random:
    return rng if rng is not None else _random


def movement_duration(distance: float, mouse_quickness: float,
                      resolution_scale: float,
                      rng: Optional[_random.Random] = None) -> float:
    """Duration for one mouse-movement leg, scaled by distance (Fitts-like)
    and MOUSE_QUICKNESS.

    Shared by drags and click-moves so both gestures move the cursor at the
    same apparent speed. Clamped to [0.04, 0.12] seconds.
    """
    r = _rng(rng)
    jitter = 0.8 + 0.4 * r.random()
    base = 0.023 + mouse_quickness / 3000.0 * jitter * math.sqrt(
        max(float(distance), 0.0) / resolution_scale)
    return min(max(base, 0.04), 0.12)


def drag_settle_sleep(rng: Optional[_random.Random] = None) -> float:
    """Pause between reaching the from-square and picking up the piece."""
    return 0.015 + 0.015 * _rng(rng).random()


def click_settle_sleep(rng: Optional[_random.Random] = None) -> float:
    """Pause after the first click so Lichess registers piece selection."""
    return 0.025 + 0.025 * _rng(rng).random()


def drag_probability(own_time: float) -> float:
    """Chance the move is a drag rather than two clicks; clicking is faster
    so it dominates under time pressure."""
    if own_time < 20:
        return own_time / 25
    return 0.8


def ponder_response_wait(initial_time: float, quickness: float,
                         rng: Optional[_random.Random] = None,
                         pace_sf: float = 1.0) -> float:
    """Wait before playing a pre-pondered (PONDER_DIC) response — the human
    'recognise the expected reply' pause, scaled by time control.

    ``pace_sf`` is the engine's combined per-game draw for this wait
    (Engine.ponder_pace_sf = game_pace_sf x game_ponder_snap_sf): ponder
    hits are ~30% of all moves and sit right at the 1-second "instant move"
    boundary, so coupling this wait to the game's character is what gives
    the instant-move rate its game-to-game spread (a fast game recognises
    expected replies near-instantly, a grinding game double-checks them).
    The wide jitter serves the same feature within a game.
    """
    base_time = 0.27 * quickness * initial_time ** 1.1 / (100 + initial_time ** 0.7)
    return base_time * pace_sf * (0.45 + 0.9 * _rng(rng).random())


def scramble_response_wait(rng: Optional[_random.Random] = None) -> float:
    """Wait before firing a pondered move in a <10s time scramble."""
    return 0.1 * (0.8 + 0.4 * _rng(rng).random())


def resign_pause(rng: Optional[_random.Random] = None) -> float:
    """Human hesitation between deciding to resign and clicking resign."""
    return 2 + 3 * _rng(rng).random()
