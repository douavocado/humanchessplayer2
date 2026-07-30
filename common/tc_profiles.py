"""Per-time-control phase envelopes for move-time pacing.

`decision_logic._get_time_taken` scales a base think time by the phase of the
game. Those multipliers were fitted at 60+0 -- 60 *seconds*, one minute of
bullet -- and one of them is wrong at longer controls: the opening form
`(base ** 0.2)/2` compresses toward a constant while the midgame scales
linearly, so the opening/midgame ratio falls as the clock grows. Humans hold
that ratio at 0.374 (60+0) -> 0.377 (180+0) across a 3x clock change, measured
over 3.14M moves.

**Resolution is exact-match-else-LEGACY, deliberately.** A "nearest row" rule
would change behaviour at controls nobody has measured: with rows at 60 and
180, a 90-second game would take the 60 row's midgame *1.4 where today's code
gives *1.7 (its branch is `> 60`). Only a clock with a fitted row behaves
differently from before.

Design and measurements:
`docs/superpowers/specs/2026-07-30-longer-tc-pacing-calibration-design.md`.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class TCProfile:
    """Phase envelopes for one time control.

    Each field maps a base think time to a phase-adjusted one. They are
    callables rather than plain multipliers because the legacy opening is a
    power-law compression, not a multiplier, and the legacy midgame branches
    on the clock -- storing them as functions lets LEGACY reproduce the
    pre-table arithmetic exactly for every control.
    """
    opening: Callable[[float], float]
    midgame: Callable[[float, float], float]   # (base, initial_time)
    endgame: Callable[[float], float]
    fitted_at: float | None  # clock this row was fitted at; None = unfitted
    label: str


LEGACY = TCProfile(
    opening=lambda base: (base ** 0.2) / 2,
    midgame=lambda base, t: base * (1.7 if t > 60 else 1.4),
    endgame=lambda base: base * 0.7,
    fitted_at=None,
    label="legacy (pre-2026-07-30 inline envelopes)",
)

# Fitted rows, keyed by exact initial clock in seconds. Task 5 adds 180.0.
TC_PROFILES: dict[float, TCProfile] = {}


def resolve_tc(initial_time: float) -> TCProfile:
    """The profile for this clock: its fitted row, or LEGACY if unfitted."""
    return TC_PROFILES.get(float(initial_time), LEGACY)


def apply_envelope(profile: TCProfile, base_time: float, phase: str,
                   initial_time: float) -> float:
    """Base think time scaled by `phase`'s envelope under `profile`.

    `phase` uses the engine's own vocabulary from
    `common.board_information.phase_of_game`: "opening", "midgame", or
    anything else (treated as endgame, matching the pre-table `else` branch).
    NOTE this is NOT cheat_detection's phase rule -- see the spec.
    """
    if phase == "opening":
        return profile.opening(base_time)
    if phase == "midgame":
        return profile.midgame(base_time, initial_time)
    return profile.endgame(base_time)
