"""SimClock: per-side remaining-time bookkeeping for simulated games.

Time never passes in real life here — moves *charge* the mover's clock with
a computed duration. Internally clocks are floats; the PGN layer quantises
to whole seconds to match what Lichess exports (the bot's real 60+0 PGNs
carry integer-second [%clk] tags only).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess


@dataclass
class SimClock:
    initial: float                     # base time in seconds
    increment: float = 0.0
    remaining: dict = field(default_factory=dict)

    def __post_init__(self):
        self.remaining = {chess.WHITE: float(self.initial),
                          chess.BLACK: float(self.initial)}

    def time_left(self, side: bool) -> float:
        return self.remaining[side]

    def charge(self, side: bool, seconds: float) -> bool:
        """Deduct ``seconds`` from ``side``'s clock, then apply increment
        (Lichess adds increment after the move completes).

        Returns True if the side flagged (ran out of time) on this move.
        """
        self.remaining[side] -= max(seconds, 0.0)
        if self.remaining[side] <= 0:
            self.remaining[side] = 0.0
            return True
        self.remaining[side] += self.increment
        return False
