"""Human-likeness diagnostic for the chess bot.

This package applies the *techniques* used by Lichess's open-source cheat
detectors (Irwin's per-move engine analysis, Kaladin's aggregate statistics)
to a self-diagnostic question: how far does this bot's play sit from a real
human distribution?

It is deliberately offline and comparison-based. It takes PGN files as input
(the bot's games plus a human reference corpus, e.g. a Lichess database dump)
and outputs a report of *feature-by-feature divergence from human play* --
where the bot looks least human and why. It does not target or score against
any particular detector.
"""

from .config import AnalysisConfig

__all__ = ["AnalysisConfig"]
