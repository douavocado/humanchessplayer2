"""Configuration for the human-likeness analyzer.

All tunables live here. Defaults: Stockfish depth 10 (dropped from 18 for a
~24x speedup), multi-PV 5, using the repo's Stockfish 17 binary.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Reuse the engine binary the bot already ships with, so the diagnostic sees
# the same evaluations. Falls back gracefully if constants can't be imported.
try:
    from common.constants import PATH_TO_STOCKFISH as _DEFAULT_SF
except Exception:  # pragma: no cover - only when run outside the repo
    _DEFAULT_SF = os.environ.get("STOCKFISH_PATH", "stockfish")


# Fractions of the initial clock that define the two clock-dependent feature
# thresholds. Both were chosen to reproduce the previously-hardcoded 60+0
# constants EXACTLY -- 60/30 = 2.0s and 60/6 = 10.0s -- so parameterising them
# leaves every existing bullet baseline, report and band table untouched.
#
# The /30 was already the documented intent. The /6 was derived backwards from
# the shipped 10.0 and lands on it exactly, which is reasonable evidence that a
# fraction is the right reading of "time pressure" rather than a coincidence.
# Open question flagged in the spec: the scramble may be partly *absolute* --
# 10s is roughly where humans stop calculating regardless of the starting clock
# -- in which case the right form is max(10.0, initial_time/6).
LONG_THINK_FRACTION = 1 / 30
TIME_PRESSURE_FRACTION = 1 / 6


@dataclass
class AnalysisConfig:
    # --- Engine analysis ---
    stockfish_path: str = _DEFAULT_SF
    depth: int = 10                 # fixed-depth analysis per position
    multipv: int = 5                # number of candidate moves ranked per position
    threads: int = 2
    hash_mb: int = 256
    # Parallel game-level analysis: worker processes, each with its own
    # Stockfish. 1 = sequential (identical to the original behaviour). Total
    # engine threads = workers * threads, so on a 16-core box workers=6-8
    # with threads=2 is about right.
    workers: int = 1
    mate_cp: int = 10000            # centipawn value assigned to a forced mate

    # --- Feature thresholds (ported from Irwin/Kaladin conventions) ---
    ambiguity_wc_window: float = 0.05   # moves within this win-prob of best are "equally good"
    instant_move_secs: float = 1.0      # emt below this counts as an "instant" move
    # Initial clock in seconds for the corpus under analysis -- 60.0 is one
    # minute of bullet, the control everything in this repo was calibrated on.
    # Set it from --tc; the two thresholds below derive from it.
    initial_time: float = 60.0
    # Set these only to override the derivation; None = derive from
    # initial_time. They are properties rather than fields so that setting
    # initial_time after construction re-derives them (analyze.py's
    # _config_from_args mutates an already-built config).
    long_think_secs_override: float | None = None
    blunder_wc_loss: float = 0.15       # win-prob drop that marks a blunder
    time_pressure_secs_override: float | None = None
    # False downgrades the TimeControl header check to a skip; see
    # pgn_loader.check_time_control. Set from --allow-tc-mismatch.
    strict_tc: bool = True

    # --- Phase boundaries ---
    opening_plies: int = 16             # first N plies = opening
    endgame_npm: int = 13               # non-pawn material (points) at/below which = endgame

    # --- Caching ---
    cache_dir: str = field(
        default_factory=lambda: os.path.join(
            os.path.dirname(__file__), "cache"
        )
    )

    # --- Reporting ---
    # Which statistic decides whether a feature is flagged:
    #   "effect_size" — |z| = |bot_mean - human_mean| / human_std >= flag_zscore.
    #     Sample-size independent; flags differences large relative to normal
    #     human game-to-game variation.
    #   "welch" — Welch two-sample t-test (bot games vs baseline games);
    #     flags when the two-sided p-value < flag_pvalue. Grows more sensitive
    #     with more games, so tiny systematic biases eventually flag.
    # Both statistics are always computed and shown; this only picks the flagger.
    test_mode: str = "effect_size"
    # A feature whose bot value is this many baseline-std's away is flagged.
    flag_zscore: float = 2.0
    # Significance level for test_mode="welch".
    flag_pvalue: float = 0.05

    @property
    def long_think_secs(self) -> float:
        """emt above this counts as a "long think": the slow tail of the
        move-time distribution, the counterpart to instant_move_secs.

        Tracked because tuning against the fast tail alone hid a larger
        divergence -- the bot measured 0.067 against a human 0.115, and
        near-zero outside the midgame (opening 0.002 vs 0.031, endgame 0.007
        vs 0.044).
        """
        if self.long_think_secs_override is not None:
            return self.long_think_secs_override
        return self.initial_time * LONG_THINK_FRACTION

    @property
    def time_pressure_secs(self) -> float:
        """Clock below this = "time pressure" for the degradation features
        (acpl/blunders in the scramble)."""
        if self.time_pressure_secs_override is not None:
            return self.time_pressure_secs_override
        return self.initial_time * TIME_PRESSURE_FRACTION

    def cache_path(self) -> str:
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(
            self.cache_dir, f"analysis_d{self.depth}_mpv{self.multipv}.sqlite"
        )

    def legacy_cache_path(self) -> str:
        """Pre-SQLite JSON cache; migrated into the .sqlite once, then unused."""
        return os.path.join(
            self.cache_dir, f"analysis_d{self.depth}_mpv{self.multipv}.json"
        )
