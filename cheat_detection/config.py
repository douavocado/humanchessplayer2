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
    blunder_wc_loss: float = 0.15       # win-prob drop that marks a blunder
    time_pressure_secs: float = 10.0    # clock below this = "time pressure" for the
                                        # degradation features (acpl/blunders in scramble)

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
