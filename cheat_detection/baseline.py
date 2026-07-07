"""Build and load a human reference baseline from a Lichess database dump.

A baseline is the distribution (mean + std across human "units") of each
feature, restricted to a rating band and time control matching the bot.
Downloaded Lichess PGN dumps (https://database.lichess.org) are the intended
input; any multi-game PGN with Elo + clock tags works.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Optional

from .config import AnalysisConfig
from .engine_analysis import EngineAnalyzer
from .features import FEATURE_KEYS
from .pipeline import Unit, iter_units


@dataclass
class Baseline:
    rating_band: tuple[int, int]
    n_units: int
    stats: dict[str, dict[str, Optional[float]]]  # feature -> {"mean","std","n"}

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({
                "rating_band": list(self.rating_band),
                "n_units": self.n_units,
                "stats": self.stats,
            }, fh, indent=2)

    @staticmethod
    def from_json(path: str) -> "Baseline":
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return Baseline(
            rating_band=tuple(d["rating_band"]),
            n_units=d["n_units"],
            stats=d["stats"],
        )


def _summarize(units: list[Unit]) -> dict[str, dict[str, Optional[float]]]:
    import math
    stats: dict[str, dict[str, Optional[float]]] = {}
    for key in FEATURE_KEYS:
        vals = [u.features[key] for u in units if u.features.get(key) is not None]
        if not vals:
            stats[key] = {"mean": None, "std": None, "n": 0}
            continue
        mean = sum(vals) / len(vals)
        std = (
            math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
            if len(vals) > 1 else 0.0
        )
        stats[key] = {"mean": mean, "std": std, "n": len(vals)}
    return stats


def build_baseline(
    pgn_path: str,
    rating_band: tuple[int, int],
    cfg: AnalysisConfig,
    max_games: Optional[int] = None,
    min_moves: int = 10,
    on_progress: Optional[Callable[[int], None]] = None,
) -> Baseline:
    """Analyse a human PGN dump and summarise it into a Baseline."""
    units: list[Unit] = []
    with EngineAnalyzer(cfg) as analyzer:
        for unit in iter_units(
            pgn_path, analyzer, cfg,
            rating_band=rating_band,
            max_games=max_games,
            min_moves=min_moves,
            on_progress=on_progress,
        ):
            units.append(unit)
    return Baseline(
        rating_band=rating_band,
        n_units=len(units),
        stats=_summarize(units),
    )
