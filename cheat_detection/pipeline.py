"""Glue: PGN -> engine analysis -> per-move features -> per-unit vectors.

A "unit" is one player's moves within one game. Both the human baseline and
the bot report are built from units, so they are compared on equal footing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Optional

from .config import AnalysisConfig
from .engine_analysis import EngineAnalyzer
from .features import MoveFeatures, aggregate_features, extract_move_features
from .pgn_loader import GameRecord, iter_games


@dataclass
class Unit:
    player: str
    elo: Optional[int]
    n_moves: int
    features: dict[str, Optional[float]]


def _sides_of_interest(
    game: GameRecord,
    player_filter: Optional[set[str]],
    rating_band: Optional[tuple[int, int]],
) -> list[tuple[bool, str, Optional[int]]]:
    """Which (color, name, elo) sides in this game we should profile."""
    import chess
    out = []
    for color, name, elo in (
        (chess.WHITE, game.white, game.white_elo),
        (chess.BLACK, game.black, game.black_elo),
    ):
        if player_filter is not None and name.lower() not in player_filter:
            continue
        if rating_band is not None:
            if elo is None or not (rating_band[0] <= elo <= rating_band[1]):
                continue
        out.append((color, name, elo))
    return out


def iter_units(
    pgn_path: str,
    analyzer: EngineAnalyzer,
    cfg: AnalysisConfig,
    player_filter: Optional[set[str]] = None,
    rating_band: Optional[tuple[int, int]] = None,
    max_games: Optional[int] = None,
    min_moves: int = 10,
    on_progress: Optional[Callable[[int], None]] = None,
) -> Iterator[Unit]:
    """Yield a Unit for each qualifying (player, game).

    ``on_progress(games_done)`` is called after each game (and the cache is
    flushed then, for crash-resumability). Pass None to stay silent.
    """
    for gi, game in enumerate(iter_games(pgn_path, max_games=max_games)):
        sides = _sides_of_interest(game, player_filter, rating_band)
        if not sides:
            continue
        for color, name, elo in sides:
            mfeats: list[MoveFeatures] = []
            for rec in game.moves:
                if rec.mover != color:
                    continue
                pe = analyzer.analyse(rec.fen_before)
                if not pe.candidates:
                    continue
                played_cp = analyzer.eval_after_move(rec.fen_before, rec.move_uci)
                mf = extract_move_features(rec, pe, played_cp, cfg)
                if mf is not None:
                    mfeats.append(mf)
            if len(mfeats) < min_moves:
                continue
            yield Unit(
                player=name,
                elo=elo,
                n_moves=len(mfeats),
                features=aggregate_features(mfeats, cfg),
            )
        if on_progress is not None:
            analyzer.flush()
            on_progress(gi + 1)
