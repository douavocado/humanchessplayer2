"""Stockfish multi-PV analysis with a persistent on-disk cache.

Re-analysing positions is by far the slowest part of the pipeline, so every
analysed position is cached (keyed by FEN) in a JSON file. Runs are resumable:
re-running over the same games reuses cached evals instantly.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import chess
import chess.engine

from .config import AnalysisConfig


@dataclass
class Candidate:
    move_uci: str
    cp: int          # centipawns from the mover's point of view (mate mapped to +/- mate_cp)


@dataclass
class PositionEval:
    """Top-k candidate moves for a position, mover's-perspective centipawns."""
    candidates: list[Candidate]

    def best_cp(self) -> int:
        return self.candidates[0].cp

    def cp_for(self, move_uci: str) -> Optional[int]:
        for c in self.candidates:
            if c.move_uci == move_uci:
                return c.cp
        return None

    def rank_of(self, move_uci: str) -> Optional[int]:
        for i, c in enumerate(self.candidates):
            if c.move_uci == move_uci:
                return i + 1  # 1-based
        return None


def _score_to_cp(score: chess.engine.PovScore, turn: bool, mate_cp: int) -> int:
    pov = score.pov(turn)
    if pov.is_mate():
        moves_to_mate = pov.mate()
        # Positive mate == good for mover; nearer mates get slightly larger magnitude.
        sign = 1 if moves_to_mate > 0 else -1
        return sign * (mate_cp - min(abs(moves_to_mate), 50))
    return int(pov.score(mate_score=mate_cp))


class EngineAnalyzer:
    """Analyse positions with Stockfish, caching results to disk."""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self._engine: Optional[chess.engine.SimpleEngine] = None
        self._cache: dict[str, dict] = {}
        self._cache_path = config.cache_path()
        self._dirty = 0
        self._load_cache()

    # --- context management ---
    def __enter__(self) -> "EngineAnalyzer":
        self._engine = chess.engine.SimpleEngine.popen_uci(self.config.stockfish_path)
        self._engine.configure({
            "Threads": self.config.threads,
            "Hash": self.config.hash_mb,
        })
        return self

    def __exit__(self, *exc) -> None:
        self.flush()
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    # --- cache ---
    def _load_cache(self) -> None:
        if os.path.exists(self._cache_path):
            try:
                with open(self._cache_path, "r", encoding="utf-8") as fh:
                    self._cache = json.load(fh)
            except (json.JSONDecodeError, OSError):
                self._cache = {}

    def flush(self) -> None:
        if self._dirty:
            tmp = self._cache_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._cache, fh)
            os.replace(tmp, self._cache_path)
            self._dirty = 0

    @staticmethod
    def _key(fen: str) -> str:
        return fen

    def _store(self, fen: str, pe: PositionEval) -> None:
        self._cache[self._key(fen)] = {
            "c": [[c.move_uci, c.cp] for c in pe.candidates]
        }
        self._dirty += 1
        if self._dirty >= 200:
            self.flush()

    def _load(self, fen: str) -> Optional[PositionEval]:
        raw = self._cache.get(self._key(fen))
        if raw is None:
            return None
        return PositionEval([Candidate(m, cp) for m, cp in raw["c"]])

    # --- analysis ---
    def analyse(self, fen: str) -> PositionEval:
        """Return top-k candidate moves for the position (cached)."""
        cached = self._load(fen)
        if cached is not None:
            return cached
        if self._engine is None:
            raise RuntimeError("EngineAnalyzer must be used as a context manager")

        board = chess.Board(fen)
        if board.is_game_over():
            pe = PositionEval([])
            self._store(fen, pe)
            return pe

        infos = self._engine.analyse(
            board,
            chess.engine.Limit(depth=self.config.depth),
            multipv=self.config.multipv,
        )
        candidates: list[Candidate] = []
        for info in infos:
            pv = info.get("pv")
            score = info.get("score")
            if not pv or score is None:
                continue
            candidates.append(Candidate(
                move_uci=pv[0].uci(),
                cp=_score_to_cp(score, board.turn, self.config.mate_cp),
            ))
        pe = PositionEval(candidates)
        self._store(fen, pe)
        return pe

    def eval_after_move(self, fen: str, move_uci: str) -> Optional[int]:
        """Mover's-perspective centipawns for a specific move.

        Uses the multi-PV ranking when the move is in the top-k; otherwise
        analyses the resulting position (multipv=1) and negates.
        """
        pe = self.analyse(fen)
        direct = pe.cp_for(move_uci)
        if direct is not None:
            return direct
        # Not in top-k: evaluate the position after the move and flip perspective.
        board = chess.Board(fen)
        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError:
            return None
        if move not in board.legal_moves:
            return None
        board.push(move)
        after = self.analyse(board.fen())
        if not after.candidates:
            return None
        return -after.best_cp()
