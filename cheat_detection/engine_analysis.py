"""Stockfish multi-PV analysis with a persistent on-disk cache.

Re-analysing positions is by far the slowest part of the pipeline, so every
analysed position is cached (keyed by FEN) in SQLite. Runs are resumable:
re-running over the same games reuses cached evals instantly.

SQLite (WAL mode) rather than one big JSON blob: lookups are indexed instead
of loading the whole cache into every process, and concurrent writers (the
parallel workers, or two separate analyses) serialise on row inserts instead
of clobbering each other's full-file rewrites. A legacy JSON cache with the
same depth/multipv is imported into the .sqlite the first time it's opened.
"""

from __future__ import annotations

import json
import os
import sqlite3
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


def open_cache_db(config: AnalysisConfig) -> sqlite3.Connection:
    """Open (creating/migrating if needed) the shared eval cache database.

    WAL mode lets many processes read while one writes; writers queue on the
    30s busy timeout instead of failing. Safe to call concurrently: the schema
    create and the legacy-JSON import are both idempotent (INSERT OR IGNORE).
    """
    db = sqlite3.connect(config.cache_path(), timeout=30.0, check_same_thread=False)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("CREATE TABLE IF NOT EXISTS evals (fen TEXT PRIMARY KEY, c TEXT NOT NULL)")
    db.commit()
    legacy = config.legacy_cache_path()
    if os.path.exists(legacy):
        (n,) = db.execute("SELECT COUNT(*) FROM evals").fetchone()
        if n == 0:
            try:
                with open(legacy, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, OSError):
                data = {}
            db.executemany(
                "INSERT OR IGNORE INTO evals VALUES (?, ?)",
                ((fen, json.dumps(entry["c"])) for fen, entry in data.items()),
            )
            db.commit()
    return db


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

    # In-session memo cap: repeated hits are same-game and opening positions,
    # so an occasional wholesale clear costs a few re-SELECTs, not re-analysis,
    # and keeps long (10k+ game) runs from growing without bound.
    _MEMO_MAX = 100_000

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self._engine: Optional[chess.engine.SimpleEngine] = None
        self._db = open_cache_db(config)
        self._memo: dict[str, PositionEval] = {}
        self._pending: list[tuple[str, str]] = []

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
        self._db.close()
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    # --- cache ---
    def flush(self) -> None:
        if self._pending:
            self._db.executemany(
                "INSERT OR IGNORE INTO evals VALUES (?, ?)", self._pending
            )
            self._db.commit()
            self._pending = []

    def _store(self, fen: str, pe: PositionEval) -> None:
        if len(self._memo) >= self._MEMO_MAX:
            self._memo.clear()
        self._memo[fen] = pe
        self._pending.append(
            (fen, json.dumps([[c.move_uci, c.cp] for c in pe.candidates]))
        )
        if len(self._pending) >= 200:
            self.flush()

    def _load(self, fen: str) -> Optional[PositionEval]:
        pe = self._memo.get(fen)
        if pe is not None:
            return pe
        row = self._db.execute("SELECT c FROM evals WHERE fen = ?", (fen,)).fetchone()
        if row is None:
            return None
        pe = PositionEval([Candidate(m, cp) for m, cp in json.loads(row[0])])
        if len(self._memo) >= self._MEMO_MAX:
            self._memo.clear()
        self._memo[fen] = pe
        return pe

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
