"""Game-level parallel analysis: N worker processes, each with its own Stockfish.

Games are striped across workers (worker i takes games where
``index % workers == i``), so every worker re-parses the PGN headers (cheap)
but the engine work (everything) is split. The analysed game set — and hence
the resulting Units — is identical to a sequential run.

Cache safety: the cache is SQLite in WAL mode, so workers read and insert
rows concurrently — writes serialise on the row level and nothing is ever
clobbered. The parent opens the DB once before spawning so the one-time
legacy-JSON migration doesn't race across workers.

The pool uses the *spawn* start method: the GUI calls this from a background
thread of a Tk process, and forking a multi-threaded process is unsafe.
"""

from __future__ import annotations

import multiprocessing as mp
import queue as _queue
from typing import Callable, Optional

from .config import AnalysisConfig
from .engine_analysis import EngineAnalyzer, open_cache_db
from .pgn_loader import iter_games
from .pipeline import Unit, iter_units, units_for_game


def _worker(
    worker_id: int,
    nworkers: int,
    pgn_path: str,
    cfg: AnalysisConfig,
    player_filter: Optional[set[str]],
    rating_band: Optional[tuple[int, int]],
    max_games: Optional[int],
    min_moves: int,
    opponent_band: Optional[tuple[int, int]],
    diff_range: Optional[tuple[Optional[int], Optional[int]]],
    progress_q,
) -> list[Unit]:
    units: list[Unit] = []
    with EngineAnalyzer(cfg) as analyzer:
        for gi, game in enumerate(iter_games(pgn_path, max_games=max_games)):
            if gi % nworkers != worker_id:
                continue
            units.extend(units_for_game(
                game, analyzer, cfg, player_filter, rating_band, min_moves,
                opponent_band, diff_range,
            ))
            analyzer.flush()  # game-boundary commit: a crash loses <=1 game
            progress_q.put(1)
    return units


def collect_units(
    pgn_path: str,
    cfg: AnalysisConfig,
    player_filter: Optional[set[str]] = None,
    rating_band: Optional[tuple[int, int]] = None,
    max_games: Optional[int] = None,
    min_moves: int = 10,
    on_progress: Optional[Callable[[int], None]] = None,
    opponent_band: Optional[tuple[int, int]] = None,
    diff_range: Optional[tuple[Optional[int], Optional[int]]] = None,
) -> list[Unit]:
    """Analyse a PGN into Units, parallelised across ``cfg.workers`` processes.

    With workers <= 1 this is exactly the old sequential path. ``on_progress``
    receives the running count of games completed (across all workers).
    """
    workers = max(1, cfg.workers)
    if workers == 1:
        with EngineAnalyzer(cfg) as analyzer:
            return list(iter_units(
                pgn_path, analyzer, cfg,
                player_filter=player_filter,
                rating_band=rating_band,
                max_games=max_games,
                min_moves=min_moves,
                on_progress=on_progress,
                opponent_band=opponent_band,
                diff_range=diff_range,
            ))

    open_cache_db(cfg).close()  # run the one-time JSON migration before forking out

    ctx = mp.get_context("spawn")
    units: list[Unit] = []
    with ctx.Manager() as mgr:
        progress_q = mgr.Queue()
        with ctx.Pool(workers) as pool:
            jobs = [
                pool.apply_async(_worker, (
                    wid, workers, pgn_path, cfg, player_filter,
                    rating_band, max_games, min_moves,
                    opponent_band, diff_range, progress_q,
                ))
                for wid in range(workers)
            ]
            pool.close()
            done = 0
            pending = list(jobs)
            while pending:
                try:
                    progress_q.get(timeout=0.5)
                    done += 1
                    if on_progress is not None:
                        on_progress(done)
                except _queue.Empty:
                    pass
                pending = [j for j in pending if not j.ready()]
            while True:  # drain ticks that raced the last ready() check
                try:
                    progress_q.get_nowait()
                    done += 1
                    if on_progress is not None:
                        on_progress(done)
                except _queue.Empty:
                    break
            for j in jobs:
                units.extend(j.get())  # re-raises worker exceptions
    return units
