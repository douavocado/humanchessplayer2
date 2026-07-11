"""CLI: simulate bot self-play games and write PGN for cheat_detection.

Example (from repo root):

    venv/bin/python -m simulation.run --games 20 --tc 60+0 --seed 1 \
        --workers 5 --out simulation/games/selfplay_60plus0.pgn

The output parses directly with cheat_detection, e.g.:

    venv/bin/python -m cheat_detection.analyze report \
        --pgn simulation/games/selfplay_60plus0.pgn \
        --player SimBotWhite SimBotBlack \
        --baseline cheat_detection/baselines/bullet_1plus0_2300_2600.json

Parallelism: games are independent, so --workers N runs N processes, each
with its own Engine pair (each pair holds several Stockfish subprocesses and
the torch models — keep N around cores/3). Seeds are partitioned
deterministically (worker w plays seeds[w::N]), so the same
(games, seed, workers) triple reproduces the same games; changing the worker
count changes which engine-state history precedes each game, so per-game
results are only guaranteed stable for a fixed worker count.
"""

from __future__ import annotations

import argparse
import datetime
import json
import multiprocessing as mp
import os
import time
from collections import Counter

from cheat_detection.fetch_lichess import parse_time_control

DEFAULT_OUT_DIR = os.path.join(os.path.dirname(__file__), "games")


def _play_chunk(task: tuple) -> list:
    """Worker: create one Engine pair and play a list of seeds on it.

    Module-level (picklable) and self-contained: runs in a spawned process.
    Returns [(SimGame, wall_secs), ...] in the order played.
    """
    worker_id, seeds, cfg, difficulty = task
    import torch
    torch.set_num_threads(1)  # N workers x default 8 threads would thrash
    from engine import Engine
    from .game_runner import GameRunner

    tag = f"[w{worker_id}]"
    print(f"{tag} loading engines (playing_level={difficulty}) "
          f"for {len(seeds)} game(s)...", flush=True)
    white_engine = Engine(playing_level=difficulty)
    black_engine = Engine(playing_level=difficulty)
    results = []
    try:
        runner = GameRunner(white_engine, black_engine, cfg)
        for seed in seeds:
            t0 = time.perf_counter()
            game = runner.play_game(seed)
            wall = time.perf_counter() - t0
            results.append((game, wall))
            kinds = Counter(m.kind for m in game.moves)
            print(f"{tag} seed {seed}: {game.result} ({game.termination}) "
                  f"in {len(game.moves)} plies, {wall:.0f}s wall | "
                  f"{dict(kinds)}", flush=True)
    finally:
        white_engine.close_engines()
        black_engine.close_engines()
    return results


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="simulation.run",
                                description="Simulate bot self-play games")
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--tc", default="60+0", help="time control, e.g. 60+0")
    p.add_argument("--seed", type=int, default=0,
                   help="base seed; game i uses seed+i")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel worker processes, each with its own engine "
                        "pair (suggest ~cores/3; 1 = in-process)")
    p.add_argument("--rating", type=int, default=2450,
                   help="Elo written to headers and fed to the engines")
    p.add_argument("--difficulty", type=int,
                   help="engine playing_level (default common.constants.DIFFICULTY)")
    p.add_argument("--max-plies", type=int, default=400)
    p.add_argument("--out", help="output PGN path")
    p.add_argument("--white-name", default="SimBotWhite")
    p.add_argument("--black-name", default="SimBotBlack")
    args = p.parse_args(argv)

    base, inc = parse_time_control(args.tc)

    # Heavy imports (torch, Stockfish) after arg validation.
    from common.constants import DIFFICULTY
    from .game_runner import SimConfig
    from .pgn_writer import write_games

    difficulty = args.difficulty if args.difficulty is not None else DIFFICULTY
    out_path = args.out or os.path.join(
        DEFAULT_OUT_DIR,
        f"selfplay_{args.tc.replace('+', 'plus')}_seed{args.seed}.pgn")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    cfg = SimConfig(initial_time=float(base), increment=float(inc),
                    white_rating=args.rating, black_rating=args.rating,
                    max_plies=args.max_plies)
    seeds = [args.seed + i for i in range(args.games)]
    workers = max(1, min(args.workers, len(seeds)))

    t_start = time.perf_counter()
    if workers == 1:
        chunk_results = [_play_chunk((0, seeds, cfg, difficulty))]
    else:
        tasks = [(w, seeds[w::workers], cfg, difficulty)
                 for w in range(workers)]
        # spawn: clean interpreter per worker (torch + subprocesses don't
        # mix well with fork).
        with mp.get_context("spawn").Pool(workers) as pool:
            chunk_results = pool.map(_play_chunk, tasks)
    total_wall = time.perf_counter() - t_start

    played = [item for chunk in chunk_results for item in chunk]
    played.sort(key=lambda gw: gw[0].seed)
    sims = [g for g, _ in played]

    manifest = {"tc": args.tc, "rating": args.rating, "difficulty": difficulty,
                "base_seed": args.seed, "workers": workers,
                "total_wall_secs": round(total_wall, 1),
                "games": [{
                    "seed": g.seed, "result": g.result,
                    "termination": g.termination, "plies": len(g.moves),
                    "wall_secs": round(wall, 1),
                    "move_kinds": dict(Counter(m.kind for m in g.moves)),
                } for g, wall in played]}

    date = datetime.date.today().strftime("%Y.%m.%d")
    write_games(sims, out_path, args.white_name, args.black_name,
                args.rating, args.rating, date=date)
    manifest_path = os.path.splitext(out_path)[0] + "_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nWrote {len(sims)} games -> {out_path} "
          f"({total_wall:.0f}s wall, {workers} worker(s))")
    print(f"Manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
