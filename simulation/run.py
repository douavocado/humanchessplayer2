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

Progress: a tqdm bar on stderr, one tick per finished game, with per-game
summary lines written through it. Workers silence stdout (the engine's raw
prints would otherwise garble the bar); pipe stderr if capturing progress.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import multiprocessing as mp
import os
import queue as queue_mod
import time
from collections import Counter


DEFAULT_OUT_DIR = os.path.join(os.path.dirname(__file__), "games")


def _play_chunk(task: tuple) -> list:
    """Worker: create one Engine pair and play a list of seeds on it.

    Module-level (picklable) and self-contained: runs in a spawned process.
    Emits progress events to ``progress_q`` ({"type": "loaded"|"game", ...})
    and returns [(SimGame, wall_secs), ...] in the order played. Engine
    stdout (raw [ENGINE] prints) is silenced so the progress bar stays clean.
    """
    worker_id, seeds, cfg, difficulty, quickness, progress_q = task
    import torch
    torch.set_num_threads(1)  # N workers x default 8 threads would thrash
    if quickness is not None:
        # engine.py binds QUICKNESS at import time; patch before importing.
        import common.constants as _constants
        _constants.QUICKNESS = quickness

    results = []
    with open(os.devnull, "w") as devnull, \
            contextlib.redirect_stdout(devnull):
        from engine import Engine
        from .game_runner import GameRunner

        white_engine = Engine(playing_level=difficulty)
        black_engine = Engine(playing_level=difficulty)
        if progress_q is not None:
            progress_q.put({"type": "loaded", "worker": worker_id,
                            "n_seeds": len(seeds)})
        try:
            runner = GameRunner(white_engine, black_engine, cfg)
            for seed in seeds:
                t0 = time.perf_counter()
                game = runner.play_game(seed)
                wall = time.perf_counter() - t0
                results.append((game, wall))
                if progress_q is not None:
                    progress_q.put({
                        "type": "game", "worker": worker_id, "seed": seed,
                        "result": game.result, "termination": game.termination,
                        "plies": len(game.moves), "wall": wall,
                        "kinds": dict(Counter(m.kind for m in game.moves)),
                    })
        finally:
            white_engine.close_engines()
            black_engine.close_engines()
    return results


def _handle_event(ev: dict, bar) -> None:
    if ev["type"] == "loaded":
        bar.write(f"[w{ev['worker']}] engines loaded, "
                  f"{ev['n_seeds']} game(s) queued")
        return
    bar.write(f"[w{ev['worker']}] seed {ev['seed']}: {ev['result']} "
              f"({ev['termination']}) in {ev['plies']} plies, "
              f"{ev['wall']:.0f}s wall | {ev['kinds']}")
    bar.update(1)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="simulation.run",
                                description="Simulate bot self-play games")
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--tc", default="60+0", help="time control, e.g. 60+0")
    p.add_argument("--seed", type=int, default=0,
                   help="base seed; game i uses seed+i")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel worker processes, each with its own engine "
                        "pair (suggest up to physical cores; 1 = in-process)")
    p.add_argument("--rating", type=int, default=2450,
                   help="Elo written to headers and fed to the engines")
    p.add_argument("--difficulty", type=int,
                   help="engine playing_level (default common.constants.DIFFICULTY)")
    p.add_argument("--quickness", type=float,
                   help="move-time pacing override (default common.constants."
                        "QUICKNESS); match the value your real games used "
                        "when comparing sim vs real")
    p.add_argument("--max-plies", type=int, default=400)
    p.add_argument("--out", help="output PGN path")
    p.add_argument("--white-name", default="SimBotWhite")
    p.add_argument("--black-name", default="SimBotBlack")
    args = p.parse_args(argv)

    from cheat_detection.fetch_lichess import parse_time_control
    from tqdm import tqdm

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
    if args.quickness is not None:
        cfg.quickness = args.quickness  # client-side waits (ponder responses)
    seeds = [args.seed + i for i in range(args.games)]
    workers = max(1, min(args.workers, len(seeds)))

    t_start = time.perf_counter()
    bar = tqdm(total=len(seeds), unit="game", desc="simulating", smoothing=0.1)
    if workers == 1:
        # In-process: a plain queue drained after each put via a tiny shim.
        class _Direct:
            @staticmethod
            def put(ev):
                _handle_event(ev, bar)
        chunk_results = [_play_chunk((0, seeds, cfg, difficulty,
                                      args.quickness, _Direct()))]
    else:
        ctx = mp.get_context("spawn")  # fork mixes badly with torch/subprocs
        with ctx.Manager() as mgr:
            progress_q = mgr.Queue()
            tasks = [(w, seeds[w::workers], cfg, difficulty,
                      args.quickness, progress_q) for w in range(workers)]
            with ctx.Pool(workers) as pool:
                async_res = pool.map_async(_play_chunk, tasks)
                while True:
                    try:
                        _handle_event(progress_q.get(timeout=0.5), bar)
                    except queue_mod.Empty:
                        if async_res.ready():
                            break
                # drain any events that raced the ready() check
                while True:
                    try:
                        _handle_event(progress_q.get_nowait(), bar)
                    except queue_mod.Empty:
                        break
                chunk_results = async_res.get()
    bar.close()
    total_wall = time.perf_counter() - t_start

    played = [item for chunk in chunk_results for item in chunk]
    played.sort(key=lambda gw: gw[0].seed)
    sims = [g for g, _ in played]

    manifest = {"tc": args.tc, "rating": args.rating, "difficulty": difficulty,
                "base_seed": args.seed, "workers": workers,
                "quickness": args.quickness,
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
