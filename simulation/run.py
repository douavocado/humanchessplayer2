"""CLI: simulate bot self-play games and write PGN for cheat_detection.

Example (from repo root):

    venv/bin/python -m simulation.run --games 20 --tc 60+0 --seed 1 \
        --workers 5 --out simulation/games/selfplay_60plus0.pgn

The output parses directly with cheat_detection, e.g.:

    venv/bin/python -m cheat_detection.analyze report \
        --pgn simulation/games/selfplay_60plus0.pgn \
        --player SimBotWhite SimBotBlack \
        --baseline cheat_detection/baselines/bullet_1plus0_2300_2600.json

Asymmetric matchups: the two bots ("a" and "b") can differ in rating,
difficulty (engine playing_level), quickness (move-time pacing) and mouse
speed via --a-*/--b-* flags; the shared --rating/--difficulty/--quickness
flags remain the default for both. --sides alternate swaps colours every
game (bot a is white in games 0, 2, 4...), so a matchup can be profiled
colour-balanced; PGN headers carry the right name/Elo per game either way.

Parallelism: games are independent, so --workers N runs N processes, each
with its own Engine pair (each pair holds several Stockfish subprocesses and
the torch models — keep N around cores/3). Seeds are partitioned
deterministically (worker w plays seeds[w::N]), so the same
(games, seed, workers) triple reproduces the same games; changing the worker
count changes which engine-state history precedes each game, so per-game
results are only guaranteed stable for a fixed worker count.

Progress: a tqdm bar on stderr, one tick per finished game, with per-game
summary lines written through it; --plain swaps the bar for plain stderr
lines (machine-readable, used by the GUI). Workers silence stdout (the
engine's raw prints would otherwise garble the bar); pipe stderr if capturing
progress.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import multiprocessing as mp
import os
import queue as queue_mod
import sys
import time
from collections import Counter
from dataclasses import asdict


DEFAULT_OUT_DIR = os.path.join(os.path.dirname(__file__), "games")


def _play_chunk(task: tuple) -> list:
    """Worker: create one Engine pair and play a list of jobs on it.

    Module-level (picklable) and self-contained: runs in a spawned process.
    ``jobs`` is [(seed, a_plays_white), ...]. Emits progress events to
    ``progress_q`` ({"type": "loaded"|"game", ...}) and returns
    [(SimGame, wall_secs), ...] in the order played. Engine stdout (raw
    [ENGINE] prints) is silenced so the progress bar stays clean.
    """
    worker_id, jobs, cfg, progress_q = task
    import torch
    torch.set_num_threads(1)  # N workers x default 8 threads would thrash

    results = []
    with open(os.devnull, "w") as devnull, \
            contextlib.redirect_stdout(devnull):
        from engine import Engine
        from .game_runner import GameRunner

        engine_a = Engine(playing_level=cfg.bot_a.difficulty,
                          quickness=cfg.bot_a.quickness)
        engine_b = Engine(playing_level=cfg.bot_b.difficulty,
                          quickness=cfg.bot_b.quickness)
        if progress_q is not None:
            progress_q.put({"type": "loaded", "worker": worker_id,
                            "n_seeds": len(jobs)})
        try:
            runner = GameRunner(engine_a, engine_b, cfg)
            for seed, a_white in jobs:
                t0 = time.perf_counter()
                game = runner.play_game(seed, a_plays_white=a_white)
                wall = time.perf_counter() - t0
                results.append((game, wall))
                if progress_q is not None:
                    progress_q.put({
                        "type": "game", "worker": worker_id, "seed": seed,
                        "white": game.white_name,
                        "result": game.result, "termination": game.termination,
                        "plies": len(game.moves), "wall": wall,
                        "kinds": dict(Counter(m.kind for m in game.moves)),
                    })
        finally:
            engine_a.close_engines()
            engine_b.close_engines()
    return results


def _handle_event(ev: dict, bar) -> None:
    if ev["type"] == "loaded":
        bar.write(f"[w{ev['worker']}] engines loaded, "
                  f"{ev['n_seeds']} game(s) queued")
        return
    bar.write(f"[w{ev['worker']}] seed {ev['seed']} (white={ev['white']}): "
              f"{ev['result']} ({ev['termination']}) in {ev['plies']} plies, "
              f"{ev['wall']:.0f}s wall | {ev['kinds']}")
    bar.update(1)


class _PlainBar:
    """tqdm-free progress: one parseable stderr line per event (GUI-friendly)."""

    def __init__(self, total: int):
        self.done, self.total = 0, total

    def write(self, s: str) -> None:
        print(s, file=sys.stderr, flush=True)

    def update(self, n: int = 1) -> None:
        self.done += n
        print(f"[progress] {self.done}/{self.total} games done",
              file=sys.stderr, flush=True)

    def close(self) -> None:
        pass


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
                   help="Elo for both bots unless --a-rating/--b-rating override")
    p.add_argument("--difficulty", type=int,
                   help="engine playing_level for both bots (default "
                        "common.constants.DIFFICULTY)")
    p.add_argument("--quickness", type=float,
                   help="move-time pacing for both bots (default common."
                        "constants.QUICKNESS); match the value your real games "
                        "used when comparing sim vs real")
    p.add_argument("--sides", choices=["fixed", "alternate"], default="fixed",
                   help="fixed: bot a is always white; alternate: colours swap "
                        "every game")
    for tag in ("a", "b"):
        p.add_argument(f"--{tag}-name", dest=f"{tag}_name",
                       help=f"bot {tag}'s PGN name")
        p.add_argument(f"--{tag}-rating", dest=f"{tag}_rating", type=int,
                       help=f"bot {tag}'s Elo (overrides --rating)")
        p.add_argument(f"--{tag}-difficulty", dest=f"{tag}_difficulty", type=int,
                       help=f"bot {tag}'s playing_level (overrides --difficulty)")
        p.add_argument(f"--{tag}-quickness", dest=f"{tag}_quickness", type=float,
                       help=f"bot {tag}'s pacing (overrides --quickness)")
        p.add_argument(f"--{tag}-mouse", dest=f"{tag}_mouse", type=float,
                       help=f"bot {tag}'s mouse quickness (default "
                            "common.constants.MOUSE_QUICKNESS)")
    p.add_argument("--max-plies", type=int, default=400)
    p.add_argument("--out", help="output PGN path")
    p.add_argument("--plain", action="store_true",
                   help="plain-line progress on stderr instead of a tqdm bar")
    args = p.parse_args(argv)

    from cheat_detection.fetch_lichess import parse_time_control

    base, inc = parse_time_control(args.tc)

    # Heavy imports (torch, Stockfish) after arg validation.
    from .game_runner import BotSpec, SimConfig
    from .pgn_writer import write_games

    def spec(tag: str, default_name: str) -> BotSpec:
        g = lambda attr: getattr(args, f"{tag}_{attr}")  # noqa: E731
        return BotSpec(
            name=g("name") or default_name,
            rating=g("rating") if g("rating") is not None else args.rating,
            difficulty=(g("difficulty") if g("difficulty") is not None
                        else args.difficulty),
            quickness=(g("quickness") if g("quickness") is not None
                       else args.quickness),
            mouse_quickness=g("mouse"),
        )

    # Default names keep the historic fixed-sides labels so existing
    # cheat_detection invocations (--player SimBotWhite SimBotBlack) still work.
    bot_a = spec("a", "SimBotWhite")
    bot_b = spec("b", "SimBotBlack")

    out_path = args.out or os.path.join(
        DEFAULT_OUT_DIR,
        f"selfplay_{args.tc.replace('+', 'plus')}_seed{args.seed}.pgn")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    cfg = SimConfig(initial_time=float(base), increment=float(inc),
                    bot_a=bot_a, bot_b=bot_b, max_plies=args.max_plies)
    jobs = [(args.seed + i, args.sides == "fixed" or i % 2 == 0)
            for i in range(args.games)]
    workers = max(1, min(args.workers, len(jobs)))

    t_start = time.perf_counter()
    if args.plain:
        bar = _PlainBar(len(jobs))
    else:
        from tqdm import tqdm
        bar = tqdm(total=len(jobs), unit="game", desc="simulating",
                   smoothing=0.1)
    if workers == 1:
        # In-process: a plain queue drained after each put via a tiny shim.
        class _Direct:
            @staticmethod
            def put(ev):
                _handle_event(ev, bar)
        chunk_results = [_play_chunk((0, jobs, cfg, _Direct()))]
    else:
        ctx = mp.get_context("spawn")  # fork mixes badly with torch/subprocs
        with ctx.Manager() as mgr:
            progress_q = mgr.Queue()
            tasks = [(w, jobs[w::workers], cfg, progress_q)
                     for w in range(workers)]
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

    manifest = {"tc": args.tc, "sides": args.sides,
                "bots": {"a": asdict(bot_a), "b": asdict(bot_b)},
                "base_seed": args.seed, "workers": workers,
                "total_wall_secs": round(total_wall, 1),
                "games": [{
                    "seed": g.seed, "white": g.white_name, "black": g.black_name,
                    "result": g.result,
                    "termination": g.termination, "plies": len(g.moves),
                    "wall_secs": round(wall, 1),
                    "move_kinds": dict(Counter(m.kind for m in g.moves)),
                } for g, wall in played]}

    date = datetime.date.today().strftime("%Y.%m.%d")
    write_games(sims, out_path, date=date)
    manifest_path = os.path.splitext(out_path)[0] + "_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nWrote {len(sims)} games -> {out_path} "
          f"({total_wall:.0f}s wall, {workers} worker(s))")
    print(f"Manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
