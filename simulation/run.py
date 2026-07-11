"""CLI: simulate bot self-play games and write PGN for cheat_detection.

Example (from repo root):

    venv/bin/python -m simulation.run --games 20 --tc 60+0 --seed 1 \
        --out simulation/games/selfplay_60plus0.pgn

The output parses directly with cheat_detection, e.g.:

    venv/bin/python -m cheat_detection.analyze report \
        --pgn simulation/games/selfplay_60plus0.pgn \
        --player SimBotWhite SimBotBlack \
        --baseline cheat_detection/baselines/bullet_1plus0_2300_2600.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import time
from collections import Counter

from cheat_detection.fetch_lichess import parse_time_control

DEFAULT_OUT_DIR = os.path.join(os.path.dirname(__file__), "games")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="simulation.run",
                                description="Simulate bot self-play games")
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--tc", default="60+0", help="time control, e.g. 60+0")
    p.add_argument("--seed", type=int, default=0,
                   help="base seed; game i uses seed+i")
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
    from engine import Engine
    from .game_runner import GameRunner, SimConfig
    from .pgn_writer import write_games

    difficulty = args.difficulty if args.difficulty is not None else DIFFICULTY
    out_path = args.out or os.path.join(
        DEFAULT_OUT_DIR,
        f"selfplay_{args.tc.replace('+', 'plus')}_seed{args.seed}.pgn")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    print(f"Loading two engines (playing_level={difficulty})...", flush=True)
    white_engine = Engine(playing_level=difficulty)
    black_engine = Engine(playing_level=difficulty)

    cfg = SimConfig(initial_time=float(base), increment=float(inc),
                    white_rating=args.rating, black_rating=args.rating,
                    max_plies=args.max_plies)
    runner = GameRunner(white_engine, black_engine, cfg)

    sims = []
    manifest = {"tc": args.tc, "rating": args.rating, "difficulty": difficulty,
                "base_seed": args.seed, "games": []}
    try:
        for i in range(args.games):
            seed = args.seed + i
            t0 = time.perf_counter()
            game = runner.play_game(seed)
            wall = time.perf_counter() - t0
            sims.append(game)
            kinds = Counter(m.kind for m in game.moves)
            sim_secs = sum(m.charged_secs for m in game.moves)
            print(f"game {i + 1}/{args.games}: {game.result} "
                  f"({game.termination}) in {len(game.moves)} plies | "
                  f"simulated {sim_secs:.0f}s of clock in {wall:.0f}s wall | "
                  f"{dict(kinds)}", flush=True)
            manifest["games"].append({
                "seed": seed, "result": game.result,
                "termination": game.termination, "plies": len(game.moves),
                "wall_secs": round(wall, 1),
                "move_kinds": dict(kinds),
            })
    finally:
        white_engine.close_engines()
        black_engine.close_engines()

    date = datetime.date.today().strftime("%Y.%m.%d")
    write_games(sims, out_path, args.white_name, args.black_name,
                args.rating, args.rating, date=date)
    manifest_path = os.path.splitext(out_path)[0] + "_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nWrote {len(sims)} games -> {out_path}")
    print(f"Manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
