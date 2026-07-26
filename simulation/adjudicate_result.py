"""Re-adjudicates games (bot self-play OR real human PGNs) from position
quality instead of the PGN's actual result, to separate "who played better
chess" from "who won/lost the clock".

Motivation: across every self-play run generated this session, 56-62% of
games ended in "timeout" (flag fall), vs. ~35-40% checkmate/resignation --
see cheat_detection/runs/ for the manifest breakdown. Grading straight PGN
results conflates "played the position better" with "managed the
clock-race tail better", which dilutes the Elo signal from anything that
only changes move *quality* (breadth, noise, Stockfish config, ...) --
and, symmetrically, makes bot-vs-human move-quality comparisons unfair if
one population's raw results are noisier than the other's.

Adjudication: cut the game at the earliest ply where *either* side's clock
first drops below CLOCK_THRESHOLD_FRACTION of their initial time (15s for a
60+0 game) -- a simple, source-agnostic proxy for "before this got
desperate", usable identically on bot self-play and real Lichess exports
(both carry [%clk]), which is the point: it lets bot and human games be cut
by the same rule instead of needing bot-specific instrumentation. Falls
back to the game's natural end if neither side ever crosses the threshold.
Evaluate the position at that cutoff with real Stockfish (no wall-clock cap
-- this is pure offline analysis, not the live engine) and adjudicate:
|eval| <= DRAW_CP_MARGIN centipawns is a draw, otherwise the side favoured
by eval wins.
"""
from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from typing import Optional

import chess
import chess.engine
import chess.pgn

from common.constants import PATH_TO_STOCKFISH

CLOCK_THRESHOLD_FRACTION = 0.25  # cut when either side's clock first drops below this fraction of initial time
ADJUDICATION_DEPTH = 20
ADJUDICATION_TIME = 1.0  # seconds, whichever limit hits first
DRAW_CP_MARGIN = 200


def _parse_clock(comment: str) -> Optional[int]:
    m = re.search(r"%clk (\d+):(\d+):(\d+)", comment or "")
    if not m:
        return None
    h, mnt, s = (int(x) for x in m.groups())
    return h * 3600 + mnt * 60 + s


def _moves_with_clock(game: chess.pgn.Game) -> list[tuple[chess.Move, Optional[int]]]:
    out = []
    node = game
    while node.variations:
        node = node.variations[0]
        out.append((node.move, _parse_clock(node.comment)))
    return out


def _find_cutoff_ply(moves: list[tuple[chess.Move, Optional[int]]], initial_time: float) -> int:
    """Earliest ply p (1-indexed count of moves played) at which either
    side's clock first reads below CLOCK_THRESHOLD_FRACTION * initial_time;
    falls back to the full game length if neither side ever crosses it."""
    threshold = CLOCK_THRESHOLD_FRACTION * initial_time
    for p, (_, clk) in enumerate(moves, start=1):
        if clk is not None and clk < threshold:
            return p
    return len(moves)


def _parse_time_control(tc: str) -> float:
    base, _, _inc = tc.partition("+")
    return float(base)


def adjudicate_position(board: chess.Board, engine: chess.engine.SimpleEngine) -> tuple[str, Optional[float]]:
    """Result string ("1-0"/"0-1"/"1/2-1/2") and cp eval (None if the game
    is already over) for the given position -- shared by the live
    early-stopping simulator (game_runner.py) and this module's post-hoc
    PGN re-adjudication."""
    if board.is_game_over():
        outcome = board.outcome()
        if outcome.winner is None:
            return "1/2-1/2", None
        return ("1-0" if outcome.winner == chess.WHITE else "0-1"), None
    info = engine.analyse(board, chess.engine.Limit(depth=ADJUDICATION_DEPTH, time=ADJUDICATION_TIME))
    cp = info["score"].pov(chess.WHITE).score(mate_score=100000)
    if abs(cp) <= DRAW_CP_MARGIN:
        return "1/2-1/2", cp
    return ("1-0" if cp > 0 else "0-1"), cp


def adjudicate_game(game: chess.pgn.Game, engine: chess.engine.SimpleEngine) -> dict:
    initial_time = _parse_time_control(game.headers.get("TimeControl", "60+0"))
    moves = _moves_with_clock(game)
    x = len(moves)
    cutoff = _find_cutoff_ply(moves, initial_time)

    board = game.board()
    for move, _ in moves[:cutoff]:
        board.push(move)

    adjudicated, cp = adjudicate_position(board, engine)

    return {
        "white": game.headers.get("White"), "black": game.headers.get("Black"),
        "actual_result": game.headers.get("Result"), "termination": game.headers.get("Termination"),
        "adjudicated_result": adjudicated, "plies": x, "cutoff_ply": cutoff, "cutoff_cp": cp,
    }


def _score(result: str, player_is_white: bool) -> float:
    if result == "1/2-1/2":
        return 0.5
    if result == "1-0":
        return 1.0 if player_is_white else 0.0
    if result == "0-1":
        return 0.0 if player_is_white else 1.0
    raise ValueError(result)


def elo_diff(score_rate: float, clip=(0.02, 0.98)) -> float:
    p = min(max(score_rate, clip[0]), clip[1])
    return -400 * math.log10(1 / p - 1)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Re-adjudicate self-play PGN results from position quality")
    p.add_argument("--pgn", required=True)
    p.add_argument("--max-games", type=int)
    p.add_argument("--out-json", help="dump the full per-game adjudication list here "
                                       "(consumed by cheat_detection.mistake_impact)")
    args = p.parse_args(argv)

    engine = chess.engine.SimpleEngine.popen_uci(PATH_TO_STOCKFISH)
    engine.configure({"Threads": 2, "Hash": 128})

    per_game = []
    with open(args.pgn) as fh:
        n = 0
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            per_game.append(adjudicate_game(game, engine))
            n += 1
            if args.max_games and n >= args.max_games:
                break
    engine.quit()

    players = sorted({g["white"] for g in per_game} | {g["black"] for g in per_game})
    raw_score = defaultdict(float)
    adj_score = defaultdict(float)
    disagreements = 0
    cutoff_before_end = 0
    for g in per_game:
        raw_score[g["white"]] += _score(g["actual_result"], True)
        raw_score[g["black"]] += _score(g["actual_result"], False)
        adj_score[g["white"]] += _score(g["adjudicated_result"], True)
        adj_score[g["black"]] += _score(g["adjudicated_result"], False)
        if g["adjudicated_result"] != g["actual_result"]:
            disagreements += 1
        if g["cutoff_ply"] < g["plies"]:
            cutoff_before_end += 1

    n = len(per_game)
    print(f"Games: {n}")
    print(f"Adjudicated cutoff before game's natural end (autopilot tail detected): "
          f"{cutoff_before_end} ({100*cutoff_before_end/n:.1f}%)")
    print(f"Adjudicated result differs from actual PGN result: "
          f"{disagreements} ({100*disagreements/n:.1f}%)")
    print()
    for player in players:
        raw_rate = raw_score[player] / n
        adj_rate = adj_score[player] / n
        print(f"{player}: raw score {raw_score[player]:.1f}/{n} ({raw_rate:.4f})  "
              f"adjudicated score {adj_score[player]:.1f}/{n} ({adj_rate:.4f})")

    if len(players) == 2:
        p1 = players[1]
        raw_elo = elo_diff(raw_score[p1] / n)
        adj_elo = elo_diff(adj_score[p1] / n)
        print()
        print(f"Raw Elo delta ({p1} - {players[0]}): {raw_elo:+.1f}")
        print(f"Adjudicated Elo delta ({p1} - {players[0]}): {adj_elo:+.1f}")

    if args.out_json:
        import json
        with open(args.out_json, "w", encoding="utf-8") as fh:
            json.dump(per_game, fh, indent=2)
        print(f"Wrote {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
