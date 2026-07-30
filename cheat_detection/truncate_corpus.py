"""Truncate a human corpus where the simulator stops, so the two are comparable.

Why this exists. `simulation.run` ends a game when either side's clock first
drops below `CLOCK_THRESHOLD_FRACTION * initial_time` (0.25, so 15s at 60+0 and
45s at 180+0) and adjudicates the result from a Stockfish eval. Simulated PGNs
therefore contain **no scramble moves at all**, while a human corpus contains
complete games. Comparing the two directly is one of the two traps written up in
`docs/position-conditioned-human-likeness.md`, and it produced confidently wrong
conclusions before it was caught: the bot's `acpl_endgame` read 21.7 against a
complete-game human 294.4, an artefact entirely of the missing scramble.

So any bot-vs-human comparison needs a baseline built from a corpus cut at the
same point. This applies the simulator's own `_find_cutoff_ply` to each game, so
the cut rule cannot drift from the simulator's.

The bullet equivalent (`bullet_1plus0_2300_plus__trunc15.pgn`) was produced ad
hoc and left no script behind, which is why this one exists as tracked code.

Results are NOT re-adjudicated: the truncated game keeps its original PGN
`Result` header. That is deliberate and it is a limitation -- the human game's
result reflects the full game including the scramble that was just removed. It
does not affect the move-level timing and accuracy features these baselines are
built for, but do not use a truncated corpus to compare *score rates*.

Usage:
    venv/bin/python -m cheat_detection.truncate_corpus \\
        --in cheat_detection/corpora/blitz_3plus0_2300_plus.pgn \\
        --out cheat_detection/corpora/blitz_3plus0_2300_plus__trunc45.pgn \\
        --tc 180+0
"""
from __future__ import annotations

import argparse
import sys

import chess.pgn

from simulation.adjudicate_result import CLOCK_THRESHOLD_FRACTION, _find_cutoff_ply

from .pgn_loader import parse_tc_seconds

_CLK = None  # lazily compiled; see _clock_of


def _clock_of(node) -> int | None:
    """Remaining clock in seconds from a node's [%clk H:MM:SS] comment."""
    global _CLK
    if _CLK is None:
        import re
        _CLK = re.compile(r"\[%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]")
    for c in node.comment.split("]"):
        m = _CLK.search(c + "]")
        if m:
            h, mi, s = m.groups()
            return int(int(h) * 3600 + int(mi) * 60 + float(s))
    return None


def truncate(in_path: str, out_path: str, initial_time: float,
             max_games: int | None = None) -> tuple[int, int, int]:
    """Returns (games written, plies kept, plies dropped)."""
    kept_games = kept_plies = dropped_plies = 0
    with open(in_path, "r", encoding="utf-8", errors="replace") as fh, \
            open(out_path, "w", encoding="utf-8") as out:
        while True:
            if max_games is not None and kept_games >= max_games:
                break
            game = chess.pgn.read_game(fh)
            if game is None:
                break

            # (move, clock) per ply, in the order the simulator sees them.
            moves: list[tuple[chess.Move, int | None]] = []
            node = game
            while node.variations:
                node = node.variations[0]
                moves.append((node.move, _clock_of(node)))
            if not moves:
                continue

            cut = _find_cutoff_ply(moves, initial_time)
            if cut >= len(moves):
                kept_plies += len(moves)
                print(game, file=out, end="\n\n")
                kept_games += 1
                continue

            # Rebuild the game with only the first `cut` plies, preserving each
            # ply's comment so [%clk] survives for the timing features.
            trunc = chess.pgn.Game()
            trunc.headers.update(game.headers)
            src, dst = game, trunc
            for _ in range(cut):
                src = src.variations[0]
                dst = dst.add_variation(src.move)
                dst.comment = src.comment
            kept_plies += cut
            dropped_plies += len(moves) - cut
            print(trunc, file=out, end="\n\n")
            kept_games += 1

            if kept_games % 5000 == 0:
                print(f"  {kept_games} games truncated ...", flush=True)
    return kept_games, kept_plies, dropped_plies


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="cheat_detection.truncate_corpus")
    p.add_argument("--in", dest="infile", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tc", required=True,
                   help="time control of the corpus, e.g. 180+0 -- sets the "
                        "clock the cutoff fraction applies to")
    p.add_argument("--max-games", type=int, default=None)
    args = p.parse_args(argv)

    initial_time = parse_tc_seconds(args.tc)
    threshold = CLOCK_THRESHOLD_FRACTION * initial_time
    print(f"Truncating {args.infile} at the simulator's cutoff "
          f"({CLOCK_THRESHOLD_FRACTION:g} * {initial_time:g}s = {threshold:g}s "
          f"remaining) ...")
    games, kept, dropped = truncate(args.infile, args.out, initial_time,
                                    args.max_games)
    total = kept + dropped
    pct = dropped / total * 100 if total else 0.0
    print(f"Wrote {games} games -> {args.out}")
    print(f"Kept {kept:,} plies, dropped {dropped:,} ({pct:.1f}% of moves were "
          f"below the cutoff and are absent from simulated games).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
