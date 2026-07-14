"""EMT-bucket decomposition of time-pressure blunders.

Under 10s of clock, split moves by elapsed move time (integer PGN
clocks: emt 0 = fired instantly -- premove/stale-ponder/prepared-reply
fires; emt >= 1 = at least glanced) and compare blunder rates per
bucket. Distinguishes "the bot's scramble decisions are too bad/good"
from "the bot fires too many blind moves".

Reference (2500-2800 bullet corpus): humans fire ~60% of sub-10s moves
at emt 0 and that is their SAFEST bucket (blunder rate 0.070 vs 0.077
thought, 0.112 for 2s+ thinks) -- quality instants come from
preparation. A bot whose emt-0 bucket is its WORST is firing blind.

Only useful on PGNs whose positions are already in the eval cache (run
a report first) -- uncached positions will be analysed from scratch.

Usage:
    venv/bin/python -m cheat_detection.emt_buckets \
        --pgn simulation/games/<run>.pgn --players SimBotWhite SimBotBlack
    # human reference:
    venv/bin/python -m cheat_detection.emt_buckets \
        --pgn <corpus.pgn> --rating 2500 2800 [--max-games N]
"""
import argparse
import sys
from collections import defaultdict

from .config import AnalysisConfig
from .engine_analysis import EngineAnalyzer
from .features import extract_move_features
from .pgn_loader import iter_games
from .pipeline import _sides_of_interest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pgn", required=True)
    ap.add_argument("--players", nargs="+", default=None)
    ap.add_argument("--rating", nargs=2, type=int, default=None)
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--tp-secs", type=float, default=10.0)
    args = ap.parse_args()

    cfg = AnalysisConfig()
    player_filter = ({p.lower() for p in args.players}
                     if args.players else None)
    band = tuple(args.rating) if args.rating else None

    buckets = defaultdict(lambda: [0, 0, 0.0])  # n, blunders, sum_wc_loss

    with EngineAnalyzer(cfg) as analyzer:
        for gi, game in enumerate(iter_games(args.pgn,
                                             max_games=args.max_games)):
            for color, name, elo in _sides_of_interest(
                    game, player_filter, band):
                for rec in game.moves:
                    if rec.mover != color:
                        continue
                    if rec.clock_before is None or rec.emt is None:
                        continue
                    if rec.clock_before >= args.tp_secs:
                        continue
                    pe = analyzer.analyse(rec.fen_before)
                    if not pe.candidates:
                        continue
                    played_cp = analyzer.eval_after_move(
                        rec.fen_before, rec.move_uci)
                    mf = extract_move_features(rec, pe, played_cp, cfg)
                    if mf is None:
                        continue
                    key = ("emt0" if rec.emt < 1 else
                           "emt1" if rec.emt < 2 else "emt2+")
                    b = buckets[key]
                    b[0] += 1
                    b[1] += int(mf.is_blunder)
                    b[2] += mf.wc_loss
            if (gi + 1) % 200 == 0:
                analyzer.flush()
                print(f"  {gi + 1} games...", file=sys.stderr)
        analyzer.flush()

    total = sum(b[0] for b in buckets.values())
    print(f"\nTime-pressure moves (clock < {args.tp_secs:g}s): {total}")
    print(f"{'bucket':<7} {'n':>7} {'share':>7} {'blunder_rate':>13} "
          f"{'mean_wc_loss':>13}")
    for key in ("emt0", "emt1", "emt2+"):
        n, bl, wc = buckets[key]
        if n == 0:
            continue
        print(f"{key:<7} {n:>7} {n / total:>7.3f} {bl / n:>13.4f} "
              f"{wc / n:>13.4f}")


if __name__ == "__main__":
    main()
