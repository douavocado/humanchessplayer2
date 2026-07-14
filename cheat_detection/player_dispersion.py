"""Between-player dispersion of per-feature means, from a human corpus.

Why this exists: a Welch t-test of the bot's feature means against the
pooled baseline asks "is the bot the AVERAGE player" -- no real account
satisfies that (every player has a persistent personal offset), so at
bot sample sizes it flags everything, innocents included. The realistic
question is "is the bot an outlier AMONG PLAYERS":

    player_z = (bot_mean - pop_mean) / between_player_std

This module estimates between_player_std per feature from per-player
feature means (players with >= --min-units games), noise-corrected by
method of moments:

    delta^2 = var(player_means) - mean(sigma_i^2 / n_i)

and also emits the empirical band of player means (2.5/97.5 and 10/90
percentiles). Judge with `python -m cheat_detection.outlier_check`.
For features where the correction removes most of the raw variance
(endgame/TP acpl: within-player noise dominates), between_std is
unreliable -- the outlier check falls back to the empirical band there.

Usage:
    venv/bin/python -m cheat_detection.player_dispersion \
        --pgn cheat_detection/corpora/bullet_1plus0_2300_plus.pgn \
        --rating 2500 2800 --min-units 20 \
        --out cheat_detection/baselines/dispersion_2500_2800.json

Evals come from the shared cache; on a fully-cached corpus this is
IO-bound (minutes).
"""
import argparse
import json
import sys
from collections import defaultdict

import numpy as np

from .config import AnalysisConfig
from .engine_analysis import EngineAnalyzer
from .pipeline import iter_units


def build(pgn, rating_band, min_units=20, max_games=None):
    cfg = AnalysisConfig()
    per_player = defaultdict(lambda: defaultdict(list))
    with EngineAnalyzer(cfg) as analyzer:
        def prog(n):
            if n % 1000 == 0:
                print(f"  {n} games...", file=sys.stderr)
        for unit in iter_units(pgn, analyzer, cfg, rating_band=rating_band,
                               max_games=max_games, on_progress=prog):
            for feat, val in unit.features.items():
                if val is not None:
                    per_player[unit.player][feat].append(float(val))

    keep = {p: f for p, f in per_player.items()
            if len(next(iter(f.values()), [])) >= min_units}
    print(f"{len(per_player)} players seen, {len(keep)} with >= "
          f"{min_units} units", file=sys.stderr)

    out = {"rating_band": list(rating_band), "min_units": min_units,
           "n_players": len(keep), "features": {}}
    feats = sorted({f for fd in keep.values() for f in fd})
    for feat in feats:
        means, sampvars = [], []
        for fd in keep.values():
            vals = fd.get(feat, [])
            if len(vals) < min_units:
                continue
            v = np.asarray(vals, dtype=float)
            means.append(v.mean())
            sampvars.append(v.var(ddof=1) / len(v))
        if len(means) < 5:
            continue
        means = np.asarray(means)
        raw_var = float(means.var(ddof=1))
        noise = float(np.mean(sampvars))
        out["features"][feat] = {
            "n_players": len(means),
            "pop_mean": float(means.mean()),
            "between_std": max(raw_var - noise, 1e-12) ** 0.5,
            "raw_std_of_means": raw_var ** 0.5,
            "mean_sampling_noise_std": noise ** 0.5,
            "band_2p5": float(np.percentile(means, 2.5)),
            "band_97p5": float(np.percentile(means, 97.5)),
            "band_10": float(np.percentile(means, 10)),
            "band_90": float(np.percentile(means, 90)),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pgn", required=True)
    ap.add_argument("--rating", nargs=2, type=int, required=True)
    ap.add_argument("--min-units", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-games", type=int, default=None)
    args = ap.parse_args()
    out = build(args.pgn, tuple(args.rating), args.min_units, args.max_games)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
