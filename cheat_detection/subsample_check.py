"""Variance-flag persistence at realistic account sizes.

The Brown-Forsythe test at n=600 bot games flags std ratios that innocent
single accounts also show (the pooled baseline's spread includes
between-player variance -- see runs/single_account_calibration.md). This
script subsamples the bot's per-game values from a report JSON down to
innocent-account size and reports how often each feature's variance flag
persists. Judge a feature by its persistence rate against the innocent
accounts' own amber rate on that feature, not by the full-sample p-value.

Usage:
    python -m cheat_detection.subsample_check runs/report.json \
        [--sub-size 50] [--draws 40] [--seed 7]
"""
import argparse
import json
import random

import numpy as np

from .report import _brown_forsythe


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report_json", help="report JSON with human_values/bot_values")
    ap.add_argument("--sub-size", type=int, default=50,
                    help="games per pseudo-account (default 50, ~ a prolific corpus account)")
    ap.add_argument("--draws", type=int, default=40,
                    help="number of pseudo-accounts to draw (default 40)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    r = json.load(open(args.report_json))
    hv, bv = r["human_values"], r["bot_values"]
    keys = [f["key"] for f in r["features"]]
    full = {f["key"]: f for f in r["features"]}

    rng = random.Random(args.seed)
    flag_counts = []
    per_feature = {k: 0 for k in keys}
    for _ in range(args.draws):
        nflags = 0
        for k in keys:
            vals = bv.get(k)
            if not vals or len(vals) < args.sub_size:
                continue
            sub = rng.sample(vals, args.sub_size)
            p = _brown_forsythe(hv[k], sub)
            if p is not None and p < 0.05:
                nflags += 1
                per_feature[k] += 1
        flag_counts.append(nflags)

    print(f"bot subsampled to {args.sub_size} games x {args.draws} draws "
          f"(from {r['bot']['n_units']} units)")
    print(f"var-flags per pseudo-account: median={int(np.median(flag_counts))}, "
          f"mean={np.mean(flag_counts):.1f}, range={min(flag_counts)}-{max(flag_counts)}")
    print("(2026-07-12 calibration, 20 innocent 2300-2600 accounts at n=32-61: "
          "median ~1, range 0-5)")
    print(f"\n{'feature':28s} {'full ratio':>10s} {'persistence':>11s}")
    for k in keys:
        rate = per_feature[k] / args.draws
        ratio = full[k].get("var_ratio")
        marker = "  <-- persistent" if rate >= 0.35 else ""
        if rate > 0 or (ratio is not None and full[k].get("var_flagged")):
            print(f"{k:28s} {ratio:10.2f} {rate:10.0%}{marker}")


if __name__ == "__main__":
    main()
