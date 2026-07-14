"""Judge a bot report as an outlier-among-players, not vs the pooled mean.

Companion to `player_dispersion` (which builds the dispersion JSON).
For each feature in a report JSON (from `analyze report --out-json`):

  player_z = (bot_mean - pop_mean) / between_player_std
  OUTLIER  : |player_z| >= 2, or bot outside the empirical 2.5-97.5 pct
             band of real player means. For features where the noise
             correction removed most of the raw variance of player means
             (between_std < 0.5 * raw_std_of_means, e.g. endgame/TP
             acpl), the z-test is unreliable and the band alone decides.
  PERSISTENT-BF : Brown-Forsythe variance flag that persists in >= 35%
             of n=50 subsample draws (the innocent-persistence rule from
             subsample_check -- raw BF flags at n=600 also hit innocents).
  TOO-AVERAGE (informational): |player_z| < 0.2. A bot that is nearly
             average on EVERY feature is itself distinctive; the target
             is mean|player_z| ~ 0.8 (= E|N(0,1)|, what a real innocent
             account shows), not 0.

Usage:
    venv/bin/python -m cheat_detection.outlier_check report.json \
        --dispersion cheat_detection/baselines/dispersion_2500_2800.json
"""
import argparse
import json
import random

from .report import _brown_forsythe

PERSIST_THRESH = 0.35
SUB_SIZE = 50
DRAWS = 40


def persistence(r, key, seed=7):
    hv, bv = r["human_values"].get(key), r["bot_values"].get(key)
    if not hv or not bv or len(bv) < SUB_SIZE:
        return 0.0
    rng = random.Random(seed)
    hits = 0
    for _ in range(DRAWS):
        p = _brown_forsythe(hv, rng.sample(bv, SUB_SIZE))
        if p is not None and p < 0.05:
            hits += 1
    return hits / DRAWS


def evaluate(report_path, dispersion_path):
    r = json.load(open(report_path))
    disp = json.load(open(dispersion_path))["features"]
    rows, n_z, n_out, n_pbf = [], 0, 0, 0
    pz_sum = pz_n = 0
    for f in r["features"]:
        k = f["key"]
        d = disp.get(k)
        row = {"key": k, "z": f["zscore"], "bot": f["bot_value"]}
        if abs(f["zscore"]) >= 2:
            n_z += 1
            row["Z"] = True
        if d:
            row["in_band"] = d["band_2p5"] <= f["bot_value"] <= d["band_97p5"]
            if d["between_std"] > 0.5 * d["raw_std_of_means"]:
                pz = (f["bot_value"] - d["pop_mean"]) / d["between_std"]
                row["player_z"] = pz
                row["too_avg"] = abs(pz) < 0.2
                pz_sum += abs(pz)
                pz_n += 1
                if abs(pz) >= 2 or not row["in_band"]:
                    n_out += 1
                    row["OUT"] = True
            elif not row["in_band"]:
                n_out += 1
                row["OUT"] = True
        if f.get("var_flagged"):
            pers = persistence(r, k)
            row["bf_persistence"] = pers
            if pers >= PERSIST_THRESH:
                n_pbf += 1
                row["PBF"] = True
        rows.append(row)
    summary = {"z_flags": n_z, "player_outliers": n_out,
               "persistent_bf": n_pbf,
               "mean_abs_player_z": pz_sum / max(pz_n, 1)}
    return summary, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report_json")
    ap.add_argument("--dispersion", required=True)
    args = ap.parse_args()
    s, rows = evaluate(args.report_json, args.dispersion)
    print(f"{args.report_json}")
    print(f"  z-flags={s['z_flags']}  player-outlier={s['player_outliers']}  "
          f"persistent-bf={s['persistent_bf']}  "
          f"mean|player_z|={s['mean_abs_player_z']:.3f} "
          f"(innocent expectation ~0.80)")
    n_avg = sum(1 for r in rows if r.get("too_avg"))
    print(f"  too-average features (|player_z|<0.2): {n_avg}/"
          f"{sum(1 for r in rows if 'player_z' in r)}")
    for r in sorted(rows, key=lambda r: -abs(r.get("player_z", 0))):
        marks = ("Z" if r.get("Z") else " ") + \
                ("O" if r.get("OUT") else " ") + \
                ("V" if r.get("PBF") else " ")
        pz = r.get("player_z")
        pz_s = f"{pz:+.2f}" if pz is not None else "  n/a"
        band = "" if r.get("in_band", True) else "  OUTSIDE BAND"
        pers = (f"  bf_persist={r['bf_persistence']:.0%}"
                if "bf_persistence" in r else "")
        print(f"    [{marks}] {r['key']:<28} player_z={pz_s} "
              f"unit_z={r['z']:+.2f}{band}{pers}")


if __name__ == "__main__":
    main()
