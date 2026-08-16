"""Read a band table against another one, on the three pre-committed measures.

Written BEFORE the 180+0 data existed, so that the comparison is defined rather
than chosen after seeing the numbers. The reads and their predictions are in
`run_band_table_180.py`'s docstring and the spec.

The three measures:

  1. Mean emt across bands -- reported two ways, because the endpoint span is
     dominated by its thinnest band and at 3+0 the 2800+ band is scarce (top
     players play far more bullet). The **weighted regression slope** over all
     usable bands uses every move and is the primary figure; the endpoint span
     is reported alongside for comparability with how bullet was described.
  2. Top-1 match across bands, same two ways.
  3. Per-phase mean emt, as the opening-to-midgame ratio per band.

Slopes are per 100 Elo and expressed as a percentage of the table's own mean, so
a 60+0 table (mean emt ~1.1s) and a 180+0 one (~3-6s) are comparable at all.

Usage:
    venv/bin/python cheat_detection/runs/elo_progression/compare_bands.py \\
        --a runs/elo_progression/report_timing.md --a-label "60+0" \\
        --b runs/elo_progression/report_timing_180.md --b-label "180+0"
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

# Band midpoints for the regression. The open-ended top band is given a
# nominal midpoint; it is the x-value, not a claim about its population.
BAND_MID = {
    "2100-2299": 2200, "2300-2399": 2350, "2400-2499": 2450,
    "2500-2599": 2550, "2600-2699": 2650, "2700-2799": 2750,
    "2800+": 2875,
}

_SECTION = re.compile(r"^#+\s+(.*)$")


def parse_sections(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    """{section heading: {band label: {column name: value}}}."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    section = "(none)"
    header: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _SECTION.match(line)
        if m:
            section = m.group(1).strip()
            header = []
            continue
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == "Band":
            header = cells
            continue
        if not header or set(cells[0]) <= set("-: ") or cells[0] not in BAND_MID:
            continue
        row: dict[str, float] = {}
        for name, cell in zip(header[1:], cells[1:]):
            try:
                row[name] = float(cell)
            except ValueError:
                pass  # "n/a" -- band below the min_n gate
        out.setdefault(section, {})[cells[0]] = row
    return out


def _wls_slope(pts: list[tuple[float, float, float]]) -> tuple[float, float]:
    """Weighted least-squares slope per 100 Elo, and the table's weighted mean.

    Weights are move counts. Returns (slope_per_100_elo, weighted_mean_y).
    """
    sw = sum(w for _, _, w in pts)
    mx = sum(w * x for x, _, w in pts) / sw
    my = sum(w * y for _, y, w in pts) / sw
    sxx = sum(w * (x - mx) ** 2 for x, _, w in pts)
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in pts)
    return (sxy / sxx * 100 if sxx else float("nan")), my


def measure(sections, col: str, section: str = "Overall (all positions)"):
    """Endpoint span and weighted slope for one column of one section."""
    rows = sections.get(section, {})
    pts = [(BAND_MID[b], r[col], r["n moves"])
           for b, r in rows.items() if col in r and "n moves" in r]
    if len(pts) < 3:
        return None
    pts.sort()
    lo_y, hi_y = pts[0][1], pts[-1][1]
    slope, mean_y = _wls_slope(pts)
    return {
        "bands": len(pts),
        "lo_band": pts[0][0], "hi_band": pts[-1][0],
        "lo": lo_y, "hi": hi_y,
        "span_abs": hi_y - lo_y,
        "span_pct": (hi_y - lo_y) / lo_y * 100 if lo_y else float("nan"),
        "slope_per_100": slope,
        "slope_pct_per_100": slope / mean_y * 100 if mean_y else float("nan"),
        "mean": mean_y,
        "n_total": sum(w for _, _, w in pts),
    }


def _fmt(m, unit=""):
    if m is None:
        return "  (too few usable bands to read)"
    return (f"  bands used: {m['bands']} ({m['lo_band']}..{m['hi_band']}), "
            f"n={m['n_total']:,.0f}\n"
            f"  endpoints: {m['lo']:.4g}{unit} -> {m['hi']:.4g}{unit}  "
            f"(span {m['span_abs']:+.4g}{unit}, {m['span_pct']:+.1f}%)\n"
            f"  weighted slope: {m['slope_per_100']:+.4g}{unit}/100 Elo  "
            f"({m['slope_pct_per_100']:+.2f}% of mean per 100 Elo)")


PHASES = ["Opening", "Middlegame", "Endgame"]


def phase_ratio(sections):
    """Opening/midgame mean-emt ratio per band, plus the pooled ratio."""
    def _find(name):
        for k in sections:
            if k.lower().startswith(name.lower()):
                return sections[k]
        return {}
    op, mid = _find("Opening"), _find("Middlegame")
    out = {}
    for b in BAND_MID:
        a, c = op.get(b, {}).get("Mean emt"), mid.get(b, {}).get("Mean emt")
        if a is not None and c:
            out[b] = a / c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--a-label", default="A")
    ap.add_argument("--b-label", default="B")
    args = ap.parse_args()

    A, B = parse_sections(Path(args.a)), parse_sections(Path(args.b))

    print("=" * 72)
    print(f"PRE-COMMITTED READS: {args.b_label} vs {args.a_label}")
    print("=" * 72)

    for n, (col, unit, pred) in enumerate([
        ("Mean emt", "s",
         "PREDICTED: proportional span materially SMALLER than the reference. "
         "If it is >= the reference, the pace-decoupling hypothesis is WRONG."),
        ("Top-1 match", "",
         "PREDICTED: span LARGER than the reference (would make accuracy "
         "levers worth sweeping at this control)."),
    ], start=1):
        print(f"\n--- Read {n}: {col} across bands ---")
        print(f"{pred}\n")
        for label, S in ((args.a_label, A), (args.b_label, B)):
            print(f"{label}:")
            print(_fmt(measure(S, col), unit))
        ma, mb = measure(A, col), measure(B, col)
        if ma and mb:
            print(f"\n  => proportional span: {ma['span_pct']:+.1f}% "
                  f"({args.a_label}) vs {mb['span_pct']:+.1f}% ({args.b_label})")
            print(f"  => %/100 Elo:        {ma['slope_pct_per_100']:+.2f}% "
                  f"vs {mb['slope_pct_per_100']:+.2f}%")

    print("\n--- Read 3: per-phase mean emt (opening/midgame ratio) ---")
    print("PREDICTED: humans are NOT flat across controls here; the bot's")
    print("(base_time**0.2)/2 opening envelope makes it nearly so.\n")
    ra, rb = phase_ratio(A), phase_ratio(B)
    print(f"  {'band':<12} {args.a_label:>10} {args.b_label:>10}")
    for b in BAND_MID:
        if b in ra or b in rb:
            fa = f"{ra[b]:.3f}" if b in ra else "n/a"
            fb = f"{rb[b]:.3f}" if b in rb else "n/a"
            print(f"  {b:<12} {fa:>10} {fb:>10}")
    for label, r in ((args.a_label, ra), (args.b_label, rb)):
        if r:
            print(f"  pooled {label}: {sum(r.values())/len(r):.3f} "
                  f"(over {len(r)} bands)")


if __name__ == "__main__":
    main()
