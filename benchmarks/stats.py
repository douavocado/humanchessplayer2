"""Shared summary statistics.

Deliberately duplicates simulation/calibrate.py's percentile helper rather
than importing it: that module is a *calibration* tool whose output feeds
simulation/latency_model.py, and its JSON schema is load-bearing for the
simulator. Benchmarks must be free to change their own reporting without
risking that contract.
"""

from __future__ import annotations

import statistics


def summarise(vals):
    """Percentile summary of a list of samples (seconds, ms, whatever)."""
    vs = sorted(v for v in vals if v is not None)
    if not vs:
        return {"n": 0}

    def pct(p):
        return vs[min(len(vs) - 1, int(p / 100 * len(vs)))]

    return {
        "n": len(vs),
        "mean": statistics.fmean(vs),
        "sd": statistics.stdev(vs) if len(vs) > 1 else 0.0,
        "min": vs[0],
        "p50": pct(50),
        "p90": pct(90),
        "p99": pct(99),
        "max": vs[-1],
    }


def fmt_ms(s):
    """Format a summary whose samples are in seconds as a milliseconds row."""
    if not s or not s.get("n"):
        return "        n/a"
    return (f"{s['mean'] * 1000:8.1f}  {s['p50'] * 1000:8.1f}  "
            f"{s['p90'] * 1000:8.1f}  {s['max'] * 1000:8.1f}")
