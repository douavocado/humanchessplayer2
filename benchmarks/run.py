"""Benchmark CLI.

    venv/bin/python -m benchmarks.run all --label thinkpad
    venv/bin/python -m benchmarks.run compute --repeat 5
    venv/bin/python -m benchmarks.run compare benchmarks/results/*.json

Results land in benchmarks/results/ as JSON. They are not gitignored, on
purpose: collecting several devices in one place is the whole point, and
committing them is the simplest way to get another machine's numbers next
to yours. They are measurements of a machine, not configuration, so
nothing reads them at runtime -- unlike simulation/calibration/*.json,
which the simulator does read and which stays gitignored.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import platform
import socket
import sys

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def _default_out(label):
    stamp = datetime.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in label)
    return os.path.join(RESULTS_DIR, f"{safe}-{stamp}.json")


def _write(payload, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\nWrote {path}")


def _suites(args):
    """Run the requested suites and assemble the payload."""
    from benchmarks import hardware

    which = args.cmd
    payload = {
        "label": args.label,
        "hostname": socket.gethostname(),
        "node": platform.node(),
        "created": datetime.datetime.now().astimezone().isoformat(
            timespec="seconds"),
        "suites": [],
    }
    print("Machine fingerprint...", flush=True)
    payload["hardware"] = hardware.describe()
    print("  {} / {} cores / {} GB".format(
        payload["hardware"]["cpu"],
        payload["hardware"]["cpu_count_logical"],
        payload["hardware"]["ram_gb"]), flush=True)

    if which in ("all", "compute"):
        from benchmarks import compute
        print("\nCompute suite...", flush=True)
        payload["compute"] = compute.run(repeat=args.repeat)
        payload["suites"].append("compute")
        print(compute.report(payload["compute"]))

    if which in ("all", "vision"):
        from benchmarks import vision
        print("\nVision suite...", flush=True)
        payload["vision"] = vision.run(repeats=args.vision_repeats)
        payload["suites"].append("vision")
        print(vision.report(payload["vision"]))

    if which in ("all", "mouse"):
        from benchmarks import mouse
        print("\nMouse suite...", flush=True)
        payload["mouse"] = mouse.run(include_live=args.mouse_live)
        payload["suites"].append("mouse")
        print(mouse.report(payload["mouse"]))

    return payload


def cmd_run(args):
    payload = _suites(args)
    _write(payload, args.out or _default_out(args.label))
    return 0


def cmd_compare(args):
    from benchmarks import compare

    paths = []
    for pattern in args.results:
        matched = sorted(glob.glob(pattern))
        paths.extend(matched or [pattern])
    if len(paths) < 1:
        print("nothing to compare", file=sys.stderr)
        return 1
    results = compare.load(paths)
    print(compare.report(results))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="benchmarks.run",
        description="Measure per-device compute, vision and mouse cost")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_run_parser(name, help_text):
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("--label", default=socket.gethostname(),
                        help="name for this device in comparisons")
        sp.add_argument("--out", help="output JSON path")
        sp.add_argument("--repeat", type=int, default=3,
                        help="passes over the position corpus (compute)")
        sp.add_argument("--vision-repeats", type=int, default=20,
                        help="timed calls per vision stage")
        sp.add_argument("--mouse-live", action="store_true",
                        help="also measure the real cursor (MOVES YOUR MOUSE)")
        sp.set_defaults(func=cmd_run)
        return sp

    add_run_parser("all", "every suite")
    add_run_parser("compute", "engine + neural network only")
    add_run_parser("vision", "screen capture + board recognition only")
    add_run_parser("mouse", "mouse gesture timing only")

    pc = sub.add_parser("compare", help="line up two or more result JSONs")
    pc.add_argument("results", nargs="+",
                    help="result JSON paths or globs; the first is baseline")
    pc.set_defaults(func=cmd_compare)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
