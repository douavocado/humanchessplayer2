"""Filter a Lichess database dump down to a usable human baseline corpus.

Lichess monthly dumps (https://database.lichess.org) mix every time control and
rating together and are huge, so feed one through this streaming filter to keep
only games that match your bot's time control and rating band. It reads a PGN
stream (typically piped from ``zstd -dc``) and writes matching games until a
cap is reached -- header-level only, so it never fully parses move text and stays
fast on multi-GB inputs.

Example:

    zstd -dc lichess_db_standard_rated_2024-01.pgn.zst | \
        venv/bin/python -m cheat_detection.fetch_corpus \
            --category blitz --rating 1800 2100 --max-games 4000 \
            --out cheat_detection/corpora/blitz_1800_2100.pgn
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Optional, TextIO

_HEADER_RE = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$')
_TC_RE = re.compile(r"^(\d+)(?:\+(\d+))?")

# Lichess time-control categories by estimated duration = base + 40 * increment.
_CATEGORY_BOUNDS = {
    "bullet": (0, 179),
    "blitz": (180, 479),
    "rapid": (480, 1499),
    "classical": (1500, 10 ** 9),
}


def _estimated_duration(base: int, inc: int) -> int:
    return base + 40 * inc


def _parse_tc(tc: str) -> Optional[tuple[int, int]]:
    if not tc or tc in ("-", "?"):
        return None
    m = _TC_RE.match(tc.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2) or 0)


def _headers_of(block: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in block.splitlines():
        if not line.startswith("["):
            break  # headers are contiguous at the top
        m = _HEADER_RE.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _keep(headers: dict[str, str], text: str, args) -> bool:
    # Rating band on both players.
    try:
        welo = int(headers.get("WhiteElo", ""))
        belo = int(headers.get("BlackElo", ""))
    except ValueError:
        return False
    lo, hi = args.rating
    if not (lo <= welo <= hi and lo <= belo <= hi):
        return False

    # Time control.
    tc = _parse_tc(headers.get("TimeControl", ""))
    if tc is None:
        return False
    base, inc = tc
    if args.category:
        clo, chi = _CATEGORY_BOUNDS[args.category]
        if not (clo <= _estimated_duration(base, inc) <= chi):
            return False
    if args.base_min is not None and base < args.base_min:
        return False
    if args.base_max is not None and base > args.base_max:
        return False

    # Clocks required (needed for the timing features).
    if not args.allow_no_clock and "%clk" not in text:
        return False

    # Skip abandoned / no-move games.
    if headers.get("Termination", "") == "Abandoned":
        return False
    return True


def filter_stream(inp: TextIO, out: TextIO, args) -> int:
    kept = 0
    scanned = 0
    buf: list[str] = []

    def flush() -> bool:
        nonlocal kept, scanned
        if not buf:
            return False
        text = "".join(buf)
        scanned += 1
        if _keep(_headers_of(text), text, args):
            out.write(text)
            if not text.endswith("\n\n"):
                out.write("\n")
            kept += 1
            return True
        return False

    for line in inp:
        if line.startswith("[Event ") and buf:
            flush()
            if kept >= args.max_games:
                buf = []
                break
            buf = [line]
        else:
            buf.append(line)
        if scanned and scanned % 100000 == 0 and buf and line.startswith("[Event "):
            print(f"  scanned {scanned}, kept {kept}...", file=sys.stderr, flush=True)
    else:
        if kept < args.max_games:
            flush()

    print(f"Kept {kept} games (scanned {scanned}).", file=sys.stderr)
    return kept


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="cheat_detection.fetch_corpus",
        description="Filter a Lichess PGN dump to a rating band + time control.",
    )
    p.add_argument("--in", dest="infile", help="input PGN (default: stdin)")
    p.add_argument("--out", required=True, help="output filtered PGN")
    p.add_argument("--rating", type=int, nargs=2, required=True, metavar=("MIN", "MAX"),
                   help="keep games where BOTH players' Elo is in this band")
    p.add_argument("--category", choices=sorted(_CATEGORY_BOUNDS),
                   help="time-control category (bullet/blitz/rapid/classical)")
    p.add_argument("--base-min", type=int, help="min base seconds (alternative to --category)")
    p.add_argument("--base-max", type=int, help="max base seconds")
    p.add_argument("--max-games", type=int, default=5000, help="stop after keeping this many")
    p.add_argument("--allow-no-clock", action="store_true",
                   help="keep games without clock tags (timing features will be sparse)")
    args = p.parse_args(argv)

    inp = open(args.infile, "r", encoding="utf-8", errors="replace") if args.infile else sys.stdin
    try:
        with open(args.out, "w", encoding="utf-8") as out:
            filter_stream(inp, out, args)
    finally:
        if args.infile:
            inp.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
