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

Multi-band mode fills several rating bands in ONE streaming pass — repeatable
``--band MIN MAX COUNT`` (MAX may be ``+`` for no upper limit; band maxima are
exclusive so adjacent bands don't overlap). Each band gets its own output file
derived from ``--out``, e.g. ``--out corpora/bullet_1plus0.pgn --band 2000 2300
17500 --band 2300 + 30000`` writes ``bullet_1plus0_2000_2300.pgn`` and
``bullet_1plus0_2300_plus.pgn``. A game counts for the first unfilled band
containing BOTH players' ratings.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
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


@dataclass
class Band:
    lo: int
    hi: Optional[int]      # inclusive max; None = unbounded
    quota: int
    path: str
    kept: int = 0
    out: Optional[TextIO] = None  # opened lazily on first kept game

    def label(self) -> str:
        return f"{self.lo}-{self.hi if self.hi is not None else '+'}"

    def _in(self, elo: int) -> bool:
        return elo >= self.lo and (self.hi is None or elo <= self.hi)

    def contains(self, welo: int, belo: int, any_player: bool = False) -> bool:
        if any_player:
            return self._in(welo) or self._in(belo)
        return self._in(welo) and self._in(belo)


def _elos_if_eligible(headers: dict[str, str], text: str, args) -> Optional[tuple[int, int]]:
    """(white_elo, black_elo) if the game passes every non-rating filter."""
    try:
        welo = int(headers.get("WhiteElo", ""))
        belo = int(headers.get("BlackElo", ""))
    except ValueError:
        return None

    # Time control.
    tc = _parse_tc(headers.get("TimeControl", ""))
    if tc is None:
        return None
    base, inc = tc
    if getattr(args, "tc_exact", None) and tc != args.tc_exact:
        return None
    if args.category:
        clo, chi = _CATEGORY_BOUNDS[args.category]
        if not (clo <= _estimated_duration(base, inc) <= chi):
            return None
    if args.base_min is not None and base < args.base_min:
        return None
    if args.base_max is not None and base > args.base_max:
        return None

    # Clocks required (needed for the timing features).
    if not args.allow_no_clock and "%clk" not in text:
        return None

    # Skip abandoned / no-move games.
    if headers.get("Termination", "") == "Abandoned":
        return None
    return welo, belo


def filter_stream(inp: TextIO, bands: list[Band], args) -> int:
    scanned = 0
    buf: list[str] = []

    def flush() -> None:
        nonlocal scanned
        if not buf:
            return
        text = "".join(buf)
        scanned += 1
        elos = _elos_if_eligible(_headers_of(text), text, args)
        if elos is None:
            return
        for band in bands:
            if band.kept >= band.quota or not band.contains(*elos, args.any_player):
                continue
            if band.out is None:
                band.out = open(band.path, "w", encoding="utf-8")
            band.out.write(text)
            if not text.endswith("\n\n"):
                band.out.write("\n")
            band.kept += 1
            break  # a game feeds at most one band

    def status() -> str:
        return ", ".join(f"{b.label()}: {b.kept}/{b.quota}" for b in bands)

    try:
        for line in inp:
            if line.startswith("[Event ") and buf:
                flush()
                if all(b.kept >= b.quota for b in bands):
                    buf = []
                    break
                buf = [line]
            else:
                buf.append(line)
            if scanned and scanned % 100000 == 0 and buf and line.startswith("[Event "):
                print(f"  scanned {scanned} | {status()}", file=sys.stderr, flush=True)
        else:
            flush()
    finally:
        for band in bands:
            if band.out is not None:
                band.out.close()

    print(f"Done (scanned {scanned}) | {status()}", file=sys.stderr)
    return sum(b.kept for b in bands)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="cheat_detection.fetch_corpus",
        description="Filter a Lichess PGN dump to a rating band + time control.",
    )
    p.add_argument("--in", dest="infile", help="input PGN (default: stdin)")
    p.add_argument("--out", required=True,
                   help="output filtered PGN; in --band mode this is the stem the "
                        "per-band file names are derived from")
    p.add_argument("--rating", type=int, nargs=2, metavar=("MIN", "MAX"),
                   help="keep games where BOTH players' Elo is in this band "
                        "(inclusive); alternative to --band")
    p.add_argument("--band", action="append", nargs=3, metavar=("MIN", "MAX", "COUNT"),
                   help="repeatable: fill a rating band with COUNT games (both "
                        "players in [MIN, MAX); MAX may be '+' for unbounded); "
                        "each band writes <out-stem>_<MIN>_<MAX>.pgn")
    p.add_argument("--category", choices=sorted(_CATEGORY_BOUNDS),
                   help="time-control category (bullet/blitz/rapid/classical)")
    p.add_argument("--tc", help="exact time control, e.g. 60+0 — keeps only that clock "
                                "so pacing matches the bot's (30+0 vs 60+0 differ ~2x)")
    p.add_argument("--base-min", type=int, help="min base seconds (alternative to --category)")
    p.add_argument("--base-max", type=int, help="max base seconds")
    p.add_argument("--max-games", type=int, default=5000, help="stop after keeping this many")
    p.add_argument("--any-player", action="store_true",
                   help="a band matches when at least ONE player's Elo is in it "
                        "(default: both must be). The off-band player's moves are "
                        "excluded later by the analysis --rating filter, so this "
                        "widens the corpus without polluting the baseline.")
    p.add_argument("--allow-no-clock", action="store_true",
                   help="keep games without clock tags (timing features will be sparse)")
    args = p.parse_args(argv)
    args.tc_exact = _parse_tc(args.tc) if args.tc else None
    if args.tc and args.tc_exact is None:
        p.error(f"bad --tc {args.tc!r}; expected e.g. 60+0")
    if bool(args.rating) == bool(args.band):
        p.error("give exactly one of --rating or --band")

    if args.band:
        stem, ext = os.path.splitext(args.out)
        bands = []
        for lo_s, hi_s, count_s in args.band:
            lo, quota = int(lo_s), int(count_s)
            unbounded = hi_s in ("+", "inf")
            hi = None if unbounded else int(hi_s) - 1  # exclusive max -> inclusive
            suffix = "plus" if unbounded else hi_s
            bands.append(Band(lo, hi, quota, f"{stem}_{lo_s}_{suffix}{ext or '.pgn'}"))
    else:
        bands = [Band(args.rating[0], args.rating[1], args.max_games, args.out)]

    inp = open(args.infile, "r", encoding="utf-8", errors="replace") if args.infile else sys.stdin
    try:
        filter_stream(inp, bands, args)
    finally:
        if args.infile:
            inp.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
