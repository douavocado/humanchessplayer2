"""Download a Lichess user's games as PGN (with clocks) via the public API.

This removes the manual ``curl`` step: point it at an account and it streams
that user's rated games for one time-control category into a PGN the analyzer
can consume directly. No auth token is required for public game export, though
one raises the rate limit if you set LICHESS_TOKEN.
"""

from __future__ import annotations

import argparse
import os
import re
import urllib.request
from typing import Callable, Optional

_API = "https://lichess.org/api/games/user/{user}"
_TC_HEADER_RE = re.compile(r'^\[TimeControl\s+"(\d+)(?:\+(\d+))?"\]', re.M)


def parse_time_control(tc: str) -> tuple[int, int]:
    """'60+0' / '180+2' / '60' -> (base_seconds, increment). Raises ValueError."""
    m = re.fullmatch(r"(\d+)(?:\+(\d+))?", tc.strip())
    if not m:
        raise ValueError(f"Bad time control {tc!r}; expected e.g. '60+0' or '180+2'.")
    return int(m.group(1)), int(m.group(2) or 0)


def _filter_pgn_by_tc(pgn_text: str, tc: tuple[int, int]) -> tuple[str, int, int]:
    """Keep only games whose TimeControl header equals ``tc`` exactly.

    Returns (filtered_pgn, kept, dropped)."""
    blocks = re.split(r"(?m)^(?=\[Event )", pgn_text)
    kept: list[str] = []
    dropped = 0
    for block in blocks:
        if not block.strip():
            continue
        m = _TC_HEADER_RE.search(block)
        if m and (int(m.group(1)), int(m.group(2) or 0)) == tc:
            kept.append(block if block.endswith("\n\n") else block + "\n")
        else:
            dropped += 1
    return "".join(kept), len(kept), dropped


def fetch_user_games(
    username: str,
    out_path: str,
    max_games: int = 300,
    perf_type: str = "bullet",
    rated: bool = True,
    clocks: bool = True,
    time_control: Optional[str] = None,
    on_log: Optional[Callable[[str], None]] = None,
) -> int:
    """Stream a user's games to ``out_path`` as PGN. Returns games written.

    ``perf_type`` is a Lichess speed category (bullet/blitz/rapid/classical).
    ``time_control`` optionally narrows to one exact clock, e.g. "60+0" —
    the API can't filter that server-side, so up to ``max_games`` games of
    ``perf_type`` are downloaded and non-matching ones discarded locally
    (60+0 and 30+0 pacing differ ~2x, so mixing them muddies timing features).
    """
    def log(msg: str) -> None:
        if on_log:
            on_log(msg)

    tc = parse_time_control(time_control) if time_control else None

    params = [
        ("rated", "true" if rated else "false"),
        ("perfType", perf_type),
        ("clocks", "true" if clocks else "false"),
        ("evals", "false"),
        ("opening", "false"),
        ("max", str(max_games)),
    ]
    query = "&".join(f"{k}={v}" for k, v in params)
    url = f"{_API.format(user=username)}?{query}"

    req = urllib.request.Request(url, headers={
        "Accept": "application/x-chess-pgn",
        "User-Agent": "humanchessplayer2-cheat_detection",
    })
    token = os.environ.get("LICHESS_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    log(f"Requesting up to {max_games} {perf_type} games for '{username}'...")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    if tc is None:
        written = 0
        with urllib.request.urlopen(req, timeout=180) as resp, \
                open(out_path, "wb") as fh:
            for chunk in iter(lambda: resp.read(65536), b""):
                fh.write(chunk)
                written += chunk.count(b"[Event ")
        log(f"Wrote {written} games -> {out_path}")
        return written

    # Exact-TC mode: buffer, filter, then write only matching games.
    with urllib.request.urlopen(req, timeout=180) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    filtered, written, dropped = _filter_pgn_by_tc(text, tc)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(filtered)
    log(f"Wrote {written} games at {time_control} -> {out_path} "
        f"(dropped {dropped} other {perf_type} games)")
    if dropped and written < max_games:
        log(f"Note: only {written}/{max_games} matched {time_control}; "
            f"raise the fetch cap to download more candidates.")
    return written


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="cheat_detection.fetch_lichess",
        description="Download a Lichess user's games as PGN (with clocks).",
    )
    p.add_argument("--user", required=True, help="Lichess username")
    p.add_argument("--out", required=True, help="output PGN path")
    p.add_argument("--max-games", type=int, default=300)
    p.add_argument("--perf", default="bullet",
                   help="speed category: bullet/blitz/rapid/classical")
    p.add_argument("--tc", help="exact time control to keep, e.g. 60+0 "
                               "(filtered locally after download)")
    args = p.parse_args(argv)
    fetch_user_games(args.user, args.out, args.max_games, args.perf,
                     time_control=args.tc, on_log=print)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
