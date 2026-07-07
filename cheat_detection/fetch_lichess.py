"""Download a Lichess user's games as PGN (with clocks) via the public API.

This removes the manual ``curl`` step: point it at an account and it streams
that user's rated games for one time-control category into a PGN the analyzer
can consume directly. No auth token is required for public game export, though
one raises the rate limit if you set LICHESS_TOKEN.
"""

from __future__ import annotations

import argparse
import os
import urllib.request
from typing import Callable, Optional

_API = "https://lichess.org/api/games/user/{user}"


def fetch_user_games(
    username: str,
    out_path: str,
    max_games: int = 300,
    perf_type: str = "bullet",
    rated: bool = True,
    clocks: bool = True,
    on_log: Optional[Callable[[str], None]] = None,
) -> int:
    """Stream a user's games to ``out_path`` as PGN. Returns games written.

    ``perf_type`` is a Lichess speed category (bullet/blitz/rapid/classical).
    """
    def log(msg: str) -> None:
        if on_log:
            on_log(msg)

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
    written = 0
    with urllib.request.urlopen(req, timeout=180) as resp, \
            open(out_path, "wb") as fh:
        for chunk in iter(lambda: resp.read(65536), b""):
            fh.write(chunk)
            written += chunk.count(b"[Event ")

    log(f"Wrote {written} games -> {out_path}")
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
    args = p.parse_args(argv)
    fetch_user_games(args.user, args.out, args.max_games, args.perf, on_log=print)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
