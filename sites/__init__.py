"""
Site registry.

A calibration profile names its site in `calibration_info.site`; this maps
that name onto the behaviour. Unknown names fall back to Lichess with a
warning rather than raising, so a typo in a profile degrades to the historic
behaviour instead of preventing the bot from starting.
"""

from .base import GameEndSignal, Site
from .chess_com import ChessComSite
from .lichess import LichessSite

__all__ = ["GameEndSignal", "Site", "LichessSite", "ChessComSite",
           "get_site", "get_site_for_config"]

_SITE_CLASSES = {
    LichessSite.name: LichessSite,
    ChessComSite.name: ChessComSite,
    # tolerate the spellings a hand-edited profile is likely to use
    "chesscom": ChessComSite,
    "chess.com": ChessComSite,
}

_INSTANCES = {}


def get_site(name):
    """Return the (cached) Site instance for a registry name."""
    key = (name or LichessSite.name).strip().lower()
    site_cls = _SITE_CLASSES.get(key)
    if site_cls is None:
        print(f"⚠️  Unknown site '{name}', falling back to {LichessSite.name}")
        site_cls = LichessSite
    if site_cls not in _INSTANCES:
        _INSTANCES[site_cls] = site_cls()
    return _INSTANCES[site_cls]


def get_site_for_config(config=None):
    """Return the Site named by the active calibration profile."""
    if config is None:
        from auto_calibration.config import get_config
        config = get_config()
    return get_site(config.get_site())
