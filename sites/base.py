"""
Site behaviour: what differs between chess sites, as opposed to what differs
between screens.

A calibration profile (auto_calibration/) answers "where are things on this
user's screen?" - board origin, clock boxes, piece and digit templates,
colours. A Site answers "how does this site behave?" - how a game start and
end are recognised, whether clock position carries state, how the result is
presented. The two axes are independent: the same monitor may be used for two
sites, and the same site runs on many screens. A profile therefore *binds* a
device layout to a site (via `calibration_info.site`) rather than implying one.

Sites deliberately import screen-capture helpers from
chessimage.image_scrape_utils directly rather than being handed a context
object. Those helpers are already profile-driven and live below the clients in
the import graph, so there is no cycle to avoid and no indirection to justify.
What sites must NOT do is import a client: detection returns a description of
what it found and lets the caller do the logging, debug-screenshot writing and
state mutation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional


@dataclass
class GameEndSignal:
    """
    Why a site concluded the game is over.

    Carries the deciding signal so the caller can log it and write the debug
    screenshot it always has - keeping that side-effecting work in the client
    rather than in detection.
    """

    method: str
    result: Optional[str] = None
    detail: Dict = field(default_factory=dict)

    def describe(self) -> str:
        bits = [self.method]
        if self.result:
            bits.append(f"result={self.result}")
        bits.extend(f"{k}={v}" for k, v in self.detail.items())
        return " ".join(bits)


class Site(ABC):
    """Base class for site-specific behaviour. See module docstring."""

    #: Registry key, and the value expected in `calibration_info.site`.
    name: str = "base"

    #: Whether the clock moves to different screen coordinates when the game
    #: state changes. True on Lichess, which is what makes "a clock is
    #: readable at the end-state coordinates" meaningful there. False on
    #: chess.com, where the clock never moves and that test carries no
    #: information at all.
    clock_position_varies_by_state: bool = True

    #: Whether a queued premove is drawn on the board at its destination
    #: before the server has confirmed it - meaning the scraped position can
    #: legitimately differ from the confirmed game state.
    renders_premoves_on_board: bool = False

    #: Logical result name -> template filename within <templates>/results/.
    result_template_files: Dict[str, str] = {}

    #: Result names that may be absent without that being a calibration fault.
    optional_result_templates: FrozenSet[str] = frozenset()

    @abstractmethod
    def detect_new_game(self, expected_time: Optional[float] = None) -> Optional[int]:
        """Return our starting time in seconds if a new game is on screen."""

    @abstractmethod
    def game_over_screen_visible(self) -> bool:
        """
        Whether an end-of-game screen is showing, ignoring any clock-derived
        signal.

        Used before seeking a new game, where a readable clock is what raised
        the question in the first place, so a clock-based end test would be
        circular.
        """

    @abstractmethod
    def detect_game_end(self, fens=None, arena: bool = False) -> Optional[GameEndSignal]:
        """
        Return a GameEndSignal if the game has ended, else None.

        Args:
            fens: the game's FEN history, most recent last. Used for the
                board-outcome check and passed through to debug output.
            arena: whether this is an arena game, which can move the result
                region.
        """
