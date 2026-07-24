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
from typing import Callable, Dict, FrozenSet, Optional, Tuple


@dataclass
class SiteActions:
    """
    The client capabilities a site needs in order to drive the UI.

    Interaction is split so that the *policy* the client owns stays in the
    client - the caps-lock human-interference guard, the log buffer, the
    humanised mouse movement - while only *where to click, and in what
    sequence* lives in the site. Passing these in rather than importing them
    keeps sites free of any client import.
    """

    click: Callable[..., None]
    sleep: Callable[[float], None]
    log: Callable[[str], None]


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

    #: Clock states worth probing to answer "is a game live on screen?".
    #: Lichess needs several because it moves the clock between the
    #: "waiting for first move" and "playing" layouts; a site whose clock
    #: never moves declares only ("play",) and a profile for it need only
    #: calibrate that one box.
    live_clock_states: Tuple[str, ...] = ("play", "start1", "start2")

    #: Clock states that only exist once a game has ended. Empty for a site
    #: where clock position carries no state, which is the same statement as
    #: clock_position_varies_by_state = False and is what makes the
    #: end-coordinate clock probe inapplicable there.
    end_clock_states: Tuple[str, ...] = ("end1", "end2", "end3")

    #: Whether the site has an arena berserk button / a back-to-lobby button.
    supports_berserk: bool = False
    supports_back_to_lobby: bool = False

    #: How far above the nominal start time a clock may read and still be a
    #: new game. A clock only ever counts *down*, so anything meaningfully
    #: above the start time is a different, longer time control - this only
    #: absorbs OCR and rounding slack.
    NEW_GAME_CLOCK_OVERSHOOT = 5

    #: How far the clock may already have run down. Detection is not
    #: instantaneous, and playing black our clock starts the moment the
    #: opponent moves, so by the first successful scan several seconds are
    #: routinely gone. The old symmetric +/- max(10, 10%) window treated
    #: "already ticked down" the same as "wrong time control" and threw away
    #: real games: a 1+0 game found nine seconds in sat right on the edge.
    #: The ratio still separates the time controls that matter - expecting
    #: 3+0 and finding a 1+0 board is rejected, and vice versa.
    NEW_GAME_CLOCK_MIN_RATIO = 0.4

    def clock_matches_new_game(self, seconds, expected_time):
        """Whether a clock reading is consistent with a game that just began."""
        if not seconds:                      # 0 is never a starting time
            return False
        if expected_time is None:
            return True
        if seconds > expected_time + self.NEW_GAME_CLOCK_OVERSHOOT:
            return False
        return seconds >= expected_time * self.NEW_GAME_CLOCK_MIN_RATIO

    _start_like_fens = None

    def start_like_board_fens(self):
        """
        Board placements that can legitimately be on screen when a new game
        is found: the starting position, or one white move into it.

        The second case is not an edge case - playing black, the opponent has
        usually moved (or premoved) before our first scan, so demanding the
        exact starting position means never detecting those games at all.
        Shared here rather than per-site because it is a fact about chess and
        scan timing, not about any one site's UI, and because keeping two
        copies is how the two implementations drift apart.
        """
        if self._start_like_fens is None:
            import chess
            fens = {chess.STARTING_BOARD_FEN}
            base = chess.Board()
            for move in list(base.legal_moves):
                base.push(move)
                fens.add(base.board_fen())
                base.pop()
            # set on the instance, not the class, so subclasses cannot share
            # a half-built cache
            self._start_like_fens = fens
        return self._start_like_fens

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

    # ------------------------------------------------------------------
    # interaction
    #
    # Each returns True if it acted. Returning False means "this site cannot
    # do that" and is a normal answer, not an error: a site with no arena has
    # no berserk button, and the caller should carry on rather than retry.
    # ------------------------------------------------------------------
    @abstractmethod
    def resign(self, actions: SiteActions) -> bool:
        """Click through this site's resign control."""

    @abstractmethod
    def start_new_game(self, actions: SiteActions, time_control: str) -> bool:
        """Click through this site's lobby to seek a game at `time_control`."""

    def berserk(self, actions: SiteActions) -> bool:
        """Click the arena berserk button. Sites without arenas need not."""
        return False

    def back_to_lobby(self, actions: SiteActions) -> bool:
        """Return to the arena lobby after a tournament game."""
        return False
