"""
chess.com site behaviour.

The important divergence from Lichess is game-end detection. Lichess moves its
clock when a game ends, which is what makes "a clock reads at the end-state
coordinates" a usable signal there. chess.com's clock sits at identical
coordinates in every game state, so that test is true during ordinary play and
carries no information - it is therefore not merely reordered here, it is
absent. chess.com instead covers the board centre with a result modal, and
that modal is what this site keys off.
"""

import time

import chess
import cv2
import numpy as np

from chessimage.image_scrape_utils import (
    SCREEN_CAPTURE,
    START_X,
    START_Y,
    STEP,
    capture_board,
    capture_bottom_clock,
    capture_top_clock,
    get_fen_from_image,
    read_clock,
)

from .base import GameEndSignal, Site

# Modal geometry, as fractions of board size. Measured across the end-state
# screenshot corpus: the modal is horizontally fixed and vertically centred on
# the board centre, 0.214 x board size wide, with only its height varying by
# ending type (392-492px on an 1872px board). The search band is padded
# generously in y to contain the tallest of them.
MODAL_HALF_WIDTH_RATIO = 0.107
MODAL_HALF_HEIGHT_RATIO = 0.15

# A pixel is "modal dark" when every channel is below this. The modal body is
# near-black (~(33,36,38)); board squares are far brighter in at least one
# channel for every theme measured.
MODAL_DARK_MAX_CHANNEL = 70

# Fraction of the centre band that must be modal-dark to call the game over.
# Measured separation across the corpus: 0.401-0.647 for the eight ended
# games, 0.000-0.067 for the twenty-one in-play frames. This threshold sits in
# a 6x gap, so it is not finely tuned and should not need to be.
MODAL_DARK_FRACTION = 0.20

# Confidence needed to name which ending the modal shows.
TITLE_MATCH_THRESHOLD = 0.75

# --- lobby geometry -------------------------------------------------------
# Offsets are in board steps from the board origin, matching how the Lichess
# client has always expressed its lobby positions. Measured from the
# find_new_game_* screenshots. The left-nav entries are really anchored to the
# window rather than the board, so board-relative is a convention borrowed for
# consistency rather than a claim about how the page reflows.
PLAY_MENU_OFFSET = (-2.71, 0.06)        # left nav "Play"
PLAY_ONLINE_OFFSET = (9.14, 0.40)       # "Play Online" card on the /play panel
TIME_CONTROL_DROPDOWN_OFFSET = (9.21, 0.25)   # the "1 min (Bullet)" selector
START_GAME_FALLBACK_OFFSET = (9.21, 0.52)     # used only if colour search fails

# How long to wait for the New Game panel after asking for it. Page loads
# vary from instant to seconds, and every click after this point is at a
# fixed position, so a fixed sleep is the wrong tool - see
# _wait_for_new_game_panel.
NEW_GAME_PANEL_TIMEOUT = 6.0
NEW_GAME_PANEL_POLL = 0.4

# The time-control grid, once the dropdown is open: three columns by category
# row. chess.com's default grid has no 5+3, so that control is unreachable
# here without "More Time Controls".
_TC_COLUMN_OFFSETS = (8.58, 9.20, 9.83)
_TC_ROW_OFFSETS = {'bullet': 1.09, 'blitz': 1.45, 'rapid': 1.81}
TIME_CONTROL_CELLS = {
    "1+0":   (0, 'bullet'),
    "1+1":   (1, 'bullet'),
    "2+1":   (2, 'bullet'),
    "3+0":   (0, 'blitz'),
    "3+2":   (1, 'blitz'),
    "5+0":   (2, 'blitz'),
    "10+0":  (0, 'rapid'),
    "10+5":  (1, 'rapid'),
    "15+10": (2, 'rapid'),
}

# Search box for the green "Start Game" button, board-relative.
START_GAME_SEARCH_BOX = (8.27, -0.15, 1.88, 3.63)   # x, y, w, h in steps

# --- "are we on a game page at all?" --------------------------------------
# The move-navigation bar (|< < > > >|) under the moves list. It exists on
# every game page and on no lobby screen, which is the one thing the clock and
# the board cannot tell us apart: the lobby renders a full-time clock over a
# starting-position preview board, and so does a game that has just begun.
#
# Board-relative like the other panel geometry here, and measured across the
# corpus at 3840x2160 (x 2700-3180, y 2040-2102 in absolute pixels).
GAME_CONTROLS_BOX = (8.18, 7.93, 2.05, 0.27)   # x, y, w, h in steps

# The bar is read as "content brighter than the panel it sits on" rather than
# as an absolute colour, so a theme change moves both together. Measured
# separation over the corpus: 0.063-0.135 of the band on every game page
# (playing, ended, aborted), and 0.000 on every lobby screen - the lobby panel
# is *perfectly* flat there, so the threshold is only guarding against stray
# antialiasing, not splitting a distribution.
GAME_CONTROLS_CONTRAST = 12
GAME_CONTROLS_MIN_FRACTION = 0.03

# --- "is a game still being played?" ---------------------------------------
# Asked before seeking, where the answer has to be liveness, not "a clock is
# readable" - chess.com's clock keeps showing the final time on a finished
# board, so the readable test is true forever after a game ends. A ticking
# clock is the one unambiguous statement that a game is in progress: at least
# one side's clock always runs during play (white's from the moment the game
# starts), and both freeze the instant it ends.
#
# The wait is affordable *here* - it happens between games - which is why the
# same test is deliberately not used in detect_new_game, where it would cost
# a second of our own clock at the start of every game.
CLOCK_TICK_WAIT = 1.2
# A tick over that wait is a second or two. Anything larger is not a tick but
# a different clock (a new game, a misread), and must not read as liveness.
CLOCK_TICK_MAX_DROP = 6


class ChessComSite(Site):
    name = "chess_com"

    # The clock never moves between game states, so clock position cannot be
    # used to infer that a game has started or ended. There is exactly one
    # clock box to calibrate and no end-state box at all, which is why
    # end_clock_states is empty rather than a repeat of the play coordinates.
    clock_position_varies_by_state = False
    live_clock_states = ("play",)
    end_clock_states = ()

    # No arena format, so neither control exists.
    supports_berserk = False
    supports_back_to_lobby = False

    # A queued premove is drawn at its destination square before the server
    # confirms it, so the scraped position can legitimately run ahead of the
    # confirmed game state.
    renders_premoves_on_board = True

    # chess.com's modal title is player-relative when we win ("You Won!") but
    # colour-relative when we lose ("White Won"), so the logical names are not
    # symmetric. Every entry is optional - naming the ending is a bonus, since
    # modal *presence* already answers whether the game is over.
    result_template_files = {
        'we_won':    'we_won_result.png',
        'white_won': 'whitewin_result.png',
        'black_won': 'blackwin_result.png',
        'draw':      'draw_result.png',
        'aborted':   'aborted_result.png',
    }
    optional_result_templates = frozenset(result_template_files)

    def __init__(self):
        self._title_refs = None

    # ------------------------------------------------------------------
    # templates
    # ------------------------------------------------------------------
    def _title_references(self):
        if self._title_refs is None:
            refs = []
            try:
                from auto_calibration.config import get_config
                from auto_calibration.template_extractor import TemplateExtractor
                extractor = TemplateExtractor(
                    template_dir=str(get_config().get_template_dir()))
                templates = extractor.load_result_templates(self.result_template_files)
                for name, ref in (templates or {}).items():
                    if ref is not None:
                        refs.append((name, cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)))
            except Exception:
                refs = []
            self._title_refs = refs
        return self._title_refs

    # ------------------------------------------------------------------
    # modal
    # ------------------------------------------------------------------
    def _centre_band(self):
        """The board-centre crop the modal would occupy, or None."""
        board = capture_board()
        if board is None or board.size == 0:
            return None
        h, w = board.shape[:2]
        half_w = int(w * MODAL_HALF_WIDTH_RATIO)
        half_h = int(h * MODAL_HALF_HEIGHT_RATIO)
        cx, cy = w // 2, h // 2
        return board[cy - half_h:cy + half_h, cx - half_w:cx + half_w]

    @staticmethod
    def _dark_fraction(band):
        return float((band.max(axis=2) < MODAL_DARK_MAX_CHANNEL).mean())

    def _identify_ending(self, band):
        """Best-matching modal title within the band, or None."""
        refs = self._title_references()
        if not refs:
            return None, 0.0
        try:
            band_gray = cv2.cvtColor(np.ascontiguousarray(band), cv2.COLOR_BGR2GRAY)
        except Exception:
            return None, 0.0
        best_name, best_score = None, 0.0
        for name, ref in refs:
            if band_gray.shape[0] < ref.shape[0] or band_gray.shape[1] < ref.shape[1]:
                continue
            score = float(cv2.matchTemplate(band_gray, ref, cv2.TM_CCOEFF_NORMED).max())
            if score > best_score:
                best_name, best_score = name, score
        if best_score > TITLE_MATCH_THRESHOLD:
            return best_name, best_score
        return None, best_score

    # ------------------------------------------------------------------
    # new game
    # ------------------------------------------------------------------
    def detect_new_game(self, expected_time=None):
        """
        A new game is a readable clock at the (only) clock position, showing
        the expected starting time, over a starting-position board, on a page
        that is actually a game.

        Unlike Lichess there is no start-state clock position to look for and
        so no vertical-offset check to distinguish it from the play clock -
        the board and the game-page test are doing that work here instead.
        """
        res = read_clock(capture_bottom_clock(state="play"))
        if res is None:
            return None
        if not self.clock_matches_new_game(res, expected_time):
            return None

        # A start-like board plus a full clock is not enough on its own here,
        # because two screens that are not a live game look exactly like one.
        # An *aborted* game leaves the starting position on the board with the
        # clock still at full time, and the lobby renders a preview board at
        # the starting position beside a clock showing the selected time
        # control. Lichess separates these with its start-state clock
        # coordinates; chess.com's clock never moves, so the discriminators
        # have to come from the rest of the screen.
        if self.game_over_screen_visible():
            return None

        # Positive evidence that this is a game page, not a lobby one. The
        # Start Game button alone is not that evidence: it is absent on the
        # /play panel once a seek is running ("Searching.." replaces it) and
        # on any other lobby state that does not happen to show it, and the
        # board and clock look identical throughout. Both were live false
        # positives - the bot "found" a game on the Play Online page and again
        # while queueing, then played into a board nobody was moving.
        if self._game_controls_visible() is False:
            return None
        if self._find_start_game_button() is not None:
            # The Start Game button only exists in the lobby.
            return None

        try:
            start_like = self.start_like_board_fens()
            board_img = capture_board()
            for bottom in ("w", "b"):
                fen = get_fen_from_image(board_img, bottom=bottom, fast_mode=True)
                if chess.Board(fen).board_fen() in start_like:
                    return res
        except Exception:
            return None
        return None

    # ------------------------------------------------------------------
    # is a game still being played?
    # ------------------------------------------------------------------
    def _read_clocks(self):
        readings = []
        for capture in (capture_bottom_clock, capture_top_clock):
            try:
                readings.append(read_clock(capture(state="play")))
            except Exception:
                readings.append(None)
        return readings

    def _clock_is_ticking(self, wait=CLOCK_TICK_WAIT):
        """Whether either clock counts down over `wait` seconds."""
        before = self._read_clocks()
        if all(reading is None for reading in before):
            return False
        time.sleep(wait)
        after = self._read_clocks()
        for was, now in zip(before, after):
            if was is None or now is None:
                continue
            if 0 < was - now <= CLOCK_TICK_MAX_DROP:
                return True
        return False

    def live_game_on_screen(self):
        """
        Whether a game is still being played, for the pre-seek guard.

        The inherited "a clock is readable at a live clock position" test
        cannot answer this here: chess.com has one clock position, it never
        moves, and a finished game goes on showing the time it ended with. So
        the test was true on every dead board, and the only thing standing
        between it and a refused seek was the result modal - which chess.com
        replaces with its analysis panel a few seconds after the game ends.
        The bot therefore refused to seek after most of its games, and only
        got away with it on the *retry* after the new-game wait timed out.

        Ticking is asked instead, which is what the guard actually means.
        """
        # Conclusive and free, and it skips the wait below.
        if self.game_over_screen_visible():
            return None
        if not self._clock_is_ticking():
            return None
        return "a clock is ticking at the play position"

    # ------------------------------------------------------------------
    # interaction
    # ------------------------------------------------------------------
    def resign(self, actions):
        """
        Click the resign control, whose position comes from the profile.

        A single click resigns outright on chess.com - there is no
        confirmation step to handle.
        """
        try:
            from auto_calibration.config import get_config
            x, y = get_config().get_resign_button_position()
        except Exception:
            actions.log("chess.com resign: no calibrated resign button position. \n")
            return False
        actions.log(f"chess.com resign click at ({x}, {y}). \n")
        actions.click(x, y, tolerance=10, clicks=1, duration=np.random.uniform(0.3, 0.7))
        return True

    @staticmethod
    def _at(offset):
        dx, dy = offset
        return START_X + dx * STEP, START_Y + dy * STEP

    def _find_start_game_button(self):
        """
        Locate the green "Start Game" button by colour.

        Its position is not fixed: opening the time-control dropdown pushes it
        down the panel (measured at y=308 collapsed vs y=788 expanded), and
        whether selecting a control collapses the dropdown again is not
        something the screenshots settle. Searching for the button rather than
        assuming a position sidesteps that entirely - and a click that missed
        would land on the variant selector, silently changing the game type.
        """
        x0 = int(START_X + START_GAME_SEARCH_BOX[0] * STEP)
        y0 = int(START_Y + START_GAME_SEARCH_BOX[1] * STEP)
        w = int(START_GAME_SEARCH_BOX[2] * STEP)
        h = int(START_GAME_SEARCH_BOX[3] * STEP)
        try:
            panel = SCREEN_CAPTURE.capture((x0, y0, w, h)).copy()[:, :, :3]
        except Exception:
            return None
        b = panel[:, :, 0].astype(int)
        g = panel[:, :, 1].astype(int)
        r = panel[:, :, 2].astype(int)
        # chess.com's action green: green channel dominant, mid-brightness.
        mask = ((g > 110) & (g < 200) & (g - b > 40) & (g - r > 25)).astype(np.uint8) * 255
        n_labels, _labels, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
        min_area = 0.03 * STEP * STEP
        best = None
        for i in range(1, n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area and (best is None or area > best[0]):
                best = (area, i)
        if best is None:
            return None
        _area, i = best
        bx, by, bw, bh = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                          int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        return x0 + bx + bw // 2, y0 + by + bh // 2

    def _game_controls_visible(self):
        """
        Whether the move-navigation bar is on screen, i.e. whether this is a
        game page at all.

        Returns None - not False - when the band could not be captured, so
        that a window narrower than the calibration (where the panel is off
        the captured area entirely) degrades to the old behaviour rather than
        wedging the bot into never finding a game.
        """
        x0 = int(START_X + GAME_CONTROLS_BOX[0] * STEP)
        y0 = int(START_Y + GAME_CONTROLS_BOX[1] * STEP)
        w = int(GAME_CONTROLS_BOX[2] * STEP)
        h = int(GAME_CONTROLS_BOX[3] * STEP)
        try:
            band = SCREEN_CAPTURE.capture((x0, y0, w, h))
        except Exception:
            return None
        if band is None or band.size == 0:
            return None
        band = band.copy()[:, :, :3]
        # A clipped capture is not the region we measured the thresholds on.
        if band.shape[0] < 0.8 * h or band.shape[1] < 0.8 * w:
            return None
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY).astype(np.int16)
        background = np.median(gray)
        fraction = float((gray >= background + GAME_CONTROLS_CONTRAST).mean())
        return fraction >= GAME_CONTROLS_MIN_FRACTION

    def _wait_for_new_game_panel(self, actions, timeout=NEW_GAME_PANEL_TIMEOUT):
        """
        Wait for the New Game panel, returning the Start Game button position.

        The green Start Game button exists only on that panel, so finding it
        is the same question as "did the page we asked for actually load?".
        Everything clicked after this point is at a fixed board-relative
        position, so getting this wrong does not merely miss - it clicks
        whatever else the page happens to be showing there.
        """
        deadline = time.time() + timeout
        while True:
            found = self._find_start_game_button()
            if found is not None:
                return found
            if time.time() >= deadline:
                return None
            actions.sleep(NEW_GAME_PANEL_POLL)

    def start_new_game(self, actions, time_control="1+0"):
        """
        Seek a game: Play -> Play Online -> time-control dropdown -> the
        control -> Start Game.

        "Play" in the left nav does open a hover submenu containing its own
        Play Online entry, but clicking it navigates to /play, and the
        submenu goes with the old page. The Play Online that matters is
        therefore the card on the right-hand panel of /play, not the submenu
        entry - driving the submenu left the bot clicking a menu that was no
        longer on screen and it never reached the time-control grid.

        Unsupported time controls are refused rather than approximated. The
        default grid has no 5+3, and clicking the nearest cell would silently
        start a game at the wrong time control, which would then mis-pace
        every move the engine makes.
        """
        cell = TIME_CONTROL_CELLS.get(time_control)
        if cell is None:
            actions.log(
                "chess.com: time control {} is not on the default grid; not seeking. \n".format(
                    time_control))
            return False

        def click(pos, tolerance=10):
            actions.click(pos[0], pos[1], tolerance=tolerance, clicks=1,
                          duration=np.random.uniform(0.3, 0.7))

        # 1. left nav "Play" -> the /play page and its right-hand panel
        click(self._at(PLAY_MENU_OFFSET))
        actions.sleep(0.8)

        # 2. "Play Online" on that panel opens the New Game panel. Skipped if
        #    that panel is already up, which is the normal case when a
        #    previous seek left us there - clicking Play Online again from
        #    the New Game panel would land on the time-control selector.
        if self._find_start_game_button() is None:
            click(self._at(PLAY_ONLINE_OFFSET))
            if self._wait_for_new_game_panel(actions) is None:
                actions.log(
                    "chess.com: New Game panel never appeared after Play -> Play Online; "
                    "not seeking rather than clicking blind. \n")
                return False

        # 3. open the time-control dropdown
        click(self._at(TIME_CONTROL_DROPDOWN_OFFSET))
        actions.sleep(0.8)

        # 4. the time control itself
        column, row = cell
        click((START_X + _TC_COLUMN_OFFSETS[column] * STEP,
               START_Y + _TC_ROW_OFFSETS[row] * STEP), tolerance=15)
        actions.sleep(0.5)

        # 5. Start Game, located by colour because step 3 moves it
        start = self._find_start_game_button()
        if start is None:
            start = self._at(START_GAME_FALLBACK_OFFSET)
            actions.log("chess.com: Start Game button not found by colour, using fallback position. \n")
        else:
            actions.log("chess.com: found Start Game button at {}. \n".format(start))
        click(start)
        return True

    def game_over_screen_visible(self):
        try:
            band = self._centre_band()
            if band is None or band.size == 0:
                return False
            return self._dark_fraction(band) >= MODAL_DARK_FRACTION
        except Exception:
            return False

    def detect_game_end(self, fens=None, arena=False):
        fens = fens or []

        # Board outcome first - site-independent and the most reliable signal
        # when it is available at all. It cannot fire for an aborted game,
        # where the board is still the starting position.
        if len(fens) > 0:
            try:
                outcome = chess.Board(fens[-1]).outcome()
                if outcome is not None:
                    return GameEndSignal(
                        method="board_outcome",
                        detail={'termination': outcome.termination, 'fen': fens[-1]},
                    )
            except Exception:
                pass

        # The result modal covering the board centre. Presence answers whether
        # the game ended; the title only names which ending it was, so an
        # unrecognised or missing title still ends the game rather than
        # leaving the bot playing on into a dead board.
        try:
            band = self._centre_band()
            if band is None or band.size == 0:
                return None
            dark = self._dark_fraction(band)
            if dark < MODAL_DARK_FRACTION:
                return None
            name, score = self._identify_ending(band)
            return GameEndSignal(
                method="modal",
                result=name,
                detail={'dark_fraction': round(dark, 3), 'title_score': round(score, 3)},
            )
        except Exception:
            return None
