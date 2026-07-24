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

import chess
import cv2
import numpy as np

from chessimage.image_scrape_utils import (
    capture_board,
    capture_bottom_clock,
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
        the expected starting time, over a starting-position board.

        Unlike Lichess there is no start-state clock position to look for and
        so no vertical-offset check to distinguish it from the play clock -
        the board is doing that work here instead.
        """
        res = read_clock(capture_bottom_clock(state="play"))
        if res is None:
            return None

        if expected_time is not None:
            if res == 0 or abs(res - expected_time) > max(10, expected_time * 0.1):
                return None
        elif res == 0:
            return None

        try:
            board_img = capture_board()
            for bottom in ("w", "b"):
                fen = get_fen_from_image(board_img, bottom=bottom, fast_mode=True)
                if chess.Board(fen).board_fen() == chess.STARTING_BOARD_FEN:
                    return res
        except Exception:
            return None
        return None

    # ------------------------------------------------------------------
    # game end
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # interaction
    # ------------------------------------------------------------------
    def resign(self, actions):
        """
        Click the resign control, whose position comes from the profile.

        ⚠️ Unverified against a live session. The button coordinate was
        measured from a real chess.com screenshot, but whether chess.com then
        raises a confirmation dialog - and where its confirm button sits -
        has not been observed, so a single click may leave the game
        unresigned. Failing that way is the safe direction (the bot plays on
        rather than resigning something unintended), but it needs checking
        against a live game before it can be relied on.
        """
        try:
            from auto_calibration.config import get_config
            x, y = get_config().get_resign_button_position()
        except Exception:
            actions.log("chess.com resign: no calibrated resign button position. \n")
            return False
        actions.log(f"chess.com resign click at ({x}, {y}); confirmation flow unverified. \n")
        actions.click(x, y, tolerance=10, clicks=1, duration=np.random.uniform(0.3, 0.7))
        return True

    def start_new_game(self, actions, time_control="1+0"):
        """
        Not implemented: seeking a game needs chess.com's lobby layout, and
        no lobby screenshot exists in the corpus to calibrate against.

        Returning False rather than guessing is deliberate - the alternative
        is clicking at coordinates derived from Lichess's lobby, which on
        chess.com would land on arbitrary page furniture. The end-of-game
        modal does carry "New <tc>" and "Rematch" buttons, but their position
        moves with the modal's height (which varies by ending), so they need
        to be located per-ending rather than assumed.
        """
        actions.log(
            "chess.com start_new_game is not implemented (no lobby calibration); "
            "not clicking. \n")
        return False

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
