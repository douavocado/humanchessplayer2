"""
Lichess site behaviour.

This is a move of logic that previously lived in clients/mp_original.py, with
its ordering, thresholds and (deliberately broad) exception handling
preserved exactly. It is the behavioural anchor for the site split: any change
in Lichess's offline readback numbers while this class is in play is a bug in
the extraction, not an improvement.
"""

import chess
import cv2

from chessimage.image_scrape_utils import (
    capture_board,
    capture_bottom_clock,
    capture_result,
    capture_white_notation,
    compare_result_images,
    get_fen_from_image,
    read_clock,
)

from .base import GameEndSignal, Site

import numpy as np

# Confidence needed to accept a notation-panel game-over message.
GAME_OVER_MESSAGE_MATCH_THRESHOLD = 0.75

# Profile-extracted result templates are matched strictly; the legacy
# chessimage/ references may come from a different screen layout, so they keep
# their historic looser threshold.
PROFILE_RESULT_THRESHOLD = 0.8
LEGACY_RESULT_THRESHOLD = 0.70

_LEGACY_RESULT_FILES = {
    'black_win': 'blackwin_result',
    'white_win': 'whitewin_result',
    'draw': 'draw_result',
}


class LichessSite(Site):
    name = "lichess"

    # Lichess repositions the clock between start, play and end states, which
    # is what makes the end-coordinate clock probe a usable end-of-game signal.
    clock_position_varies_by_state = True

    result_template_files = {
        'white_win': 'whitewin_result.png',
        'black_win': 'blackwin_result.png',
        'draw': 'draw_result.png',
    }
    optional_result_templates = frozenset()

    game_over_message_files = {
        'aborted': 'aborted_result.png',
        'didnt_move': 'didnt_move_result.png',
    }

    def __init__(self):
        self._result_refs = None
        self._message_refs = None
        self._start_like_fens = None

    # ------------------------------------------------------------------
    # templates
    # ------------------------------------------------------------------
    def _template_dir(self):
        from auto_calibration.config import get_config
        return get_config().get_template_dir()

    def _result_references(self):
        """Result reference images as (name, image, threshold) triples."""
        if self._result_refs is not None:
            return self._result_refs
        refs = []
        try:
            from auto_calibration.template_extractor import TemplateExtractor
            extractor = TemplateExtractor(template_dir=str(self._template_dir()))
            profile_templates = extractor.load_result_templates(self.result_template_files)
            for name, ref in (profile_templates or {}).items():
                if ref is not None:
                    refs.append((name, ref, PROFILE_RESULT_THRESHOLD))
        except Exception:
            pass
        if not refs:
            for name, stem in _LEGACY_RESULT_FILES.items():
                ref = cv2.imread(f"chessimage/{stem}.png")
                if ref is not None:
                    refs.append((name, ref, LEGACY_RESULT_THRESHOLD))
        self._result_refs = refs
        return refs

    def _message_references(self):
        if self._message_refs is None:
            try:
                from auto_calibration.template_extractor import TemplateExtractor
                extractor = TemplateExtractor(template_dir=str(self._template_dir()))
                templates = extractor.load_game_over_message_templates()
                self._message_refs = list(templates.items())
            except Exception:
                self._message_refs = []
        return self._message_refs

    # ------------------------------------------------------------------
    # new game
    # ------------------------------------------------------------------
    def _start_like_board_fens(self):
        """
        Board placements that can be on screen when a new game is found: the
        starting position, or one white move into it - as black the opponent
        often moves (or premoves) before our first scan.
        """
        if self._start_like_fens is None:
            fens = {chess.STARTING_BOARD_FEN}
            base = chess.Board()
            for move in list(base.legal_moves):
                base.push(move)
                fens.add(base.board_fen())
                base.pop()
            self._start_like_fens = fens
        return self._start_like_fens

    def detect_new_game(self, expected_time=None):
        # try to read bot clock for start position. if none is found, then
        # haven't started the game
        for state in ["start1", "start2"]:
            res, details = read_clock(capture_bottom_clock(state=state), return_details=True)
            if res is not None:
                # In 'start' states, the digits should be roughly vertically
                # centered in the crop. If they are significantly shifted, it
                # might be the 'play' clock being seen.
                v_center = details.get('v_center', 0)
                orig_h = details.get('original_height', 66)
                v_error = abs(v_center - orig_h / 2)

                if v_error > orig_h * 0.1:  # Tightened to 10% off-center
                    continue

                # Validate against expected time
                if expected_time is not None:
                    # Accept if within 10% of expected time, and NOT 0
                    if res == 0 or abs(res - expected_time) > max(10, expected_time * 0.1):
                        continue
                elif res == 0:
                    # Always ignore 0 as a starting time
                    continue

                # Starting Board Verification (the "strict" check). If we
                # found a valid clock, verify the board is at (or one white
                # move into) the starting position: as black the opponent may
                # already have moved before we scan
                board_img = capture_board()
                # Side is not known yet; try 'w' first as it's most common
                try:
                    start_like = self._start_like_board_fens()
                    test_fen = get_fen_from_image(board_img, bottom="w", fast_mode=True)
                    if chess.Board(test_fen).board_fen() not in start_like:
                        # Could be we are playing as black, try other orientation
                        test_fen_b = get_fen_from_image(board_img, bottom="b", fast_mode=True)
                        if chess.Board(test_fen_b).board_fen() not in start_like:
                            continue  # Not a start-like board in either orientation
                except Exception:
                    continue

                return res

        return None  # either returns None, no clock found

    # ------------------------------------------------------------------
    # game end
    # ------------------------------------------------------------------
    def clock_readable_at_end_positions(self):
        """
        Whether any end-state clock coordinate yields a reading.

        Only meaningful because Lichess moves its clock when the game ends.
        Kept public because the client uses it as a guarded fallback.
        """
        for state in ("end1", "end2", "end3"):
            if read_clock(capture_bottom_clock(state=state)) is not None:
                return True
        return False

    def find_game_over_message(self):
        """
        Detect a game that ended without a result box, via its notation-panel
        message: "White/Black aborted the game" or "White/Black didn't move".

        These endings leave none of the usual game-end signals: the board is
        still start-like (so no board outcome, and the clock fallback is
        guarded off) and there is no result box for the result templates to
        match - only an italic message. The panel is more compact than after
        a normal game (zero or one move in the list) and shifts with layout,
        so the message is template-searched anywhere within the notation
        region rather than compared at a fixed spot.

        Returns the matched message name (truthy) or None.
        """
        refs = self._message_references()
        if not refs:
            return None
        try:
            region = capture_white_notation()
            if region is None or region.size == 0:
                return None
            region_gray = cv2.cvtColor(np.ascontiguousarray(region), cv2.COLOR_BGR2GRAY)
            for name, ref in refs:
                ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
                if (region_gray.shape[0] < ref_gray.shape[0]
                        or region_gray.shape[1] < ref_gray.shape[1]):
                    continue
                score = cv2.matchTemplate(region_gray, ref_gray, cv2.TM_CCOEFF_NORMED).max()
                if score > GAME_OVER_MESSAGE_MATCH_THRESHOLD:
                    return name
            return None
        except Exception:
            return None

    def game_over_screen_visible(self):
        if self.find_game_over_message() is not None:
            return True
        try:
            result_img = capture_result(arena=False)
            if result_img is not None and result_img.size > 0:
                for _name, ref, threshold in self._result_references():
                    if compare_result_images(result_img, ref) > threshold:
                        return True
        except Exception:
            pass
        return False

    def detect_game_end(self, fens=None, arena=False):
        fens = fens or []

        # Method 1: board outcome (checkmate/stalemate) - most reliable
        if len(fens) > 0:
            board = chess.Board(fens[-1])
            outcome = board.outcome()
            if outcome is not None:
                return GameEndSignal(
                    method="board_outcome",
                    detail={'termination': outcome.termination, 'fen': fens[-1]},
                )

        # Method 2: result image comparison, using calibrated coordinates
        try:
            result_img = capture_result(arena=arena)
            if result_img is not None and result_img.size > 0:
                for name, ref, threshold in self._result_references():
                    score = compare_result_images(result_img, ref)
                    if score > threshold:
                        return GameEndSignal(
                            method="result_template",
                            result=name,
                            detail={'score': round(float(score), 3), 'threshold': threshold},
                        )
        except Exception:
            # Don't fail the game check if result image comparison fails
            pass

        # Method 2b: aborted / didn't-move endings show no result box at all -
        # match their message in the notation panel instead
        message = self.find_game_over_message()
        if message:
            return GameEndSignal(method="message", detail={'message': message})

        # Method 3: fallback - clock readable at end positions but NOT at play
        # position, catching a UI change caused by the game ending.
        # Guard: at game start ("play the first move" state) the clock also
        # sits away from the play position and can bleed into an end-state
        # region, so a start-position board is never treated as a game end here.
        try:
            start_placement = chess.STARTING_FEN.split()[0]
            board_start_like = (len(fens) > 0 and fens[-1].split()[0] == start_placement)
            if not board_start_like:
                play_clock = read_clock(capture_bottom_clock(state="play"))
                if play_clock is None and self.clock_readable_at_end_positions():
                    return GameEndSignal(method="clock_fallback")
        except Exception:
            pass

        return None
