"""Premove search, extracted verbatim from engine.Engine.

get_premove finds a move to queue while it isn't our turn: takeback
premoves first (best-takeback check), then an opponent-best-reply
simulation (opening book if in book, else a stockfish-move search on the
resulting position), filtered for human plausibility (not moving into an
en-pris square) and, in the opening or a flag-race scramble, vetted with
check_safe_premove before being allowed to fire.

Eighth slice of the engine.py strangler-fig extraction (see
testing/engine_parity/ for the regression harness). Verbatim move
(self -> engine), no interface redesign.
"""
import numpy as np

import chess
import chess.engine

from common.board_information import (
    PIECE_VALS, phase_of_game, calculate_threatened_levels, check_best_takeback_exists,
)
from common.utils import check_safe_premove
from common.constants import FLAG_RACE_TIME
from common.search_constants import PREMOVE_SCAN_MULTIPV


def get_premove(engine, board, side, takeback_only=False):
    """ Given a position which is not our turn, using only computer evaluations return
        a premove which we may make. The nature of this function is so that it
        may be called to spot immediate takebacks, and also be used in last second
        time scramble situations. Side is the side which we are on.

        If takeback_only is True, then we only return a premove if we found a
        takeback premove, otherwise we return None.

        Returns a move_uci string.
    """
    # First check that the position given is indeed not our turn
    assert board.turn != side

    # Next check for takeback premoves
    # We define takebacks here to be moves that the opposition can make that
    # capture material, and it is in our best interest to capture back.
    for move_obj in board.legal_moves:
        from_value = PIECE_VALS[board.piece_type_at(move_obj.from_square)]
        piece_type_to = board.piece_type_at(move_obj.to_square)
        if piece_type_to is not None:
            to_value = PIECE_VALS[piece_type_to]
            if to_value - from_value > -0.6: # roughly similar trade
                exists, takeback_move_uci = check_best_takeback_exists(board.copy(), move_obj.uci(), engine=engine.stockfish_engine)
                if exists == True:
                    # then we have a best takeback
                    premove_uci = takeback_move_uci
                    engine.log += "Detected and returning takeback premove: {}. \n".format(premove_uci)
                    return premove_uci

    if takeback_only == True: # if we are only looking for takeback premoves
        return None

    # If no takebacks, then use computer moves to determine best premove to make.
    # We pretend opponent makes top engine move, and we respond using get_stockfish_move
    # perform analysis on current position
    analysis = engine.stockfish_engine.analyse(board, limit=chess.engine.Limit(depth=8, time=0.02))
    opp_best_move_obj = analysis["pv"][0]
    dummy_board = board.copy()
    dummy_board.push(opp_best_move_obj)
    candidate_premove = None
    # if we are in the opening, then check whether we could respond with opening database move
    game_phase = phase_of_game(dummy_board)
    if game_phase == "opening":
        result = list(engine.opening_book.find_all(dummy_board))
        if len(result) != 0:
            engine.log += "Found matching position in opening database during premove search. Using it to find premove in opening.\n"
            excluded_moves = [res.move for res in result[5:]]
            # Now get weighted choice of move to play
            played_move_obj = engine.opening_book.weighted_choice(dummy_board, exclude_moves=excluded_moves).move
            engine.log += "Chosen premove from opening book: {} \n".format(dummy_board.san(played_move_obj))
            # we need to check whether this is a safe premove or not
            if check_safe_premove(board, played_move_obj.uci()) == True:
                engine.log += "Double checked that premove is safe. \n"
                candidate_premove = played_move_obj.uci()
            else:
                engine.log += "Opening book premove is not a safe premove. Resorting to stockfish premove. \n"

        else:
            engine.log += "Even though opening phase, did not find matching position in opening database. Resorting to stockfish premove. \n"
    if candidate_premove is None: # didn't find opening book premove
        next_analysis = engine.stockfish_engine.analyse(dummy_board, limit=chess.engine.Limit(depth=8, time=0.02), multipv=PREMOVE_SCAN_MULTIPV)
        if isinstance(next_analysis, dict):
            next_analysis = [next_analysis]
        # now use get_stockfish move on this position
        # Of course, we can only get a stockfish move if the game is not over
        if dummy_board.outcome() is None:
            candidate_premove = engine.get_stockfish_move(board=dummy_board, analysis=next_analysis, last_move_uci=opp_best_move_obj.uci(), log=False)
            engine.log += "Detected premove from stockfish evals: {} \n".format(candidate_premove)
        else:
            # game is over
            engine.log += "Cannot get premove for position {} as it is game over. \n".format(dummy_board.fen())
            return None

    # Now that we have found our premove, we have to do some extra checks to make sure it is a human
    # premove. for example, a human wouldn't premove a piece into a square which would be enpris
    dummy_board_2 = board.copy()
    dummy_board_2.turn = side # pretend it is our turn in this situation
    move_obj = chess.Move.from_uci(candidate_premove)
    piece_at = dummy_board_2.piece_type_at(move_obj.to_square)
    colour_at = dummy_board_2.color_at(move_obj.to_square)
    if colour_at == side: # if we are just premoving a takeback
        return candidate_premove
    elif piece_at is None:
        piece_val_at = 0
    else:
        piece_val_at = PIECE_VALS[piece_at]
    dummy_board_2.push(move_obj)
    if calculate_threatened_levels(move_obj.to_square, dummy_board_2) - piece_val_at > 0.6:
        # premove moves to be enpris
        engine.log += "Premove {} moves to an enpris square, filtering the premove out. \n".format(dummy_board.san(move_obj))
        premove_uci = None
    else:
        # if we are in the opening, then check the premove is safe
        if game_phase == "opening":
            if check_safe_premove(board, candidate_premove) == True:
                premove_uci = candidate_premove
            else:
                engine.log += "Discovered game phase is opening, and premove {} is not considered a safe premove. Filtering out. \n".format(candidate_premove)
                premove_uci = None
        elif engine._own_time_or_none() is not None and engine._own_time_or_none() < FLAG_RACE_TIME \
                and np.random.random() < engine.scramble_veto_p:
            # In a flag race a queued premove fires on the server
            # unconditionally, so unsafe ones are a main source of
            # instant blunders (measured: bot emt-0 TP blunder rate
            # 0.136 vs human 0.072 -- humans' scramble premoves are
            # instinct-safe takebacks and king steps). Vet like the
            # opening branch, but gated on this game's scramble skill:
            # unconditional vetting collapsed the catastrophe tail
            # (endgame ACPL std ratio fell to 0.37x) -- a panicky game
            # must still queue the occasional howler. Takeback premoves
            # returned earlier are deliberately exempt.
            if check_safe_premove(board, candidate_premove) == True:
                premove_uci = candidate_premove
            else:
                engine.log += "Scramble premove {} is not a safe premove, filtering out. \n".format(candidate_premove)
                premove_uci = None
        else:
            premove_uci = candidate_premove
    return premove_uci
