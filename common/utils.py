# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 13:11:33 2024

@author: xusem
"""
import chess
import chess.engine
import time

from common.board_information import calculate_threatened_levels, get_threatened_board, PIECE_VALS

class InvalidPositionError(Exception):
    """ A board given to the engine is structurally impossible (e.g. a king
        missing from a mid-animation screen scrape). Callers should discard
        the position and rescan rather than treat this as fatal.
    """
    pass


# Piece-placement Status flags that can never occur in a real game, in
# contrast to turn/castling/en-passant flags which the screen scraper
# legitimately gets wrong on transient frames.
_IMPOSSIBLE_PLACEMENT_STATUS = (
    chess.Status.NO_WHITE_KING
    | chess.Status.NO_BLACK_KING
    | chess.Status.TOO_MANY_KINGS
    | chess.Status.TOO_MANY_WHITE_PAWNS
    | chess.Status.TOO_MANY_BLACK_PAWNS
    | chess.Status.PAWNS_ON_BACKRANK
    | chess.Status.TOO_MANY_WHITE_PIECES
    | chess.Status.TOO_MANY_BLACK_PIECES
    | chess.Status.EMPTY
)

def scraped_fen_sanity_issues(fen_or_board, turn_reliable=False):
    """ Check a screen-scraped position for piece placements that are
        impossible in a real game, e.g. a missing king when a capture
        animation covers it. Stockfish segfaults if asked to analyse such
        a position, so these must never be adopted as the tracked board.

        Only placement is checked by default: turn, castling rights and
        en-passant are synthesised by the scraper and may be transiently
        wrong on perfectly usable scrapes. In particular the raw scrape FEN
        always claims white to move, so any position where black stands in
        check would false-positive an OPPOSITE_CHECK test (that exact bug
        froze a live game on 2026-07-12: we were black, in check from
        Nxe5+, and every scan was rejected until we flagged).

        Pass turn_reliable=True only where the board's turn is
        authoritative (e.g. the engine after update_info has established
        side to move). Then OPPOSITE_CHECK -- the side NOT to move in
        check -- is also flagged: it arises from a mid-animation scrape
        (our checking move rendered, the opponent's king-move reply not
        yet) and Stockfish segfaults on it deterministically, defeating
        the restart-and-retry handler (live crash, also 2026-07-12:
        31.Bh4+ Kc8 scraped with the king still on d8, white to move).

        Returns a (possibly empty) list of human-readable issue strings.
    """
    board = fen_or_board if isinstance(fen_or_board, chess.Board) else chess.Board(fen_or_board)
    mask = _IMPOSSIBLE_PLACEMENT_STATUS
    if turn_reliable:
        mask |= chess.Status.OPPOSITE_CHECK
    bad = board.status() & mask
    return [flag.name for flag in chess.Status if flag & bad]


def extend_mate_score(score, mate_score=2500, extension=100):
    """ Given we are close to mating opponent, extend mate score to be such that
        each move closer to mate is not 1 eval difference but rather extension amount
        in difference.
        
        Returns altered score.
    """
    if score >= mate_score - 15:
        # can see mate in 15 or fewer
        return score + (score+15 -mate_score)*extension
    else:
        return score

def scramble_fire_veto(board_now: chess.Board, move_uci: str) -> bool:
    """ Whether a blind scramble fire (a stale pondered move played on the
        current board without thought) must be suppressed: the moved piece
        would land on a square where it can be captured at a profit.

        Humans banging out stale premoves in a flag race still see the board
        in front of them -- they blunder to *changes* they missed, not by
        dropping a piece onto a square that has been attacked for a full
        second. This keeps the deliberately-fallible blind fires (missed
        zwischenzugs, wrong plans) while cutting the see-it-and-hang-it ones
        humans do not produce. board_now must have our side to move.

        Returns True when the fire should be suppressed.
    """
    move_obj = chess.Move.from_uci(move_uci)
    piece_at = board_now.piece_type_at(move_obj.to_square)
    gain = PIECE_VALS[piece_at] if piece_at is not None else 0
    dummy_board = board_now.copy()
    dummy_board.push(move_obj)
    return calculate_threatened_levels(move_obj.to_square, dummy_board) - gain > 0.6

def check_safe_premove(board:chess.Board, premove_uci: str):
    """ Given a position and a generated premove_uci, decide whether the move is deemed
        'safe'. That is opponent cannot/unlikely to play a move which leads to a significant
        advantage after our move. We shall only calculate opponent moves which do not
        immediately give away material (do not capture not enpris piece, move to enpris
        square).
        
        Returns True if premove is safe, else returns False
    """
    move_obj = chess.Move.from_uci(premove_uci)
    # check turn is correct
    if board.turn == board.color_at(move_obj.from_square):
        raise Exception("Premove uci {} not valid for board turn with fen {}".format(premove_uci, board.fen()))
    
    # check premove is valid move
    if board.color_at(move_obj.from_square) is None:
        raise Exception("Premove uci {} is not valid for board with fen {}".format(premove_uci, board.fen()))
    
    # calculate current threatened levels
    current_threatened_board = get_threatened_board(board, colour=not board.turn, piece_types=[1,2,3,4,5])
    current_threatened_levels = sum(current_threatened_board)
    
    # curr_analysis_obj = STOCKFISH.analyse(board, limit=chess.engine.Limit(time=0.01), multipv=10)
    # current_eval = curr_analysis_obj[0]["score"].pov(not board.turn).score(mate_score=2500)
    # consider_moves = [entry["pv"][0] for entry in curr_analysis_obj]
    for opp_move_obj in board.legal_moves:
        # not move that is not enpris.
        to_material = board.piece_type_at(opp_move_obj.to_square)
        if to_material is None:
            to_mat = 0
        else:
            to_mat = PIECE_VALS[to_material]
        
        dummy_board= board.copy()
        dummy_board.push(opp_move_obj)
        if calculate_threatened_levels(opp_move_obj.to_square, dummy_board) - to_mat > 0.6:
            continue
        else:
            # An opponent reply can invalidate the premove (capture or
            # displace the moving piece, block a pawn push): the site just
            # cancels the premove then, so it contributes no risk -- and
            # push() would raise on the non-pseudo-legal move.
            if move_obj not in dummy_board.legal_moves:
                continue
            # see if eval much worse
            dummy_board.push(move_obj)
            new_threatened_board = get_threatened_board(dummy_board, colour=not board.turn, piece_types=[1,2,3,4,5])
            new_threatened_levels = sum(new_threatened_board)
            if new_threatened_levels > current_threatened_levels + 0.6:
                return False
    return True
        
    
def patch_fens(fen_before, fen_after, depth_lim:int=3):
    """ If get_move_made function is not able to find legal move to link the two fens
        we try to find in between fens to link the two fens.
        
        If no in between board are found, return None. Else return the fens and
        the moves made in between.
        
        Note when looking to patch fens with 3 or more plies missing, no longer becomes accurate
    """    
    moves_found = _recurse_patch_fens(fen_before, fen_after, depth_lim, [])
    if moves_found is not None:
        return_fens = [fen_before]
        dummy_board = chess.Board(fen_before)
        for move_uci in moves_found:
            dummy_board.push_uci(move_uci)
            return_fens.append(dummy_board.fen())
        return moves_found, return_fens
    else:
        return None
    
def _recurse_patch_fens(fen_before, fen_after, depth_lim, move_stack):
    before_b = chess.Board(fen_before)
    after_b = chess.Board(fen_after)
    if before_b.board_fen() == after_b.board_fen() and before_b.turn == after_b.turn: # terminating condition
        return move_stack
    elif depth_lim <= 0: # second terminating condition to make sure search doesn't go on forever
        return None
    else:
        changed_squares = chess.SquareSet(before_b.occupied ^ after_b.occupied)
        # find which squares were empty and which squares were occupied in the before board    
        if before_b.turn == chess.WHITE:
            moved_square_from = [sq for sq in changed_squares if before_b.color_at(sq) == chess.WHITE]
        else:
            moved_square_from = [sq for sq in changed_squares if before_b.color_at(sq) == chess.BLACK]
        
        test_moves = [move for move in before_b.legal_moves if move.from_square in moved_square_from]
        for move in test_moves:
            new_move_stack = move_stack[:] + [move.uci()]
            dummy_board = before_b.copy()
            dummy_board.push(move)
            res = _recurse_patch_fens(dummy_board.fen(), fen_after, depth_lim -1, new_move_stack)
            if res is not None:
                return res
        return None

def flip_uci(move_uci):
    """ Given a move uci, return a uci which is the move flipped. For example
        g2g3 is flipped to g7g6. """
    move_obj = chess.Move.from_uci(move_uci)
    from_sq = move_obj.from_square
    to_sq = move_obj.from_uci(move_uci).to_square
    promotion  = move_obj.promotion
    flipped_move_obj = chess.Move(chess.square_mirror(from_sq), chess.square_mirror(to_sq), promotion=promotion)
    return flipped_move_obj.uci()


if __name__ == "__main__":
    before = chess.Board("r1bqrnk1/pp3pb1/6pp/8/1P1N4/P5P1/1B1QP1BP/1R2R1K1 w - - 2 20")
    start = time.time()
    print(check_safe_premove(before, "d8b6"))
    end = time.time()
    print(end-start)