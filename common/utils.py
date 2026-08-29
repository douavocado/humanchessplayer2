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

def premove_render_placement(fen: str, move_uci: str, premove_uci: str):
    """
    The board placement a site *draws* while an unconfirmed premove sits on it.

    chess.com (and Lichess) show a queued premove immediately, with the piece
    already on its destination square, before the opponent has moved and
    before the server has accepted anything. The result is frequently a
    position that never existed and could never exist: queue Bg7xe5 expecting
    a recapture, and if the opponent plays something else the board is drawn
    with our bishop standing on our own knight's square.

    Scraping that and treating it as reality is how a phantom position enters
    the game state - it links to nothing, so the history gets wiped and the
    engine is asked to move from a board that is not on the server. Computing
    the expected drawing here lets a scan recognise its own premove and
    discard the frame instead.

    The premove is applied *as a rendering*, not as a chess move: the piece is
    relocated whatever occupies the destination, because that is what the site
    draws. Our own move is the opposite - it must be genuinely legal, or we
    are not describing a board the site ever drew, and a wrong prediction here
    is worse than none: it could match a real scan and discard it. Returns the
    board placement (no side/castling fields), or None if the inputs do not
    describe a premove that could be drawn.
    """
    if not premove_uci:
        return None
    try:
        board = chess.Board(fen)
        if move_uci:
            move = chess.Move.from_uci(move_uci)
            if move not in board.legal_moves:
                return None
            board.push(move)
        premove = chess.Move.from_uci(premove_uci)
        piece = board.piece_at(premove.from_square)
        if piece is None:
            return None
        board.remove_piece_at(premove.from_square)
        if premove.promotion:
            piece = chess.Piece(premove.promotion, piece.color)
        board.set_piece_at(premove.to_square, piece)
        return board.board_fen()
    except (ValueError, AssertionError):
        return None


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
        
    
# A single ply changes at most four squares (castling, which moves both the
# king and the rook); every other move changes two or three. So a placement
# diff wider than 4 squares per remaining ply can never be closed, which is
# what keeps the full-width search below affordable.
_MAX_SQUARES_PER_PLY = 4

# Ceiling on legal moves generated per patch_fens call. The search runs
# inside the client's ~14 scans/second loop, so a pathological position must
# fail fast rather than eat the clock. Measured worst case over the 24 real
# link failures in one session batch was ~1200 moves generated.
PATCH_FENS_MOVE_BUDGET = 20000


def _placement_diff_mask(board, target):
    """ Bitboard of the squares whose piece differs between two positions.

        Zero exactly when the placements are identical, so it doubles as the
        terminating test and as the search's pruning heuristic.
    """
    diff = board.occupied ^ target.occupied
    for colour in (chess.WHITE, chess.BLACK):
        for piece_type in chess.PIECE_TYPES:
            diff |= (board.pieces_mask(piece_type, colour)
                     ^ target.pieces_mask(piece_type, colour))
    return diff


def patch_fens(fen_before, fen_after, depth_lim:int=3):
    """ If get_move_made function is not able to find legal move to link the two fens
        we try to find in between fens to link the two fens.

        If no in between board are found, return None. Else return the fens and
        the moves made in between.

        The search is full width: restricting it to moves whose from-square
        changed occupancy misses every sequence where the square is refilled
        before the target position, which is what our own premoves look like
        (f1g2 then e1g1 puts the rook back on f1, so f1 never reads as
        vacated). That filter cost ~1.3 history wipes per game live.

        Deepening one ply at a time makes the shortest link win. That matters
        because patch_fens can also "confirm" a longer piece-shuffle line
        through the same position, and callers - the unreadable-turn fallback
        above all - rank candidates on ply count.

        Note when looking to patch fens with 3 or more plies missing, no longer becomes accurate
    """
    target = chess.Board(fen_after)
    for depth in range(0, depth_lim + 1):
        board = chess.Board(fen_before)
        budget = [PATCH_FENS_MOVE_BUDGET]
        moves_found = _recurse_patch_fens(board, target, depth, [], budget)
        if moves_found is not None:
            return_fens = [fen_before]
            dummy_board = chess.Board(fen_before)
            for move_uci in moves_found:
                dummy_board.push_uci(move_uci)
                return_fens.append(dummy_board.fen())
            return moves_found, return_fens
    return None


def _recurse_patch_fens(board, target, depth_lim, move_stack, budget):
    """ Depth-limited search for a move sequence turning board into target.

        board is mutated and restored in place - rebuilding it from a fen at
        every node dominated the runtime (seconds per call, on a clock the
        bot is trying not to flag).
    """
    diff = _placement_diff_mask(board, target)
    if diff == 0 and board.turn == target.turn: # terminating condition
        return list(move_stack)
    if depth_lim <= 0: # second terminating condition to make sure search doesn't go on forever
        return None
    if chess.popcount(diff) > _MAX_SQUARES_PER_PLY * depth_lim:
        return None

    for move in _ordered_patch_moves(board, diff):
        budget[0] -= 1
        if budget[0] < 0:
            return None
        board.push(move)
        move_stack.append(move.uci())
        res = _recurse_patch_fens(board, target, depth_lim - 1, move_stack, budget)
        move_stack.pop()
        board.pop()
        if res is not None:
            return res
    return None


def _ordered_patch_moves(board, diff):
    """ Legal moves, those touching a differing square first.

        Ordering only - the search stays full width, because the linking move
        can also be a transit move onto and off a square that ends up
        matching. Trying the moves that visibly close the diff first finds
        the real line sooner and cuts the work by roughly an order of
        magnitude.
    """
    touching, other = [], []
    for move in board.legal_moves:
        if (chess.BB_SQUARES[move.from_square] | chess.BB_SQUARES[move.to_square]) & diff:
            touching.append(move)
        else:
            other.append(move)
    return touching + other

def highlight_squares_to_chess(highlighted, bottom):
    """ Map image-space square indices (0-63, row-major from the top-left of
        the board crop) onto python-chess squares for the given orientation. """
    if bottom == "w":
        return {chess.square_mirror(square) for square in highlighted}
    return {chess.square_mirror(63 - square) for square in highlighted}


def highlights_contradict_move(highlighted, move_uci, bottom):
    """ Whether the site's last-move highlights disagree with the move a scan
        has just been linked to.

        A linked scan is the one path that is adopted with no second look at
        the board, so a mid-animation frame that happens to spell a legal
        move sails straight through. That is how 19...Rae8 became 19...Rab8
        in the 2026-08-22 game: the rook was caught over b8 a quarter of the
        way through its slide, a8b8 linked, and the engine spent the rest of
        the position unable to see the rook it was allowed to take on e8.

        The highlights are an independent reading of the same frame - they
        mark where the move really came from and went to, and a piece in
        flight does not carry them with it. Castling and en passant both
        highlight exactly the moving piece's own from/to pair, so the plain
        uci squares are the right thing to compare against.

        Only an unambiguous pair is trusted. Anything else - nothing
        detected, a square hidden under a premove overlay, a stray tint -
        counts as no signal rather than as disagreement, so this can raise a
        doubt about a reading but can never invent one out of a blank one.
    """
    if len(highlighted) != 2:
        return False
    move = chess.Move.from_uci(move_uci)
    marked = highlight_squares_to_chess(highlighted, bottom)
    return marked != {move.from_square, move.to_square}


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