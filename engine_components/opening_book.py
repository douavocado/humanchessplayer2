"""Opening-book consultation shared by Engine.get_human_move and
engine_components.premover.get_premove.

Layers two polyglot books: a small hand-curated `engine.repertoire_book`
(one account's real recurring opening habits, mined from Lichess's player
opening-explorer -- see assets/data/Opening_books/build_repertoire.py) is
tried first; if the current position isn't covered there, `engine.opening_book`
(the broad bullet.bin) is tried next. Either way the actual move is sampled
with weighted_choice, so a book's own stored weights (not this module) decide
the odds among the eligible candidates at a matched position.
"""
from common.search_constants import OPENING_REPERTOIRE_TOP_N, OPENING_BOOK_TOP_N


def _weighted_book_move(book, board, top_n):
    """ Look up board in book, keep only the top_n entries (by file order,
        matching the book-building convention of writing entries
        weight-descending within a position), and weighted_choice among
        them. Returns a chess.Move, or None if book has no entry here. """
    result = list(book.find_all(board))
    if not result:
        return None
    excluded_moves = [res.move for res in result[top_n:]]
    return book.weighted_choice(board, exclude_moves=excluded_moves).move


def consult_book(engine, board):
    """ Given a position, return a move from the repertoire book if covered,
        else from the general opening book, else None. """
    move = _weighted_book_move(engine.repertoire_book, board, OPENING_REPERTOIRE_TOP_N)
    if move is not None:
        engine.log += "Found repertoire book move: {} \n".format(board.san(move))
        return move
    move = _weighted_book_move(engine.opening_book, board, OPENING_BOOK_TOP_N)
    if move is not None:
        engine.log += "Found general opening book move: {} \n".format(board.san(move))
        return move
    engine.log += "No opening book match found for current position. \n"
    return None
