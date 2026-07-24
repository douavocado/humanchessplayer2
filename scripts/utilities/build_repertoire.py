# -*- coding: utf-8 -*-
"""Builds repertoire.bin: a small, hand-curated polyglot opening book
modelling one real account's recurring bullet habits (Mishka_The_Great,
Lichess IM, 1+0/1/2+0 rated bullet), mined via the Lichess player
opening-explorer API (https://explorer.lichess.org/player).

python-chess can only *read* polyglot books, so entries are packed here
directly per the format python-chess's reader expects (see
chess/polyglot.py: ENTRY_STRUCT = struct.Struct(">QHHI"), key = zobrist_hash).

LINES entries are (prefix, move, weight): prefix is the UCI move sequence
already played (from the game start) leading to the position where it is
our move; move is the UCI move we play there; weight is that move's real
measured share (as a percentage of games reaching that position) among the
sibling moves also listed for the same prefix. Where only one entry is
listed for a prefix, that reply covers the large majority of real games
reaching it (~75-100%) and is the sole repertoire branch on purpose --
everything else falls through to the general opening book / NN move
selection, by design (see engine_components/opening_book.py).

Coverage is deliberately narrow: only positions with directly-measured
percentages are included. Re-run this script (from the repo root) after
editing LINES to regenerate repertoire.bin (gitignored, like bullet.bin).
"""
import struct

import chess

ENTRY_STRUCT = struct.Struct(">QHHI")

OUT_PATH = "assets/data/Opening_books/repertoire.bin"

# (prefix_uci_moves, our_move_uci, weight)
LINES = [
    # White: 1.Nf3 (65.4% of her White games), then 2.g3 into a King's
    # Indian Attack / Reti-style fianchetto system regardless of Black's
    # setup (86.3% vs 1...Nf6, 86.3% vs 1...g6, 74.6% vs 1...d5).
    ([], "g1f3", 100),
    (["g1f3", "g8f6"], "g2g3", 100),
    (["g1f3", "g7g6"], "g2g3", 100),
    (["g1f3", "d7d5"], "g2g3", 100),

    # Black vs 1.e4: 1...g6 (56.0%) -- Modern Defense -- then a genuine
    # 3-way fork depending on White's 2nd move (not noise: near-even split
    # measured both ways round).
    (["e2e4"], "g7g6", 100),
    (["e2e4", "g7g6", "d2d4"], "f8g7", 36),
    (["e2e4", "g7g6", "d2d4"], "c7c5", 33),
    (["e2e4", "g7g6", "d2d4"], "d7d5", 27),
    (["e2e4", "g7g6", "b1c3"], "d7d5", 36),
    (["e2e4", "g7g6", "b1c3"], "f8g7", 32),
    (["e2e4", "g7g6", "b1c3"], "c7c5", 29),

    # Black vs 1.d4: 1...g6 (60.5%) -- Modern Defense -- forking similarly.
    (["d2d4"], "g7g6", 100),
    (["d2d4", "g7g6", "g1f3"], "f8g7", 49),
    (["d2d4", "g7g6", "g1f3"], "c7c5", 41),
    (["d2d4", "g7g6", "c2c3"], "c7c5", 59),
    (["d2d4", "g7g6", "c2c3"], "f8g7", 34),
    (["d2d4", "g7g6", "c2c4"], "c7c5", 55),
    (["d2d4", "g7g6", "c2c4"], "f8g7", 35),
]


def encode_move(move: chess.Move) -> int:
    """ Pack a move into polyglot's 16-bit raw_move representation (see
        chess/polyglot.py MemoryMappedReader.__getitem__ for the inverse). """
    promotion_part = (move.promotion - 1) if move.promotion else 0
    return (promotion_part << 12) | (move.from_square << 6) | move.to_square


def build_entries():
    from chess.polyglot import zobrist_hash

    entries = []
    for prefix, move_uci, weight in LINES:
        board = chess.Board()
        for uci in prefix:
            board.push(chess.Move.from_uci(uci))
        key = zobrist_hash(board)
        move = chess.Move.from_uci(move_uci)
        assert move in board.legal_moves, f"illegal move {move_uci} after {prefix}"
        entries.append((key, encode_move(move), weight, 0))

    # Polyglot readers bisect on key, so entries must be sorted ascending by
    # key; sibling entries (same key) are sorted weight-descending to match
    # the "top_n by file order" convention _weighted_book_move relies on.
    entries.sort(key=lambda e: (e[0], -e[2]))
    return entries


def main():
    entries = build_entries()
    with open(OUT_PATH, "wb") as f:
        for key, raw_move, weight, learn in entries:
            f.write(ENTRY_STRUCT.pack(key, raw_move, weight, learn))
    print(f"Wrote {len(entries)} entries to {OUT_PATH}")


if __name__ == "__main__":
    main()
