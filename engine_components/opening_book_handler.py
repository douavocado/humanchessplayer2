import chess
import chess.polyglot
import os
from .logger import Logger # Use relative import

class OpeningBookHandler:
    def __init__(self, logger: Logger, opening_book_path: str = "assets/data/Opening_books/bullet.bin"):
        self.logger = logger
        self.opening_book = None
        self.book_path = opening_book_path
        self._load_book()

    def _load_book(self):
        """Loads the Polyglot opening book."""
        self.logger.add_log(f"Attempting to load opening book from: {self.book_path}\n")
        if not os.path.exists(self.book_path):
            self.logger.add_log(f"ERROR: Opening book file not found at {self.book_path}. Opening book disabled.\n")
            self.opening_book = None
            return

        try:
            self.opening_book = chess.polyglot.open_reader(self.book_path)
            self.logger.add_log("Opening book loaded successfully.\n")
        except Exception as e:
            self.logger.add_log(f"ERROR: Failed to load opening book from {self.book_path}: {e}\n")
            self.opening_book = None # Ensure book is None on failure

    def is_book_loaded(self) -> bool:
        """Checks if the opening book was loaded successfully."""
        return self.opening_book is not None

    def find_book_moves(self, board: chess.Board) -> list[chess.polyglot.Entry]:
        """Finds all opening book entries for the given board position."""
        if not self.is_book_loaded():
            self.logger.add_log("Attempted to find book moves, but opening book is not loaded.\n")
            return []
        try:
            return list(self.opening_book.find_all(board))
        except Exception as e:
            # Handle potential errors during lookup, though less common with chess.polyglot
            self.logger.add_log(f"ERROR: An error occurred while querying the opening book for FEN {board.fen()}: {e}\n")
            return []

    def get_weighted_choice(self, board: chess.Board, exclude_moves: list[chess.Move] = None) -> chess.polyglot.Entry | None:
        """Gets a weighted random move from the opening book for the position."""
        if not self.is_book_loaded():
            self.logger.add_log("Attempted to get weighted choice, but opening book is not loaded.\n")
            return None
        try:
            # Ensure exclude_moves is a list if None
            if exclude_moves is None:
                exclude_moves = []
            # The weighted_choice method might raise IndexError if no moves are available after exclusion
            if not list(self.opening_book.find_all(board, minimum_weight=0)):
                 self.logger.add_log(f"No book moves found for FEN {board.fen()} to make a weighted choice.\n")
                 return None
            return self.opening_book.weighted_choice(board, exclude_moves=exclude_moves)
        except IndexError:
             self.logger.add_log(f"No book moves available for weighted choice after exclusions for FEN {board.fen()}.\n")
             return None
        except Exception as e:
            self.logger.add_log(f"ERROR: An error occurred during weighted choice for FEN {board.fen()}: {e}\n")
            return None

    def close_book(self):
        """Closes the opening book file handle."""
        if self.is_book_loaded():
            try:
                self.opening_book.close()
                self.logger.add_log("Opening book closed successfully.\n")
                self.opening_book = None
            except Exception as e:
                self.logger.add_log(f"ERROR: Failed to close opening book: {e}\n")

# Ensure the book is closed when the module is potentially unloaded or program exits
import atexit
# Create a global instance or manage instances carefully if multiple handlers are needed
# For simplicity here, assume a single instance might be created implicitly
# A better approach would be explicit instance management and cleanup calls.
# This atexit registration might not work perfectly if the handler instance isn't kept alive.
# Consider managing cleanup in the main Engine class's __del__ or a dedicated cleanup function.
# atexit.register(lambda: OpeningBookHandler(None).close_book()) # Placeholder - needs proper instance management