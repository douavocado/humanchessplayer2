import chess
import chess.engine
import random

from .logger import Logger
from .state_manager import StateManager
from .analyzer import Analyzer, STOCKFISH # Use main STOCKFISH for analysis
from .opening_book_handler import OpeningBookHandler
from .stockfish_move_logic import StockfishMoveLogic # Needed for fallback premove logic
from common.board_information import (
    PIECE_VALS, check_best_takeback_exists, calculate_threatened_levels, phase_of_game
)
from common.utils import check_safe_premove

class Premover:
    def __init__(self, logger: Logger, state_manager: StateManager, analyzer: Analyzer, opening_book_handler: OpeningBookHandler, stockfish_move_logic: StockfishMoveLogic):
        self.logger = logger
        self.state_manager = state_manager
        self.analyzer = analyzer
        self.opening_book_handler = opening_book_handler
        self.stockfish_move_logic = stockfish_move_logic

    def get_premove(self, board: chess.Board, takeback_only: bool = False) -> str | None:
        """
        Determines a suitable premove for the given board state (opponent's turn).

        Args:
            board: The current board state (must be opponent's turn).
            takeback_only: If True, only return a premove if it's a takeback.

        Returns:
            The UCI string of the premove, or None if no suitable premove is found.
        """
        side = self.state_manager.get_side()
        if side is None:
            self.logger.add_log("ERROR: Cannot get premove, side not set.\n")
            return None
        if board.turn == side:
            self.logger.add_log(f"ERROR: get_premove called when it's our turn (Side: {side}, Board Turn: {board.turn}).\n")
            # Attempt to recover by using the previous board state if available
            prev_board = self.state_manager.get_prev_board()
            if prev_board and prev_board.turn != side:
                 self.logger.add_log("WARNING: Using previous board state for premove calculation.\n")
                 board = prev_board
            else:
                 return None # Cannot proceed

        self.logger.add_log(f"Calculating premove for FEN: {board.fen()} (Takeback only: {takeback_only})\n")

        # 1. Check for Obvious Takeback Premoves
        # Iterate through opponent's legal moves to see if any allow a good takeback for us.
        self.logger.add_log("Checking for takeback premoves...\n")
        for opp_move_obj in board.legal_moves:
            # Check if the opponent's move is a capture of roughly equal or greater value
            moving_piece_type = board.piece_type_at(opp_move_obj.from_square)
            captured_piece_type = board.piece_type_at(opp_move_obj.to_square)

            if captured_piece_type is not None and moving_piece_type is not None:
                from_value = PIECE_VALS.get(moving_piece_type, 0)
                to_value = PIECE_VALS.get(captured_piece_type, 0)

                # Consider it a potential setup for takeback if capture value >= moving piece value (approx)
                if to_value >= from_value - 0.6:
                    # Check if there's a good takeback available *after* this opponent move
                    exists, takeback_move_uci = check_best_takeback_exists(board.copy(), opp_move_obj.uci())
                    if exists:
                        # Found a potential takeback premove
                        # Basic safety check: ensure the takeback itself isn't immediately blundering
                        temp_board_after_opp = board.copy()
                        temp_board_after_opp.push(opp_move_obj)
                        if takeback_move_uci: # Ensure takeback move is valid
                             takeback_move_obj = chess.Move.from_uci(takeback_move_uci)
                             if takeback_move_obj in temp_board_after_opp.legal_moves:
                                  temp_board_after_takeback = temp_board_after_opp.copy()
                                  temp_board_after_takeback.push(takeback_move_obj)
                                  # Check if the square we moved to is now heavily attacked
                                  threat_level_after = calculate_threatened_levels(takeback_move_obj.to_square, temp_board_after_takeback)
                                  piece_value_moved = PIECE_VALS.get(temp_board_after_opp.piece_type_at(takeback_move_obj.from_square), 0)

                                  if threat_level_after < piece_value_moved + 0.5: # Simple safety check
                                       self.logger.add_log(f"Detected safe takeback premove: {takeback_move_uci} in response to {opp_move_obj.uci()}.\n")
                                       return takeback_move_uci
                                  else:
                                       self.logger.add_log(f"Takeback premove {takeback_move_uci} seems unsafe (Threat: {threat_level_after}). Skipping.\n")
                             else:
                                  self.logger.add_log(f"WARNING: check_best_takeback_exists returned illegal move {takeback_move_uci}.\n")

        self.logger.add_log("No obvious takeback premoves found.\n")
        if takeback_only:
            return None # Stop here if only looking for takebacks

        # 2. Determine Candidate Premove based on Predicted Opponent Move
        self.logger.add_log("Predicting opponent's best move and our response...\n")
        candidate_premove = None
        engine = STOCKFISH # Use main engine for this prediction
        if engine is None:
             self.logger.add_log("ERROR: Stockfish engine not available for premove prediction.\n")
             return None

        try:
            # Analyze opponent's likely best move
            analysis = engine.analyse(board, limit=chess.engine.Limit(time=0.02))
            if not analysis or 'pv' not in analysis or not analysis['pv']:
                 self.logger.add_log("WARNING: Could not get opponent's predicted move from analysis.\n")
                 return None # Cannot predict response

            opp_best_move_obj = analysis['pv'][0]
            dummy_board_after_opp = board.copy()
            dummy_board_after_opp.push(opp_best_move_obj)

            # Check if game ended
            if dummy_board_after_opp.outcome() is not None:
                self.logger.add_log(f"Predicted game end after opponent move {opp_best_move_obj.uci()}. No premove.\n")
                return None

            # Check opening book for our response first
            game_phase = phase_of_game(dummy_board_after_opp) # Phase after opponent's predicted move
            if game_phase == "opening" and self.opening_book_handler.is_book_loaded():
                book_entry = self.opening_book_handler.get_weighted_choice(dummy_board_after_opp)
                if book_entry:
                    book_move_uci = book_entry.move.uci()
                    self.logger.add_log(f"Found opening book response {book_move_uci} to predicted opponent move.\n")
                    # Check safety before accepting book premove
                    if check_safe_premove(board, book_move_uci):
                        self.logger.add_log("Book premove deemed safe.\n")
                        candidate_premove = book_move_uci
                    else:
                        self.logger.add_log("Book premove deemed unsafe. Falling back to Stockfish.\n")
                else:
                     self.logger.add_log("No opening book response found.\n")


            # If no book move, get Stockfish response
            if candidate_premove is None:
                self.logger.add_log("Getting Stockfish response to predicted opponent move...\n")
                # Use the dedicated Stockfish move logic component
                # Pass the board *after* the predicted opponent move
                next_analysis = engine.analyse(dummy_board_after_opp, limit=chess.engine.Limit(time=0.02), multipv=10)
                if isinstance(next_analysis, dict): next_analysis = [next_analysis]

                candidate_premove = self.stockfish_move_logic.get_stockfish_move(
                    board=dummy_board_after_opp,
                    analysis=next_analysis,
                    last_opp_move_uci=opp_best_move_obj.uci() # Pass the predicted opponent move
                )
                if candidate_premove:
                     self.logger.add_log(f"Stockfish candidate premove: {candidate_premove}\n")
                else:
                     self.logger.add_log("Stockfish logic failed to provide a candidate premove.\n")
                     return None # Failed to get any candidate

        except (chess.engine.EngineError, ValueError, Exception) as e:
            self.logger.add_log(f"ERROR during premove prediction: {e}\n")
            return None


        # 3. Safety Checks for the Candidate Premove
        if candidate_premove is None:
             return None # Should not happen if logic above is correct, but safety first

        self.logger.add_log(f"Performing safety checks on candidate premove: {candidate_premove}\n")
        try:
            premove_obj = chess.Move.from_uci(candidate_premove)

            # Check if the premove destination square is safe *now* (before opponent moves)
            # This is a basic check against blundering into an existing attack
            temp_board_our_turn = board.copy()
            temp_board_our_turn.turn = side # Pretend it's our turn to check attacks correctly
            is_capture = temp_board_our_turn.is_capture(premove_obj)
            piece_val_at_dest = 0
            if is_capture:
                 dest_piece = temp_board_our_turn.piece_at(premove_obj.to_square)
                 if dest_piece: piece_val_at_dest = PIECE_VALS.get(dest_piece.piece_type, 0)


            # Check if the destination square is attacked by the opponent
            if temp_board_our_turn.is_attacked_by(not side, premove_obj.to_square):
                 # If it's attacked, is it a capture of equal/greater value, or defended?
                 moving_piece = temp_board_our_turn.piece_at(premove_obj.from_square)
                 moving_piece_val = PIECE_VALS.get(moving_piece.piece_type, 0) if moving_piece else 0

                 # Allow if it's a capture of equal/greater value OR if the destination is defended by us
                 if not (piece_val_at_dest >= moving_piece_val - 0.5 or temp_board_our_turn.is_attacked_by(side, premove_obj.to_square)):
                      self.logger.add_log(f"Premove {candidate_premove} moves to a currently attacked square and is not a good capture or defended. Unsafe.\n")
                      return None # Basic blunder check failed

            # Use the common check_safe_premove utility for more thorough checks (especially in opening)
            if not check_safe_premove(board, candidate_premove):
                 self.logger.add_log(f"check_safe_premove utility deemed premove {candidate_premove} unsafe.\n")
                 return None

            self.logger.add_log(f"Premove {candidate_premove} passed safety checks.\n")
            return candidate_premove

        except ValueError:
             self.logger.add_log(f"ERROR: Invalid UCI {candidate_premove} during safety check.\n")
             return None