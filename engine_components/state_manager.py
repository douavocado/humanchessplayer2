import chess
from common.utils import patch_fens
from .logger import Logger # Use relative import

class StateManager:
    def __init__(self, logger: Logger, initial_fen: str = chess.STARTING_FEN):
        self.logger = logger
        self.input_info = {
            "side": None,
            "fens": [initial_fen], # Start with the initial fen
            "self_clock_times": [],
            "opp_clock_times": [],
            "self_initial_time": None,
            "opp_initial_time": None,
            "last_moves": [],
        }
        # Initialize current_board based on the last fen
        self.current_board = chess.Board(self.input_info["fens"][-1])
        self.analytics_updated = False # Track analytics status here

    def update_info(self, info_dic: dict):
        """Updates the engine state based on the provided info_dic."""
        self.logger.add_log("Received and updating info_dic: \n")
        self.logger.add_log(str(info_dic) + "\n")
        self.input_info.update(info_dic)

        # Ensure fens and last_moves are lists
        if not isinstance(self.input_info.get("fens"), list):
            self.input_info["fens"] = [self.input_info.get("fens", chess.STARTING_FEN)]
        if not isinstance(self.input_info.get("last_moves"), list):
             self.input_info["last_moves"] = [] # Default to empty list if not provided or wrong type

        # Trim history if it gets too long (e.g., keep last 10 states)
        max_history = 10
        if len(self.input_info["fens"]) > max_history:
             self.input_info["fens"] = self.input_info["fens"][-max_history:]
        if len(self.input_info["last_moves"]) > max_history -1: # Moves are one less than fens
             self.input_info["last_moves"] = self.input_info["last_moves"][-(max_history-1):]
        if len(self.input_info["self_clock_times"]) > max_history:
             self.input_info["self_clock_times"] = self.input_info["self_clock_times"][-max_history:]
        if len(self.input_info["opp_clock_times"]) > max_history:
             self.input_info["opp_clock_times"] = self.input_info["opp_clock_times"][-max_history:]


        # Reconstruct board state from history to ensure consistency
        if len(self.input_info["fens"]) >= 1 and len(self.input_info["last_moves"]) >= 1:
            # Find the fen corresponding to the start of the last_moves sequence
            start_fen_index = -len(self.input_info["last_moves"]) - 1
            if abs(start_fen_index) > len(self.input_info["fens"]):
                 self.logger.add_log(f"ERROR: Mismatch between fens ({len(self.input_info['fens'])}) and last_moves ({len(self.input_info['last_moves'])}). Resetting board.\n")
                 self.current_board = chess.Board(self.input_info["fens"][-1])
            else:
                try:
                    start_fen = self.input_info["fens"][start_fen_index]
                    test_board = chess.Board(start_fen)
                    valid_history = True
                    for move_uci in self.input_info["last_moves"]:
                        try:
                            move = chess.Move.from_uci(move_uci)
                            if move in test_board.legal_moves:
                                test_board.push(move)
                            else:
                                self.logger.add_log(f"ERROR: Illegal move {move_uci} found in history for FEN {test_board.fen()}. Resetting board.\n")
                                valid_history = False
                                break
                        except ValueError:
                             self.logger.add_log(f"ERROR: Invalid UCI move string '{move_uci}' in history. Resetting board.\n")
                             valid_history = False
                             break

                    if valid_history:
                        # Check if the final reconstructed board matches the last provided FEN
                        final_expected_fen = self.input_info["fens"][-1]
                        if test_board.fen() == final_expected_fen:
                            self.current_board = test_board.copy()
                            self.logger.add_log("Successfully reconstructed board from history.\n")
                        else:
                            self.logger.add_log(f"ERROR: Reconstructed FEN {test_board.fen()} does not match last provided FEN {final_expected_fen}. Using last FEN.\n")
                            self.current_board = chess.Board(final_expected_fen)
                    else:
                         self.current_board = chess.Board(self.input_info["fens"][-1])

                except (ValueError, IndexError) as e:
                    self.logger.add_log(f"ERROR: Could not sync history: {e}. Defaulting to last known fen.\n")
                    self.current_board = chess.Board(self.input_info["fens"][-1])

        elif len(self.input_info["fens"]) >= 1:
             self.logger.add_log("No last moves provided or history mismatch, resorting to last fen in history.\n")
             self.current_board = chess.Board(self.input_info["fens"][-1])
        else:
             self.logger.add_log("ERROR: No FENs provided. Resetting to starting position.\n")
             self.current_board = chess.Board()
             self.input_info["fens"] = [chess.STARTING_FEN]


        # Validate side and turn consistency
        if self.input_info["side"] is not None and self.current_board.turn != self.input_info["side"]:
            self.logger.add_log(f"WARNING: Current board turn ({self.current_board.turn}) does not match engine side ({self.input_info['side']}). This might indicate an issue with the provided FEN history or side information.\n")
            # Depending on the desired behavior, you might force the side or log a more critical error.
            # For now, we'll trust the FEN's turn indicator but log the discrepancy.

        self.analytics_updated = False # Reset analytics status after update

    def get_board(self) -> chess.Board:
        return self.current_board.copy()

    def get_info(self, key: str, default=None):
        return self.input_info.get(key, default)

    def get_side(self):
        return self.input_info.get("side")

    def get_fen_history(self) -> list:
        return self.input_info.get("fens", [])

    def get_move_history(self) -> list:
        return self.input_info.get("last_moves", [])

    def get_prev_board(self) -> chess.Board | None:
        fens = self.get_fen_history()
        if len(fens) >= 2:
            try:
                return chess.Board(fens[-2])
            except ValueError:
                self.logger.add_log(f"ERROR: Invalid FEN '{fens[-2]}' found in history at index -2.\n")
                return None
        return None

    def get_prev_prev_board(self) -> chess.Board | None:
        fens = self.get_fen_history()
        if len(fens) >= 3:
            try:
                return chess.Board(fens[-3])
            except ValueError:
                 self.logger.add_log(f"ERROR: Invalid FEN '{fens[-3]}' found in history at index -3.\n")
                 return None
        return None

    def set_analytics_updated(self, status: bool):
        self.analytics_updated = status

    def is_analytics_updated(self) -> bool:
        return self.analytics_updated