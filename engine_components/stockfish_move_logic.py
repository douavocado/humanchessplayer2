import random
import numpy as np
import chess
from .logger import Logger
from .state_manager import StateManager
from .analyzer import Analyzer
from .mood_manager import MoodManager # Needed for mood-based adjustments
from common.utils import extend_mate_score

class StockfishMoveLogic:
    def __init__(self, logger: Logger, state_manager: StateManager, analyzer: Analyzer, mood_manager: MoodManager):
        self.logger = logger
        self.state_manager = state_manager
        self.analyzer = analyzer
        self.mood_manager = mood_manager

    def get_stockfish_move(self, board: chess.Board = None, analysis: list = None, last_opp_move_uci: str = None) -> str | None:
        """
        Selects a move based purely on Stockfish analysis, often used in time pressure.
        Applies some randomization based on move distance and evaluation.

        Args:
            board: The board state to analyze. Defaults to current board from state_manager.
            analysis: Pre-computed Stockfish analysis. Defaults to current analysis from analyzer.
            last_opp_move_uci: The UCI string of the opponent's last move (used for distance calc).

        Returns:
            The UCI string of the selected move, or None if no move can be determined.
        """
        self.logger.add_log("Selecting move using pure Stockfish logic...\n")

        current_board = board if board is not None else self.state_manager.get_board()
        current_analysis = analysis if analysis is not None else self.analyzer.get_stockfish_analysis()
        side = self.state_manager.get_side()
        mood = self.mood_manager.get_mood()

        if side is None:
            self.logger.add_log("ERROR: Cannot get Stockfish move, side not set.\n")
            return None
        if not current_analysis:
            self.logger.add_log("ERROR: No Stockfish analysis available to select a move.\n")
            # As a last resort, pick a random legal move if possible
            legal_moves = list(current_board.legal_moves)
            if legal_moves:
                 move = random.choice(legal_moves).uci()
                 self.logger.add_log(f"WARNING: No analysis, returning random legal move: {move}\n")
                 return move
            else:
                 self.logger.add_log("ERROR: No analysis and no legal moves.\n")
                 return None


        # Sample a subset of moves from the analysis for consideration
        total_moves = len(current_analysis)
        sample_n = max(int(total_moves * 0.6), 1) # Sample ~60% of moves, at least 1
        self.logger.add_log(f"Sampling {sample_n} moves from {total_moves} available analysis lines.\n")
        # Ensure sample_n is not larger than total_moves
        sample_n = min(sample_n, total_moves)
        sampled_analysis = random.sample(current_analysis, sample_n)

        # Extract evaluations for sampled moves
        move_eval_dic = {}
        for entry in sampled_analysis:
            if 'pv' in entry and entry['pv']:
                move_uci = entry['pv'][0].uci()
                # Ensure the move is actually legal in the current position
                try:
                    move_obj = chess.Move.from_uci(move_uci)
                    if move_obj in current_board.legal_moves:
                         score = entry.get('score')
                         if score:
                              move_eval_dic[move_uci] = extend_mate_score(score.pov(side).score(mate_score=2500))
                         else:
                              self.logger.add_log(f"WARNING: No score found for move {move_uci} in analysis entry.\n")
                    else:
                         self.logger.add_log(f"WARNING: Sampled move {move_uci} is not legal in current position {current_board.fen()}. Skipping.\n")

                except ValueError:
                     self.logger.add_log(f"WARNING: Invalid UCI {move_uci} in analysis. Skipping.\n")
            else:
                 self.logger.add_log("WARNING: Analysis entry missing 'pv' or PV is empty.\n")


        if not move_eval_dic:
            self.logger.add_log("ERROR: No valid moves found after sampling and validation.\n")
            legal_moves = list(current_board.legal_moves)
            return random.choice(legal_moves).uci() if legal_moves else None


        # --- Special Cases (Mate / Endgame / Time Pressure) ---
        own_time = max(self.state_manager.get_info("self_clock_times", [1])[-1], 1)
        opp_time = max(self.state_manager.get_info("opp_clock_times", [1])[-1], 1)
        top_engine_move = max(move_eval_dic, key=move_eval_dic.get)
        top_engine_eval = move_eval_dic[top_engine_move]

        # If finding mate and under time pressure vs opponent, strongly prefer the mating line
        if opp_time > own_time and top_engine_eval >= 2490 and mood == "hurry": # Mate in <= 10
            if np.random.random() < 0.8: # High probability
                self.logger.add_log(f"Spotted mate in {2500 - top_engine_eval} moves while opponent has more time ({opp_time:.1f}s > {own_time:.1f}s). Playing top engine move {top_engine_move}.\n")
                return top_engine_move
            else:
                self.logger.add_log(f"Spotted mate in {2500 - top_engine_eval}, but not playing top move by chance.\n")

        # If in endgame (<10 pieces) and hurrying, increase chance of playing best move
        no_pieces = len(chess.SquareSet(current_board.occupied))
        if no_pieces < 10 and mood == "hurry":
            if np.random.random() < 0.4: # Moderate probability
                self.logger.add_log(f"Endgame ({no_pieces} pieces) and hurrying. Playing top engine move {top_engine_move}.\n")
                return top_engine_move
            else:
                self.logger.add_log(f"Endgame ({no_pieces} pieces) and hurrying, but not playing top move by chance.\n")

        # --- Calculate Move Appeal (Eval + Distance + Noise) ---
        self.logger.add_log(f"Sampled moves and evals: { {current_board.san(chess.Move.from_uci(k)): v for k,v in move_eval_dic.items()} }\n")

        # Calculate move distances (lower is better)
        move_distance_dic = {}
        last_own_move_uci = None
        move_history = self.state_manager.get_move_history()
        # Need the move before the opponent's last move (our last move)
        if len(move_history) >= 2:
             last_own_move_uci = move_history[-2]

        for move_uci in move_eval_dic.keys():
            distance = 0
            try:
                move_obj = chess.Move.from_uci(move_uci)
                # Distance from where our *last* move ended to where *this* move starts
                if last_own_move_uci:
                    last_own_move_obj = chess.Move.from_uci(last_own_move_uci)
                    distance += chess.square_distance(last_own_move_obj.to_square, move_obj.from_square)
                # Distance travelled by the piece in *this* move (weighted less)
                distance += 0.5 * chess.square_distance(move_obj.from_square, move_obj.to_square)
                move_distance_dic[move_uci] = distance
            except ValueError:
                 self.logger.add_log(f"WARNING: Invalid UCI {move_uci} during distance calculation. Assigning high distance.\n")
                 move_distance_dic[move_uci] = 99 # Assign high distance if UCI invalid


        self.logger.add_log(f"Move distances: { {current_board.san(chess.Move.from_uci(k)): v for k,v in move_distance_dic.items()} }\n")

        # Combine evaluation and distance into an 'appeal' score
        # Higher eval is better, lower distance is better
        # Scale eval influence by remaining time (more time = more weight on eval)
        move_appealing_dic = {}
        for move_uci in move_eval_dic.keys():
             eval_component = move_eval_dic[move_uci] * (own_time + 5) / 2000
             distance_component = move_distance_dic.get(move_uci, 99) # Use high distance if missing
             move_appealing_dic[move_uci] = 10 + eval_component - distance_component # Base appeal + eval - distance

        self.logger.add_log(f"Combined appeal scores (pre-noise): { {current_board.san(chess.Move.from_uci(k)): v for k,v in move_appealing_dic.items()} }\n")

        # Add noise based on time pressure (more noise when time is lower)
        noise_level = max(0, (15 - own_time)) / 15 # Noise increases linearly below 15s
        noisy_appealing_dic = {}
        for move_uci in move_appealing_dic.keys():
             noise = noise_level * np.random.randn() * 5 # Scale noise effect
             noisy_appealing_dic[move_uci] = move_appealing_dic[move_uci] + noise

        self.logger.add_log(f"Appeal scores after noise (Level: {noise_level:.2f}): { {current_board.san(chess.Move.from_uci(k)): v for k,v in noisy_appealing_dic.items()} }\n")

        # Choose the move with the highest noisy appeal score
        chosen_move = max(noisy_appealing_dic, key=noisy_appealing_dic.get)
        self.logger.add_log(f"Chosen Stockfish move (low time logic): {chosen_move} ({current_board.san(chess.Move.from_uci(chosen_move))})\n")

        return chosen_move