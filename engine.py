# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:29:08 2024

@author: xusem
"""
import os
import datetime
import chess
import random
import atexit
import threading
import signal
import sys

# Import all the components
from engine_components.logger import Logger
from engine_components.state_manager import StateManager
from engine_components.scorers import Scorers
from engine_components.analyzer import Analyzer
from engine_components.opening_book_handler import OpeningBookHandler
from engine_components.mood_manager import MoodManager
from engine_components.decision_logic import (
    decide_resign, decide_human_filters, decide_breadth,
    get_time_taken, check_obvious_move
)
from engine_components.human_move_logic import HumanMoveLogic
from engine_components.stockfish_move_logic import StockfishMoveLogic
from engine_components.ponderer import Ponderer
from engine_components.premover import Premover

from common.board_information import phase_of_game

# Flag to track if cleanup has been performed
_cleanup_performed = False

class Engine:
    """
    Class for engine instance.
    
    The Engine is responsible for the following things ONLY:
    receiving board information -> outputting move and premoves
    
    All other history related data to do with past moves etc are not handled
    in the Engine instance. They are handled in the client wrapper
    """
    def __init__(self, playing_level: int = 6, 
                 log_file: str = None, 
                 opening_book_path: str = "Opening_books/bullet.bin"):
        """
        Initialize the Engine with all its components.
        
        Args:
            playing_level: Strength level of the engine (1-10)
            log_file: Path to log file. If None, a default path is generated.
            opening_book_path: Path to the opening book file.
        """
        # Set default log file path if not provided
        if log_file is None:
            log_dir = os.path.join(os.getcwd(), 'Engine_logs')
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, str(datetime.datetime.now()).replace(" ", "").replace(":", "_") + '.txt')
        
        # Initialize components
        self.playing_level = playing_level
        self.logger = Logger(log_file)
        self.state_manager = StateManager(self.logger)
        self.scorers = Scorers(self.logger)
        self.analyzer = Analyzer(self.logger, self.state_manager)
        self.opening_book_handler = OpeningBookHandler(self.logger, opening_book_path)
        self.mood_manager = MoodManager(self.logger, self.state_manager, self.analyzer)
        self.human_move_logic = HumanMoveLogic(self.logger, self.state_manager, self.scorers, self.mood_manager, self.analyzer)
        self.stockfish_move_logic = StockfishMoveLogic(self.logger, self.state_manager, self.analyzer, self.mood_manager)
        self.ponderer = Ponderer(self.logger, self.state_manager, self.analyzer, self.human_move_logic, self.mood_manager)
        self.premover = Premover(self.logger, self.state_manager, self.analyzer, self.opening_book_handler, self.stockfish_move_logic)
        
        self.logger.add_log(f"Engine initialized with playing level {playing_level}\n")
        self.logger.add_log(f"Opening book path: {opening_book_path}\n")
        self.logger.write_log()
        
        # Register cleanup with atexit to ensure resources are released even if __del__ isn't called
        atexit.register(self.cleanup)

    def _write_log(self):
        """Writes down thinking into a log file for debugging."""
        self.logger.write_log()
    
    def update_info(self, info_dic: dict, auto_update_analytics: bool = True):
        """
        Updates the engine state with new board information.
        
        Args:
            info_dic: Dictionary containing board state information:
                - side: either chess.WHITE or chess.BLACK - indicates what side we are
                - fens: List of fens ordered with most recent fen last
                - self_clock_times and opp_clock_times: List of past clock times
                - self_initial_time and opp_initial_time: Starting clock times
                - last_moves: A list of moves made with most recent last (UCI format)
            auto_update_analytics: Whether to automatically calculate analytics after update
        """
        self.state_manager.update_info(info_dic)
        
        if auto_update_analytics:
            self.calculate_analytics()
    
    def calculate_analytics(self):
        """
        Calculates analytics for the current board position.
        This includes Stockfish analysis and Lucas analytics.
        """
        self.analyzer.calculate_analytics()
        # After analytics are calculated, determine the mood
        self.mood_manager.determine_mood()
    
    def get_stockfish_move(self, board: chess.Board = None, analysis = None, last_move_uci: str = None):
        """
        Gets a move using only Stockfish (no human filters).
        
        Args:
            board: The board to analyze. If None, uses current board.
            analysis: Pre-computed analysis. If None, uses current analysis.
            last_move_uci: Last move UCI string for distance calculation.
            
        Returns:
            UCI string of the selected move
        """
        return self.stockfish_move_logic.get_stockfish_move(board, analysis, last_move_uci)
    
    def get_human_move(self, target_time: float = 0.5):
        """
        Gets a move using human-like evaluation and filtering.
        
        Args:
            target_time: Target time to spend on move calculation
            
        Returns:
            UCI string of the selected move
        """
        current_board = self.state_manager.get_board()
        game_phase = self.state_manager.get_info("game_phase", None)
        
        # If game phase not provided in state, calculate it
        if game_phase is None:
            from common.board_information import phase_of_game
            game_phase = phase_of_game(current_board)
            self.logger.add_log(f"Calculated game phase: {game_phase}\n")
        
        # Check opening book first
        if game_phase == "opening" and self.opening_book_handler.is_book_loaded():
            self.logger.add_log("Checking opening book for current position...\n")
            book_moves = self.opening_book_handler.find_book_moves(current_board)
            
            if book_moves:
                self.logger.add_log(f"Found {len(book_moves)} matching positions in opening database.\n")
                top_results = book_moves[:5]
                for res in top_results:
                    self.logger.add_log(f"{current_board.san(res.move)}: {res.weight}\n")
                
                excluded_moves = [res.move for res in book_moves[5:]]
                book_entry = self.opening_book_handler.get_weighted_choice(current_board, exclude_moves=excluded_moves)
                
                if book_entry:
                    book_move = book_entry.move.uci()
                    self.logger.add_log(f"Chosen move from opening book: {current_board.san(book_entry.move)}\n")
                    return book_move
            
            self.logger.add_log("No suitable opening book move found. Using human move logic.\n")
        
        # Get human probabilities and alter them
        un_altered_move_dic = self.human_move_logic.get_human_probabilities(current_board, game_phase)
        
        # Handle case where too few human moves are found
        if len(un_altered_move_dic) == 0 or (
            len(list(current_board.legal_moves)) - len(un_altered_move_dic) > 2 and 
            len(un_altered_move_dic) <= 3
        ):
            self.logger.add_log("Too few human probability moves found. Defaulting to Stockfish move.\n")
            return self.get_stockfish_move()
        
        # Alter probabilities based on heuristics
        altered_move_dic = self.human_move_logic.alter_move_probabilties(un_altered_move_dic, current_board)
        
        # Decide search breadth
        no_root_moves = decide_breadth(
            self.logger, self.state_manager, self.analyzer, 
            self.mood_manager, self.playing_level, target_time
        )
        
        # Get top moves based on altered probabilities
        human_move_ucis = list(altered_move_dic.keys())
        root_moves = human_move_ucis[:no_root_moves]
        
        # Get evaluations for these moves from Stockfish analysis
        stockfish_analysis = self.analyzer.get_stockfish_analysis()
        human_move_evals = {}
        
        for analysis_object in stockfish_analysis:
            if 'pv' in analysis_object and analysis_object['pv']:
                move_uci = analysis_object['pv'][0].uci()
                if move_uci in root_moves:
                    from common.utils import extend_mate_score
                    side = self.state_manager.get_side()
                    eval_ = extend_mate_score(analysis_object['score'].pov(side).score(mate_score=2500))
                    human_move_evals[move_uci] = [eval_, 0]  # [eval, depth_considered]
        
        # Re-evaluate top moves with deeper search if time allows
        re_evaluations = int(max(target_time / 0.12 - 1, 0))
        self.logger.add_log(f"Planning to re-evaluate {re_evaluations} of the top human variations.\n")
        
        top_human_moves = sorted(human_move_evals.keys(), key=lambda x: human_move_evals[x][0], reverse=True)
        depth = (re_evaluations // no_root_moves) + 1
        
        re_evaluate_moves = random.sample(top_human_moves, min(re_evaluations, len(top_human_moves)))
        time_allowed = target_time - 0.1  # Reserve some time for final processing
        
        if re_evaluate_moves:
            re_evaluations_dic = self.ponderer.re_evaluate_moves(
                current_board, re_evaluate_moves, no_root_moves,
                depth=depth, prev_board=self.state_manager.get_prev_board(),
                limit=[depth * no_root_moves, time_allowed], log=True
            )
            
            # Update evaluations with re-evaluated results
            for move_uci, (eval_, depth_considered) in re_evaluations_dic.items():
                if eval_ is not None:
                    human_move_evals[move_uci] = [eval_, depth_considered]
        
        # Apply noise and other adjustments to make final selection
        from common.board_information import phase_of_game, calculate_threatened_levels
        import numpy as np
        
        noise_factors = {"opening": 0.8, "midgame": 1.2, "endgame": 0.3}
        noise_phase = noise_factors.get(game_phase, 1.0)
        
        eval_with_noise = {}
        for move_uci, (eval_, depth_considered) in human_move_evals.items():
            # Base noise calculation
            base_noise_sd = 40 * (np.tanh(eval_ / (self.playing_level * 50)))**2 + 20
            noise_sd = 4 * base_noise_sd / (max(target_time, 0.1) * (depth_considered + 4))
            
            # Apply noise
            noise = np.random.randn() * noise_sd * noise_phase
            
            # Penalties for shallow evaluation
            depth_penalty = 20 * (2 - depth_considered) if depth_considered > 0 else 70
            
            # Bonus for captures
            move_obj = chess.Move.from_uci(move_uci)
            capture_bonus = 0
            if current_board.is_capture(move_obj):
                capture_bonus = 40 * calculate_threatened_levels(move_obj.to_square, current_board)
            
            # Final score
            eval_with_noise[move_uci] = eval_ + noise - depth_penalty + capture_bonus
        
        # Select the move with the highest adjusted evaluation
        top_move = max(eval_with_noise.keys(), key=lambda x: eval_with_noise[x])
        self.logger.add_log(f"Selected human move: {current_board.san(chess.Move.from_uci(top_move))}\n")
        
        return top_move
    
    def check_obvious_move(self):
        """
        Checks if there's an obvious move in the current position.
        
        Returns:
            (move_uci, is_obvious): The move UCI string and whether it's obvious
        """
        return check_obvious_move(
            self.logger, self.state_manager, self.analyzer, self.mood_manager
        )
    
    def decide_resign(self):
        """
        Decides whether to resign the current position.
        
        Returns:
            bool: True if the engine should resign, False otherwise
        """
        return decide_resign(
            self.logger, self.state_manager, self.analyzer
        )
    
    def get_premove(self, board: chess.Board, takeback_only: bool = False):
        """
        Gets a premove for the given board position.
        
        Args:
            board: The board position (opponent's turn)
            takeback_only: If True, only return a premove if it's a takeback
            
        Returns:
            UCI string of the premove, or None if no suitable premove is found
        """
        return self.premover.get_premove(board, takeback_only)
    
    def ponder(self, board: chess.Board, time_allowed: float, search_width: int, 
               time_per_position: float = 0.1, prev_board: chess.Board = None, 
               ponder_width: int = None, use_ponder: bool = False):
        """
        Ponders possible opponent moves and prepares responses using human-like evaluation.
        
        Args:
            board: The board position (opponent's turn)
            time_allowed: Maximum time to spend pondering
            search_width: Number of candidate moves to consider for each position
            time_per_position: Approximate time to spend per position
            prev_board: Previous board state for context
            ponder_width: Number of opponent moves to consider
            use_ponder: Whether to use dedicated ponder engine
            
        Returns:
            Dictionary mapping board FENs to response move UCIs
        """
        return self.ponderer.human_ponder(
            board, time_allowed, search_width, time_per_position, 
            prev_board, ponder_width, use_ponder
        )
    
    def stockfish_ponder(self, board: chess.Board, time_allowed: float, ponder_width: int, 
                         use_ponder: bool = False, root_moves: list = None):
        """
        Ponders using only Stockfish analysis (faster than human pondering).
        
        Args:
            board: The board position (opponent's turn)
            time_allowed: Maximum time to spend pondering
            ponder_width: Number of lines to analyze
            use_ponder: Whether to use dedicated ponder engine
            root_moves: Specific moves to consider
            
        Returns:
            Dictionary mapping board FENs to response move UCIs
        """
        return self.ponderer.stockfish_ponder(
            board, time_allowed, ponder_width, use_ponder, root_moves
        )
    
    def make_move(self, log: bool = True):
        """
        Main function to generate a move from the current position.
        
        Args:
            log: Whether to write logs to file
            
        Returns:
            Dictionary containing:
            - move_made: UCI string of the selected move
            - time_take: Time to display for the move
            - premove: Optional premove to make immediately after
            - ponder_dic: Optional dictionary of pre-calculated responses
        """
        if log:
            self._write_log()
        
        self.logger.add_log("Make move function called.\n")
        return_dic = {}
        
        # Check if analytics are up to date
        if not self.state_manager.is_analytics_updated():
            self.logger.add_log("WARNING: Making move with outdated analytics. Running calculate_analytics().\n")
            self.calculate_analytics()
        
        # Check for obvious moves first
        obvious_move, obvious_move_found = self.check_obvious_move()
        
        if obvious_move_found:
            return_dic["move_made"] = obvious_move
            use_human_filters = False
            # Calculate display time for the move
            return_dic["time_take"] = get_time_taken(
                self.logger, self.state_manager, self.analyzer, 
                self.mood_manager, obvious=True, human_filters=False
            )
        else:
            # Decide whether to use human filters based on time pressure
            use_human_filters = decide_human_filters(self.logger, self.state_manager)
            
            # Calculate display time for the move
            return_dic["time_take"] = get_time_taken(
                self.logger, self.state_manager, self.analyzer, 
                self.mood_manager, obvious=False, human_filters=use_human_filters
            )
            
            # Get the move
            if use_human_filters:
                # Set target time for human evaluation
                total_time = return_dic["time_take"]
                target_time = min(total_time * 0.6, 2.5) if total_time > 1 else total_time * 0.8
                self.logger.add_log(f"Using human filters with target time {target_time:.3f}s\n")
                
                return_dic["move_made"] = self.get_human_move(target_time=target_time)
            else:
                self.logger.add_log("Using pure Stockfish move due to time pressure.\n")
                return_dic["move_made"] = self.get_stockfish_move()
        
        # Check if opponent blundered on their last move
        self.mood_manager.check_opponent_blunder()
        
        # Consider premove
        current_board = self.state_manager.get_board()
        after_board = current_board.copy()
        after_board.push_uci(return_dic["move_made"])
        
        if after_board.outcome() is None:  # Game not over after our move
            own_time = max(self.state_manager.get_info("self_clock_times", [1])[-1], 1)
            self_initial_time = self.state_manager.get_info("self_initial_time", 60)
            mood = self.mood_manager.get_mood()
            
            # Decide whether to calculate a premove
            if mood == "hurry" and own_time < 20:
                # With some probability return a premove in time trouble
                if random.random() < 0.3 * self_initial_time / (own_time + 0.3 * self_initial_time):
                    return_dic["premove"] = self.get_premove(after_board, takeback_only=False)
                else:
                    # Look for takeback premoves only
                    return_dic["premove"] = self.get_premove(after_board, takeback_only=True)
            elif self_initial_time <= 60 and phase_of_game(current_board) == "opening":
                # With high probability return a premove in bullet opening
                if random.random() < 0.9:
                    return_dic["premove"] = self.get_premove(after_board, takeback_only=False)
                else:
                    # Look for takeback premoves only
                    return_dic["premove"] = self.get_premove(after_board, takeback_only=True)
            else:
                # Look for takeback premoves only in normal situations
                return_dic["premove"] = self.get_premove(after_board, takeback_only=True)
        else:
            return_dic["premove"] = None
        
        # If we have extra time, do some pondering
        time_spent = 0.1  # Approximate time spent so far
        time_left = return_dic["time_take"] - time_spent
        search_width = decide_breadth(
            self.logger, self.state_manager, self.analyzer, 
            self.mood_manager, self.playing_level, time_left
        )
        time_per_position = 0.1
        if time_left > time_per_position * search_width * 1.15:
            self.logger.add_log(f"Have enough time ({time_left:.3f}s) to ponder for the next position.\n")
            ponder_dic = self.ponder(
                after_board, time_left / 1.15, search_width, 
                time_per_position=0.1, prev_board=current_board
            )
        else:
            self.logger.add_log(f"Not enough time ({time_left:.3f}s) to ponder for the next position.\n")
            ponder_dic = None
            
        return_dic["ponder_dic"] = ponder_dic
        
        if log:
            self._write_log()
            
        self.logger.add_log(f"Returning move result: {return_dic}\n")
        return return_dic
    
    def cleanup(self):
        """
        Comprehensive cleanup of all resources.
        This method ensures all components release their resources properly.
        """
        global _cleanup_performed
        
        # Prevent multiple cleanups
        if _cleanup_performed:
            return
            
        _cleanup_performed = True
        
        self.logger.add_log("Performing comprehensive cleanup of all engine resources...\n")
        
        # 1. Find and terminate any non-daemon threads
        try:
            self.logger.add_log("Checking for active threads to terminate...\n")
            for thread in threading.enumerate():
                if thread != threading.current_thread() and not thread.daemon:
                    self.logger.add_log(f"Found active thread: {thread.name}\n")
                    # For threads that support it, try to signal termination
                    if hasattr(thread, 'terminate'):
                        thread.terminate()
                    elif hasattr(thread, 'stop'):
                        thread.stop()
                    # Don't try to join threads here as it might block
        except Exception as e:
            self.logger.add_log(f"Error while terminating threads: {e}\n")
        
        # 2. Clean up Stockfish and other external processes - very important to do this first
        # 2a. Ponderer (closes dedicated Stockfish instance)
        if hasattr(self, 'ponderer'):
            try:
                self.logger.add_log("Cleaning up ponderer resources...\n")
                self.ponderer.close_ponder_engine()
            except Exception as e:
                self.logger.add_log(f"Error during ponderer cleanup: {e}\n")
        
        # 2b. Analyzer (closes main Stockfish instance)
        if hasattr(self, 'analyzer'):
            try:
                self.logger.add_log("Cleaning up analyzer resources...\n")
                self.analyzer.close_engine()
            except Exception as e:
                self.logger.add_log(f"Error during analyzer cleanup: {e}\n")
                
        # 2c. Scorers (may have Stockfish or other engines)
        if hasattr(self, 'scorers'):
            try:
                self.logger.add_log("Cleaning up scorer resources...\n")
                self.scorers.close_engines()
            except Exception as e:
                self.logger.add_log(f"Error during scorers cleanup: {e}\n")
        
        # 3. Opening book handler (file resources)
        if hasattr(self, 'opening_book_handler'):
            try:
                self.logger.add_log("Cleaning up opening book resources...\n")
                self.opening_book_handler.close_book()
            except Exception as e:
                self.logger.add_log(f"Error during opening book cleanup: {e}\n")
        
        # 4. Any other components that might need cleanup
        # Add more component cleanup here if needed
        
        # 5. Final log write
        try:
            self.logger.add_log("Engine cleanup completed.\n")
            self.logger.write_log()
        except Exception as e:
            # If logger fails, print to console as last resort
            print(f"Error writing final log during cleanup: {e}")
    
    def __del__(self):
        """
        Clean up resources when the engine is destroyed.
        Calls the comprehensive cleanup method.
        """
        try:
            self.cleanup()
        except Exception as e:
            # If cleanup fails during __del__, try to at least log it
            try:
                self.logger.add_log(f"Error during Engine __del__ cleanup: {e}\n")
                self.logger.write_log()
            except:
                # Last resort if logger also fails
                print(f"Fatal error during Engine cleanup: {e}")





if __name__ == "__main__":
    import time
    import numpy as np
    # Set random seeds for reproducibility
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    try:
        import torch
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
    except ImportError:
        pass  # torch not available
    # Add signal handlers for graceful termination
    def signal_handler(sig, frame):
        print("\nCleaning up resources before exit...")
        # Force exit after a timeout to prevent hanging
        def force_exit():
            print("Forced exit due to timeout during cleanup")
            os._exit(1)
        
        # Set timeout to force exit if cleanup takes too long
        timer = threading.Timer(5.0, force_exit)
        timer.daemon = True
        timer.start()
        
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    engine = Engine(playing_level=3)
    # b = chess.Board("3r2k1/3r1p1p/PQ2p1p1/8/5q2/2P2N2/1P3PP1/R3K2R w KQ - 1 24")
    input_dic ={'fens': ['2q1r1k1/Rp1n1pb1/1Pp3pp/2P5/2pP4/2N5/5PPP/2BQ2K1 w - - 0 26', '2q1r1k1/Rp1n1pb1/1Pp3pp/2P5/2pP4/2N5/3B1PPP/3Q2K1 b - - 1 26', '2q1r1k1/Rp1n1p2/1Pp3pp/2P5/2pb4/2N5/3B1PPP/3Q2K1 w - - 0 27', '2q1r1k1/Rp1n1p2/1Pp3pp/2P5/2pb4/8/3BNPPP/3Q2K1 b - - 1 27', '2q1r1k1/Rp1n1p2/1Pp3pp/2b5/2p5/8/3BNPPP/3Q2K1 w - - 0 28', '2q1r1k1/Rp1n1p2/1Pp3pB/2b5/2p5/8/4NPPP/3Q2K1 b - - 0 28', '2q1r1k1/Rp1n1p2/1bp3pB/8/2p5/8/4NPPP/3Q2K1 w - - 0 29', '2q1r1k1/Rp1n1p2/1bp3pB/8/2p5/8/4NPPP/Q5K1 b - - 1 29'], 'self_clock_times': [39, 37, 36, 34, 34, 32, 31, 29], 'opp_clock_times': [46, 42, 42, 41, 39, 37, 36, 35], 'last_moves': ['c1d2', 'g7d4', 'c3e2', 'd4c5', 'd2h6', 'c5b6', 'd1a1'], 'side': False, 'self_initial_time': 60, 'opp_initial_time': 60}
    start = time.time()
    engine.update_info(input_dic)
    print(engine.make_move(log=False))
    end = time.time()
    print("finished in {} seconds".format(end-start))
    print("Engine log contents:")
    print(engine.logger.log_buffer)
    
    # Explicitly clean up engine resources and exit
    engine.cleanup()
    del engine
    
    # Force exit to ensure no hanging threads
    os._exit(0)