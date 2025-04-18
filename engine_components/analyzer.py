import time
import chess
import chess.engine
from common.constants import PATH_TO_STOCKFISH
from common.board_information import get_lucas_analytics
from .logger import Logger
from .state_manager import StateManager # Use relative imports

# Initialize Stockfish engine instance here
try:
    STOCKFISH = chess.engine.SimpleEngine.popen_uci(PATH_TO_STOCKFISH)
except Exception as e:
    # Log error and potentially raise or handle it
    print(f"CRITICAL: Failed to initialize Stockfish engine at {PATH_TO_STOCKFISH}: {e}")
    STOCKFISH = None # Indicate failure

class Analyzer:
    def __init__(self, logger: Logger, state_manager: StateManager):
        self.logger = logger
        self.state_manager = state_manager
        self.stockfish_analysis = None # Stores the latest analysis result
        self.lucas_analytics = {
            "complexity": None,
            "win_prob": None,
            "eff_mob": None,
            "narrowness": None,
            "activity": None,
        }
        if STOCKFISH is None:
             self.logger.add_log("CRITICAL: Stockfish engine failed to initialize. Analyzer will not function correctly.\n")


    def calculate_analytics(self, analysis_time_limit: float = 0.05):
        """
        Performs Stockfish analysis and calculates Lucas analytics for the current board state.
        Updates the state_manager's analytics_updated flag.
        """
        if STOCKFISH is None:
            self.logger.add_log("ERROR: Stockfish engine not available. Cannot calculate analytics.\n")
            self.state_manager.set_analytics_updated(False) # Ensure flag reflects failure
            return

        current_board = self.state_manager.get_board()
        self.logger.add_log("Calculating analytics for current board state.\n")
        self.logger.add_log(f"Evaluating from the board (FEN: {current_board.fen()}):\n")
        self.logger.add_log(str(current_board) + "\n")

        # Performing a quick initial analysis of the position
        self.logger.add_log(f"Performing initial analysis with time limit {analysis_time_limit}s.\n")
        start_time = time.time()
        try:
            # Determine number of lines based on legal moves, capped for performance
            legal_moves = list(current_board.legal_moves)
            no_lines = min(len(legal_moves), 50) # Cap multipv for performance

            if no_lines == 0:
                 self.logger.add_log("No legal moves available. Skipping analysis.\n")
                 self.stockfish_analysis = [] # Empty analysis
                 self.lucas_analytics = {k: 0 for k in self.lucas_analytics} # Reset analytics
                 self.state_manager.set_analytics_updated(True) # Mark as 'updated' (though empty)
                 return

            analysis = STOCKFISH.analyse(current_board, limit=chess.engine.Limit(time=analysis_time_limit), multipv=no_lines)

            # Ensure analysis is always a list
            if isinstance(analysis, dict):
                analysis = [analysis]
            elif analysis is None: # Handle potential None return
                 analysis = []

            self.stockfish_analysis = analysis
            end_time = time.time()
            self.logger.add_log(f"Stockfish analysis computed in {end_time - start_time:.4f} seconds.\n")

            # Getting lucas analytics for the position
            self.logger.add_log("Calculating Lucas analytics for the position.\n")
            if not self.stockfish_analysis:
                 self.logger.add_log("WARNING: No Stockfish analysis available to calculate Lucas analytics. Setting to defaults.\n")
                 self.lucas_analytics = {k: 0 for k in self.lucas_analytics} # Reset analytics
            else:
                try:
                    xcomp, xmlr, xemo, xnar, xact = get_lucas_analytics(current_board, analysis=self.stockfish_analysis)
                    lucas_dict = {"complexity": xcomp, "win_prob": xmlr, "eff_mob": xemo, "narrowness": xnar, "activity": xact}
                    self.lucas_analytics.update(lucas_dict)
                    self.logger.add_log(f"Lucas analytics: {lucas_dict}\n")
                except Exception as e:
                    self.logger.add_log(f"ERROR: Failed to calculate Lucas analytics: {e}\n")
                    self.lucas_analytics = {k: 0 for k in self.lucas_analytics} # Reset on error

            self.state_manager.set_analytics_updated(True) # Mark analytics as updated

        except chess.engine.EngineTerminatedError:
             self.logger.add_log("ERROR: Stockfish engine terminated unexpectedly during analysis.\n")
             self.stockfish_analysis = None
             self.state_manager.set_analytics_updated(False)
        except Exception as e:
            self.logger.add_log(f"ERROR: An unexpected error occurred during analysis: {e}\n")
            self.stockfish_analysis = None
            self.state_manager.set_analytics_updated(False)


    def get_stockfish_analysis(self):
        """Returns the stored Stockfish analysis."""
        return self.stockfish_analysis

    def get_lucas_analytics(self):
        """Returns the stored Lucas analytics."""
        return self.lucas_analytics

    def get_lucas_metric(self, metric_name: str, default=None):
        """Returns a specific Lucas metric."""
        return self.lucas_analytics.get(metric_name, default)

    def close_engine(self):
        """Safely closes the Stockfish engine."""
        if STOCKFISH:
            try:
                STOCKFISH.quit()
                self.logger.add_log("Stockfish engine closed successfully.\n")
            except chess.engine.EngineTerminatedError:
                 self.logger.add_log("Stockfish engine already terminated.\n")
            except Exception as e:
                 self.logger.add_log(f"ERROR: Failed to close Stockfish engine: {e}\n")

# Ensure the engine is closed when the module is potentially unloaded or program exits
# This might require more robust handling in a larger application (e.g., using atexit)
def cleanup():
     if STOCKFISH:
         try:
             STOCKFISH.quit()
         except: # Ignore errors during cleanup
             pass

import atexit
atexit.register(cleanup)