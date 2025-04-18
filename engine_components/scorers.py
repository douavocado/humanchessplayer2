import chess.engine
from models.models import MoveScorer, StockFishSelector
from common.constants import (
    PATH_TO_STOCKFISH,
    MOVE_FROM_WEIGHTS_OP_PTH, MOVE_FROM_WEIGHTS_MID_PTH, MOVE_FROM_WEIGHTS_END_PTH,
    MOVE_TO_WEIGHTS_OP_PTH, MOVE_TO_WEIGHTS_MID_PTH, MOVE_TO_WEIGHTS_END_PTH
)
from .logger import Logger # Use relative import

class Scorers:
    def __init__(self, logger: Logger):
        self.logger = logger
        self.logger.add_log("Initializing move scorers...\n")
        try:
            # Setting up scorers for moves
            self.human_scorers = {
                "opening": MoveScorer(MOVE_FROM_WEIGHTS_OP_PTH, MOVE_TO_WEIGHTS_OP_PTH),
                "midgame": MoveScorer(MOVE_FROM_WEIGHTS_MID_PTH, MOVE_TO_WEIGHTS_MID_PTH),
                "endgame": MoveScorer(MOVE_FROM_WEIGHTS_END_PTH, MOVE_TO_WEIGHTS_END_PTH),
            }
            self.stockfish_scorer = StockFishSelector(PATH_TO_STOCKFISH)
            self.logger.add_log("Move scorers initialized successfully.\n")
        except Exception as e:
            self.logger.add_log(f"ERROR: Failed to initialize scorers: {e}\n")
            # Depending on severity, might want to raise the exception or handle differently
            self.human_scorers = {}
            self.stockfish_scorer = None
            raise RuntimeError(f"Failed to initialize scorers: {e}") from e


    def get_human_scorer(self, game_phase: str) -> MoveScorer | None:
        """Returns the appropriate human scorer based on the game phase."""
        scorer = self.human_scorers.get(game_phase)
        if scorer is None:
            self.logger.add_log(f"WARNING: No human scorer found for game phase '{game_phase}'.\n")
        return scorer

    def get_stockfish_scorer(self) -> StockFishSelector | None:
        """Returns the Stockfish scorer instance."""
        if self.stockfish_scorer is None:
             self.logger.add_log("WARNING: Stockfish scorer was not initialized.\n")
        return self.stockfish_scorer

    def close_engines(self):
        self.stockfish_scorer.close_engine()
