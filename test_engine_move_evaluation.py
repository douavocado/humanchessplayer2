import unittest
from unittest.mock import Mock, patch, MagicMock
import chess
import chess.engine
import chess.polyglot
import numpy as np
import os
import sys
import random

from engine import Engine


class TestEngineEvaluation(unittest.TestCase):
    """Test suite for the move evaluation methods of the Engine class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create mock objects for external dependencies
        self.mock_stockfish_scorer = Mock()
        self.mock_move_scorer = Mock()
        self.mock_opening_book = Mock()
        self.mock_stockfish = Mock()
        self.mock_ponder_stockfish = Mock()
        
        # Patch external dependencies - updated paths to match new structure
        self.patcher1 = patch('engine_components.stockfish_move_logic.StockfishMoveLogic', return_value=self.mock_stockfish_scorer)
        self.patcher2 = patch('engine_components.scorers.Scorers', return_value=self.mock_move_scorer)
        self.patcher3 = patch('chess.polyglot.open_reader', return_value=self.mock_opening_book)
        self.patcher4 = patch('engine_components.analyzer.STOCKFISH', self.mock_stockfish)
        self.patcher5 = patch('engine_components.ponderer.PONDER_STOCKFISH', self.mock_ponder_stockfish)
        
        # Start the patchers
        self.mock_stockfish_selector_class = self.patcher1.start()
        self.mock_move_scorer_class = self.patcher2.start()
        self.mock_open_reader = self.patcher3.start()
        self.patcher4.start()  # No need to store reference, we're using self.mock_stockfish directly
        self.patcher5.start()  # No need to store reference, we're using self.mock_ponder_stockfish directly
        
        # Patch the __init__ method to disable writing logs at startup
        with patch('engine_components.logger.Logger.write_log'):
            # Create a test engine instance
            self.log_file = os.path.join(os.getcwd(), 'test_log.txt')
            self.engine = Engine(playing_level=6, log_file=self.log_file)
        
        # Flag to track test failure
        self.test_failed = False
        
        # Set up basic input info for testing
        self.basic_input_info = {
            'side': chess.WHITE,
            'fens': ['rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'],
            'self_clock_times': [180],
            'opp_clock_times': [180],
            'self_initial_time': 180,
            'opp_initial_time': 180,
            'last_moves': []
        }
        
        # Set up mock analysis results
        self.mock_analysis = [
            {
                'pv': [chess.Move.from_uci('e2e4')],
                'score': chess.engine.PovScore(chess.engine.Cp(50), chess.WHITE)
            },
            {
                'pv': [chess.Move.from_uci('d2d4')],
                'score': chess.engine.PovScore(chess.engine.Cp(40), chess.WHITE)
            }
        ]
        
        # Configure mock stockfish to return the mock analysis
        self.mock_stockfish.analyse.return_value = self.mock_analysis
        
        # Update engine state manager with basic info
        self.engine.state_manager.input_info = self.basic_input_info.copy()
        self.engine.state_manager.current_board = chess.Board()
        self.engine.analyzer.stockfish_analysis = self.mock_analysis
        self.engine.state_manager.analytics_updated = True
        
        # Initialize lucas_analytics in the engine
        self.engine.state_manager.lucas_analytics = {
            "complexity": 10,
            "win_prob": 0.5,
            "eff_mob": 20,
            "narrowness": 5,
            "activity": 15,
        }
        
        # Initialize mood for tests
        self.engine.mood_manager.current_mood = "confident"

    def run(self, result=None):
        """Override the run method to track test failures and output logs immediately."""
        self.test_failed = False
        test_method_name = getattr(self, '_testMethodName', None)
        test_method = getattr(self, test_method_name, None)
        
        # Call the original run method
        super(TestEngineEvaluation, self).run(result)
        
        # Check if the test failed
        if result and test_method_name in [failure[0]._testMethodName for failure in result.failures + result.errors]:
            self.test_failed = True
            # Output engine logs immediately
            if os.path.exists(self.log_file):
                print(f"\nTest {test_method_name} failed. Engine log contents:")
                print("-" * 50)
                with open(self.log_file, 'r') as f:
                    print(f.read())
                print("-" * 50)

    def tearDown(self):
        """Clean up after each test method."""
        # Print log if test failed
        if self.test_failed and os.path.exists(self.log_file):
            print("\nTest failed. Engine log contents:")
            print("-" * 50)
            with open(self.log_file, 'r') as f:
                print(f.read())
            print("-" * 50)
        
        # Stop the patchers
        self.patcher1.stop()
        self.patcher2.stop()
        self.patcher3.stop()
        self.patcher4.stop()
        self.patcher5.stop()
        
        # Remove test log file if it exists
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def test_alter_move_probabilties(self):
        """Test the alter_move_probabilties method."""
        # Set up test data
        self.engine.state_manager.input_info = self.basic_input_info.copy()
        move_dic = {'e2e4': 0.7, 'd2d4': 0.3}
        board = chess.Board()
        
        # Mock extend_mate_score to return the score unchanged
        with patch('common.utils.extend_mate_score', side_effect=lambda x: x):
            # Test normal adjustment
            self.engine.mood_manager.current_mood = "confident"
            self.engine.analyzer.stockfish_analysis = self.mock_analysis
            
            # Call the method through the human_move_logic component
            adjusted_dic = self.engine.human_move_logic.alter_move_probabilties(move_dic, board)
            
            # Verify that probabilities were adjusted but still sum to approximately 1
            self.assertAlmostEqual(sum(adjusted_dic.values()), 1.0, places=5)
            
            # Test hurry mood
            self.engine.mood_manager.current_mood = "hurry"
            self.engine.state_manager.input_info["self_clock_times"] = [10]
            
            # Call the method through the human_move_logic component
            adjusted_dic = self.engine.human_move_logic.alter_move_probabilties(move_dic, board)
            
            # Verify that probabilities were adjusted but still sum to approximately 1
            self.assertAlmostEqual(sum(adjusted_dic.values()), 1.0, places=5)


if __name__ == '__main__':
    unittest.main()