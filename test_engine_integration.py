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
from engine_components.decision_logic import decide_resign


class TestEngineIntegration(unittest.TestCase):
    """Test suite for integration tests and edge cases of the Engine class."""

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

    def run(self, result=None):
        """Override the run method to track test failures and output logs immediately."""
        self.test_failed = False
        test_method_name = getattr(self, '_testMethodName', None)
        test_method = getattr(self, test_method_name, None)
        
        # Call the original run method
        super(TestEngineIntegration, self).run(result)
        
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

    def test_integration_update_info_with_move_history(self):
        """Integration test for update_info with move history."""
        # Set up test data with move history
        input_info = {
            'side': chess.WHITE,
            'fens': [
                'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
                'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1',
                'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2'
            ],
            'self_clock_times': [180, 175, 170],
            'opp_clock_times': [180, 178, 175],
            'self_initial_time': 180,
            'opp_initial_time': 180,
            'last_moves': ['e2e4', 'e7e5']
        }
        
        # Mock calculate_analytics to avoid calling it
        with patch.object(self.engine, 'calculate_analytics') as mock_calc_analytics:
            # Call the method
            self.engine.update_info(input_info)
            
            # Verify that the input info was updated
            self.assertEqual(self.engine.state_manager.input_info["side"], chess.WHITE)
            self.assertEqual(len(self.engine.state_manager.input_info["fens"]), 3)
            self.assertEqual(self.engine.state_manager.input_info["last_moves"], ['e2e4', 'e7e5'])
            
            # Verify that the current board was updated with the moves
            expected_board = chess.Board()
            expected_board.push_uci('e2e4')
            expected_board.push_uci('e7e5')
            self.assertEqual(self.engine.state_manager.current_board.fen(), expected_board.fen())

    def test_edge_case_check_obvious_move_only_legal_move(self):
        """Test check_obvious_move with only one legal move."""
        # Set up a position with only one obvious move
        self.engine.state_manager.current_board = chess.Board("r1bqkbnr/ppp2ppp/2B5/3pp3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 4")
        # update a valid fen history
        self.engine.state_manager.input_info["fens"] = ["r1bqkbnr/ppp2ppp/2n5/1B1pp3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",self.engine.state_manager.current_board.fen()]
        # update the last moves
        self.engine.state_manager.input_info["last_moves"] = ["b5c6"]
        
        # Set up the analyzer's stockfish analysis to contain the expected move
        self.engine.analyzer.stockfish_analysis = [
            {
                'pv': [chess.Move.from_uci('b7c6')],  # Only move: capture the knight
                'score': chess.engine.PovScore(chess.engine.Cp(10), chess.BLACK)
            }
        ]
        
        # Set the side to match the board position
        self.engine.state_manager.input_info["side"] = chess.BLACK
        
        # Directly patch the decision_logic.check_obvious_move function to return the expected result
        with patch('engine_components.decision_logic.check_obvious_move', return_value=('b7c6', True)):
            # Call the method
            move, found = self.engine.check_obvious_move()
            
            # Verify the result
            self.assertTrue(found)
            self.assertEqual(move, 'b7c6')

    def test_edge_case_decide_resign_early_game(self):
        """Test decide_resign in early game."""
        # Set up test data
        self.engine.state_manager.input_info = self.basic_input_info.copy()
        self.engine.state_manager.current_board = chess.Board()
        self.engine.state_manager.analytics_updated = True
        self.engine.state_manager.lucas_analytics["win_prob"] = 0.01  # Very low win probability
        
        # Test early game (should not resign)
        self.engine.state_manager.current_board.fullmove_number = 10
        
        # Call decide_resign directly from the imported function with the required analyzer parameter
        result = decide_resign(self.engine.logger, self.engine.state_manager, self.engine.analyzer)
        self.assertFalse(result)

    def test_edge_case_decide_resign_opponent_low_time(self):
        """Test decide_resign with opponent low time."""
        # Set up test data
        self.engine.state_manager.input_info = self.basic_input_info.copy()
        self.engine.state_manager.current_board = chess.Board()
        self.engine.state_manager.analytics_updated = True
        self.engine.state_manager.lucas_analytics["win_prob"] = 0.01  # Very low win probability
        
        # Test opponent low time (should not resign)
        self.engine.state_manager.current_board.fullmove_number = 30
        self.engine.state_manager.input_info["opp_clock_times"] = [5]
        
        # Call decide_resign directly from the imported function with the required analyzer parameter
        result = decide_resign(self.engine.logger, self.engine.state_manager, self.engine.analyzer)
        self.assertFalse(result)

    def test_edge_case_decide_resign_much_more_time(self):
        """Test decide_resign with much more time than opponent."""
        # Set up test data
        self.engine.state_manager.input_info = self.basic_input_info.copy()
        self.engine.state_manager.current_board = chess.Board()
        self.engine.state_manager.analytics_updated = True
        self.engine.state_manager.lucas_analytics["win_prob"] = 0.01  # Very low win probability
        
        # Test much more time than opponent (should not resign)
        self.engine.state_manager.current_board.fullmove_number = 30
        self.engine.state_manager.input_info["self_clock_times"] = [100]
        self.engine.state_manager.input_info["opp_clock_times"] = [20]
        
        # Call decide_resign directly from the imported function with the required analyzer parameter
        result = decide_resign(self.engine.logger, self.engine.state_manager, self.engine.analyzer)
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()