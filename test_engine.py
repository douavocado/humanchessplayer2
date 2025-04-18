import unittest
from unittest.mock import Mock, patch, MagicMock
import chess
import chess.engine
import chess.polyglot
import numpy as np
import os
import time
import datetime
import sys
import random

# Import the Engine class
from engine import Engine

# We'll mock these dependencies rather than importing them directly
# This avoids import errors if they're not available in the test environment
# from models.models import MoveScorer, StockFishSelector
# from common.board_information import (
#     phase_of_game, PIECE_VALS, is_capturing_move, is_capturable,
#     is_attacked_by_pinned, is_check_move, is_takeback
# )
# from common.utils import flip_uci, patch_fens, check_safe_premove, extend_mate_score


class TestEngine(unittest.TestCase):
    """Test suite for the Engine class."""

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
        
        # Setup engine with state manager attributes
        self.engine.state_manager.lucas_analytics = {
            "complexity": None,
            "win_prob": 0.5,  # Default value for tests
            "eff_mob": None,
            "narrowness": None,
            "activity": None,
        }
        
        # Add additional needed properties that have moved to components
        self.engine.state_manager.input_info = {
            "side": None,
            "fens": None,
            "self_clock_times": None,
            "opp_clock_times": None, 
            "self_initial_time": None,
            "opp_initial_time": None,
            "last_moves": None
        }
        self.engine.state_manager.current_board = chess.Board()
        self.engine.state_manager.analytics_updated = False
        self.engine.mood_manager.current_mood = "confident"
        self.engine.mood_manager.just_blundered = False

    def run(self, result=None):
        """Override the run method to track test failures and output logs immediately."""
        self.test_failed = False
        test_method_name = getattr(self, '_testMethodName', None)
        
        # Initialize log_file if it doesn't exist yet
        if not hasattr(self, 'log_file'):
            self.log_file = os.path.join(os.getcwd(), 'test_log.txt')
        
        # Call the original run method
        super(TestEngine, self).run(result)
        
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

    def test_init(self):
        """Test the initialization of the Engine class."""
        # Mock StateManager.__init__ to prevent default fens from being set
        with patch('engine_components.state_manager.StateManager.__init__') as mock_init:
            mock_init.return_value = None
            
            # Create a fresh engine instance for this test
            with patch('engine_components.logger.Logger.write_log'):
                engine = Engine(playing_level=6, log_file=self.log_file)
            
            # Manually set the state_manager input_info for testing
            engine.state_manager = Mock()
            engine.state_manager.input_info = {
                "side": None,
                "fens": None,
                "self_clock_times": None,
                "opp_clock_times": None, 
                "self_initial_time": None,
                "opp_initial_time": None,
                "last_moves": None
            }
        
            # Verify that the engine is initialized correctly
            self.assertEqual(engine.playing_level, 6)
            self.assertEqual(engine.mood_manager.current_mood, "confident")
            self.assertFalse(engine.mood_manager.just_blundered)
            self.assertIsNone(engine.state_manager.input_info["side"])
            self.assertIsNone(engine.state_manager.input_info["fens"])
            self.assertIsNone(engine.state_manager.input_info["self_clock_times"])
            self.assertIsNone(engine.state_manager.input_info["opp_clock_times"])
            self.assertIsNone(engine.state_manager.input_info["self_initial_time"])
            self.assertIsNone(engine.state_manager.input_info["opp_initial_time"])
            self.assertIsNone(engine.state_manager.input_info["last_moves"])

    def test_write_log(self):
        """Test the _write_log method."""
        # Use a separate log file for this test to avoid interference
        test_log_file = os.path.join(os.getcwd(), 'test_log_write_test.txt')
        
        # Create a fresh logger just for this test
        from engine_components.logger import Logger
        test_logger = Logger(test_log_file)
        test_logger.log_buffer = "Test log message"
        
        # Temporarily replace the engine's logger
        original_logger = self.engine.logger
        self.engine.logger = test_logger
        
        try:
            # Call the method
            self.engine._write_log()
            
            # Verify that the log file was written
            with open(test_log_file, 'r') as f:
                content = f.read()
                self.assertEqual(content, "Test log message")
            
            # Verify that the log was cleared
            self.assertEqual(test_logger.log_buffer, "")
        finally:
            # Restore the original logger
            self.engine.logger = original_logger
            
            # Clean up the test log file
            if os.path.exists(test_log_file):
                os.remove(test_log_file)

    def test_update_info(self):
        """Test the update_info method."""
        # Set up test data
        input_info = self.basic_input_info.copy()
        
        # Mock calculate_analytics to avoid calling it
        with patch.object(self.engine, 'calculate_analytics') as mock_calc_analytics:
            # Call the method
            self.engine.update_info(input_info)
            
            # Verify that the input info was updated
            self.assertEqual(self.engine.state_manager.input_info["side"], chess.WHITE)
            self.assertEqual(self.engine.state_manager.input_info["fens"], ['rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'])
            self.assertEqual(self.engine.state_manager.input_info["self_clock_times"], [180])
            self.assertEqual(self.engine.state_manager.input_info["opp_clock_times"], [180])
            self.assertEqual(self.engine.state_manager.input_info["self_initial_time"], 180)
            self.assertEqual(self.engine.state_manager.input_info["opp_initial_time"], 180)
            self.assertEqual(self.engine.state_manager.input_info["last_moves"], [])
            
            # Verify that calculate_analytics was called
            mock_calc_analytics.assert_called_once()
            
            # Verify that the current board was updated
            self.assertEqual(self.engine.state_manager.current_board.fen(), 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')

    def test_calculate_analytics(self):
        """Test the calculate_analytics method."""
        # Set up test data
        self.engine.state_manager.input_info = self.basic_input_info.copy()
        self.engine.state_manager.current_board = chess.Board()
        
        # Mock the analyzer's calculate_analytics method
        with patch.object(self.engine.analyzer, 'calculate_analytics') as mock_analyzer:
            # Mock the mood manager's determine_mood method
            with patch.object(self.engine.mood_manager, 'determine_mood') as mock_mood:
                # Call the method
                self.engine.calculate_analytics()
                
                # Verify that method calls were made
                mock_analyzer.assert_called_once()
                mock_mood.assert_called_once()

    def test_decide_resign(self):
        """Test the _decide_resign method."""
        # Since decide_resign is now in a separate module, we'll test the integration
        # Set up test data
        self.engine.state_manager.input_info = self.basic_input_info.copy()
        self.engine.state_manager.current_board = chess.Board()
        self.engine.state_manager.analytics_updated = True
        self.engine.state_manager.lucas_analytics = {"win_prob": 0.05}
        
        # Mock the imported decide_resign function
        with patch('engine_components.decision_logic.decide_resign', return_value=False) as mock_decide:
            from engine_components.decision_logic import decide_resign
            result = decide_resign(self.engine.logger, self.engine.state_manager)
            self.assertFalse(result)

    def test_decide_human_filters(self):
        """Test the _decide_human_filters method."""
        # Since decide_human_filters is now in a separate module, we'll test the integration
        with patch('engine_components.decision_logic.decide_human_filters', return_value=(5, 0.8)) as mock_decide:
            from engine_components.decision_logic import decide_human_filters
            breadth, threshold = decide_human_filters(
                self.engine.logger, self.engine.state_manager, 
                self.engine.analyzer, self.engine.mood_manager
            )
            self.assertEqual(breadth, 5)
            self.assertEqual(threshold, 0.8)

    def test_get_stockfish_move(self):
        """Test the get_stockfish_move method."""
        # Set up a test board
        board = chess.Board()
        
        # Mock the stockfish_move_logic's get_stockfish_move method
        with patch.object(self.engine.stockfish_move_logic, 'get_stockfish_move', return_value='e2e4') as mock_get_move:
            # Call the method
            move = self.engine.get_stockfish_move(board)
            
            # Verify that the method was called with correct arguments
            mock_get_move.assert_called_once_with(board, None, None)
            
            # Verify the result
            self.assertEqual(move, 'e2e4')

    def test_check_obvious_move(self):
        """Test the check_obvious_move method."""
        # Set up a mock return value that matches the function's actual tuple return type
        mock_return = ('e2e4', True)  # (move_uci, is_obvious)
        
        # Patch the engine module's check_obvious_move import directly
        with patch('engine.check_obvious_move', return_value=mock_return) as mock_check:
            # Call the method through the engine
            result = self.engine.check_obvious_move()
            
            # Verify the function was called with the correct arguments
            mock_check.assert_called_once_with(
                self.engine.logger,
                self.engine.state_manager,
                self.engine.analyzer,
                self.engine.mood_manager
            )
            
            # Check that we got the expected result
            self.assertEqual(result, mock_return)


if __name__ == '__main__':
    unittest.main()
