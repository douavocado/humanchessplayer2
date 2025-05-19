import unittest
from unittest.mock import Mock, patch, MagicMock
import chess
import chess.engine
import chess.polyglot
import numpy as np
import os
import time
import datetime

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
        
        # Patch external dependencies
        self.patcher1 = patch('engine.StockFishSelector', return_value=self.mock_stockfish_scorer)
        self.patcher2 = patch('engine.MoveScorer', return_value=self.mock_move_scorer)
        self.patcher3 = patch('chess.polyglot.open_reader', return_value=self.mock_opening_book)
        self.patcher4 = patch('engine.STOCKFISH', self.mock_stockfish)
        self.patcher5 = patch('engine.PONDER_STOCKFISH', self.mock_ponder_stockfish)
        
        # Start the patchers
        self.mock_stockfish_selector_class = self.patcher1.start()
        self.mock_move_scorer_class = self.patcher2.start()
        self.mock_open_reader = self.patcher3.start()
        self.patcher4.start()  # No need to store reference, we're using self.mock_stockfish directly
        self.patcher5.start()  # No need to store reference, we're using self.mock_ponder_stockfish directly
        
        # Create a test engine instance
        self.engine = Engine(playing_level=6, log_file=os.path.join(os.getcwd(), 'test_log.txt'))
        
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
        
        # Initialize lucas_analytics in the engine
        self.engine.lucas_analytics = {
            "complexity": None,
            "win_prob": 0.5,  # Default value for tests
            "eff_mob": None,
            "narrowness": None,
            "activity": None,
        }

    def tearDown(self):
        """Clean up after each test method."""
        # Stop the patchers
        self.patcher1.stop()
        self.patcher2.stop()
        self.patcher3.stop()
        self.patcher4.stop()
        self.patcher5.stop()
        
        # Remove test log file if it exists
        if os.path.exists('test_log.txt'):
            os.remove('test_log.txt')

    def test_init(self):
        """Test the initialization of the Engine class."""
        # Verify that the engine is initialized correctly
        self.assertEqual(self.engine.playing_level, 6)
        self.assertEqual(self.engine.mood, "confident")
        self.assertIsNone(self.engine.just_blundered)
        self.assertIsNone(self.engine.input_info["side"])
        self.assertIsNone(self.engine.input_info["fens"])
        self.assertIsNone(self.engine.input_info["self_clock_times"])
        self.assertIsNone(self.engine.input_info["opp_clock_times"])
        self.assertIsNone(self.engine.input_info["self_initial_time"])
        self.assertIsNone(self.engine.input_info["opp_initial_time"])
        self.assertIsNone(self.engine.input_info["last_moves"])
        
        # Verify that the scorers are initialized
        self.mock_move_scorer_class.assert_called()
        self.mock_stockfish_selector_class.assert_called_once()
        self.mock_open_reader.assert_called_once()

    def test_write_log(self):
        """Test the _write_log method."""
        # Set up test data
        self.engine.log = "Test log message"
        
        # Call the method
        self.engine._write_log()
        
        # Verify that the log file was written
        with open(self.engine.log_file, 'r') as f:
            content = f.read()
            self.assertEqual(content, "Test log message")
        
        # Verify that the log was cleared
        self.assertEqual(self.engine.log, "")

    def test_update_info(self):
        """Test the update_info method."""
        # Set up test data
        input_info = self.basic_input_info.copy()
        
        # Mock calculate_analytics to avoid calling it
        with patch.object(self.engine, 'calculate_analytics') as mock_calc_analytics:
            # Call the method
            self.engine.update_info(input_info)
            
            # Verify that the input info was updated
            self.assertEqual(self.engine.input_info["side"], chess.WHITE)
            self.assertEqual(self.engine.input_info["fens"], ['rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'])
            self.assertEqual(self.engine.input_info["self_clock_times"], [180])
            self.assertEqual(self.engine.input_info["opp_clock_times"], [180])
            self.assertEqual(self.engine.input_info["self_initial_time"], 180)
            self.assertEqual(self.engine.input_info["opp_initial_time"], 180)
            self.assertEqual(self.engine.input_info["last_moves"], [])
            
            # Verify that calculate_analytics was called
            mock_calc_analytics.assert_called_once()
            
            # Verify that the current board was updated
            self.assertEqual(self.engine.current_board.fen(), 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')

    def test_calculate_analytics(self):
        """Test the calculate_analytics method."""
        # Set up test data
        self.engine.input_info = self.basic_input_info.copy()
        self.engine.current_board = chess.Board()
        
        # Mock get_lucas_analytics to return test values
        with patch('engine.get_lucas_analytics', return_value=(10, 0.6, 20, 5, 15)) as mock_get_lucas:
            # Mock _set_mood to return a fixed mood
            with patch.object(self.engine, '_set_mood', return_value="confident") as mock_set_mood:
                # Call the method
                self.engine.calculate_analytics()
                
                # Verify that stockfish analysis was called
                self.mock_stockfish.analyse.assert_called_once()
                
                # Verify that lucas analytics were updated
                self.assertEqual(self.engine.lucas_analytics["complexity"], 10)
                self.assertEqual(self.engine.lucas_analytics["win_prob"], 0.6)
                self.assertEqual(self.engine.lucas_analytics["eff_mob"], 20)
                self.assertEqual(self.engine.lucas_analytics["narrowness"], 5)
                self.assertEqual(self.engine.lucas_analytics["activity"], 15)
                
                # Verify that mood was set
                mock_set_mood.assert_called_once()
                self.assertEqual(self.engine.mood, "confident")
                
                # Verify that analytics_updated was set to True
                self.assertTrue(self.engine.analytics_updated)

    def test_decide_resign(self):
        """Test the _decide_resign method."""
        # Set up test data
        self.engine.input_info = self.basic_input_info.copy()
        self.engine.current_board = chess.Board()
        self.engine.analytics_updated = True
        self.engine.lucas_analytics["win_prob"] = 0.05
        
        # Test early game (should not resign)
        self.engine.current_board.fullmove_number = 10
        result = self.engine._decide_resign()
        self.assertFalse(result)
        
        # Test opponent low time (should not resign)
        self.engine.current_board.fullmove_number = 30
        self.engine.input_info["opp_clock_times"] = [5]
        result = self.engine._decide_resign()
        self.assertFalse(result)
        
        # Test much more time than opponent (should not resign)
        self.engine.input_info["self_clock_times"] = [100]
        self.engine.input_info["opp_clock_times"] = [20]
        result = self.engine._decide_resign()
        self.assertFalse(result)

    def test_decide_human_filters(self):
        """Test the _decide_human_filters method."""
        # Set up test data
        self.engine.input_info = self.basic_input_info.copy()
        
        # Test normal time (should use human filters)
        self.engine.input_info["self_clock_times"] = [30]
        result = self.engine._decide_human_filters()
        self.assertTrue(result)
        
        # Test low time with random chance to not use filters
        self.engine.input_info["self_clock_times"] = [5]
        
        # Mock random to return a value that would skip human filters
        with patch('numpy.random.random', return_value=0.1):
            result = self.engine._decide_human_filters()
            self.assertFalse(result)
        
        # Mock random to return a value that would use human filters
        with patch('numpy.random.random', return_value=0.9):
            result = self.engine._decide_human_filters()
            self.assertTrue(result)

    def test_get_stockfish_move(self):
        """Test the get_stockfish_move method."""
        # Set up test data
        self.engine.input_info = self.basic_input_info.copy()
        self.engine.current_board = chess.Board()
        self.engine.stockfish_analysis = self.mock_analysis
        
        # Mock extend_mate_score to return the score unchanged
        with patch('engine.extend_mate_score', side_effect=lambda x: x):
            # Mock random.sample to return a controlled subset of moves
            with patch('random.sample', return_value=self.mock_analysis):
                # Call the method
                move = self.engine.get_stockfish_move()
                
                # Verify that the correct move was returned
                self.assertEqual(move, 'e2e4')  # Top move from mock analysis

    def test_check_obvious_move(self):
        """Test the check_obvious_move method."""
        # Set up test data
        self.engine.input_info = self.basic_input_info.copy()
        self.engine.current_board = chess.Board()
        
        # Test only one legal move
        self.engine.stockfish_analysis = [
            {
                'pv': [chess.Move.from_uci('e2e4')],
                'score': chess.engine.PovScore(chess.engine.Cp(50), chess.WHITE)
            }
        ]
        
        move, found = self.engine.check_obvious_move()
        self.assertTrue(found)
        self.assertEqual(move, 'e2e4')
        
        # Test multiple moves (no obvious move)
        self.engine.stockfish_analysis = self.mock_analysis
        
        # Mock is_takeback to return False
        with patch('engine.is_takeback', return_value=False):
            move, found = self.engine.check_obvious_move()
            self.assertFalse(found)
            self.assertIsNone(move)


if __name__ == '__main__':
    unittest.main()
