import unittest
from unittest.mock import Mock, patch, MagicMock
import chess
import chess.engine
import chess.polyglot
import numpy as np
import os
import time
import sys

from engine import Engine


class TestEngineTimePondering(unittest.TestCase):
    """Test suite for the time management and pondering methods of the Engine class."""

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
        self.patcher4.start()
        self.patcher5.start()
        
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
        
        # Update engine with basic info
        self.engine.input_info = self.basic_input_info.copy()
        self.engine.current_board = chess.Board()
        self.engine.stockfish_analysis = self.mock_analysis
        self.engine.analytics_updated = True
        
        # Initialize lucas_analytics in the engine
        self.engine.lucas_analytics = {
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
        super(TestEngineTimePondering, self).run(result)
        
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

    def test_set_target_time(self):
        """Test the _set_target_time method."""
        # Test very low total time
        total_time = 0.5
        target_time = self.engine._set_target_time(total_time)
        self.assertAlmostEqual(target_time, 0.4, places=1)  # 80% of total time
        
        # Test medium total time
        total_time = 2.0
        target_time = self.engine._set_target_time(total_time)
        self.assertLess(target_time, total_time)
        self.assertGreater(target_time, 0.1)
        
        # Test high total time
        total_time = 5.0
        target_time = self.engine._set_target_time(total_time)
        self.assertAlmostEqual(target_time, 3.0, places=1)  # 60% of total time

    def test_get_time_taken_obvious_move(self):
        """Test the _get_time_taken method with obvious move."""
        # Set up test data
        self.engine.input_info = self.basic_input_info.copy()
        
        # Mock numpy.random functions to return controlled values
        with patch('numpy.random.randn', return_value=0):
            with patch('numpy.random.random', return_value=0.5):
                # Test obvious move
                time_taken = self.engine._get_time_taken(obvious=True)
                self.assertGreater(time_taken, 0)
                
                # Time for obvious move should be less than for non-obvious move
                non_obvious_time = self.engine._get_time_taken(obvious=False)
                self.assertLess(time_taken, non_obvious_time)

    def test_get_time_taken_different_moods(self):
        """Test the _get_time_taken method with different moods."""
        # Set up test data
        self.engine.input_info = self.basic_input_info.copy()
        
        # Mock numpy.random functions to return controlled values
        with patch('numpy.random.randn', return_value=0):
            with patch('numpy.random.random', return_value=0.5):
                with patch('engine.phase_of_game', return_value="midgame"):
                    # Test different moods
                    moods = ["confident", "cocky", "cautious", "tilted", "hurry"]
                    times = {}
                    
                    for mood in moods:
                        self.engine.mood = mood
                        times[mood] = self.engine._get_time_taken(obvious=False, human_filters=True)
                    
                    # Cautious should take more time than cocky
                    self.assertGreater(times["cautious"], times["cocky"])
                    
                    # Hurry should take less time than confident
                    self.assertLess(times["hurry"], times["confident"])

    def test_ponder_moves_basic(self):
        """Test the _ponder_moves method with basic functionality."""
        # Set up test data
        board = chess.Board()
        move_ucis = ['e2e4']
        search_width = 2
        
        # Mock stockfish analysis
        self.mock_stockfish.analyse.return_value = {
            'pv': [chess.Move.from_uci('e7e5')],
            'score': chess.engine.PovScore(chess.engine.Cp(-50), chess.BLACK)
        }
        
        # Mock the get_move_dic method to return a tuple of (score, move_dic)
        mock_move_dic = {'e7e5': 0.8, 'e7e6': 0.2}
        self.mock_move_scorer.get_move_dic.return_value = (0.5, mock_move_dic)
        
        # Mock extend_mate_score to return the score unchanged
        with patch('engine.extend_mate_score', side_effect=lambda x: x):
            # Call the method
            result = self.engine._ponder_moves(board, move_ucis, search_width)
            
            # Verify that stockfish was called
            self.mock_stockfish.analyse.assert_called()
            
            # Verify the result format
            self.assertIn('e2e4', result)
            self.assertEqual(len(result['e2e4']), 2)  # [response, eval]
            self.assertEqual(result['e2e4'][0], 'e7e5')  # Response move


if __name__ == '__main__':
    unittest.main()