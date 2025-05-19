import unittest
from unittest.mock import Mock, patch, MagicMock
import chess
import chess.engine
import chess.polyglot
import numpy as np
import os

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

    def test_get_human_probabilities(self):
        """Test the get_human_probabilities method."""
        # Set up test data
        board = chess.Board()
        game_phase = "opening"
        
        # Mock the scorer to return controlled values
        self.mock_move_scorer.get_move_dic.return_value = (None, {'e2e4': 0.7, 'd2d4': 0.3})
        
        # Mock adjust_human_prob to return controlled values
        with patch.object(self.engine, 'adjust_human_prob', return_value={'e2e4': 0.75, 'd2d4': 0.25}) as mock_adjust:
            # Call the method
            result = self.engine.get_human_probabilities(board, game_phase)
            
            # Verify that the scorer was called
            self.mock_move_scorer.get_move_dic.assert_called_once()
            
            # Verify that adjust_human_prob was called
            mock_adjust.assert_called_once()
            
            # Verify the result
            self.assertEqual(result, {'e2e4': 0.75, 'd2d4': 0.25})

    def test_adjust_human_prob(self):
        """Test the adjust_human_prob method."""
        # Set up test data
        self.engine.input_info = self.basic_input_info.copy()
        move_dic = {'e2e4': 0.7, 'd2d4': 0.3}
        board = chess.Board()
        
        # Mock extend_mate_score to return the score unchanged
        with patch('engine.extend_mate_score', side_effect=lambda x: x):
            # Test normal adjustment
            self.engine.mood = "confident"
            self.engine.stockfish_analysis = self.mock_analysis
            
            adjusted_dic = self.engine.adjust_human_prob(move_dic, board)
            
            # Verify that probabilities were adjusted but still sum to approximately 1
            self.assertAlmostEqual(sum(adjusted_dic.values()), 1.0, places=5)
            
            # Test hurry mood
            self.engine.mood = "hurry"
            self.engine.input_info["self_clock_times"] = [10]
            
            adjusted_dic = self.engine.adjust_human_prob(move_dic, board)
            
            # Verify that probabilities were adjusted but still sum to approximately 1
            self.assertAlmostEqual(sum(adjusted_dic.values()), 1.0, places=5)


if __name__ == '__main__':
    unittest.main()