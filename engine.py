# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:29:08 2024

@author: xusem
"""
import datetime
import os
import time
import numpy as np
import random

import chess
import chess.polyglot

from models.models import MoveScorer, StockFishSelector
from common.constants import (PATH_TO_STOCKFISH, MOVE_FROM_WEIGHTS_OP_PTH, MOVE_FROM_WEIGHTS_MID_PTH,
                              MOVE_FROM_WEIGHTS_END_PTH, MOVE_TO_WEIGHTS_MID_PTH, 
                              MOVE_TO_WEIGHTS_END_PTH, MOVE_TO_WEIGHTS_OP_PTH,
                              QUICKNESS
                              )

from common.board_information import (
    phase_of_game, PIECE_VALS, STOCKFISH, get_lucas_analytics, is_capturing_move, is_capturable,
    is_attacked_by_pinned, is_check_move, is_takeback, is_newly_attacked, get_threatened_board,
    is_offer_exchange, king_danger, is_open_file, calculate_threatened_levels, check_best_takeback_exists
            )
from common.utils import flip_uci, get_move_made

# TODO: Openings
# TODO: time control, make things less regular
# TODO: allow premoves for obvious moves
# TODO: Add resign functionality
# TODO: 3-fold repetition logic

class Engine:
    """ Class for engine instance.
    
        The Engine is responsible for the following things ONLY
        receiving board information -> outputting move and premoves
        
        All other history related data to do with past moves etc are not handled
        in the Engine instance. They are handled in the client wrapper
    """
    def __init__(self, playing_level:int = 6, log_file: str = os.path.join(os.getcwd(), 'Engine_logs',str(datetime.datetime.now()).replace(" ", "").replace(":","_") + '.txt'), opening_book_path:str = "Opening_books/bullet.bin"):
        self.input_info = {
            "side": None,
            "fens": None,
            "self_clock_times" : None,
            "opp_clock_times": None,
            "self_initial_time": None,
            "opp_initial_time": None,
            "last_moves": None,
                           }
        self.current_board = chess.Board()
        self.log = ""
        self.log_file = log_file
        
        # setting up scorers for moves
        self.human_scorers = {
            "opening": MoveScorer(MOVE_FROM_WEIGHTS_OP_PTH, MOVE_TO_WEIGHTS_OP_PTH),
            "midgame": MoveScorer(MOVE_FROM_WEIGHTS_MID_PTH, MOVE_TO_WEIGHTS_MID_PTH),
            "endgame": MoveScorer(MOVE_FROM_WEIGHTS_END_PTH, MOVE_TO_WEIGHTS_END_PTH),
            }
        self.stockfish_scorer = StockFishSelector(PATH_TO_STOCKFISH)
        self.stockfish_analysis = None
        
        # Getting opening books
        self.opening_book = chess.polyglot.open_reader(opening_book_path)
        
        # lucas statistics for the current position
        self.lucas_analytics = {
            "complexity": None,
            "win_prob": None,
            "eff_mob": None,
            "narrowness": None,
            "activity": None,
            }
        # A bool to track whether we have updated analytics following updating info
        self.analytics_updated = False
        
        self.playing_level = playing_level
        self.mood = None
        self.just_blundered = None
        
    def _write_log(self):
        """ Writes down thinking into a log file for debugging. """
        with open(self.log_file,'a') as log:
            log.write(self.log)
            log.close()
        self.log = ""
    
    def _decide_resign(self):
        """ The decision making function which decides whether to resign.
        
            Returns bool
        """
        self.log += "Deciding whether to resign the current position. \n"
        # Can only decide to resign if analytics are updated
        if self.analytics_updated == False:
            self.log += "WARNING: _decide_resign() function has been called with outdated analytics. Please call .calculate_analytics() first to get accurate output. \n"
        
        own_time = self.input_info["self_clock_times"][-1]
        opp_time = self.input_info["opp_clock_times"][-1]
        starting_time = self.input_info["self_initial_time"]
        
        # does not resign in the first 15 moves
        if self.current_board.fullmove_number < 15:
            return False
        elif opp_time <= 10:
            return False
        elif own_time/(opp_time-10) > 3:
            return False
        win_pct = self.lucas_analytics["win_prob"]
        
        # We shall only resign if our win percentage is sufficiently low, and the material
        # imbalance is obvious
        mat_dic = {1:1, 2:3.1, 3:3.5, 4:5.5, 5:9.9, 6:3}
        our_mat = sum([len(self.current_board.pieces(x, self.current_board.turn))*mat_dic[x] for x in range(1,7)])
        opp_mat = sum([len(self.current_board.pieces(x, not self.current_board.turn))*mat_dic[x] for x in range(1,7)])
        if win_pct < np.log(starting_time)* opp_time/(500*starting_time) and opp_mat - our_mat > 5:
            self.log += "Win percentage too low, material imbalance too large and opponent time too high, resigning... \n"
            return True
        else:
            return False
    
    def _decide_human_filters(self):
        """ Decide if given the position we want to use human filters or not. When
            using human filters we would need to use more processing time, which
            may not be suitable when we have low time. If possible however, we always
            want to use human filters.
            
            Returns True/False
        """
        
        # TODO: for now just return true with high probability if in time trouble
        own_time = self.input_info["self_clock_times"][-1]
        if own_time < 10:
            if np.random.random() < (10-own_time)/25:
                return False
            else:
                return True
        return True
    
    def get_stockfish_move(self, board:chess.Board = None, analysis=None, last_move_uci:str = None, log:bool = True, ):
        """ Uses board information to get a move strictly from stockfish with no human
            filters. Very fast, and only called for in necessary situations (when in
            super low time)
            
            Returns a move_uci string of move made
        """
        # If no kwargs given, then assume we are using the current position
        if board is None:
            board = self.current_board.copy()
        
        if analysis is None :
            analysis = self.stockfish_analysis
        
        if last_move_uci is None:
            if len(self.input_info["last_moves"]) >= 2:
                last_move_uci = self.input_info["last_moves"][-2]
        # take a random sample from the moves given by our stockfish analysis object
        # grade each move by their evaluation, and if previous moves are given (our
        # own moves) then grade them also based on proximity and distance moved by mouse
        total_moves = len(analysis)
        sample_n = max(int(total_moves*0.6),1)
        if log:
            self.log += "Choosing to sample {} moves from analysis object. \n".format(sample_n)
        sampled_moves = random.sample(analysis, sample_n)
        move_eval_dic = {entry["pv"][0].uci(): entry["score"].pov(board.turn).score(mate_score=2500) for entry in sampled_moves}
        if log:
            self.log += "Sampled the following moves and their corresponding evals: {} \n".format(move_eval_dic)
        move_distance_dic = {}
        for move_uci in move_eval_dic.keys():
            distance = 0
            move_obj = chess.Move.from_uci(move_uci)
            # if we are given information about our own previous move, then include that distance too
            if last_move_uci is not None:
                own_last_move_obj = chess.Move.from_uci(last_move_uci)
                to_square = own_last_move_obj.to_square
                distance += chess.square_distance(to_square, move_obj.from_square)
            distance += chess.square_distance(move_obj.from_square, move_obj.to_square)
            move_distance_dic[move_uci] = distance
        
        if log:
            self.log += "Evaluated the moves square distance to move: {} \n".format(move_distance_dic)
        own_time = self.input_info["self_clock_times"][-1]
        move_appealing_dic = {move_uci : 10 + move_eval_dic[move_uci]*own_time/2000 - move_distance_dic[move_uci] for move_uci in move_eval_dic.keys()}
        if log:
            self.log += "Combining both the dictionary preferences, we have their move preferences: {} \n".format(move_appealing_dic)
        # Add noise to introduce randomness. The lower our time, the more the noise
        # note own time has to be less than 15 seconds for valid calculation.
        noise_level = (15-own_time)/15
        move_appealing_dic = {move_uci: move_appealing_dic[move_uci] + noise_level*np.random.randn() for move_uci in move_appealing_dic.keys()}
        if log:
            self.log += "Appealingness after adding noise: {} \n".format(move_appealing_dic)
        
        move_chosen = max(move_appealing_dic.keys(), key=lambda x: move_appealing_dic[x])
        if log:
            self.log += "Chosen stockfish move under time pressure: {} \n".format(move_chosen)
        return move_chosen
        
    def get_human_probabilities(self, board : chess.Board, game_phase: str, log:bool = True):
        """ Given a chess.Board item, returns the top move ucis along with their
            human move probabilties, evaluated from neural network only. These bare
            no extra tinkering methods and are purely from the neural net. """
        if log:
            self.log += "Getting human probabilities. \n"
        scorer = self.human_scorers[game_phase]
        # The model has been trained to only score from the perspective of white pieces,
        # we flip the board if it is black's turn
        if board.turn == chess.WHITE:
            dummy_board = board.copy()
        if board.turn == chess.BLACK:
            dummy_board = board.mirror()
        _, nn_top_move_dic = scorer.get_move_dic(dummy_board, san=False, top=100)
        
        # if we were black, we need to convert all the ucis to be flipped
        if board.turn == chess.BLACK:
            nn_top_move_dic = {flip_uci(k): v for k,v in nn_top_move_dic.items()}
        
        # Normalising probabilities such that they add to 1
        total = sum(nn_top_move_dic.values())
        nn_top_move_dic = {k: v/total for k, v in nn_top_move_dic.items()}
        
        if log:
            log_move_dic = {board.san(chess.Move.from_uci(k)) : round(v, 5) for k,v in nn_top_move_dic.items()}
            self.log += "Move_dic before alteration: {} \n".format(log_move_dic)
        return nn_top_move_dic
    
    def _alter_move_probabilties(self, move_dic : dict, board:chess.Board, prev_board:chess.Board = None, prev_prev_board:chess.Board = None, log:bool = True):
        """ Given a move dictionary with move uci as key and value as their unaltered
            probabilities, we alter the probabilties to make moves stick out more
            (for example hanging pieces more likely to be moved etc).
            
            Returns an altered move_dic. 
        """
        start = time.time()        
        # Moves which protect/block/any way that make our pieces less en pris are appealing
        # Factor is given by factor = np.exp(-lvl_diff/10)        
        # Moves which make opponent pieces more en pris (attacks/threatens) are appealing
        # Factor is given by factor = np.exp(lvl_diff/10)
        lower_threshold_prob = sum(move_dic.values())/len(move_dic)
        
        # Capturing enpris pieces are more appealing, the more the piece is enpris the more appealing
        capture_en_pris_sf = 1.5
        # Squares that pinned pieces attack that break the pin are more desirable
        break_pin_sf = 3.0
        # Captures are just generally more appealing. The bigger the capture the more appealing
        capture_sf = 1.5
        # Capturable pieces are more appealling to move
        capturable_sf = 1.3
        
        # Checks are more appealing (particularly under time pressure)
        check_sf_dic = {"confident": 1.8,
                        "cocky": 2.3,
                        "cautious": 1.6,
                        "tilted": 2.8,
                        "hurry": 2.5,
                        "flagging": 2.4}
        # Takebacks are more appealing
        takeback_sf = 2.5
        # Newly threatened en-pris pieces are more appealling to move
        new_threatened_sf_dic = {"confident": 3.5,
                        "cocky": 2.7,
                        "cautious": 3.9,
                        "tilted": 2.1,
                        "hurry": 2.4,
                        "flagging": 3.2}
        
        # Offering exchanges/exchanging when material up appealing
        exchange_sf_dic = {"confident": 3.2,
                        "cocky": 2.0,
                        "cautious": 2.2,
                        "tilted": 1.5,
                        "hurry": 3.8,
                        "flagging": 0.8}
        # Offering exchanges/exchanging when king danger is high appealing
        exchange_k_danger_sf_dic = {"confident": 3.4,
                        "cocky": 2.1,
                        "cautious": 3.9,
                        "tilted": 1.0,
                        "hurry": 3.0,
                        "flagging": 2.5}
        # Pushing passed pawns in endgame more appealing
        passed_pawn_end_sf = 3.0
        
        # Repeat moves are undesirable
        repeat_sf_dic = {"confident": 0.5,
                        "cocky": 0.7,
                        "cautious": 0.5,
                        "tilted": 0.8,
                        "hurry": 0.7,
                        "flagging": 0.6}
        
###############################################################################
        
        # Moves which protect/block/any way that make our pieces less en pris are appealing        
        # Likewise moves that make our pieces more en pris are less appealing
        # Moves which make opponent pieces more en pris (attacks/threatens) are appealing
        strenghening_moves = []
        weakening_moves = []        
        # For time computation sake, we ignore the threatened levels of pawns
        self_curr_threatened_levels = sum(get_threatened_board(board, colour=board.turn, piece_types=[1,2,3,4,5]))
        opp_curr_threatened_levels = sum(get_threatened_board(board, colour=(not board.turn), piece_types=[1,2,3,4,5]))
        for move_uci in move_dic.keys():
            move_obj = chess.Move.from_uci(move_uci)
            dummy_board = board.copy()
            dummy_board.push(move_obj)
            new_threatened_board = get_threatened_board(dummy_board, colour=board.turn, piece_types=[1,2,3,4,5])
            self_new_threatened_levels = sum(new_threatened_board)
            # if new move makes our piece en_pris, then we calculate enemy threatened levels
            # as if there was no such piece. This prevents scenarios where we attack
            # opposition pieces of higher value at the cost of sacrificing our original
            # attacking piece.
            if new_threatened_board[move_obj.to_square] > 0.6:
                dummy_board.remove_piece_at(move_obj.to_square)
            
            opp_new_threatened_levels = sum(get_threatened_board(dummy_board, colour=(not board.turn), piece_types=[1,2,3,4,5]))
            self_lvl_diff = self_new_threatened_levels - self_curr_threatened_levels
            opp_lvl_diff = opp_new_threatened_levels - opp_curr_threatened_levels
            # psychologically, protecting pieces is more favorable than attacking pieces
            # therefore we weight being in a safer position more heavily than being in a 
            # more pressuring situation
            # however this happens only if our move is not a capture.
            lvl_diff = opp_lvl_diff - self_lvl_diff*1.5
            # if our move was a capture, we must take that into account that we gained some material
            piece_type = board.piece_type_at(move_obj.to_square)
            if piece_type is not None: # if we captured something
                lvl_diff += PIECE_VALS[piece_type] * 1.5          
            factor = np.exp(lvl_diff/2)     
            
            # Some moves of the strengthening moves are weird and unlikely due to the
            # fault of the nn. We want these moves to be visible, so we set a threshold
            # for which if these moves have lower prob, we raise them to so that the factor
            # changes are significant.
            if lvl_diff > 0.9:
                move_dic[move_uci] = max(move_dic[move_uci], lower_threshold_prob)
            move_dic[move_uci] *= factor
            
            if lvl_diff > 0.9:
                strenghening_moves.append(board.san(chess.Move.from_uci(move_uci)))
            elif lvl_diff < -0.9:
                weakening_moves.append(board.san(chess.Move.from_uci(move_uci)))
        if log:
            self.log += "Found moves that are weakening and make our pieces more enpris/opp pieces less enpris: {} \n".format(weakening_moves)
            self.log += "Found moves that protect our pieces more or apply more pressure to opponent: {} \n".format(strenghening_moves)
        
        # Squares that pinned pieces attack that break the pin are more desirable to move to
        adv_pinned_moves = []
        for move_uci in move_dic.keys():
            to_square = chess.Move.from_uci(move_uci).to_square
            no_pinned_atks = is_attacked_by_pinned(board, to_square, not board.turn)
            if no_pinned_atks > 0:
                # it is a capturing move
                if move_dic[move_uci] > lower_threshold_prob:
                    move_dic[move_uci] = lower_threshold_prob
                move_dic[move_uci] *= (break_pin_sf**no_pinned_atks)
                adv_pinned_moves.append(board.san(chess.Move.from_uci(move_uci)))
        if log:
            self.log += "Found moves that take advantage of pinned pieces: {} \n".format(adv_pinned_moves)
        
        # Capturing moves are more appealing
        capturing_moves = []
        for move_uci in move_dic.keys():
            if is_capturing_move(board, move_uci):
                # it is a capturing move
                piece_value = PIECE_VALS[board.piece_type_at(chess.Move.from_uci(move_uci).to_square)]
                move_dic[move_uci] *= capture_sf * (piece_value**0.25)
                capturing_moves.append(board.san(chess.Move.from_uci(move_uci)))
        if log:
            self.log += "Found capturing moves from position: {} \n".format(capturing_moves)
        
        # Capturing enpris pieces are more appealing
        capturing_enpris_moves = []
        for move_uci in move_dic.keys():
            if is_capturing_move(board, move_uci):
                # it is a capturing move
                move_obj = chess.Move.from_uci(move_uci)
                threatened_lvls = calculate_threatened_levels(move_obj.to_square, board)
                if threatened_lvls > 0.6:  # captured piece is enpris
                    move_dic[move_uci] *= capture_en_pris_sf * (threatened_lvls**0.25)
                    capturing_enpris_moves.append(board.san(move_obj))
        if log:
            self.log += "Found capturing enpris moves from position: {} \n".format(capturing_moves)
        
        # Capturable pieces are more appealling to move
        capturable_moves = []
        for move_uci in move_dic.keys():
            from_square = chess.Move.from_uci(move_uci).from_square
            if is_capturable(board, from_square):
                # it is a capturing move
                move_dic[move_uci] *= capturable_sf
                capturable_moves.append(board.san(chess.Move.from_uci(move_uci)))
        if log:
            self.log += "Found moves that move capturable pieces: {} \n".format(capturable_moves)
        
        # Checks are more appealing (particularly under time pressure)        
        checking_moves = []
        for move_uci in move_dic.keys():
            if is_check_move(board, move_uci):
                # depending what mood we are, checks are more attractive
                move_dic[move_uci] *= check_sf_dic[self.mood]
                checking_moves.append(board.san(chess.Move.from_uci(move_uci)))
        if log:
            self.log += "Found checking moves: {} \n".format(checking_moves)
        
        # Takebacks are more appealing
        # We may only calculate this criterion if we have information of previous move
        if prev_board is not None:
            takeback_moves = []
            last_move_uci = get_move_made(prev_board.fen(), board.fen())
            for move_uci in move_dic.keys():
                if is_takeback(prev_board, last_move_uci, move_uci):
                    move_dic[move_uci] *= takeback_sf
                    takeback_moves.append(board.san(chess.Move.from_uci(move_uci)))
            if log:
                self.log += "Found takeback moves: {} \n".format(takeback_moves)
        
        # calc threatened of prev board and board of from square
        # Newly threatened en-pris pieces are more appealling to move
        # We may only calculate this criterion if we have information of previous positions
        if prev_board is not None:            
            new_threatened_moves = []
            # first get all the squares our own pieces and work out whether they are
            # newly threatened or not
            from_squares = [sq for piece_type in range(1,6) for sq in board.pieces(piece_type, board.turn) ]
            from_sq_dic = {from_sq : is_newly_attacked(prev_board, board, from_sq) for from_sq in from_squares}
            newly_attacked_squares = [sq for sq in from_sq_dic if from_sq_dic[sq] > 0.6]
            for move_uci in move_dic.keys():
                dummy_board = board.copy()
                dummy_board.push_uci(move_uci)
                for square in newly_attacked_squares:
                    threatened_levels = calculate_threatened_levels(square, dummy_board)
                    difference_in_threatened = from_sq_dic[square] - threatened_levels
                    if difference_in_threatened > 0.6:
                        # depending what mood we are, we may be more sensitive to new attacks
                        # also depends on the value of what is newly being attacked
                        move_dic[move_uci] *= new_threatened_sf_dic[self.mood] * (1 + difference_in_threatened)**0.2
                        new_threatened_moves.append(board.san(chess.Move.from_uci(move_uci)))
            if log:
                self.log += "Found moves that respond to newly threatened pieces: {} \n".format(new_threatened_moves)
            
        # Offering exchanges/exchanging when material up appealing
        # likewise offering exchanges when material down unappealing
        # calculates threatened levels of every move after its moved at to_square when material imbalance
        good_exchanges = []
        bad_exchanges = []
        # first sum the material on our side compared with opponent side
        mat_dic = {1:1, 2:3.1, 3:3.5, 4:5.5, 5:9.9, 6:3}
        own_mat = sum([len(board.pieces(x, board.turn))*mat_dic[x] for x in range(1,6)])
        opp_mat = sum([len(board.pieces(x, not board.turn))*mat_dic[x] for x in range(1,6)])
        if own_mat - opp_mat > 2.9: # if we are more than a knight up, encourage trades
            for move_uci in move_dic.keys():
                if is_offer_exchange(board, move_uci) == True:
                    move_dic[move_uci] *= exchange_sf_dic[self.mood]  
                    good_exchanges.append(board.san(chess.Move.from_uci(move_uci)))
        elif opp_mat - own_mat > 2.9: # if we are more than a knight down, discourage trades
            for move_uci in move_dic.keys():
                if is_offer_exchange(board, move_uci) == True:
                    move_dic[move_uci] /= exchange_sf_dic[self.mood] 
                    bad_exchanges.append(board.san(chess.Move.from_uci(move_uci)))
        if log:
            self.log += "Found moves that encourage exchanges as we are material up: {} \n".format(good_exchanges)
            self.log += "Found moves that trade when we are material down: {} \n".format(bad_exchanges)
      
        # Offering exchanges/exchanging when king danger is high appealing
        # However if opponent is in even more danger, than we pieces on board
        # calculates threatened levels of every move after its moved at to_square when king danger imbalance
        good_king_exchanges = []
        bad_king_exchanges = []
        game_phase = phase_of_game(board)
        self_king_danger_lvl = king_danger(board, board.turn, game_phase)
        opp_king_danger_lvl = king_danger(board, not board.turn, game_phase)
        if abs(opp_king_danger_lvl - self_king_danger_lvl) < 400:
            # indifferent, keep pieces on
            pass
        elif opp_king_danger_lvl - self_king_danger_lvl >= 400 and opp_king_danger_lvl > 500:
            # opponent is in much higher danger level than we are
            # discourage trades
            for move_uci in move_dic.keys():
                if is_offer_exchange(board, move_uci) == True:
                    move_dic[move_uci] /= exchange_k_danger_sf_dic[self.mood]
                    bad_king_exchanges.append(board.san(chess.Move.from_uci(move_uci)))
        elif self_king_danger_lvl - opp_king_danger_lvl >= 400 and self_king_danger_lvl > 500:
            # we are in much more king danger than opponent
            # encourage trades
            for move_uci in move_dic.keys():
                if is_offer_exchange(board, move_uci) == True:
                    move_dic[move_uci] *= exchange_k_danger_sf_dic[self.mood]
                    good_king_exchanges.append(board.san(chess.Move.from_uci(move_uci)))
        if log:
            self.log += "Found moves that encourage exchanges when our king is in danger: {} \n".format(good_king_exchanges)
            self.log += "Found moves that trade when enemy king is in danger: {} \n".format(bad_king_exchanges)
            
        # Pushing passed pawns in endgame more appealing
        if game_phase == "endgame":
            passed_pawn_moves = []
            for move_uci in move_dic.keys():
                move_obj = chess.Move.from_uci(move_uci)
                move_piece_type = board.piece_type_at(move_obj.from_square)
                if move_piece_type == chess.PAWN:
                    passed_result = is_open_file(board, chess.square_file(move_obj.from_square))
                    if board.turn == chess.WHITE and passed_result == -2:                    
                        move_dic[move_uci] *= passed_pawn_end_sf
                        passed_pawn_moves.append(board.san(move_obj))
                    elif board.turn == chess.BLACK and passed_result == 2:
                        move_dic[move_uci] *= passed_pawn_end_sf
                        passed_pawn_moves.append(board.san(move_obj))
            if log:
                self.log += "Found moves that push passed pawns in the endgame: {} \n".format(passed_pawn_moves)
            
        # Normalising and sorting finalising probabilities
        total = sum(move_dic.values()) + 10**(-10)
        move_dic = {k : move_dic[k]/total for k in sorted(move_dic.keys(), reverse=True, key=lambda x: move_dic[x])}
        
        # Repeat moves are undesirable
        # These moves are defined to be moves which repeatedly move a piece, and goto a square 
        # which we could have gone to with the piece last move (including moving piece back and forth)
        if prev_board is not None and prev_prev_board is not None:
            repeat_moves = []
            last_own_move_uci = get_move_made(prev_prev_board.fen(), prev_board.fen())
            last_own_move_obj = chess.Move.from_uci(last_own_move_uci)
            previous_reach = set([move.to_square for move in prev_prev_board.legal_moves if move.from_square == last_own_move_obj.from_square] + [last_own_move_obj.from_square])
            for move_uci in move_dic.keys():
                move_obj = chess.Move.from_uci(move_uci)
                if move_obj.from_square == last_own_move_obj.to_square: # if we are moving the same piece as before
                    to_square = move_obj.to_square
                    if to_square in previous_reach: # then we have found repeating move
                        move_dic[move_uci] *= repeat_sf_dic[self.mood]
                        repeat_moves.append(board.san(move_obj))
            
            if log:
                self.log += "Found moves that are repetitive, and waste time: {} \n".format(repeat_moves)
        
        end = time.time()
        if log:
            # make the move_dic and prob look nicely formatted in the log
            log_move_dic = {board.san(chess.Move.from_uci(k)) : round(v, 5) for k,v in move_dic.items()}
            self.log += "Move_dic after alteration: {} \n".format(log_move_dic)        
            self.log += "Move_dic alterations performed in {} seconds. \n".format(end-start)
        return move_dic
       
    def _decide_breadth(self):
        """ Given our current board information, decide how many of our human moves
            to consider and pass onto the engine.
            
            Returns integer
        """
        game_phase = phase_of_game(self.current_board)
        king_dang = king_danger(self.current_board, self.current_board.turn, game_phase)
        eff_mob = self.lucas_analytics["eff_mob"]
        if eff_mob < 10:
            # Either the best move is obvious (e.g. a takeback, mate in one) or
            # there is a tactic involved. Decrease search width to mimic human
            # behaviour in tactics under pressure
            
            if eff_mob > 5 and game_phase == 'midgame' and self.mood != "hurry":
                no_moves = self.playing_level
            else:
                no_moves = max(self.playing_level-1, 1)
        else:
            # There are plenty of good moves, search wide so close the game out effectively
            if game_phase != 'endgame' and king_dang < 500:
                no_moves = max(5, self.playing_level)
            elif king_dang > 500: 
                # king is in danger, need to pay attention
                no_moves = self.playing_level+1
            else:
                # in the endgame, there aren't many moves available anyways
                no_moves = max(self.playing_level-2,1)
        
        # Now for mood dependent logic
        if self.mood in ["cocky", "hurry"]:
            no_moves = max(no_moves - 1, 1)
        elif self.mood == "cautious":
            no_moves = no_moves + 1
        elif self.mood == "tilted":
            no_moves = max(no_moves - 2, 1)
        
        self.log += "Calculated number of root moves in current position: {} \n".format(no_moves)
        return no_moves
    
    def get_human_move(self, target_time:float = 0.5):
        """ Uses board information to get move based on human filters from machine
            learning model, and probabilities outputted by the model. Performs evaluations
            on self.current_board
            
            Return move_uci of move made
        """
        
        game_phase = phase_of_game(self.current_board)
        self.log += "Evaluated current game phase: {} \n".format(game_phase)
        
        # If the game phase is in the opening, we check to see if we can use our opening
        # book to return a move
        if game_phase == "opening":
            self.log += "Detected game phase is opening, consulting opening book for matching positions. \n"
            result = list(self.opening_book.find_all(self.current_board))
            if len(result) != 0:
                self.log += "Found matching position in opening database. Outputting top results: \n"
                top_results = result[:5]
                for res in top_results:
                    self.log += "{} : {} \n".format(self.current_board.san(res.move), res.weight)
                excluded_moves = [res.move for res in result[5:]]
                # Now get weighted choice of move to play
                played_move_obj = self.opening_book.weighted_choice(self.current_board, exclude_moves=excluded_moves).move
                self.log += "Chosen move from opening book: {} \n".format(self.current_board.san(played_move_obj))
                return played_move_obj.uci()
            else:
                self.log += "Did not find matching position in opening database. Resorting to human move. \n"
        
        # Now get the human moves from the position and their probabilities
        start = time.time()
        un_altered_move_dic = self.get_human_probabilities(self.current_board, game_phase)
        
        # In the rare case where we have not manage to find any human moves, we 
        # substitute a computer made one
        if len(un_altered_move_dic) == 0 or (len(list(self.current_board.legal_moves)) - len(un_altered_move_dic) > 2 and len(un_altered_move_dic) <= 3) :
            self.log += "We have found too little human prob moves, defaulting to computer made move. \n"
            top_move = self.get_stockfish_move(log=False)
            self.log += "Decided output move from computer backup is: {} \n".format(self.current_board.san(chess.Move.from_uci(top_move)))
            return top_move
        
        # Now get altered_probabilities
        if len(self.input_info["fens"]) >= 2:
            prev_board = chess.Board(self.input_info["fens"][-2])
        else:
            prev_board = None
        if len(self.input_info["fens"]) >= 3:
            prev_prev_board = chess.Board(self.input_info["fens"][-3])
        else:
            prev_prev_board = None
        altered_move_dic = self._alter_move_probabilties(un_altered_move_dic, self.current_board, prev_board=prev_board, prev_prev_board=prev_prev_board)
        
        # Now decide how many of these top moves we shall consider for calculation
        no_root_moves = self._decide_breadth()
        self.log += "Decided search breath for current position: {}. \n".format(no_root_moves)
        # We now piece together the top moves from human search and our stockfish_analysis
        human_move_ucis = list(altered_move_dic.keys())
        root_moves = human_move_ucis[:no_root_moves]
        self.log += "Decided root human moves are: {} \n".format([self.current_board.san(chess.Move.from_uci(x)) for x in root_moves])
        
        # Now cross reference with already computed analysis object to find evaluations
        # Eval scores are from the perspective of the board turn, so from ourselves
        human_move_evals = {}
        for analysis_object in self.stockfish_analysis:
            move_uci = analysis_object['pv'][0].uci()
            if move_uci in root_moves:
                eval_ = analysis_object['score'].pov(self.current_board.turn).score(mate_score=2500)
                #clipping
                eval_ = min(eval_, 2500)
                eval_ = max(eval_, -2500)
                human_move_evals[move_uci] = eval_
        
        san_human_move_evals = {self.current_board.san(chess.Move.from_uci(k)): v for k, v in human_move_evals.items()}
        self.log += "Computed human move evals: {} \n".format(san_human_move_evals)
        
        end = time.time()
        human_calc_time = end-start
        BUFFER = 0.08 # the minimum extra time of human_calc time to deal with instances where we got a surprising quick time
        self.log += "Human probabilities including alterations evaluated in {} seconds. \n". format(human_calc_time)
        # Now simply selecting the best of these human evals is natural, however
        # a better way would be to cloud the judgement of the evaluations
        # by computing the again in a human manner. This takes time (roughly 0.12 secs
        # per move) so we can't do this for every move. The number of moves we perform
        # this re-evaluation will depend on our target time, the time we set at
        # the beginning of the move to try and make our move by.
        re_evaluations = int(max(target_time//max(human_calc_time, BUFFER) - 1, 0))
        self.log += "Plan to re-evaluate {} of the top human variations to cloud judgement. \n".format(re_evaluations)
        top_human_moves = sorted(human_move_evals.keys(), reverse=True, key= lambda x: human_move_evals[x])
        
        # If the number of re-evaluations far exceeds the numbre of top moves, we may keep 
        # re-evaluating each seed move with greater depth        
        depth = re_evaluations // no_root_moves
        
        re_evaluate_moves = random.sample(top_human_moves, min(re_evaluations, len(top_human_moves)))
        san_re_evaluate_moves = [self.current_board.san(chess.Move.from_uci(x)) for x in re_evaluate_moves]
        self.log += "Re-evaluating moves: {} with depth {} \n".format(san_re_evaluate_moves, depth)
        re_evaluations_dic = self._re_evaluate(self.current_board, re_evaluate_moves, no_root_moves, depth=depth, prev_board=prev_board)
        san_re_evaluations_dic = {self.current_board.san(chess.Move.from_uci(k)):v for k,v in re_evaluations_dic.items()}
        self.log += "Re-evaluated evals are: {} \n".format(san_re_evaluations_dic)
        human_move_evals.update(re_evaluations_dic)
        
        san_human_move_evals = {self.current_board.san(chess.Move.from_uci(k)): v for k, v in human_move_evals.items()}
        self.log += "Updated human move evaluations are: {} \n".format(san_human_move_evals)
        
        # To further randomise and avoid repetitional play, we cloud the evaluations further by some Gaussian noise
        # To incentivise re-evaluated moves (so that spending longer on moves actually means better judgement)
        # we have larger noise levels for non re-evaluated moves. The greater the depth 
        # we re-evaluated the moves the lesser the noise
        for move_uci in human_move_evals.keys():
            if move_uci in re_evaluate_moves:
                noise = (np.abs(human_move_evals[move_uci])**0.2)*np.random.randn()*100/(5*target_time*(depth + 1))
            else:
                # if we haven't re-evaluated the move, wehave slight negative bias towards it, hence the
                # slight negative mean
                noise = -70 + (1+np.abs(human_move_evals[move_uci])**0.2)*np.random.randn()*100/(5*target_time)
            
            human_move_evals[move_uci] += noise
        
        san_human_move_evals = {self.current_board.san(chess.Move.from_uci(k)): v for k, v in human_move_evals.items()}
        self.log += "Updated human move evaluations after noise are: {} \n".format(san_human_move_evals)
        self._write_log()
        top_move = max(human_move_evals.keys(), key= lambda x: human_move_evals[x])
        self.log += "Decided output move from human move function: {} \n".format(self.current_board.san(chess.Move.from_uci(top_move)))
        return top_move
    
    def _ponder_moves(self, board:chess.Board, move_ucis: list, search_width:int, prev_board: chess.Board = None, log:bool = True):
        """ We ponder on the given board position, and consider the moves given by the list
            of move_ucis. We again use human probabilities to narrow our search width.
            
            Returns a dictionary with:
                key: the move uci from move_ucis
                value: [move uci of response, eval of response]
            eval of response is given from the perspective of ourselves.
        """
        if log:
            self.log += "Pondering the moves {} for the fen {} \n".format(move_ucis, board.fen())
            
        return_dic = {}
        for move_uci in move_ucis:
            dummy_board = board.copy()
            move_obj = chess.Move.from_uci(move_uci)
            dummy_board.push(move_obj)
            
            # First check that the game has not ended after this move
            outcome = dummy_board.outcome()
            if outcome is not None:
                # Then game has ended, return corresonding eval
                winner = outcome.winner
                if winner is None:
                    return_dic[move_uci] = [None, 0]
                elif winner == self.current_board.turn:
                    return_dic[move_uci] = [None, 2500]
                elif winner == (not self.current_board.turn):
                    return_dic[move_uci] = [None, -2500]
                else:
                    raise Exception("Unrecgonised outcome winner: {}".format(winner))
                continue
            
            # Now get human probabilities of this new position
            game_phase = phase_of_game(dummy_board)
            un_altered_move_dic = self.get_human_probabilities(dummy_board, game_phase, log=False)
            # if we discover too few moves from getting human probabilities, then 
            # instead the root moves will be just the legal moves instead
            if len(un_altered_move_dic) <= 2:
                root_moves = list(dummy_board.legal_moves)
            else:
                if prev_board is not None:
                    prev_prev_board = prev_board.copy()
                else:
                    prev_prev_board = None
                
                altered_move_dic = self._alter_move_probabilties(un_altered_move_dic, board=dummy_board, prev_board=board, prev_prev_board=prev_prev_board, log=False)
                # for the sake of time saving, we shall not be altering these move probabilities
                human_move_ucis = list(altered_move_dic.keys())
                root_moves = [chess.Move.from_uci(x) for x in human_move_ucis[:search_width]]
            
            single_analysis = STOCKFISH.analyse(dummy_board, chess.engine.Limit(time=0.02), root_moves=root_moves)
            response = single_analysis["pv"][0].uci()
            eval_ = single_analysis['score'].pov(self.current_board.turn).score(mate_score=2500)
            #clipping
            eval_ = min(eval_, 2500)
            eval_ = max(eval_, -2500)
            return_dic[move_uci] = [response, eval_]
        
        if log:
            self.log += "Returning ponder results: {} \n".format(return_dic)
        return return_dic
            
    def _recursive_ponder(self, board: chess.Board, move_uci : str, no_root_moves, depth: int, prev_board: chess.Board = None):
        """ Recursive function for getting evaluations during pondering. """
        ponder_results = self._ponder_moves(board, [move_uci], no_root_moves, prev_board=prev_board, log=False)
                 
        if depth >= 1: # keep going
            new_board = board.copy()
            new_board.push_uci(move_uci)
            consider_move = ponder_results[move_uci][0]
            if consider_move is not None:
                # if we actually have a valid move
                # sometimes the game has already ended at this point
                return self._recursive_ponder(new_board, consider_move, no_root_moves, depth-1, prev_board=board.copy())
            else:
                return ponder_results[move_uci][1]
        else:   
            return ponder_results[move_uci][1]
    
    def _re_evaluate(self, board:chess.Board, re_evaluate_moves: list, no_root_moves: int, depth:int = 0, prev_board:chess.Board = None):
        """ Given a list of move_ucis, apply them to the current board and re_evaluate
            them using top human_moves only. This gives a non_accurate evaluation
            and simulates human foresight not being exhaustive.
            
            Returns a dictionary with key move_uci and value the evaluation (from our pov)
        """       
        
        return_dic = {}
        for move_uci in re_evaluate_moves:
            return_dic[move_uci] = self._recursive_ponder(board, move_uci, no_root_moves, depth, prev_board=prev_board)
        
        return return_dic
            
    
    def update_info(self, info_dic : dict, auto_update_analytics:bool = True):
        """ The engine is fed the following thins in the info_dic, which a dictionary
            of board information: 
                - side: either chess.WHITE or chess.BLACK - indicates what side we are
                - fens: List of fens ordered with most recent fen last. Engine makes
                        use of at most 5 previous fens.
                - self_clock_times and opp_clock_times: List of past clock times
                        with most recent last. From this we can also work out last move times.
                - self_initial_time and opp_initial_time: Starting clock times for self and opp
                - last_moves: A list of moves made with most recent last. These moves are in uci
                        string format.    
            
            This function should be called the first thing before making any calculations.
        """
        self.log += "Received and updating info_dic: \n"
        self.log += str(info_dic) + "\n"        
        self.input_info.update(info_dic)
        self.current_board = chess.Board(self.input_info["fens"][-1])
        # make sure the last fen entry is indeed our turn
        assert self.current_board.turn == self.input_info["side"]
        
        self.analytics_updated = False
        
        if auto_update_analytics == True:
            self.calculate_analytics()
        
    def calculate_analytics(self):
        """ Before any move making or human analysis is performed, statistics must be computed for
            the infomation dict self.input_info. This function must be called after every
            update_info, or everytime the info dic is updated.
            
            Returns None
        """
        self.log += "Calculating analytics for current information dictionary. \n"
        self.log += "Evaluating from the board (capital letters are white pieces): \n"
        self.log += str(self.current_board) + "\n"
        
        # Performing a quick initial analysis of the position
        self.log += "Performing initial quick analysis perhaps used later by stockfish. \n"
        start = time.time()
        analysis = STOCKFISH.analyse(self.current_board, limit=chess.engine.Limit(time=0.05), multipv=25)
        if isinstance(analysis, dict): # sometimes analysis only gives one result and is not a list.
            analysis = [analysis]
        self.stockfish_analysis = analysis
        end = time.time()
        self.log += "Analysis computed in {} seconds. \n".format(end-start)
        self.log += "Printing stockfish analysis object: {} \n".format(self.stockfish_analysis)
        # Getting lucas analytics for the position
        self.log += "Calculating lucas analytics for the position. \n"
        xcomp, xmlr, xemo, xnar, xact = get_lucas_analytics(self.current_board, analysis=self.stockfish_analysis)
        lucas_dict = {"complexity": xcomp, "win_prob": xmlr, "eff_mob": xemo, "narrowness": xnar, "activity": xact}
        self.lucas_analytics.update(lucas_dict)
        self.log += "Lucas analytics: {} \n".format(lucas_dict)
        
        # Now determine our player "mood" and set it as our mode for the rest of the calculations
        self.mood = self._set_mood()
        self.log += "Setted mood to be: {} \n".format(self.mood)
        self.analytics_updated = True
    
    def check_obvious_move(self):
        """ Given input information, check whether there is an obvious move in the
            position that we may play immediately.
            
            Returns [obvious_move: uci_str, obvious_move_found : bool]
            In the case that no obvious is found, obvious_move is None"""
        # We shall define obvious moves as the following:
        # 1. Only moves
        # 2. Takebacks
        # 3. Forced mate in ones
        self.opponent_just_blundered = False
        
        # Check for only legal moves
        if len(self.stockfish_analysis) == 1:
            self.log += "Found only legal move {}, so playing it as obvious move. \n".format(self.stockfish_analysis[0]["pv"][0].uci())
            return self.stockfish_analysis[0]["pv"][0].uci(), True
        
        # Check for takebacks. These moves are takebacks that are both the best move
        # and is also a takeback. If the takeback is not by far the best move (>100 cp)
        # then with some probability return no.
        if len(self.input_info["last_moves"]) >= 1:
            prev_board = chess.Board(self.input_info["fens"][-2])
            last_move_uci = self.input_info["last_moves"][-1]
            best_move_uci = self.stockfish_analysis[0]["pv"][0].uci()
            cp_diff = self.stockfish_analysis[0]["score"].pov(self.current_board.turn).score(mate_score=2500) - self.stockfish_analysis[1]["score"].pov(self.current_board.turn).score(mate_score=2500)
            if is_takeback(prev_board, last_move_uci, best_move_uci):
                self.log += "Detected top move {} is a takeback. \n".format(best_move_uci)
                # check if the takeback is a blunder, if it is then capturing it is not an obvious moved
                last_move_obj = chess.Move.from_uci(last_move_uci)
                if calculate_threatened_levels(last_move_obj.to_square, prev_board)  <= 0 and PIECE_VALS[self.current_board.piece_type_at(last_move_obj.to_square)] - PIECE_VALS[prev_board.piece_type_at(last_move_obj.to_square)] > 0.6:
                    # then it is blunder
                    self.log += "Opponent has captured a piece that was not enpris: {}, not considering it as takeback. \n".format(prev_board.san(last_move_obj))
                    self.opponent_just_blundered = True
                else:                    
                    if cp_diff > 100:
                        self.log += "Top move is also by far best option, returning as obvious move. \n"
                        return best_move_uci, True                    
                    else:
                        if np.random.random() < 0.7:
                            self.log += "Top move is not by far best, but by chance we still return it as obvious move. \n"
                            return best_move_uci, True
                        else:
                            self.log += "Top move is not by far the best, by chance we don't return it as obvious move. \n"
            
        # Check for forced mate in ones
        # We only return obvious move if previously it was mate in 2 and no matter what
        # enemy plays our obvious move would have been mate in one
        if len(self.input_info["fens"]) >= 2 and self.stockfish_analysis[0]["score"].pov(self.current_board.turn).mate() == 1:
            last_board = chess.Board(self.input_info["fens"][-2])
            last_analysis = STOCKFISH.analyse(last_board, limit=chess.engine.Limit(time=0.01,mate=1))
            last_mate = last_analysis["score"].pov(self.current_board.turn).mate()            
            if last_mate == 2:
                last_mate_move_uci = last_analysis["pv"][1].uci()
                if self.stockfish_analysis[0]["pv"][0].uci() == last_mate_move_uci:
                    self.log += "Found obvious move that no matter what gives mate next move: {} \n". format(last_mate_move_uci)
                    return last_mate_move_uci, True
                
        # Otherwise no obvious move
        self.log += "No obvious move found. \n"
        return None, False
    
    def _set_target_time(self):
        """ Given we are using human approaches to decide the move, we set and initial
            target time which we try to compute our human move. This is supposedly a
            reflection of how hurredly the player is before they've even thought
            about any moves. 
        
            Returns target time, a non-negative float.
        """
        self_initial_time = self.input_info["self_initial_time"]
        own_time = self.input_info["self_clock_times"][-1]
        
        base_time = QUICKNESS*self_initial_time/(85 + self_initial_time*0.25)
        
        # if we are in hurry mode (i.e. we are in low time), then we adjust our base 
        # time accordingly
        mood_sf_dic = {"confident": 1,
                        "cocky": 0.6,
                        "cautious": 1.6,
                        "tilted": 0.4,
                        "hurry": (own_time/self_initial_time)**0.7,
                        "flagging": 0.8}
        
        # we need to leverage information from lucas analytics
        complexity = self.lucas_analytics["complexity"]
        complexity_sf = ((complexity+15)/30)**0.4
        
        target_time = base_time * complexity_sf * mood_sf_dic[self.mood]
        self.log += "Decided target time for human evaluation to be: {} \n".format(target_time)
        return target_time
    
    def _get_time_taken(self, obvious:bool=False, human_filters:bool=True):
        """ Calculates the amount of time in total we should spend on a move.
            obvious is whether we made a quick obvious move.  
            human_filters is whether we used human filters
        
            Returns time_taken : float
        """
        self.log += "Deciding time taken to make the move from receiving input. \n"
        self_initial_time = self.input_info["self_initial_time"]
        base_time = QUICKNESS*self_initial_time/(85 + self_initial_time*0.25)
        self.log += "Initial base time without calculations: {} \n".format(base_time)
        # we move faster depending on whether we are proportionally behind on time
        # or move slower if we are ahead
        own_time = max(self.input_info["self_clock_times"][-1],1)
        opp_time = max(self.input_info["opp_clock_times"][-1],1)
        base_time *= (own_time/opp_time)**(20/self_initial_time)
        self.log += "Base time after relative time vs opponent: {} \n".format(base_time)
        
        obvious_sf = 0.8
        # default value for time_taken
        time_taken = 1.0
        if obvious == True:
            # if we have made obvious move, we don't need to take much time
            time_taken = base_time * (obvious_sf + np.clip(0.2*np.random.randn(), -0.5, 0.5))
            self.log += "We have made obvious mode, so move quickly in {} seconds. \n".format(time_taken)
            return time_taken
        elif human_filters == True:
            # we have used human probabilities calculation.
            # our main analysis is for this case
            game_phase = phase_of_game(self.current_board)
            # in the opening and endgame we spend less time on average than the mid game
            if game_phase == "opening":
                base_time *= 0.2
            elif game_phase == "midgame":
                base_time *= 1.2
            else:
                base_time *=0.9
            
            self.log += "Base time after game phase analysis: {} \n".format(base_time)
            
            # The more activity we have in the position, the more we think
            activity = self.lucas_analytics["activity"]
            activity_sf = ((activity+12)/25)**0.4
            base_time *= activity_sf
            
            self.log += "Base time after activity analysis: {} \n".format(base_time)
            
            # The greater the proportion of good moves in a position, the less we think
            eff_mob = self.lucas_analytics["eff_mob"]
            if eff_mob > 25:
                base_time *= 0.7
                self.log += "Base time after eff_mob analysis: {} \n".format(base_time)
            
            # If opponent has just blundered, then act startled
            if self.opponent_just_blundered == True:
                self.log += "Opponent has just blundered, acting startled with longer think time. \n"
                base_time *= 2
            
            # Now sort according to moods
            if self.mood == "confident":
                # low variation, solid move times.                
                if np.random.random() < 0.3: # then we have a quick low variation move
                    time_taken = base_time * (1.0+ np.clip(0.1*np.random.randn(), -0.3, 0.3))
                elif np.random.random() < 0.7: # medium low variation move
                    time_taken = base_time * (1.3+ np.clip(0.15*np.random.randn(), -0.4, 0.4))
                else:
                    # slightly longer think, larger variation
                    time_taken = base_time * (3.5 + np.clip(0.7*np.random.randn(), -1.7, 2.0))
            elif self.mood == "cocky":
                # medium variation, quick move times
                if np.random.random() < 0.8: # then we have a quick low variation move
                    time_taken = base_time * (0.9 + np.clip(0.1*np.random.randn(), -0.3, 0.3))
                else:
                    # slightly longer think
                    time_taken = base_time * (3.3 + np.clip(0.4*np.random.randn(), -0.8, 0.8))
            elif self.mood == "cautious":
                # medium variation, slow moves
                if np.random.random() < 0.5: # then we have a quick low variation move
                    time_taken = base_time * (1.3+ np.clip(0.1*np.random.randn(), -0.3, 0.3))
                elif np.random.random() < 0.6: # medium low variation move
                    time_taken = base_time * (2.1+ np.clip(0.15*np.random.randn(), -0.4, 0.4))
                else:
                    # slightly longer think, large variation
                    time_taken = base_time * (6.5 + np.clip(1.4*np.random.randn(), -3.9, 4.5))
            elif self.mood == "tilted":
                # if we have just made the blunder, pause for long time to reflect on it
                # otherwise low variation, quick move times
                if self.just_blundered == True:
                    time_taken = base_time * (4.5 + np.clip(0.7*np.random.randn(), -1.5, 1.5))
                else:
                    time_taken = base_time * (0.8 + np.clip(0.08*np.random.randn(), -0.2, 0.2))
            elif self.mood == "hurry":
                # medium variation, quick move times
                if np.random.random() < 0.4:
                    time_taken = base_time * (0.8 + np.clip(0.1*np.random.randn(), -0.3, 0.3))
                elif np.random.random() < 0.6:
                    time_taken = base_time * (1.1 + np.clip(0.1*np.random.randn(), -0.3, 0.3))
                else:
                    # slightly longer think
                    time_taken = base_time * (1.7 + np.clip(0.2*np.random.randn(), -0.4, 0.6))
                
                # if we are in hurry mode (i.e. we are in low time), then our time taken
                # depends on how much time we have left
                time_taken *= (3*own_time/self_initial_time)**0.9
            elif self.mood == "flagging":
                # large variation, quick move times
                if np.random.random() < 0.4:
                    time_taken = base_time * (1.2 + np.clip(0.1*np.random.randn(), -0.3, 0.3))
                elif np.random.random() < 0.7:
                    time_taken = base_time * (1.6 + np.clip(0.3*np.random.randn(), -0.5, 0.5))
                else:
                    # slightly longer think
                    time_taken = base_time * (3.1 + np.clip(0.4*np.random.randn(), -0.8, 1.0))
            
            self.log += "Decided time taken after mood analysis: {} \n".format(time_taken)
        self.log += "Decided time taken for move: {} \n".format(time_taken)
        return time_taken
    
    def _set_mood(self):
        """ Given input information, we determine our mood which influences the rest of our
            calculations. The calculation of these moods should ideally not depend on engine
            analytics such as lucas analytics and should rely on more human heuristics
            (Material up/down, time situation etc). Moods can be the following, and roughly 
            correspond to the following behaviours:
            
            Confident - Default mood. Plays move at normal speeds relative to the game phase
            Cocky - Plays moves faster and considers less moves in general in a situation.
                    Usually occurs when player is winning heavily in material and not under big threats.
            Cautious - Plays moves slower and considers more moves and ponders more.
                    Usually occurs in complex even positions.
            Tilted - Plays moves either really fast or takes one big think and considers less
                    moves. Happens when we have recently blundered some material,
                    or are losing so badly and close to resigning.
            Hurry - Plays moves faster, consider less moves and does more pondering
                    Occurs when we are in time trouble.
            Flagging - Plays moves faster, considers less sometimes or is cautious.
                    Occurs when we have a lot more time than opponent and they are in time
                    trouble.
        """
        
        self.log += "Setting mood from given input information. \n"
        self.just_blundered = False
        # First check our time situation
        # If we are low in time, we are in hurry mode.
        # We define low time to be normally distributed about (initial_time*0.1 + 15*0.7)
        # with standard deviation initial_time/30
        self_initial_time = self.input_info["self_initial_time"]
        opp_initial_time = self.input_info["opp_initial_time"]
        own_time = self.input_info["self_clock_times"][-1]
        opp_time = self.input_info["opp_clock_times"][-1]
        self_low_time_threshold = self_initial_time*0.1 + 15*0.7 + self_initial_time*np.random.randn()/30
        opp_low_time_threshold = opp_initial_time*0.1 + 15*0.7 + opp_initial_time*np.random.randn()/30
        
        if own_time < self_low_time_threshold:
            return "hurry"
        self.log += "We have more than the threshold {} time, not in hurry mode. \n".format(self_low_time_threshold)
        
        # Next we consider whether we are tilted from a past blunder
        # this requires information of evaluations from previous positions
        # If we made the blunder one move ago (so eval from 3 positions ago was much higher than it is now)
        # Then we make a big pause and set just_blundered to be True
        # First determine whether we have made a mistake in the last few moves.
        current_eval = self.stockfish_analysis[0]["score"].pov(self.current_board.turn).score(mate_score=2500)
        if len(self.input_info["fens"]) >= 3:
            self.log += "Checking to see if we have made a blunder recently. \n"            
            last_avail_board = chess.Board(self.input_info["fens"][0])
            last_eval = STOCKFISH.analyse(last_avail_board, limit=chess.engine.Limit(time=0.01))["score"].pov(self.current_board.turn).score(mate_score=2500)
            if last_eval - current_eval > 300 and current_eval < 200: # if our previous position is much better than it is now, and we are not massively winning still
                # if the blunder happened exactly a move ago
                # then with some probability we recognise we have just made a blunder, and this
                # fact is then used in _get_time_taken()
                if np.random.random() < 0.8:
                    self.just_blundered = True
                else:
                    self.just_blundered = False
                return "tilted"
            self.log += "Haven't made a blunder recently. Last eval: {}, Current eval: {} \n".format(last_eval, current_eval)
        
        # Next check opponent time. If opponent is in time pressure we would be in flagging mode
        if opp_time < opp_low_time_threshold:
            # we won't always be in flagging mode, particularly if we have enough time.
            if np.random.random() < 0.7:
                return "flagging"
            else:
                self.log += "Opponent has less than the threshold {} time, but by chance not in flagging mode. \n".format(opp_low_time_threshold)
        else:
            self.log += "Opponent has more than the threshold {} time, not in flagging mode. \n".format(opp_low_time_threshold)
        
        # If we are up on time lots and winning by alot on the position, then we are in cocky mode
        # Also in the initial stages of the game when we are blitzing out opening moves.
        # We define "up on time by a lot" by initial_time/6
        # We define initial stages as  time > initial_time - low_time_threshold/2
        if own_time > self_initial_time - self_low_time_threshold/2:
            return "cocky"
        self.log += "Not in initial stages and blitzing opening moves. Not in cocky mode. \n"
        time_gap = self_initial_time/6
        if current_eval > 300 and own_time - opp_time > time_gap:
            return "cocky"
        self.log += "Time gap {} and on current eval {}. We are not in cocky mode. \n".format(own_time - opp_time, current_eval)
        
        # If position is relatively even, and complex (lots of good moves)
        # the more complex the position, the more likely we are cautious
        complexity = self.lucas_analytics["complexity"]
        eff_mob = self.lucas_analytics["eff_mob"]
        if abs(current_eval) < 250:
            if np.random.random() < (0.2 + complexity/(100*eff_mob + 100))**0.6:
                return "cautious"
            else:
                self.log += "Postion is close to even (current eval {}) and complex (complexity {}). But by chance not cautious. \n".format(current_eval, complexity)
        else:
            self.log += "Position not even enough (current eval {}) or not complex enough (complexity {}). Not in cautious mode. \n".format(current_eval, complexity)
        
        # If no previous criteria is satisfied, resort to default mood
        return "confident"
            
    def get_premove(self, board:chess.Board, side:bool, takeback_only:bool=False):
        """ Given a position which is not our turn, using only computer evaluations return
            a premove which we may make. The nature of this function is so that it
            may be called to spot immediate takebacks, and also be used in last second
            time scramble situations. Side is the side which we are on.
            
            If takeback_only is True, then we only return a premove if we found a
            takeback premove, otherwise we return None.
            
            Returns a move_uci string.
        """
        # First check that the position given is indeed not our turn
        assert board.turn != side
        
        # Next check for takeback premoves
        # We define takebacks here to be moves that the opposition can make that
        # capture material, and it is in our best interest to capture back.
        for move_obj in board.legal_moves:
            from_value = PIECE_VALS[board.piece_type_at(move_obj.from_square)]
            piece_type_to = board.piece_type_at(move_obj.to_square)            
            if piece_type_to is not None:
                to_value = PIECE_VALS[piece_type_to]
                if to_value - from_value > -0.6: # roughly similar trade
                    exists, takeback_move_uci = check_best_takeback_exists(board.copy(), move_obj.uci())
                    if exists == True:
                        # then we have a best takeback
                        premove_uci = takeback_move_uci
                        self.log += "Detected and returning takeback premove: {}. \n".format(premove_uci)
                        return premove_uci
        
        if takeback_only == True: # if we are only looking for takeback premoves
            return None
        
        # If no takebacks, then use computer moves to determine best premove to make.
        # We pretend opponent makes top engine move, and we respond using get_stockfish_move
        # perform analysis on current position 
        analysis = STOCKFISH.analyse(board, limit=chess.engine.Limit(time=0.01))
        opp_best_move_obj = analysis["pv"][0]
        dummy_board = board.copy()
        dummy_board.push(opp_best_move_obj)
        next_analysis = STOCKFISH.analyse(dummy_board, limit=chess.engine.Limit(time=0.01), multipv=10)
        if isinstance(next_analysis, dict):
            next_analysis = [next_analysis]
        # now use get_stockfish move on this position
        # Of course, we can only get a stockfish move if the game is not over
        if dummy_board.outcome() is None:
            premove_uci = self.get_stockfish_move(board=dummy_board, analysis=next_analysis, last_move_uci=opp_best_move_obj.uci(), log=False)
            self.log += "Detected premove from stockfish evals: {} \n".format(premove_uci)
        else:
            # game is over
            self.log += "Cannot get premove for position {} as it is game over. \n".format(dummy_board.fen())
            return None
        \
        return premove_uci
    
    def ponder(self, board: chess.Board, time_allowed : float, time_per_position : float = 0.15, prev_board:chess.Board = None):
        """ Given a board position that is not our (side) turn, we ponder possible moves
            and return a dictionary which has key board_fen (so position only), and value uci in response.
            Time allowed represents how much we may ponder. Time per position is roughly 
            the amount of time to form probabilies and alter them.
        """
        self.log += "Ponder position with fen {} \n".format(board.fen())
        self.log += "Pondering time allowed: {} \n".format(time_allowed)
        variations_allowed = max(1, int(time_allowed/time_per_position))
        
        # First double check our position has not ended. If it has no moves, return None
        if board.outcome() is not None:
            return None
        
        # As a maximum number of ponder moves (to prevent too quick responses),
        # depends on the time control.
        initial_time = self.input_info["self_initial_time"]
        if initial_time <= 180:
            max_ponder_no = 1
        else:
            max_ponder_no = 2
        
        search_width = self._decide_breadth() # this is slightly incorrect, but close enough
        ponder_depth = max(variations_allowed // (max_ponder_no * search_width) - 1, 0)
        
        analysis_object = STOCKFISH.analyse(board, limit=chess.engine.Limit(time=0.01), multipv=min(variations_allowed, max_ponder_no))
        if isinstance(analysis_object, dict):
            analysis_object = [analysis_object]
        
        opp_moves_considered = [entry["pv"][0].uci() for entry in analysis_object]
        san_opp_moves_considered = [board.san(chess.Move.from_uci(x)) for x in opp_moves_considered]
        self.log += "Considering with ponder depth {}, moves in this position: {} \n".format(ponder_depth, san_opp_moves_considered)
        
        # Now execute or recursive function to find response dictionary        
        return_dic = {}
        san_return_dic = {}
        for move_uci in opp_moves_considered:
            dummy_board = board.copy()
            dummy_board.push_uci(move_uci)
            # first check that the game is not over
            if dummy_board.outcome() is not None:
                # then game is over
                # we can't respond to this
                continue            
            board_fen = dummy_board.board_fen()
            game_phase = phase_of_game(dummy_board)
            top_human_move_dic = self.get_human_probabilities(dummy_board, game_phase, log=False)
            if len(top_human_move_dic) <= 2:
                top_human_moves = [move.uci() for move in dummy_board.legal_moves]
            else:
                top_human_move_dic = self._alter_move_probabilties(top_human_move_dic, dummy_board, prev_board = board, prev_prev_board=prev_board, log= False)
                top_human_moves = sorted(top_human_move_dic.keys(), key=lambda x: top_human_move_dic[x], reverse=True)[:search_width]
                
            re_evaluate_dic = self._re_evaluate(dummy_board, top_human_moves, search_width, depth=ponder_depth, prev_board = board.copy())
            best_response = max(re_evaluate_dic.keys(), key=lambda x : re_evaluate_dic[x])            
            return_dic[board_fen] = best_response
            san_return_dic[board_fen] = dummy_board.san(chess.Move.from_uci(best_response))
        self.log += "Computed responses for these moves: {} \n".format(san_return_dic)
        
        if len(return_dic) == 0:
            return None
        else:
            return return_dic
        
    def _check_opp_blunder(self):
        """ Check from last couple of positions whether opponent has made a blunder on the
            last move, and did so by hanging a valuable piece. 
            
            Returns None. We only update the self.opp_just_blundered variable
        """
        self.opponent_just_blundered = False
        # We can only do this if we have the previous positions excluding our current position
        if len(self.input_info["fens"]) < 2:
            self.log += "Can't detect whether opponent has blundered as we don not have enough previous position info. \n"
            return
        # First check opponent just blundered based on eval
        curr_eval = self.stockfish_analysis[0]["score"].pov(self.current_board.turn).score(mate_score=2500)
        prev_board = chess.Board(self.input_info["fens"][-2])
        prev_analysis = STOCKFISH.analyse(prev_board, limit=chess.engine.Limit(time=0.01))
        prev_eval = prev_analysis["score"].pov(self.current_board.turn).score(mate_score=2500)
        if curr_eval - prev_eval > 150: # then opponent just blundered
            self.log += "Opponent has blundered, checking to see if it is from hung piece. \n"
            # now check if opponent just hung piece they played
            last_move_obj = chess.Move.from_uci(self.input_info["last_moves"][-1])
            if calculate_threatened_levels(last_move_obj.to_square, self.current_board) >= 3:
                # then opponent has hung a piece
                self.log += "Opponent has hung a piece, acting startled. \n"
                self.opponent_just_blundered = True
            else:
                self.log += "Opponent has not hung a piece. \n"
        else:
            self.log += "Opponent has not blundered. Current eval {}, previous eval {} \n".format(curr_eval, prev_eval)
    
    def make_move(self, log:bool=True):
        """ This is the main function for prompting a move output from the engine. 

            Returns a dictionary with the following outputs:
                - move_made: uci string of move made
                - time_take: time taken to execute the move, excludes the time in processing the move
                - premove: (Optional) A premove to make immediately after
                - ponder_dic: (Optional) A dictionary of responses representing pre-thought out lines 
                        that we can respond quickly by without needing to consult the engine again.
                        This entry tends to get returned when we have had long think time for
                        our move.
        """
        if log == True:
            self._write_log()
        
        move_start = time.time()
        return_dic = {}
        self.log += "Make move function called. \n"
        
        # If analytics for the position hasn't been called, issue warning.
        if self.analytics_updated == False:
            self.log += "WARNING: calculating move with outdated analytics. Please run .calculate_analytics() function to update for the new information dict. \n"
        
        # We should check for obvious moves that don't require much thought
        obvious_move, obvious_move_found = self.check_obvious_move()
        if log == True:
            self._write_log()
        if obvious_move_found == True:
            return_dic["move_made"] = obvious_move
            use_human_filters = False
        else:
            # Now need to decide base on information whether we are using strictly engine move
            # Or using human filters
            self.log += "Deciding whether to use human filters. \n"
            use_human_filters = self._decide_human_filters()
            if use_human_filters == True:
                human_start = time.time()
                self.log += "Using human filters. \n"
                # first let us decide how much time we shall try spend on the move
                # key word "try"
                target_time = self._set_target_time()
                return_dic["move_made"] = self.get_human_move(target_time=target_time)
                human_end = time.time()
                self.log += "Human move gotten in {} seconds. \n".format(human_end-human_start)
            else:
                self.log += "Not using human filters. \n"
                stockfish_start = time.time()
                return_dic["move_made"] = self.get_stockfish_move()
                stockfish_end = time.time()
                self.log += "Stockfish move gotten in {} seconds. \n".format(stockfish_end-stockfish_start)
        
        if log == True:
            self._write_log()
        # Now that we have decided what move are going to make, lets check whether the opponent
        # hung a big piece the previous move (and it was a blunder), so we can act startled
        self.log += "Checking for opponent blunders. \n"
        self._check_opp_blunder()
        if log == True:
            self._write_log()
        
        # If we are in a hurry, and our time is absolutely low then we also return 
        # a premove for the next move.
        # If we are not in a hurry, look for takeback premoves
        # Sometimes if the time control is bullet, and we are in the opening, we
        # may also premove with some probability
        own_time = self.input_info["self_clock_times"][-1]        
        after_board = self.current_board.copy()
        after_board.push_uci(return_dic["move_made"])
        if after_board.outcome() is None:
            self_initial_time = self.input_info["self_initial_time"]
            premove_start = time.time()
            if self.mood == "hurry" and own_time < 20:
                # with some probability we return a premove
                if np.random.random() < 0.4*self_initial_time/(own_time + 0.4*self_initial_time):
                    return_dic["premove"] = self.get_premove(after_board, self.current_board.turn)
                else:
                    # look for takeback premoves only
                    return_dic["premove"] = self.get_premove(after_board, self.current_board.turn, takeback_only=True)
            elif self_initial_time <= 60 and phase_of_game(self.current_board) == "opening":
                # with some probability return a forced premove, otherwise just look for takeback premoves
                if np.random.random() < 0.2:
                    return_dic["premove"] = self.get_premove(after_board, self.current_board.turn)
                else:
                    # look for takeback premoves only
                    return_dic["premove"] = self.get_premove(after_board, self.current_board.turn, takeback_only=True)
            else:
                # look for takeback premoves only
                return_dic["premove"] = self.get_premove(after_board, self.current_board.turn, takeback_only=True)
            premove_end = time.time()
            self.log += "Premove search took {} seconds. \n".format(premove_end-premove_start)
        else:
            return_dic["premove"] = None
        
        self.log += "Gotten premove: {} \n".format(return_dic["premove"])
        if log == True:
            self._write_log()
        # Finally decide how much time we are going to spend (including thinking time)
        return_dic["time_take"] = self._get_time_taken(obvious=obvious_move_found, human_filters=use_human_filters)
        move_end = time.time()
        if log == True:
            self._write_log()
        # If we have extra time than that of alloted then we may do some pondering for the position after our move
        time_spent = move_end - move_start
        
        if return_dic["time_take"] - time_spent > 0.6:
            self.log += "Have enough time to ponder for he next position. Time taken so far: {} \n".format(time_spent)
            ponder_start = time.time()
            ponder_dic = self.ponder(after_board, return_dic["time_take"] - time_spent- 0.05, prev_board=self.current_board.copy())
            ponder_end = time.time()
            self.log += "Took {} seconds for pondering. \n".format(ponder_end - ponder_start)
        else:
            ponder_dic = None
        return_dic["ponder_dic"] = ponder_dic
        
        if log == True:
            self._write_log()
        self.log += "Returning return dic with all of our engine's calculations: \n"
        self.log += "{} \n".format(return_dic)
        
        # log our calculating information for this move
        if log == True:
            self._write_log()
        
        return return_dic

if __name__ == "__main__":
    engine = Engine()
    # b = chess.Board("3r2k1/3r1p1p/PQ2p1p1/8/5q2/2P2N2/1P3PP1/R3K2R w KQ - 1 24")
    input_dic ={'fens': ['4r3/p1p3pp/8/1p2k3/2p2p1P/2P2KP1/P7/8 w - - 0 31', '4r3/p1p3pp/8/1p2k3/2p2pPP/2P2K2/P7/8 b - - 0 31', '4r3/p1p4p/6p1/1p2k3/2p2pPP/2P2K2/P7/8 w - - 0 32', '4r3/p1p4p/6p1/1p2k2P/2p2pP1/2P2K2/P7/8 b - - 0 32', '4r3/p1p4p/8/1p2k1pP/2p2pP1/2P2K2/P7/8 w - - 0 33', '4r3/p1p4p/7P/1p2k1p1/2p2pP1/2P2K2/P7/8 b - - 0 33', '4r3/2p4p/7P/pp2k1p1/2p2pP1/2P2K2/P7/8 w - - 0 34', '4r3/2p4p/7P/pp2k1p1/P1p2pP1/2P2K2/8/8 b - - 0 34'], 'self_clock_times': [31, 30, 29, 29, 28, 27, 26, 25], 'opp_clock_times': [35, 34, 34, 33, 33, 32, 31, 30], 'last_moves': ['g3g4', 'g7g6', 'h4h5', 'g6g5', 'h5h6', 'a7a5', 'a2a4'], 'side': False, 'self_initial_time': 60, 'opp_initial_time': 60}
    start = time.time()
    engine.update_info(input_dic)
    print(engine.make_move(log=False))
    end = time.time()
    print("finished in {} seconds".format(end-start))