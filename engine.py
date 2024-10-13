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
import chess.engine
import chess.polyglot

from models.models import MoveScorer, StockFishSelector
from common.constants import (PATH_TO_STOCKFISH, MOVE_FROM_WEIGHTS_OP_PTH, MOVE_FROM_WEIGHTS_MID_PTH,
                              MOVE_FROM_WEIGHTS_END_PTH, MOVE_TO_WEIGHTS_MID_PTH, 
                              MOVE_TO_WEIGHTS_END_PTH, MOVE_TO_WEIGHTS_OP_PTH,
                              QUICKNESS, PATH_TO_PONDER_STOCKFISH
                              )

from common.board_information import (
    phase_of_game, PIECE_VALS, STOCKFISH, get_lucas_analytics, is_capturing_move, is_capturable,
    is_attacked_by_pinned, is_check_move, is_takeback, is_newly_attacked, get_threatened_board,
    is_offer_exchange, king_danger, is_open_file, calculate_threatened_levels, check_best_takeback_exists,
    is_weird_move
            )
from common.utils import flip_uci, patch_fens, check_safe_premove, extend_mate_score

# TODO: Intelligent premoves
# TODO: 3-fold repetition logic
PONDER_STOCKFISH = chess.engine.SimpleEngine.popen_uci(PATH_TO_PONDER_STOCKFISH) # separate stockfish object used entirely for pondering

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
        self.mood = "confident"
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
        
        own_time = max(self.input_info["self_clock_times"][-1],1)
        opp_time = max(self.input_info["opp_clock_times"][-1],1)
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
        our_mat = sum([len(self.current_board.pieces(x, self.input_info["side"]))*mat_dic[x] for x in range(1,7)])
        opp_mat = sum([len(self.current_board.pieces(x, not self.input_info["side"]))*mat_dic[x] for x in range(1,7)])
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
        own_time = max(self.input_info["self_clock_times"][-1],1)
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
        move_eval_dic = {entry["pv"][0].uci(): extend_mate_score(entry["score"].pov(board.turn).score(mate_score=2500)) for entry in sampled_moves}
        # if we are winning by a big margin such that we have mate in less than 10 moves, and we have sufficient time, than with large probability we play the
        # zeroing moves. This helps us out in closing games
        own_time = max(self.input_info["self_clock_times"][-1],1)
        opp_time = max(self.input_info["opp_clock_times"][-1],1)
        top_engine_move = max(move_eval_dic.keys(), key= lambda x: move_eval_dic[x])
        if opp_time > own_time and max(move_eval_dic.values()) >= 2490 and self.mood == "hurry": #mate in less than 15 and we are under time pressure
            if np.random.random() < 0.8:                
                if log:
                    self.log += "We have sufficient time ({}) and we have spotted mate in {}. Playing top engine move to close the game. \n".format(own_time, 2500-move_eval_dic[top_engine_move])
                return top_engine_move
            else:
                if log:
                    self.log += "We spotted mate in {}, but by chance we don't go for top move. \n".format(2500-move_eval_dic[top_engine_move])
        else:
            no_pieces = len(chess.SquareSet(board.occupied))
            if no_pieces < 10 and self.mood == "hurry": # less than 10 pieces including kings
                # with some probability play best move
                if np.random.random() < 0.4:                
                    if log:
                        self.log += "Less than 10 pieces on the board, playing top engine move to close the game. \n"
                    return top_engine_move
                else:
                    if log:
                        self.log += "Less than 10 pieces on the board, but by chance not playing top engine move. \n"
        if log:
            self.log += "Sampled the following moves and their corresponding evals: {} \n".format(move_eval_dic)
        move_distance_dic = {}
        for move_uci in move_eval_dic.keys():
            distance = 0
            move_obj = chess.Move.from_uci(move_uci)
            # if we are given information about our own previous move, then include that distance too
            # we weight the move getting to our next piece than the distance travelled by that distance
            if last_move_uci is not None:
                own_last_move_obj = chess.Move.from_uci(last_move_uci)
                to_square = own_last_move_obj.to_square
                distance += chess.square_distance(to_square, move_obj.from_square)
            distance += 0.5*chess.square_distance(move_obj.from_square, move_obj.to_square)
            move_distance_dic[move_uci] = distance
        
        if log:
            self.log += "Evaluated the moves square distance to move: {} \n".format(move_distance_dic)
        own_time = max(self.input_info["self_clock_times"][-1],1)
        move_appealing_dic = {move_uci : 10 + move_eval_dic[move_uci]*(own_time+5)/2000 - move_distance_dic[move_uci] for move_uci in move_eval_dic.keys()}
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
    
    def adjust_human_prob(self, move_dic, board : chess.Board = None):
        """ Given move_dic from human probabilities, we normalise the probabilities
            i.e. make them less extreme depending on how low on time we are as well
            as how far in the game we are (remaining pieces) as well as how far we
            are winning by (helps with closing out games).
            
            Returns normalised move_dic
        """
        if board is None:
            board = chess.Board(self.input_info["fens"][-1])
        power_factor = 1
        # first make sure all probabilities are in the range (0, 1)
        eps = 10**(-10)
        return_move_dic = {k: np.clip(v, eps,1-eps) for k, v in move_dic.items()}
        
        if self.mood == "hurry":
            # the less time we have, the more we normalise
            own_time = max(self.input_info["self_clock_times"][-1], 1)
            initial_time = self.input_info["self_initial_time"]
            power_factor *= np.sqrt(initial_time/own_time)
        
        pieces_left = len(list(chess.SquareSet(board.occupied)))
        if pieces_left <= 18:
            power_factor *= (19-pieces_left)/5
        
        eval_ = extend_mate_score(self.stockfish_analysis[0]["score"].pov(self.input_info["side"]).score(mate_score=2500))
        # although not quite correct, we shall use to ease computation
        if eval_ > 500: # massively winning
            power_factor *= (eval_ - 100)/400
        
        return_move_dic = {k: v**(1/power_factor) for k, v in return_move_dic.items()}
        # normalise to add to 1
        total = sum(return_move_dic.values()) + eps
        return_move_dic = {k: v/total for k, v in return_move_dic.items()}
        
        return return_move_dic
        
    
    def get_human_probabilities(self, board : chess.Board, game_phase: str, log:bool = True):
        """ Given a chess.Board item, returns the top move ucis along with their
            human move probabilties, evaluated from neural network only. These bare
            no extra tinkering methods and are purely from the neural net.    
        """
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
            
        # adjust according to time
        nn_top_move_dic = self.adjust_human_prob(nn_top_move_dic, board=board)
        if log:
            log_move_dic = {board.san(chess.Move.from_uci(k)) : round(v, 5) for k,v in nn_top_move_dic.items()}
            self.log += "Move_dic after normalising alteration: {} \n".format(log_move_dic)
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
        
        # Punish weird moves
        weird_move_sd_dic = {"opening" : 0.1,
                             "midgame" : 0.3,
                             "endgame" : 1.0,
            }
        
        lower_threshold_prob = sum(move_dic.values())/len(move_dic)
        
        # If king danger high, then moves that defend our king are more attractive
        protect_king_sf = 2.8
        # Capturing enpris pieces are more appealing, the more the piece is enpris the more appealing
        capture_en_pris_sf = 1.5
        # Squares that pinned pieces attack that break the pin are more desirable
        break_pin_sf = 3.0
        # Captures are just generally more appealing. The bigger the capture the more appealing
        capture_sf = 1.5
        # Capturable pieces are more appealling to move
        capturable_sf = 1.3
        
        # Checks are more appealing (particularly under time pressure)
        check_sf_dic = {"confident": 2.3,
                        "cocky": 3.3,
                        "cautious": 2.1,
                        "tilted": 3.3,
                        "hurry": 3.0,
                        "flagging": 2.9}
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
        repeat_sf_dic = {"confident": 0.3,
                        "cocky": 0.5,
                        "cautious": 0.3,
                        "tilted": 0.6,
                        "hurry": 0.5,
                        "flagging": 0.4}
        
        
###############################################################################
        game_phase = phase_of_game(board) # useful for function
        self_king_danger_lvl = king_danger(board, board.turn, game_phase)
        opp_king_danger_lvl = king_danger(board, not board.turn, game_phase)
        
        # Moves which protect/block/any way that make our pieces less en pris are appealing        
        # Likewise moves that make our pieces more en pris are less appealing
        # Moves which make opponent pieces more en pris (attacks/threatens) are appealing
        strenghening_moves = []
        weakening_moves = []     
        weird_moves = []
        # For time computation sake, we ignore the threatened levels of pawns
        curr_threatened_board = get_threatened_board(board, colour=board.turn, piece_types=[1,2,3,4,5])
        self_curr_threatened_levels = sum(curr_threatened_board)
        # if out of the our threatened pieces there was only one thing that was threatened, then
        # exaggerate the levels.
        if sum(np.array(curr_threatened_board) > 0.5) == 1:
            solo_factor = 2
        else:
            solo_factor = 1
        
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
            
            
            lvl_diff = opp_lvl_diff - self_lvl_diff*1.5*solo_factor
            # if our move was a capture, we must take that into account that we gained some material
            piece_type = board.piece_type_at(move_obj.to_square)
            if piece_type is not None: # if we captured something
                lvl_diff += PIECE_VALS[piece_type] * 1.5 *solo_factor         
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
            else:
                # if move doesn't seem to do anything that threatens or protects pieces, then we check
                # whether it is a "weird" move
                if is_weird_move(board, game_phase, move_uci, self_king_danger_lvl):
                    # then incur penalty
                    move_dic[move_uci] *= weird_move_sd_dic[game_phase]
                    weird_moves.append(board.san(chess.Move.from_uci(move_uci)))
        if log:
            self.log += "Found moves that are weakening and make our pieces more enpris/opp pieces less enpris: {} \n".format(weakening_moves)
            self.log += "Found moves that protect our pieces more or apply more pressure to opponent: {} \n".format(strenghening_moves)
            self.log += "Found weird moves: {} \n".format(weird_moves)
        
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
        
        # If king danger high, then moves that defend our king are more attractive
        before_king_danger = king_danger(board, board.turn, game_phase)
        # if king is not in danger, pass
        if before_king_danger < 250:
            if log:
                self.log += "King danger {} not high to consider protecting king moves. Skipping... \n".format(before_king_danger)
        else:
            protect_king_moves = []            
            for move_uci in move_dic.keys():
                dummy_board= board.copy()
                dummy_board.push_uci(move_uci)
                new_king_danger = king_danger(dummy_board, board.turn, game_phase)
                if new_king_danger <= 0:
                    # found move
                    if move_dic[move_uci] > lower_threshold_prob:
                        move_dic[move_uci] = lower_threshold_prob
                    move_dic[move_uci] *= protect_king_sf*(before_king_danger/50)**(1/4)
                    protect_king_moves.append(board.san(chess.Move.from_uci(move_uci)))
                elif before_king_danger/new_king_danger > 1.5:
                    # found move
                    if move_dic[move_uci] > lower_threshold_prob:
                        move_dic[move_uci] = lower_threshold_prob
                    denom = max(new_king_danger, 50)
                    move_dic[move_uci] *= protect_king_sf*(before_king_danger/denom)**(1/4)
                    protect_king_moves.append(board.san(chess.Move.from_uci(move_uci)))
            if log:
                self.log += "Found moves that take protect our vulnerable king: {} \n".format(protect_king_moves)
            
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
            res = patch_fens(prev_board.fen(), board.fen(), depth_lim=1)
            last_move_uci = res[0][0]
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
            res = patch_fens(prev_prev_board.fen(), prev_board.fen(), depth_lim=1)
            last_own_move_uci = res[0][0]
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
        king_dang = king_danger(self.current_board, self.input_info["side"], game_phase)
        eff_mob = self.lucas_analytics["eff_mob"]
        if eff_mob < 15:
            # Either the best move is obvious (e.g. a takeback, mate in one) or
            # there is a tactic involved. Decrease search width to mimic human
            # behaviour in tactics under pressure
            
            if eff_mob > 5 and game_phase == "midgame":
                no_moves = max(self.playing_level-1,1)
            elif game_phase == "endgame":
                no_moves = no_moves = max(8, self.playing_level+3)
            else:
                no_moves = max(self.playing_level-2, 1)
        else:
            # There are plenty of good moves, search wide so close the game out effectively
            if game_phase == 'endgame' and king_dang < 500:
                no_moves = max(7, self.playing_level+2)
            elif king_dang > 500: 
                # king is in danger, need to pay attention
                no_moves = self.playing_level+2
            else:
                # not in the endgame, lots of good moves
                no_moves = max(self.playing_level-1,1)
        
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
        start = time.time()
        
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
                eval_ = extend_mate_score(analysis_object['score'].pov(self.input_info["side"]).score(mate_score=2500))

                human_move_evals[move_uci] = eval_
        
        san_human_move_evals = {self.current_board.san(chess.Move.from_uci(k)): v for k, v in human_move_evals.items()}
        self.log += "Computed human move evals: {} \n".format(san_human_move_evals)
        
        end = time.time()
        human_calc_time = end-start
        BUFFER = 0.06 # the minimum extra time of human_calc time to deal with instances where we got a surprising quick time
        self.log += "Human probabilities including alterations evaluated in {} seconds. \n". format(human_calc_time)
        # Now simply selecting the best of these human evals is natural, however
        # a better way would be to cloud the judgement of the evaluations
        # by computing the again in a human manner. This takes time (roughly 0.12 secs
        # per move) so we can't do this for every move. The number of moves we perform
        # this re-evaluation will depend on our target time, the time we set at
        # the beginning of the move to try and make our move by.
        # if target time is too much, then shorten it
        if target_time > no_root_moves * 6 * human_calc_time:
            target_time = no_root_moves * 6 * human_calc_time
            self.log += "Shortening human target time to {} \n".format(target_time)
            
        re_evaluations = int(max(target_time//max(human_calc_time +0.02, BUFFER) - 1, 0))
        self.log += "Plan to re-evaluate {} of the top human variations to cloud judgement. \n".format(re_evaluations)
        top_human_moves = sorted(human_move_evals.keys(), reverse=True, key= lambda x: human_move_evals[x])
        
        # If the number of re-evaluations far exceeds the numbre of top moves, we may keep 
        # re-evaluating each seed move with greater depth        
        depth = (re_evaluations // no_root_moves) + 1
        
        reval_start = time.time()
        re_evaluate_moves = random.sample(top_human_moves, min(re_evaluations, len(top_human_moves)))
        san_re_evaluate_moves = [self.current_board.san(chess.Move.from_uci(x)) for x in re_evaluate_moves]
        time_allowed = target_time - (reval_start - start)
        self.log += "Re-evaluating moves: {} with depth {} with time allowed {} \n".format(san_re_evaluate_moves, depth, time_allowed)
        re_evaluations_dic = self._re_evaluate(self.current_board, re_evaluate_moves, no_root_moves, depth=depth, prev_board=prev_board, limit=[depth*no_root_moves, time_allowed])
        san_re_evaluations_dic = {self.current_board.san(chess.Move.from_uci(k)):v for k,v in re_evaluations_dic.items()}
        self.log += "Re-evaluated evals with depth considered statistics are: {} \n".format(san_re_evaluations_dic)
        # some evals in re_evaluations_dic may be None if we didn't have time to consider them. Filter these out
        re_evaluations_dic = {k:v for k,v in re_evaluations_dic.items() if v[0] is not None}
        
        new_human_move_evals = {k: [v,0] for k,v in human_move_evals.items()} # includes depth considered statistics
        new_human_move_evals.update(re_evaluations_dic)
        reval_end = time.time()
        self.log += "Re-evaluations performed in {} seconds. \n".format(reval_end - reval_start)
        
        san_human_move_evals = {self.current_board.san(chess.Move.from_uci(k)): v for k, v in new_human_move_evals.items()}
        self.log += "Updated human move evaluations are: {} \n".format(san_human_move_evals)
        
        # To further randomise and avoid repetitional play, we cloud the evaluations further by some Gaussian noise
        # To incentivise re-evaluated moves (so that spending longer on moves actually means better judgement)
        # we have larger noise levels for non re-evaluated moves. The greater the depth 
        # we re-evaluated the moves the lesser the noise
        # Also the more we considered a move, the more bias we have towards it
        # furthermore if we only used computer evaluation (i.e. depth = 0), then we have extra
        # negative penalty
        depth_penalty = 20
        zero_depth_penalty = 50
        
        # Furthermore, humans won't delay capturing pieces if they're free and enpris. So like
        # we gave capturing moves higher spotting chance, we shall also encourage capturing moves here too
        eval_only_dic = {}
        for move_uci in new_human_move_evals.keys():
            eval_, depth_considered = new_human_move_evals[move_uci]            
            base_noise_sd = 40*(np.tanh(eval_/(self.playing_level*50)))**2 + 20            
            noise_sd = 4*base_noise_sd/(target_time*(depth_considered+4))
            
            noise = np.random.randn()*noise_sd - depth_penalty*(2- depth_considered)
            if depth_considered == 0:
                noise -= zero_depth_penalty       
            
            # encourage capture moves, depending on how enpris the piece is
            move_obj = chess.Move.from_uci(move_uci)
            if self.current_board.is_capture(move_obj):
                capture_bonus = 40* int(calculate_threatened_levels(move_obj.to_square, self.current_board.copy()))
            else:
                capture_bonus = 0
            
            eval_only_dic[move_uci] = eval_ + noise + capture_bonus
        
        san_human_move_evals = {self.current_board.san(chess.Move.from_uci(k)): v for k, v in eval_only_dic.items()}
        self.log += "Updated human move evaluations after noise and capture bonuses are: {} \n".format(san_human_move_evals)
        #self._write_log()
        top_move = max(eval_only_dic.keys(), key= lambda x: eval_only_dic[x])
        self.log += "Decided output move from human move function: {} \n".format(self.current_board.san(chess.Move.from_uci(top_move)))
        return top_move
    
    def _ponder_moves(self, board:chess.Board, move_ucis: list, search_width:int, prev_board: chess.Board = None, log:bool = True, use_ponder:bool=False):
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
                elif winner == self.input_info["side"]:
                    return_dic[move_uci] = [None, extend_mate_score(2500)]
                elif winner == (not self.input_info["side"]):
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
                
                altered_move_dic = self._alter_move_probabilties(un_altered_move_dic, board=dummy_board, prev_board=board.copy(), prev_prev_board=prev_prev_board, log=False)
                
                human_move_ucis = list(altered_move_dic.keys())
                root_moves = [chess.Move.from_uci(x) for x in human_move_ucis[:search_width]]
            
            if use_ponder == True:
                single_analysis = PONDER_STOCKFISH.analyse(dummy_board, chess.engine.Limit(time=0.05), root_moves=root_moves)
            else:
                single_analysis = STOCKFISH.analyse(dummy_board, chess.engine.Limit(time=0.02), root_moves=root_moves)
            if "pv" in single_analysis:
                response = single_analysis["pv"][0].uci()
                eval_ = extend_mate_score(single_analysis['score'].pov(self.input_info["side"]).score(mate_score=2500))
            else:
                self.log += "ERROR: KeyError pv not in analysis object {}. Returning random response as well as retaining current_eval. \n".format(single_analysis)
                response = root_moves[0].uci() # pick first one
                eval_ = extend_mate_score(self.stockfish_analysis[0]["score"].pov(self.input_info["side"]).score(mate_score=2500))
            
            return_dic[move_uci] = [response, eval_]
        
        if log:
            self.log += "Returning ponder results: {} \n".format(return_dic)
        return return_dic
            
    def _recursive_ponder(self, board: chess.Board, move_uci : str, no_root_moves, depth: int, prev_board: chess.Board = None, limit = None, use_ponder:bool= False):
        """ Recursive function for getting evaluations during pondering. 
        
            Returns [move_uci eval, depth_considered]
            If limit is None, then depth_considered is None. If there are time limit
            constraints, then depth_considered is the depth that move_uci has been considered.
        """
        start = time.time()
        ponder_results = self._ponder_moves(board, [move_uci], no_root_moves, prev_board=prev_board, log=False, use_ponder=use_ponder)
        end = time.time()
        # self.log += "Ponder position fen {} with move {} at depth {} took {} seconds to calculate. \n".format(board.fen(), board.san(chess.Move.from_uci(move_uci)), depth, end-start)
        if limit is not None:
            re_evaluations_left, time_left, depth_considered, total_depth, comparison_eval = limit
            # self.log += "We have {} time left to calculate {} variations. \n".format(time_left - (end-start), re_evaluations_left-1)
            if depth > 1:
                # then we have time constraint
                # we shall adaptively
                new_time_left = time_left - (end-start)
                if new_time_left <= 0.07: # no time left for one more
                    # return result
                    return [ponder_results[move_uci][1], depth_considered]
                
                # otherwise, forecast whether we are on track for finish or not
                new_re_evaluations_left = re_evaluations_left - 1
                # pretend that the rest of the evaluations will take end-start
                forecast_evaluations_left = new_time_left / (end-start)
                new_board = board.copy()
                new_board.push_uci(move_uci)
                consider_move = ponder_results[move_uci][0]
                if consider_move is not None:
                    # we are running out of time, OR if the line does/doesn't seem that promising compared to comparison eval
                    if ponder_results[move_uci][1] > comparison_eval + 100:
                        # proceed as usual, because line is quite promising
                        return self._recursive_ponder(new_board, consider_move, no_root_moves, depth-1, prev_board=board.copy(), limit=[new_re_evaluations_left, new_time_left, depth_considered + 1, total_depth, comparison_eval], use_ponder=use_ponder)
                    elif forecast_evaluations_left < new_re_evaluations_left - 1:
                        # then proceed onto next jump 1 depth less
                        if depth == 2:
                            return [ponder_results[move_uci][1], depth_considered]
                        else:
                            return self._recursive_ponder(new_board, consider_move, no_root_moves, depth-2, prev_board=board.copy(), limit=[new_re_evaluations_left-1, new_time_left, depth_considered+1, total_depth, comparison_eval], use_ponder=use_ponder)
                    elif ponder_results[move_uci][1] < comparison_eval - 250:
                        # variation not promising enough. Stop variation here
                        return [ponder_results[move_uci][1], depth_considered]
                    else:
                        # continue as usual
                        return self._recursive_ponder(new_board, consider_move, no_root_moves, depth-1, prev_board=board.copy(), limit=[new_re_evaluations_left, new_time_left, depth_considered + 1, total_depth, comparison_eval], use_ponder=use_ponder)
                else: # depth <=1 and limit is not None
                    return [ponder_results[move_uci][1], depth_considered]
            else:
                return [ponder_results[move_uci][1], depth_considered]
        else: # not using limits
            if depth > 1:
                new_board = board.copy()
                new_board.push_uci(move_uci)
                consider_move = ponder_results[move_uci][0]
                if consider_move is not None:
                    # if we actually have a valid move
                    # sometimes the game has already ended at this point
                    return self._recursive_ponder(new_board, consider_move, no_root_moves, depth-1, prev_board=board.copy(), use_ponder=use_ponder)
                else:
                    return [ponder_results[move_uci][1], None]
            else:
                return [ponder_results[move_uci][1], None]
            
    
    def _re_evaluate(self, board:chess.Board, re_evaluate_moves: list, no_root_moves: int, depth:int = 0, prev_board:chess.Board = None, limit=None, use_ponder:bool=False):
        """ Given a list of move_ucis, apply them to the current board and re_evaluate
            them using top human_moves only. This gives a non_accurate evaluation
            and simulates human foresight not being exhaustive.
            
            Returns a dictionary with key move_uci and value the evaluation (from our pov)
        """       
        # to avoid bias, scramble the moves
        random.shuffle(re_evaluate_moves)
        return_dic = {}
        if limit is None:            
            for move_uci in re_evaluate_moves:
                eval_, _ = self._recursive_ponder(board, move_uci, no_root_moves, depth, prev_board=prev_board, use_ponder=use_ponder)
                return_dic[move_uci] = eval_
        else:
            comparison_eval = -9999
            evaluations_left, time_left = limit
            time_allowed = time_left
            start = time.time()
            for move_uci in re_evaluate_moves:
                if time_allowed <= 0.07:
                    # that's it, no more time left                    
                    return_dic[move_uci] = [None, 0]
                    continue
                eval_, depth_considered = self._recursive_ponder(board, move_uci, no_root_moves, depth, prev_board=prev_board, limit=[evaluations_left, time_allowed, 1, depth, comparison_eval], use_ponder=use_ponder)
                return_dic[move_uci] = [eval_, depth_considered]
                comparison_eval = max(eval_, comparison_eval)
                evaluations_left -= depth
                time_allowed = time_left - (time.time() - start)
                
        
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
        
        # if more than one historic fen and moves are given, then use this movestack to
        # also account for repetition
        if len(self.input_info["last_moves"]) >= 1:
            test_board = chess.Board(self.input_info["fens"][-len(self.input_info["last_moves"])-1])
            for move_uci in self.input_info["last_moves"]:
                if chess.Move.from_uci(move_uci) in test_board.legal_moves:
                    test_board.push_uci(move_uci)
                else:
                    self.log += "ERROR: Could not sync last moves of input info with fen history at move {}. Defaulting to last known fen and wiping history. \n".format(move_uci)
                    self.current_board = chess.Board(self.input_info["fens"][-1])
                    self.input_info["fens"] = self.input_info["fens"][-1:]
                    self.input_info["last_moves"] = []
                    break
            if test_board.fen() == self.input_info["fens"][-1]:
                self.current_board = test_board.copy()
            elif len(self.input_info["last_moves"]) > 0:
                self.log += "ERROR: Played through last_move list but got fen {} instead of {}. Defaulting to last known fen. \n".format(test_board.fen(), chess.Board(self.input_info["fens"][-1]))
                self.current_board = chess.Board(self.input_info["fens"][-1])
        else:
            self.log += "No last moves given, so resorting to last fen in history with no move_stack. \n"
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
        no_lines = len(list(self.current_board.legal_moves))
        analysis = STOCKFISH.analyse(self.current_board, limit=chess.engine.Limit(time=0.05), multipv=no_lines)
        if isinstance(analysis, dict): # sometimes analysis only gives one result and is not a list.
            analysis = [analysis]
        self.stockfish_analysis = analysis
        end = time.time()
        self.log += "Analysis computed in {} seconds. \n".format(end-start)
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
            cp_diff = self.stockfish_analysis[0]["score"].pov(self.input_info["side"]).score(mate_score=2500) - self.stockfish_analysis[1]["score"].pov(self.input_info["side"]).score(mate_score=2500)
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
        if len(self.input_info["fens"]) >= 2 and self.stockfish_analysis[0]["score"].pov(self.input_info["side"]).mate() == 1:
            last_board = chess.Board(self.input_info["fens"][-2])
            last_analysis = STOCKFISH.analyse(last_board, limit=chess.engine.Limit(time=0.02,mate=1))
            last_mate = last_analysis["score"].pov(self.input_info["side"]).mate()            
            if last_mate == 2:
                last_mate_move_uci = last_analysis["pv"][1].uci()
                if self.stockfish_analysis[0]["pv"][0].uci() == last_mate_move_uci:
                    self.log += "Found obvious move that no matter what gives mate next move: {} \n". format(last_mate_move_uci)
                    return last_mate_move_uci, True
                
        # Otherwise no obvious move
        self.log += "No obvious move found. \n"
        return None, False
    
    def _set_target_time(self, total_time):
        """ Given we are using human approaches to decide the move, we set and initial
            target time which we try to compute our human move. This is supposedly a
            reflection of how hurredly the player is before they've even thought
            about any moves. total_time is the time limit we cannot exceed.
        
            Returns target time, a non-negative float.
        """
        # self_initial_time = self.input_info["self_initial_time"]
        # own_time = max(self.input_info["self_clock_times"][-1],1)
        
        # base_time = max(0.6*QUICKNESS*self_initial_time/(85 + self_initial_time*0.25), 0.1)
        
        # # if we are in hurry mode (i.e. we are in low time), then we adjust our base 
        # # time accordingly
        # mood_sf_dic = {"confident": 1,
        #                 "cocky": 0.6,
        #                 "cautious": 1.6,
        #                 "tilted": 0.4,
        #                 "hurry": (own_time/self_initial_time)**0.7,
        #                 "flagging": 0.8}
        
        # # we need to leverage information from lucas analytics
        # activity = self.lucas_analytics["activity"]
        # activity_sf = ((activity+12)/25)**0.4
        
        # target_time = base_time * activity_sf * mood_sf_dic[self.mood]
        # while target_time > total_time*0.9: # cannot exceed total time
        #     target_time /= 1.2
        
        # # likewise if the human consider time is too low
        # while target_time < total_time*0.5:
        #     target_time *= 1.2
        
        # Base entirely on max achievable time for greatest human calculation depth
        if total_time <= 1:
            # then we would never have enough time to ponder anyway, so human evaluation takes up majority.
            target_time = total_time*0.8
        elif total_time <= 2.5:
            # grey area where we have a bit of time but not too much time to ponder.
            ratio = 0.6*(total_time-1) + 0.8*(2-total_time)
            target_time = ratio*total_time
        else:
            # maximum amount of human time capped, use rest for ponder
            target_time = 2.5*0.6
            
        self.log += "Decided target time for human evaluation to be: {} \n".format(target_time)
        return max(target_time,0.1)
    
    def _get_time_taken(self, obvious:bool=False, human_filters:bool=True):
        """ Calculates the amount of time in total we should spend on a move.
            obvious is whether we made a quick obvious move.  
            human_filters is whether we used human filters
        
            Returns time_taken : float
        """
        self.log += "Deciding time taken to make the move from receiving input. \n"
        self_initial_time = self.input_info["self_initial_time"]
        base_time = max(QUICKNESS*self_initial_time/(85 + self_initial_time*0.25),0.1)
        self.log += "Initial base time without calculations: {} \n".format(base_time)
        # we move faster depending on whether we are proportionally behind on time
        # or move slower if we are ahead
        own_time = max(self.input_info["self_clock_times"][-1],1)
        opp_time = max(self.input_info["opp_clock_times"][-1],1)
        base_time *= (own_time/opp_time)**(10/self_initial_time)
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
                base_time *= 0.4
            elif game_phase == "midgame":
                base_time *= 1.2
            else:
                base_time *=0.6
            
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
                if np.random.random() < 0.4: # then we have a quick low variation move
                    time_taken = base_time * (1.0+ np.clip(0.1*np.random.randn(), -0.3, 0.3))
                elif np.random.random() < 0.7: # medium low variation move
                    time_taken = base_time * (1.3+ np.clip(0.2*np.random.randn(), -0.4, 0.7))
                else:
                    # slightly longer think, larger variation
                    time_taken = base_time * (3.5 + np.clip(0.7*np.random.randn(), -1.7, 2.0))
            elif self.mood == "cocky":
                # medium variation, quick move times
                if np.random.random() < 0.9: # then we have a quick low variation move
                    time_taken = base_time * (0.9 + np.clip(0.3*np.random.randn(), -0.2, 0.7))
                else:
                    # slightly longer think
                    time_taken = base_time * (3.3 + np.clip(0.4*np.random.randn(), -0.8, 0.8))
            elif self.mood == "cautious":
                # medium variation, slow moves
                if np.random.random() < 0.6: # then we have a quick low variation move
                    time_taken = base_time * (1.3+ np.clip(0.2*np.random.randn(), -0.3, 0.5))
                elif np.random.random() < 0.6: # medium low variation move
                    time_taken = base_time * (2.1+ np.clip(0.25*np.random.randn(), -0.4, 0.7))
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
                elif np.random.random() < 0.7:
                    time_taken = base_time * (1.1 + np.clip(0.1*np.random.randn(), -0.3, 0.3))
                else:
                    # slightly longer think
                    time_taken = base_time * (1.7 + np.clip(0.2*np.random.randn(), -0.4, 0.6))
                
                # if we are in hurry mode (i.e. we are in low time), then our time taken
                # depends on how much time we have left
                time_taken *= (3*own_time/self_initial_time)**0.9
            elif self.mood == "flagging":
                # large variation, quick move times
                if np.random.random() < 0.5:
                    time_taken = base_time * (1.2 + np.clip(0.2*np.random.randn(), -0.3, 0.6))
                elif np.random.random() < 0.7:
                    time_taken = base_time * (1.6 + np.clip(0.3*np.random.randn(), -0.5, 0.8))
                else:
                    # slightly longer think
                    time_taken = base_time * (3.1 + np.clip(0.4*np.random.randn(), -0.8, 1.0))
            
            self.log += "Decided time taken after mood analysis: {} \n".format(time_taken)
        
        # we also use reflective moving, which is to play faster when opponent has been playing
        # fast over the last few moves (as we feel the pressure to keep up with the pace)
        # and slower when opponents are moving slower
        if len(self.input_info["opp_clock_times"]) >= 4:
            recent_time = (self.input_info["opp_clock_times"][-4] - self.input_info["opp_clock_times"][-1])/3 # opponents average move time in last 3 moves
            non_recent_time = (self.input_info["opp_clock_times"][0] - self.input_info["opp_clock_times"][-1])/(len(self.input_info["opp_clock_times"])-1)
            target_time_spent = 0.2*non_recent_time + 0.8*recent_time
            
            # sometimes we don't use this
            if np.random.random() < 0.8:
                time_taken = 0.3*target_time_spent + time_taken * 0.7
                self.log += "Decided time taken after opponent speed consideration: {} \n".format(time_taken)
        
        time_taken = max(time_taken, 0.1)
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
        own_time = max(self.input_info["self_clock_times"][-1],1)
        opp_time = max(self.input_info["opp_clock_times"][-1],1)
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
        current_eval = extend_mate_score(self.stockfish_analysis[0]["score"].pov(self.input_info["side"]).score(mate_score=2500))
        if len(self.input_info["fens"]) >= 3:
            self.log += "Checking to see if we have made a blunder recently. \n"            
            last_avail_board = chess.Board(self.input_info["fens"][0])
            last_eval = extend_mate_score(STOCKFISH.analyse(last_avail_board, limit=chess.engine.Limit(time=0.02))["score"].pov(self.input_info["side"]).score(mate_score=2500))
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
        
        # If position is relatively even, and there are exactly a few good moves
        # (not 1 good move but between 2-4)
        complexity = self.lucas_analytics["complexity"]
        eff_mob = self.lucas_analytics["eff_mob"]
        if abs(current_eval) < 250 and eff_mob < 15 and eff_mob > 2:
            # top two moves cannot be takebacks. These cases are not considered complicated
            if len(self.input_info["last_moves"]) >= 1 and len(self.input_info["fens"]) >= 2:
                prev_move_uci = self.input_info["last_moves"][-1]
                top_move_uci = self.stockfish_analysis[0]["pv"][0].uci()
                second_move_uci = self.stockfish_analysis[1]["pv"][0].uci()
                prev_board = chess.Board(self.input_info["fens"][-2])
                if is_takeback(prev_board, prev_move_uci, top_move_uci) and is_takeback(prev_board, prev_move_uci, second_move_uci):
                    # not cautious
                    self.log += "Top two engine moves are both takebacks. Cannot be cautious. \n"
                    return "confident"
            if np.random.random() < (0.35 + complexity/(100*eff_mob + 100))**0.6:
                return "cautious"
            else:
                self.log += "Postion is close to even (current eval {}) with complexity {} and eff_mob {}. But by chance not cautious. \n".format(current_eval, complexity, eff_mob)
        else:
            self.log += "Position not even enough (current eval {}) or did not satisify eff_mob conditions (eff_mob {}) . Not in cautious mode. \n".format(current_eval, eff_mob)
        
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
        analysis = STOCKFISH.analyse(board, limit=chess.engine.Limit(time=0.02))
        opp_best_move_obj = analysis["pv"][0]
        dummy_board = board.copy()
        dummy_board.push(opp_best_move_obj)
        candidate_premove = None
        # if we are in the opening, then check whether we could respond with opening database move
        game_phase = phase_of_game(dummy_board)
        if game_phase == "opening":
            result = list(self.opening_book.find_all(dummy_board))
            if len(result) != 0:
                self.log += "Found matching position in opening database during premove search. Using it to find premove in opening.\n"
                excluded_moves = [res.move for res in result[5:]]
                # Now get weighted choice of move to play
                played_move_obj = self.opening_book.weighted_choice(dummy_board, exclude_moves=excluded_moves).move
                self.log += "Chosen premove from opening book: {} \n".format(dummy_board.san(played_move_obj))
                # we need to check whether this is a safe premove or not
                if check_safe_premove(board, played_move_obj.uci()) == True:
                    self.log += "Double checked that premove is safe. \n"
                    candidate_premove = played_move_obj.uci()
                else:
                    self.log += "Opening book premove is not a safe premove. Resorting to stockfish premove. \n"                
                
            else:
                self.log += "Even though opening phase, did not find matching position in opening database. Resorting to stockfish premove. \n"
        if candidate_premove is None: # didn't find opening book premove
            next_analysis = STOCKFISH.analyse(dummy_board, limit=chess.engine.Limit(time=0.02), multipv=10)
            if isinstance(next_analysis, dict):
                next_analysis = [next_analysis]
            # now use get_stockfish move on this position
            # Of course, we can only get a stockfish move if the game is not over
            if dummy_board.outcome() is None:
                candidate_premove = self.get_stockfish_move(board=dummy_board, analysis=next_analysis, last_move_uci=opp_best_move_obj.uci(), log=False)
                self.log += "Detected premove from stockfish evals: {} \n".format(candidate_premove)
            else:
                # game is over
                self.log += "Cannot get premove for position {} as it is game over. \n".format(dummy_board.fen())
                return None
        
        # Now that we have found our premove, we have to do some extra checks to make sure it is a human
        # premove. for example, a human wouldn't premove a piece into a square which would be enpris
        dummy_board_2 = board.copy()
        dummy_board_2.turn = side # pretend it is our turn in this situation
        move_obj = chess.Move.from_uci(candidate_premove)
        piece_at = dummy_board_2.piece_type_at(move_obj.to_square)
        colour_at = dummy_board_2.color_at(move_obj.to_square)
        if colour_at == side: # if we are just premoving a takeback
            return candidate_premove
        elif piece_at is None:
            piece_val_at = 0
        else:
            piece_val_at = PIECE_VALS[piece_at]
        dummy_board_2.push(move_obj)
        if calculate_threatened_levels(move_obj.to_square, dummy_board_2) - piece_val_at > 0.6:
            # premove moves to be enpris
            self.log += "Premove {} moves to an enpris square, filtering the premove out. \n".format(dummy_board.san(move_obj))
            premove_uci = None
        else:
            # if we are in the opening, then check the premove is safe
            if game_phase == "opening":
                if check_safe_premove(board, candidate_premove) == True:
                    premove_uci = candidate_premove
                else:
                    self.log += "Discovered game phase is opening, and premove {} is not considered a safe premove. Filtering out. \n".format(candidate_premove)
                    premove_uci = None
            else:
                premove_uci = candidate_premove
        return premove_uci
    
    def stockfish_ponder(self, board:chess.Board, time_allowed : float, ponder_width:int, use_ponder:bool = False, root_moves:list= None):
        """ Given a board position that is not our side turn, we ponder moves using
            stockfish and return a dictionary with has key board_fen and value uci.
            This method is much faster than self.ponder()
        
        """
        self.log += "Stockfish ponder position with fen {} \n".format(board.fen())
        self.log += "Stockfish pondering time allowed: {} \n".format(time_allowed)
        
        # First double check our position has not ended. If it has no moves, return None
        if board.outcome() is not None:
            return None
        
        if root_moves is None:
            root_moves = list(board.legal_moves)
            
        if use_ponder:
            analysis_object = PONDER_STOCKFISH.analyse(board, limit=chess.engine.Limit(time=time_allowed), multipv=ponder_width, root_moves=root_moves)
        else:
            analysis_object = STOCKFISH.analyse(board, limit=chess.engine.Limit(time=time_allowed), multipv=ponder_width, root_moves=root_moves)
        if isinstance(analysis_object, dict):
            analysis_object = [analysis_object]
        
        return_dic = {}
        for line in analysis_object:
            if "pv" in line:
                if len(line["pv"]) >=2:
                    dummy_board = board.copy()
                    opp_move = line["pv"][0]
                    dummy_board.push(opp_move)
                    board_fen = dummy_board.board_fen()
                    response_move_uci = line["pv"][1].uci()
                    return_dic[board_fen] = response_move_uci
                
        if len(return_dic) == 0:
            return None
        
        self.log += "Stockfish ponder moves returned ponder dic: {} \n".format(return_dic)
        return return_dic
    
    def ponder(self, board: chess.Board, time_allowed : float, search_width : int, time_per_position : float = 0.1, prev_board:chess.Board = None, ponder_width: int = None, use_ponder:bool=False):
        """ Given a board position that is not our (side) turn, we ponder possible moves
            and return a dictionary which has key board_fen (so position only), and value uci in response.
            Time allowed represents how much we may ponder. Time per position is roughly 
            the amount of time to form probabilies and alter them.
            
            Returns a dictionary with key board_fen and value move_uci response
        """
        self.log += "Ponder position with fen {} \n".format(board.fen())
        self.log += "Pondering time allowed: {} \n".format(time_allowed)
        variations_allowed = max(1, int(time_allowed/time_per_position))
        
        # First double check our position has not ended. If it has no moves, return None
        if board.outcome() is not None:
            return None
        
        if ponder_width is None:
            # As a maximum number of ponder moves (to prevent too quick responses),
            # depends on the time control.
            initial_time = self.input_info["self_initial_time"]
            if initial_time <= 180:
                max_ponder_no = 2
            else:
                max_ponder_no = 3
        
            # decide ponder depth and width
            ponder_width = 1 # min ponder width        
            for i in range(max_ponder_no):
                ponder_depth = round(variations_allowed / ((max_ponder_no-i) * search_width))
                if ponder_depth >= 2:
                    ponder_width = max_ponder_no-i
                    break
        else:
            # ponder width has been preset
            ponder_depth = round(variations_allowed / (ponder_width * search_width))
        if use_ponder:
            analysis_object = PONDER_STOCKFISH.analyse(board, limit=chess.engine.Limit(time=0.05), multipv=ponder_width)
        else:
            analysis_object = STOCKFISH.analyse(board, limit=chess.engine.Limit(time=0.02), multipv=ponder_width)
        if isinstance(analysis_object, dict):
            analysis_object = [analysis_object]
        
        if len(analysis_object) == 0:
            self.log += "ERROR: Couldn't fetch stockfish analysis object, returning None. \n"
            return None
        elif "pv" not in analysis_object[0]:
            self.log += "ERROR: pv KeyError in stockfish analysis object: {} \n".format(analysis_object)
            return None
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
                
            re_evaluate_dic = self._re_evaluate(dummy_board, top_human_moves, search_width, depth=ponder_depth, prev_board = board.copy(), limit=[ponder_depth*search_width, time_allowed/2], use_ponder=use_ponder)
            # adding noise
            for move_uci in re_evaluate_dic.keys():
                eval_, depth_considered = re_evaluate_dic[move_uci]
                if eval_ is  None:
                    # move never got considered
                    # Get stockfish evaluation of move, but penalise heavily
                    if use_ponder:
                        an_obj = PONDER_STOCKFISH.analyse(dummy_board, limit=chess.engine.Limit(time=0.05), root_moves=[chess.Move.from_uci(move_uci)])
                        if "score" in an_obj:
                            new_eval = extend_mate_score(an_obj["score"].pov(self.input_info["side"]).score(mate_score=2500))
                        else:
                            # something went wrong
                            self.log += "Something went wrong with analysis object with use_ponder: {}. Returning no ponder dic. \n".format(an_obj)
                            return None
                    else:
                        an_obj = STOCKFISH.analyse(dummy_board, limit=chess.engine.Limit(time=0.01), root_moves=[chess.Move.from_uci(move_uci)])
                        if "score" in an_obj:
                            new_eval = extend_mate_score(an_obj["score"].pov(self.input_info["side"]).score(mate_score=2500))
                        else:
                            # something went wrong
                            self.log += "Something went wrong with analysis object: {}. Returning no ponder dic. \n".format(an_obj)
                            return None
                    eval_ = new_eval - 100
                    re_evaluate_dic[move_uci][0] = new_eval - 100 # penalty
                base_noise_sd = 40*(np.tanh(eval_/(self.playing_level*50)))**2 + 20                             
                noise_sd = 4*base_noise_sd/(time_allowed*(depth_considered+4))
                noise = np.random.randn()*noise_sd                
                re_evaluate_dic[move_uci][0] += noise                
                # encourage capture moves, depending on how enpris the piece is
                move_obj = chess.Move.from_uci(move_uci)
                if dummy_board.is_capture(move_obj):
                    capture_bonus = 40* calculate_threatened_levels(move_obj.to_square, dummy_board)
                    re_evaluate_dic[move_uci][0] += capture_bonus
                    
            best_response = max(re_evaluate_dic.keys(), key=lambda x : re_evaluate_dic[x][0])            
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
        curr_eval = extend_mate_score(self.stockfish_analysis[0]["score"].pov(self.input_info["side"]).score(mate_score=2500))
        prev_board = chess.Board(self.input_info["fens"][-2])
        prev_analysis = STOCKFISH.analyse(prev_board, limit=chess.engine.Limit(time=0.02))
        prev_eval = extend_mate_score(prev_analysis["score"].pov(self.input_info["side"]).score(mate_score=2500))
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
        obvious_move_start = time.time()
        obvious_move, obvious_move_found = self.check_obvious_move()
        obvious_move_end= time.time()
        self.log += "Obvious move check performed in {} seconds. \n".format(obvious_move_end-obvious_move_start)
        if log == True:
            self._write_log()
        if obvious_move_found == True:
            return_dic["move_made"] = obvious_move
            use_human_filters = False
            # Decide how much time we are going to spend (including thinking time)
            return_dic["time_take"] = self._get_time_taken(obvious=obvious_move_found, human_filters=use_human_filters)
        else:
            # Now need to decide base on information whether we are using strictly engine move
            # Or using human filters
            time_taken_start = time.time()
            self.log += "Deciding whether to use human filters. \n"
            use_human_filters = self._decide_human_filters()
            # Decide how much time we are going to spend (including thinking time)
            return_dic["time_take"] = self._get_time_taken(obvious=obvious_move_found, human_filters=use_human_filters)
            time_taken_end = time.time()
            self.log += "Get time taken function took {} seconds to evaluate. \n".format(time_taken_end - time_taken_start)
            if use_human_filters == True:
                human_start = time.time()
                self.log += "Using human filters. \n"
                # first let us decide how much time we shall try spend on the move
                # key word "try"
                total_time = return_dic["time_take"] # can't be more than decided max time
                target_time = self._set_target_time(total_time)
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
        opp_blunder_check_start = time.time()
        self.log += "Checking for opponent blunders. \n"
        self._check_opp_blunder()
        opp_blunder_check_end = time.time()
        self.log += "Opponent blunder check took {} seconds. \n".format(opp_blunder_check_end- opp_blunder_check_start)
        if log == True:
            self._write_log()
        
        # If we are in a hurry, and our time is absolutely low then we also return 
        # a premove for the next move.
        # If we are not in a hurry, look for takeback premoves
        # Sometimes if the time control is bullet, and we are in the opening, we
        # may also premove with some probability
        own_time = max(self.input_info["self_clock_times"][-1],1)
        after_board = self.current_board.copy()
        after_board.push_uci(return_dic["move_made"])
        if after_board.outcome() is None:
            self_initial_time = self.input_info["self_initial_time"]
            premove_start = time.time()
            if self.mood == "hurry" and own_time < 20:
                # with some probability we return a premove
                if np.random.random() < 0.3*self_initial_time/(own_time + 0.3*self_initial_time):
                    return_dic["premove"] = self.get_premove(after_board, self.input_info["side"])
                else:
                    # look for takeback premoves only
                    return_dic["premove"] = self.get_premove(after_board, self.input_info["side"], takeback_only=True)
            elif self_initial_time <= 60 and phase_of_game(self.current_board) == "opening":
                # with some probability return a forced premove, otherwise just look for takeback premoves
                if np.random.random() < 0.9:
                    return_dic["premove"] = self.get_premove(after_board, self.input_info["side"])
                else:
                    # look for takeback premoves only
                    return_dic["premove"] = self.get_premove(after_board, self.input_info["side"], takeback_only=True)
            else:
                # look for takeback premoves only
                return_dic["premove"] = self.get_premove(after_board, self.input_info["side"], takeback_only=True)
            premove_end = time.time()
            self.log += "Premove search took {} seconds. \n".format(premove_end-premove_start)
        else:
            return_dic["premove"] = None
        
        self.log += "Gotten premove: {} \n".format(return_dic["premove"])
        if log == True:
            self._write_log()
        
        move_end = time.time()
        if log == True:
            self._write_log()
        # If we have extra time than that of alloted then we may do some pondering for the position after our move
        time_spent = move_end - move_start
        
        time_per_position = 0.1 # max time spent per ponder position
        search_width = self._decide_breadth() # this is slightly incorrect, but close enough
        if return_dic["time_take"] - time_spent > time_per_position*search_width * 1.15:
            self.log += "Have enough time to ponder for he next position. Time taken so far: {} \n".format(time_spent)
            ponder_start = time.time()
            ponder_dic = self.ponder(after_board, (return_dic["time_take"] - time_spent)/1.15, search_width, time_per_position=time_per_position, prev_board=self.current_board.copy())
            ponder_end = time.time()
            self.log += "Took {} seconds for pondering. \n".format(ponder_end - ponder_start)
        else:
            self.log += "Do not have enough time to ponder for he next position. Time taken so far: {} \n".format(time_spent)
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
    engine = Engine(playing_level=5)
    # b = chess.Board("3r2k1/3r1p1p/PQ2p1p1/8/5q2/2P2N2/1P3PP1/R3K2R w KQ - 1 24")
    input_dic ={'fens': ['2rq1k1r/1b1n1ppp/p1n1p3/1pb1P1N1/8/2NB2QP/PPP3P1/R1B2R1K b - - 6 15', '2rq1k1r/1b3ppp/p1n1p3/1pb1n1N1/8/2NB2QP/PPP3P1/R1B2R1K w - - 0 16', '2rq1k1r/1b3ppp/p1n1N3/1pb1n3/8/2NB2QP/PPP3P1/R1B2R1K b - - 0 16', '2rq3r/1b2kppp/p1n1N3/1pb1n3/8/2NB2QP/PPP3P1/R1B2R1K w - - 1 17', '2rq3r/1b2kppp/p1n5/1pN1n3/8/2NB2QP/PPP3P1/R1B2R1K b - - 0 17', '2r4r/1b2kppp/p1nq4/1pN1n3/8/2NB2QP/PPP3P1/R1B2R1K w - - 1 18', '2r4r/1N2kppp/p1nq4/1p2n3/8/2NB2QP/PPP3P1/R1B2R1K b - - 0 18', '2r4r/1N2kppp/p1n5/1p2n3/3q4/2NB2QP/PPP3P1/R1B2R1K w - - 1 19'], 'self_clock_times': [45, 45, 43, 41, 38, 36, 35, 32], 'opp_clock_times': [52, 51, 45, 36, 35, 33, 29, 25], 'last_moves': ['d7e5', 'g5e6', 'f8e7', 'e6c5', 'd8d6', 'c5b7', 'd6d4'], 'side': True, 'self_initial_time': 60, 'opp_initial_time': 60}
    start = time.time()
    engine.update_info(input_dic)
    print(engine.make_move(log=False))
    end = time.time()
    print("finished in {} seconds".format(end-start))
