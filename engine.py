# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:29:08 2024

@author: xusem
"""
import chess
import datetime
import os
import time
import numpy as np
import random

from models.models import MoveScorer, StockFishSelector
from constants import MOVE_FROM_WEIGHTS_PTH, MOVE_TO_WEIGHTS_MID_PTH, MOVE_TO_WEIGHTS_END_PTH, MOVE_TO_WEIGHTS_OP_PTH

from board_information import (
    phase_of_game, PIECE_VALS, STOCKFISH, get_lucas_analytics, is_capturing_move, is_capturable,
    is_attacked_by_pinned, is_check_move, is_takeback, is_newly_attacked, get_threatened_board,
    is_offer_exchange, king_danger, is_open_file, calculate_threatened_levels
            )
from utils import flip_uci, get_move_made

class Engine:
    """ Class for engine instance.
    
        The Engine is responsible for the following things ONLY
        receiving board information -> outputting move and premoves
        
        All other history related data to do with past moves etc are not handled
        in the Engine instance. They are handled in the client wrapper
    """
    def __init__(self, playing_level:int = 6, log_file: str = os.path.join(os.getcwd(), 'Engine_Logs',str(datetime.datetime.now()).replace(" ", "").replace(":","_") + '.txt')):
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
            "opening": MoveScorer(MOVE_FROM_WEIGHTS_PTH, MOVE_TO_WEIGHTS_OP_PTH),
            "midgame": MoveScorer(MOVE_FROM_WEIGHTS_PTH, MOVE_TO_WEIGHTS_MID_PTH),
            "endgame": MoveScorer(MOVE_FROM_WEIGHTS_PTH, MOVE_TO_WEIGHTS_END_PTH),
            }
        self.stockfish_scorer = StockFishSelector('Engines/stockfish_14_x64.exe')
        self.stockfish_analysis = None
        
        # lucas statistics for the current position
        self.lucas_analytics = {
            "complexity": None,
            "win_prob": None,
            "eff_mob": None,
            "narrowness": None,
            "activity": None,
            }
        
        self.playing_level = playing_level
        self.mood = None
        
    def _write_log(self):
        """ Writes down thinking into a log file for debugging. """
        with open(self.log_file,'a') as log:
            log.write(self.log)
            log.close()
        self.log = ""
        
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
            if np.random() < (15-own_time)/16:
                return False
            else:
                return True
        return True
    
    def get_stockfish_move(self):
        """ Uses board information to get a move strictly from stockfish with no human
            filters. Very fast, and only called for in necessary situations (when in
            super low time)
            
            Returns a move_uci string of move made
        """
        # TODO
        # take a random sample from the moves given by our stockfish analysis object
        # grade each move by their evaluation, and if previous moves are given (our
        # own moves) then grade them also based on proximity and distance moved by mouse
        total_moves = len(self.stockfish_analysis)
        sample_n = max(int(total_moves*0.6),1)
        self.log += "Choosing to sample {} moves from stockfish analysis object. \n".format(sample_n)
        sampled_moves = random.sample(self.stockfish_analysis, sample_n)
        move_eval_dic = {entry["pv"][0].uci(): entry["score"].pov(self.current_board.turn).score(mate_score=2500) for entry in sampled_moves}
        self.log += "Sampled the following moves and their corresponding evals: {} \n".format(move_eval_dic)
        move_distance_dic = {}
        for move_uci in move_eval_dic.keys():
            distance = 0
            move_obj = chess.Move.from_uci(move_uci)
            # if we are given information about our own previous move, then include that distance too
            if len(self.input_info["last_moves"]) >= 2:
                own_last_move_obj = chess.Move.from_uci(self.input_info["last_moves"][-2])
                to_square = own_last_move_obj.to_square
                distance += chess.square_distance(to_square, move_obj.from_square)
            distance += chess.square_distance(move_obj.from_square, move_obj.to_square)
            move_distance_dic[move_uci] = distance
        
        self.log += "Evaluated the moves square distance to move: {} \n".format(move_distance_dic)
        own_time = self.input_info["self_clock_times"][-1]
        move_appealing_dic = {move_uci : 10 + move_eval_dic[move_uci]*own_time/2000 - move_distance_dic[move_uci]}
        self.log += "Combining both the dictionary preferences, we have their move preferences: {} \n".format(move_appealing_dic)
        # Add noise to introduce randomness. The lower our time, the more the noise
        # note own time has to be less than 15 seconds for valid calculation.
        noise_level = (15-own_time)/15
        move_appealing_dic = {move_uci: move_appealing_dic[move_uci] + noise_level*np.random.randn() for move_uci in move_appealing_dic.keys()}
        self.log += "Appealingness after adding noise: {} \n".format(move_appealing_dic)
        
        move_chosen = max(move_appealing_dic.keys(), key=lambda x: move_appealing_dic[x])
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
            dummy_board = board.mirror().copy()
        _, nn_top_move_dic = scorer.get_move_dic(dummy_board, san=False, top=100)
        
        # if we were black, we need to convert all the ucis to be flipped
        if board.turn == chess.BLACK:
            nn_top_move_dic = {flip_uci(k): v for k,v in nn_top_move_dic.items()}
        if log:
            log_move_dic = {board.san(chess.Move.from_uci(k)) : round(v, 5) for k,v in nn_top_move_dic.items()}
            self.log += "Move_dic before alteration: {} \n".format(log_move_dic)
        return nn_top_move_dic
    
    def _alter_move_probabilties(self, move_dic : dict, board:chess.Board = None, prev_board:chess.Board = None, log:bool = True):
        """ Given a move dictionary with move uci as key and value as their unaltered
            probabilities, we alter the probabilties to make moves stick out more
            (for example hanging pieces more likely to be moved etc).
            
            Returns an altered move_dic. 
        """
        start = time.time()
        # TODO
        if board is None: # if there is no board, assume we are working with the current one
            board = self.current_board.copy()
        if prev_board is None and len(self.input_info["fens"]) >= 2:
            prev_board = chess.Board(self.input_info["fens"][-2])
        
        # Captures are just generally more appealing. The bigger the capture the more appealing
        capture_sf = 2.0
        # Capturable pieces are more appealling to move
        capturable_sf = 1.3
        # Squares that pinned pieces attack that break the pin are more desirable
        break_pin_sf = 2.2
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
        # Moves which protect/block/any way that make our pieces less en pris are appealing
        # Factor is given by factor = np.exp(-lvl_diff/10)
        
        # Moves which make opponent pieces more en pris (attacks/threatens) are appealing
        # Factor is given by factor = np.exp(lvl_diff/10)
        
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
        
###############################################################################
        
        # Capturing moves are more appealling
        capturing_moves = []
        for move_uci in move_dic.keys():
            if is_capturing_move(board, move_uci):
                # it is a capturing move
                piece_value = PIECE_VALS[board.piece_type_at(chess.Move.from_uci(move_uci).to_square)]
                move_dic[move_uci] *= capture_sf * (piece_value**0.25)
                capturing_moves.append(board.san(chess.Move.from_uci(move_uci)))
        if log:
            self.log += "Found capturing moves from position: {} \n".format(capturing_moves)
        
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
        
        # Squares that pinned pieces attack that break the pin are more desirable to move to
        adv_pinned_moves = []
        for move_uci in move_dic.keys():
            to_square = chess.Move.from_uci(move_uci).to_square
            no_pinned_atks = is_attacked_by_pinned(board, to_square, not board.turn)
            if no_pinned_atks > 0:
                # it is a capturing move
                move_dic[move_uci] *= (break_pin_sf**no_pinned_atks)
                adv_pinned_moves.append(board.san(chess.Move.from_uci(move_uci)))
        if log:
            self.log += "Found moves that take advantage of pinned pieces: {} \n".format(adv_pinned_moves)
        
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
                if is_takeback(last_move_uci, move_uci):
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
            lvl_diff = opp_lvl_diff - self_lvl_diff*1.5
            # if our move was a capture, we must take that into account that we gained some material
            piece_type = board.piece_type_at(move_obj.to_square)
            if piece_type is not None: # if we captured something
                lvl_diff += PIECE_VALS[piece_type]            
            factor = np.exp(lvl_diff/2)            
            move_dic[move_uci] *= factor
            if lvl_diff > 0.9:
                strenghening_moves.append(board.san(chess.Move.from_uci(move_uci)))
            elif lvl_diff < -0.9:
                weakening_moves.append(board.san(chess.Move.from_uci(move_uci)))
        if log:
            self.log += "Found moves that are weakening and make our pieces more enpris/opp pieces less enpris: {} \n".format(weakening_moves)
            self.log += "Found moves that protect our pieces more or apply more pressure to opponent: {} \n".format(strenghening_moves)
        
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
        # TODO
        return self.playing_level
    
    def get_human_move(self, target_time:float = 0.5):
        """ Uses board information to get move based on human filters from machine
            learning model, and probabilities outputted by the model.
            
            Return move_uci of move made
        """
        
        game_phase = phase_of_game(self.current_board)
        self.log += "Evaluated current game phase: {} \n".format(game_phase)
        
        # Now get the human moves from the position and their probabilities
        un_altered_move_dic = self.get_human_probabilities(self.current_board, game_phase)
        
        # Now get altered_probabilities
        altered_move_dic = self._alter_move_probabilties(un_altered_move_dic)
        
        # Now decide how many of these top moves we shall consider for calculation
        no_root_moves = self._decide_breadth()
        self.log += "Decided search breath for current position: {}. \n".format(no_root_moves)
        # We now piece together the top moves from human search and our stockfish_analysis
        human_move_ucis = list(altered_move_dic.keys())
        root_moves = human_move_ucis[:no_root_moves]
        self.log += "Decided root human moves are: {} \n".format(root_moves)
        
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
        self.log += "Computed human move evals: {} \n".format(human_move_evals)
        
        # Now simply selecting the best of these human evals is natural, however
        # a better way would be to cloud the judgement of the evaluations
        # by computing the again in a human manner. This takes time (roughly 0.12 secs
        # per move) so we can't do this for every move. The number of moves we perform
        # this re-evaluation will depend on our target time, the time we set at
        # the beginning of the move to try and make our move by.
        re_evaluations = int(max(target_time//0.19 - 1, 0))
        self.log += "Plan to re-evaluate {} of the top human variations to cloud judgement. \n".format(re_evaluations)
        top_human_moves = sorted(human_move_evals.keys(), reverse=True, key= lambda x: human_move_evals[x])
        re_evaluate_moves = top_human_moves[:re_evaluations]
        self.log += "Re-evaluating moves: {} \n".format(re_evaluate_moves)
        re_evaluations_dic = self._re_evaluate(re_evaluate_moves, no_root_moves)
        self.log += "Re-evaluated evals are: {} \n".format(re_evaluations_dic)
        human_move_evals.update(re_evaluations_dic)
        self.log += "Updated human move evaluations are: {} \n".format(human_move_evals)
        
        # To further randomise and avoid repetitional play, we cloud the evaluations further by some Gaussian noise
        for move_uci in human_move_evals.keys():
            noise = (1+np.abs(human_move_evals[move_uci])**0.2)*np.random.randn()*100/np.sqrt(6)
            human_move_evals[move_uci] += noise
        
        self.log += "Updated human move evaluations after noise are: {} \n".format(human_move_evals)
        top_move = max(human_move_evals.keys(), key= lambda x: human_move_evals[x])
        self.log += "Decided output move from human move function: {} \n".format(self.current_board.san(chess.Move.from_uci(top_move)))
        return top_move
    
    def _ponder_moves(self, board:chess.Board, move_ucis: list, search_width:int):
        """ We ponder on the given board position, and consider the moves given by the list
            of move_ucis. We again use human probabilities to narrow our search width.
            
            Returns a dictionary with:
                key: the move uci from move_ucis
                value: [move uci of response, eval of response]
            eval of response is given from the perspective of ourselves.
        """
        self.log += "Pondering the moves {} for the fen {} \n".format(move_ucis, board.fen())
        return_dic = {}
        for move_uci in move_ucis:
            dummy_board = board.copy()
            move_obj = chess.Move.from_uci(move_uci)
            dummy_board.push(move_obj)
            
            # Now get human probabilities of this new position
            game_phase = phase_of_game(dummy_board)
            un_altered_move_dic = self.get_human_probabilities(dummy_board, game_phase, log=False)
            altered_move_dic = self._alter_move_probabilties(un_altered_move_dic, board=dummy_board, prev_board=board, log=False)
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
        
        self.log += "Returning ponder results: {} \n".format(return_dic)
        return return_dic
            
    
    def _re_evaluate(self, re_evaluate_moves: list, no_root_moves: int):
        """ Given a list of move_ucis, apply them to the current board and re_evaluate
            them using top human_moves only. This gives a non_accurate evaluation
            and simulates human foresight not being exhaustive.
            
            Returns a dictionary with key move_uci and value the evaluation (from our pov)
        """
        # To compensate for the fact we do not recalculate search width for pondering,
        # we increase the number of root moves searched by scale factor
        ponder_results = self._ponder_moves(self.current_board, re_evaluate_moves, int(no_root_moves*1.2))
        
        return_dic = {}
        for move_uci in re_evaluate_moves:
            return_dic[move_uci] = ponder_results[move_uci][1]
        
        return return_dic
            
    
    def update_info(self, info_dic : dict):
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
        
        # Now determine our player "mood" and set it as our mode for the rest of the calculations
        # The reason we execute this function here is because we want our mood
        # to solely depend on human heuristics rather than engine statistics (like evals)
        self._set_mood()
    
    def check_obvious_move(self):
        """ Given input information, check whether there is an obvious move in the
            position that we may play immediately.
            
            Returns [obvious_move: uci_str, obvious_move_found : bool]
            In the case that no obvious is found, obvious_move is None"""
        # TODO
        return None, False
    
    def _set_target_time(self):
        """ Given we are using human approaches to decide the move, we set and initial
            target time which we try to compute our human move. This is supposedly a
            reflection of how hurredly the player is before they've even thought
            about any moves. 
        
            Returns target time, a non-negative float.
        """
        # TODO
        return 1
    
    def _get_time_taken(self, obvious:bool=False, human_filters:bool=True):
        """ Calculates the amount of time in total we should spend on a move.
            obvious is whether we made a quick obvious move.  
            human_filters is whether we used human filters
        
            Returns time_taken : float
        """
        # TODO
        return np.random.random()+0.5
    
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
        
        # TODO
        self.log += "Setting mood from given input information. \n"
        self.mood = "confident"
        
        self.log += "Mood set to: {}. \n".format(self.mood)
    
    def make_move(self):
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
        self.log += "Make move function called. \n"
        self.log += "Evaluating from the board (capital letters are white pieces): \n"
        self.log += str(self.current_board) + "\n"
        return_dic = {}
        # Performing a quick initial analysis of the position
        self.log += "Performing initial quick analysis perhaps used later by stockfish. \n"
        start = time.time()
        analysis = STOCKFISH.analyse(self.current_board, limit=chess.engine.Limit(time=0.05), multipv=25)
        if isinstance(analysis, dict): # sometimes analysis only gives one result and is not a list.
            analysis = [analysis]
        self.stockfish_analysis = analysis
        end = time.time()
        self.log += "Analysis computed in {} seconds: \n".format(end-start)
        # Getting lucas analytics for the position
        self.log += "Calculating lucas analytics for the position. \n"
        xcomp, xmlr, xemo, xnar, xact = get_lucas_analytics(self.current_board, analysis=self.stockfish_analysis)
        lucas_dict = {"complexity": xcomp, "win_prob": xmlr, "eff_mob": xemo, "narrowness": xnar, "activity": xact}
        self.lucas_analytics.update(lucas_dict)
        self.log += "Lucas analytics: {} \n".format(lucas_dict)
        
        # We should check for obvious moves that don't require much thought
        obvious_move, obvious_move_found = self.check_obvious_move()
        
        if obvious_move_found == True:
            return_dic["move_made"] = obvious_move
            use_human_filters = False
        else:
            # Now need to decide base on information whether we are using strictly engine move
            # Or using human filters
            self.log += "Deciding whether to use human filters. \n"
            use_human_filters = self._decide_human_filters()
            if use_human_filters == True:
                self.log += "Using human filters. \n"
                # first let us decide how much time we shall try spend on the move
                # key word "try"
                target_time = self._set_target_time()
                return_dic["move_made"] = self.get_human_move(target_time=target_time)
            else:
                self.log += "Not using human filters. \n"
                return_dic["move_made"] = self.get_stockfish_move()
            
        # Finally decide how much time we are going to spend (including thinking time)
        return_dic["time_take"] = self._get_time_taken(obvious=obvious_move_found, human_filters=use_human_filters)
        
        # log our calculating information for this move
        #self._write_log()
        
        return return_dic

if __name__ == "__main__":
    engine = Engine()
    # b = chess.Board("3r2k1/3r1p1p/PQ2p1p1/8/5q2/2P2N2/1P3PP1/R3K2R w KQ - 1 24")
    input_dic = {'fens': ['1rbq1rk1/p3bpp1/1p2pn1p/n3N3/8/2NPB3/PPQ1PPBP/R3K1R1 b Q - 1 15',
       '1r1q1rk1/pb2bpp1/1p2pn1p/n3N3/8/2NPB3/PPQ1PPBP/R3K1R1 w Q - 2 16',
       '1r1q1rk1/pB2bpp1/1p2pn1p/n3N3/8/2NPB3/PPQ1PP1P/R3K1R1 b Q - 0 16',
       '3q1rk1/pr2bpp1/1p2pn1p/n3N3/8/2NPB3/PPQ1PP1P/R3K1R1 w Q - 0 17',
       '3q1rk1/pr2bpp1/1p2pn1B/n3N3/8/2NP4/PPQ1PP1P/R3K1R1 b Q - 0 17'],
      'side': False,
      'self_clock_times': [53.0, 51.0, 50.0, 49.0, 48.0],
      'opp_clock_times': [50.0, 49.0, 48.0, 47.0, 47.0],
      'self_initial_time': 60.0,
      'opp_initial_time': 60.0,
      'last_moves': ['f3e5', 'c8b7', 'g2b7', 'b8b7', 'e3h6']}
    start = time.time()
    engine.update_info(input_dic)
    print(engine.make_move())
    end = time.time()
    print("finished in {} seconds".format(end-start))