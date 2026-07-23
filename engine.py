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
import torch

import chess
import chess.engine
import chess.polyglot

from models.models import MoveScorer, StockFishSelector
from development.alter_move_prob_train.alter_move_prob_nn import AlterMoveProbNN

from common.constants import (PATH_TO_STOCKFISH, MOVE_FROM_WEIGHTS_OP_PTH, MOVE_FROM_WEIGHTS_MID_PTH,
                              MOVE_FROM_WEIGHTS_END_PTH, MOVE_TO_WEIGHTS_MID_PTH,
                              MOVE_TO_WEIGHTS_END_PTH, MOVE_TO_WEIGHTS_OP_PTH,
                              QUICKNESS,
                              HUMAN_EVAL_NOISE_SCALE,
                              FLAG_RACE_TIME,
                              PATH_TO_PONDER_STOCKFISH, MOVE_FROM_WEIGHTS_TACTICS_PTH,
                              MOVE_TO_WEIGHTS_TACTICS_PTH,
                              DEPTH_PENALTY, ZERO_DEPTH_PENALTY, CAPTURE_BONUS
                              )

from common.board_information import (
    phase_of_game, PIECE_VALS, calculate_threatened_levels, check_best_takeback_exists,
            )
from common.search_constants import (
    PREMOVE_SCAN_MULTIPV, MAX_CALC_DEPTH_COEFF, PONDER_TIME_PER_POSITION,
)
from common.utils import (check_safe_premove, extend_mate_score)
from common.logging import get_logger, LogLevel, LegacyLoggerAdapter
from engine_components import simple_decisions, state, mood_manager, stockfish_move_logic, human_move_logic, decision_logic, game_character

# TODO: Intelligent premoves
# TODO: 3-fold repetition logic

class Engine:
    """ Class for engine instance.
    
        The Engine is responsible for the following things ONLY
        receiving board information -> outputting move and premoves
        
        All other history related data to do with past moves etc are not handled
        in the Engine instance. They are handled in the client wrapper
    """
    def __init__(self, playing_level:int = 6, log_file: str = None, opening_book_path:str = "assets/data/Opening_books/bullet.bin", quickness: float = None):
        # Per-instance move-time pacing; None keeps the global QUICKNESS, so
        # live behaviour is unchanged. The simulator sets it to give the two
        # bots of a self-play pair different pacing.
        self.quickness = QUICKNESS if quickness is None else quickness
        self.input_info = {
            "side": None,
            "fens": None,
            "self_clock_times" : None,
            "opp_clock_times": None,
            "self_initial_time": None,
            "opp_initial_time": None,
            "opp_rating": None,
            "self_rating": None,
            "last_moves": None,
                           }
        self.current_board = chess.Board()
        
        # Logging - uses unified logging system via legacy adapter
        self.log = LegacyLoggerAdapter(channel="engine")
        
        
        # setting up scorers for moves
        self.human_scorers = {
            "opening": MoveScorer(MOVE_FROM_WEIGHTS_OP_PTH, MOVE_TO_WEIGHTS_OP_PTH),
            "midgame": MoveScorer(MOVE_FROM_WEIGHTS_MID_PTH, MOVE_TO_WEIGHTS_MID_PTH),
            "endgame": MoveScorer(MOVE_FROM_WEIGHTS_END_PTH, MOVE_TO_WEIGHTS_END_PTH),
            "defensive_tactics": MoveScorer(MOVE_FROM_WEIGHTS_TACTICS_PTH, MOVE_TO_WEIGHTS_TACTICS_PTH),
            }
        self.stockfish_scorer = StockFishSelector(PATH_TO_STOCKFISH)
        self.stockfish_engine = chess.engine.SimpleEngine.popen_uci(PATH_TO_STOCKFISH)
        self.ponder_stockfish_engine = chess.engine.SimpleEngine.popen_uci(PATH_TO_PONDER_STOCKFISH)
        self.stockfish_analysis = None
        self._stockfish_path = PATH_TO_STOCKFISH  # Store for potential engine restart

        # initialise move prob altering model
        model_weights_path = "development/alter_move_prob_train/data/alter_move_prob_nn_best.pth"
        self.move_prob_altering_model = AlterMoveProbNN()
        self.move_prob_altering_model.load_state_dict(torch.load(model_weights_path, weights_only=True))
        self.move_prob_altering_model.eval()
        self.move_prob_altering_model.load_params_dict()
        
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
        # Position "sharpness": win-probability spread across the top candidate
        # moves (see _compute_sharpness). Drives the "complicated position"
        # time-scaling in _get_time_taken. 0.25 is the neutral / critical point.
        self.sharpness = None
        # The sharpness scan itself ({move_uci: win_chance} from the deep
        # narrow scan), kept for _chosen_move_wc_loss. None if the scan failed.
        self.sharpness_scan = None
        # Per-game character (see _sample_game_character): one draw per game
        # each, resampled at every game boundary, None until the first
        # update_info. game_pace_sf scales every base think time in
        # _get_time_taken; game_premove_sf scales every premove-search
        # probability in make_move.
        self.game_pace_sf = None
        self.game_premove_sf = None
        self.game_snap_gate = None
        self.game_ponder_snap_sf = None
        self.game_scramble_skill = None
        self.game_scramble_fire_sf = None
        self.game_ponder_width = None
        self._last_seen_ply = None
        # A bool to track whether we have updated analytics following updating info
        self.analytics_updated = False
        
        self.playing_level = playing_level
        self.mood = "confident"
        self.just_blundered = None
        
    def _write_log(self):
        """ Writes buffered log messages to the log file. """
        self.log.write()
    
    def _decide_resign(self):
        """ The decision making function which decides whether to resign.

            Returns bool
        """
        return simple_decisions.decide_resign(self)

    def _decide_human_filters(self):
        """ Decide if given the position we want to use human filters or not. When
            using human filters we would need to use more processing time, which
            may not be suitable when we have low time. If possible however, we always
            want to use human filters.

            Returns True/False
        """
        return simple_decisions.decide_human_filters(self)
    
    def get_stockfish_move(self, board:chess.Board = None, analysis=None, last_move_uci:str = None, log:bool = True, ):
        """ Uses board information to get a move strictly from stockfish with no human
            filters. Very fast, and only called for in necessary situations (when in
            super low time)

            Returns a move_uci string of move made
        """
        return stockfish_move_logic.get_stockfish_move(self, board=board, analysis=analysis, last_move_uci=last_move_uci, log=log)

    def adjust_human_prob(self, move_dic, board : chess.Board = None):
        """ Given move_dic from human probabilities, we normalise the probabilities
            i.e. make them less extreme depending on how low on time we are as well
            as how far in the game we are (remaining pieces) as well as how far we
            are winning by (helps with closing out games).

            Returns normalised move_dic
        """
        return human_move_logic.adjust_human_prob(self, move_dic, board=board)

    def get_human_probabilities(self, board : chess.Board, game_phase: str, log:bool = True):
        """ Given a chess.Board item, returns the top move ucis along with their
            human move probabilties, evaluated from neural network only. These bare
            no extra tinkering methods and are purely from the neural net.
        """
        return human_move_logic.get_human_probabilities(self, board, game_phase, log=log)

    def _alter_move_prob_nn(self, move_dic : dict, board:chess.Board, prev_board:chess.Board = None, prev_prev_board:chess.Board = None, log:bool = True):
        """ Given a move dictionary with move uci as key and value as their unaltered
            probabilities, we alter the probabilties to make moves stick out more
            (for example hanging pieces more likely to be moved etc).

            Returns an altered move_dic.
        """
        return human_move_logic.alter_move_prob_nn(self, move_dic, board, prev_board=prev_board, prev_prev_board=prev_prev_board, log=log)

    def _alter_move_probabilties(self, move_dic : dict, board:chess.Board, prev_board:chess.Board = None, prev_prev_board:chess.Board = None, log:bool = True):
        """ Given a move dictionary with move uci as key and value as their unaltered
            probabilities, we alter the probabilties to make moves stick out more
            (for example hanging pieces more likely to be moved etc).

            Returns an altered move_dic.
        """
        return human_move_logic.alter_move_probabilties(self, move_dic, board, prev_board=prev_board, prev_prev_board=prev_prev_board, log=log)

    def _decide_breadth(self, total_time=None):
        """ Given our current board information and amount of time, decide how many of our human moves
            to consider and pass onto the engine.

            Returns integer
        """
        return decision_logic.decide_breadth(self, total_time=total_time)

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
        # altered_move_dic = self._alter_move_probabilties(un_altered_move_dic, self.current_board, prev_board=prev_board, prev_prev_board=prev_prev_board)
        altered_move_dic = self._alter_move_prob_nn(un_altered_move_dic, self.current_board, prev_board=prev_board, prev_prev_board=prev_prev_board)
        # Now decide how many of these top moves we shall consider for calculation
        no_root_moves = self._decide_breadth(target_time)
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
        if target_time > no_root_moves * 6 * max(human_calc_time, BUFFER):
            target_time = no_root_moves * 6 * max(human_calc_time, BUFFER)
            self.log += "Shortening human target time to {} \n".format(target_time)
            
        re_evaluations = int(max(target_time//max(human_calc_time +0.02, BUFFER) - 1, 0))
        self.log += "Plan to re-evaluate {} of the top human variations to cloud judgement. \n".format(re_evaluations)
        top_human_moves = sorted(human_move_evals.keys(), reverse=True, key= lambda x: human_move_evals[x])
        
        # If the number of re-evaluations far exceeds the numbre of top moves, we may keep 
        # re-evaluating each seed move with greater depth    
        # however there is a max depth, as probability of silly moves increases with depth
        max_depth = 1 + int(MAX_CALC_DEPTH_COEFF * no_root_moves ** 0.5) # this is the maximum depth we will consider for re-evaluation, as for too deep re-evaluation we increase the chance of silly moves.
        depth = min((re_evaluations // no_root_moves) + 1, max_depth)
        
        reval_start = time.time()
        re_evaluate_moves = random.sample(top_human_moves, min(re_evaluations, len(top_human_moves)))
        san_re_evaluate_moves = [self.current_board.san(chess.Move.from_uci(x)) for x in re_evaluate_moves]
        time_allowed = target_time - (reval_start - start)
        self.log += "Re-evaluating moves: {} with depth {} with time allowed {} \n".format(san_re_evaluate_moves, depth, time_allowed)
        # Targeted console output
        print(f"[ENGINE] Re-evaluating moves: {san_re_evaluate_moves} with depth {depth}")
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
        # We add extra parameter to control for the noise level at different stages of the game
        noise_dic = {"opening": 0.8,
                     "midgame": 1.2,
                     "endgame": 0.3,}
        
        
        # Furthermore, humans won't delay capturing pieces if they're free and enpris. So like
        # we gave capturing moves higher spotting chance, we shall also encourage capturing moves here too
        eval_only_dic = {}
        noise_phase = noise_dic[game_phase]
        for move_uci in new_human_move_evals.keys():
            eval_, depth_considered = new_human_move_evals[move_uci]
            base_noise_sd = 40*(np.tanh(eval_/(self.playing_level*50)))**2 + 20
            noise_sd = HUMAN_EVAL_NOISE_SCALE*4*base_noise_sd/(target_time*(depth_considered+4))
            
            noise = np.random.randn()*noise_sd*noise_phase - DEPTH_PENALTY*(2- depth_considered)
            if depth_considered == 0:
                noise -= ZERO_DEPTH_PENALTY       
            
            # encourage capture moves, depending on how enpris the piece is
            move_obj = chess.Move.from_uci(move_uci)
            if self.current_board.is_capture(move_obj):
                capture_bonus = CAPTURE_BONUS* int(calculate_threatened_levels(move_obj.to_square, self.current_board.copy()))
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
                
                # altered_move_dic = self._alter_move_probabilties(un_altered_move_dic, board=dummy_board, prev_board=board.copy(), prev_prev_board=prev_prev_board, log=False)
                altered_move_dic = self._alter_move_prob_nn(un_altered_move_dic, board=dummy_board, prev_board=board.copy(), prev_prev_board=prev_prev_board, log=False)
                
                human_move_ucis = list(altered_move_dic.keys())
                
                root_moves = [chess.Move.from_uci(x) for x in human_move_ucis[:search_width]]
            
            if use_ponder == True:
                single_analysis = self.ponder_stockfish_engine.analyse(dummy_board, chess.engine.Limit(time=0.05), root_moves=root_moves)
            else:
                single_analysis = self.stockfish_engine.analyse(dummy_board, chess.engine.Limit(time=0.02), root_moves=root_moves)
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
            
    
    def _sample_game_character(self):
        """ Draws this game's character multipliers -- game-to-game variation
            that per-move noise cannot produce:

            - game_pace_sf: applied to think times in _get_time_taken, so one
              game is played uniformly faster or slower than another.
            - game_premove_sf: applied to the premove-search probabilities in
              make_move, so one game premoves eagerly and another rarely.
            - game_snap_gate: the intuition-gate probability in
              _get_time_taken, so one game snaps most sharp positions on gut
              feel and another stops to think on most of them.
            - game_ponder_snap_sf: applied (with pace) to the ponder-response
              wait only, so one game bangs out recognised replies and another
              double-checks them -- the second channel of the instant-move
              rate's game-to-game spread.
            - game_scramble_skill: this game's flag-race composure, setting
              the scramble eval cap and blind-move probability in
              get_stockfish_move; one game scrambles cleanly, another throws
              a won ending.
        """
        return game_character.sample_game_character(self)

    @property
    def scramble_veto_p(self):
        """ Probability the scramble safety vetos apply this game (see
            SCRAMBLE_VETO_P_* in constants); 1.0 before the first draw.
            Clients read this so live and sim can't drift. """
        return game_character.scramble_veto_p(self)

    @property
    def ponder_pace_sf(self):
        """ Combined per-game scale for the ponder-response wait (pace x
            ponder-snap), 1.0 before the first per-game draw. Clients and the
            simulator both read this so the coupling can't drift. """
        return game_character.ponder_pace_sf(self)

    def new_game(self):
        """ Signals a game boundary: resamples per-game character (pace,
            premove propensity). update_info also detects boundaries itself
            (ply going backwards), so calling this is optional -- it only makes
            the boundary exact for games that start deeper into the book than
            the last one ended.
        """
        return state.new_game(self)

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
        return state.update_info(self, info_dic, auto_update_analytics)

    def calculate_analytics(self):
        """ Before any move making or human analysis is performed, statistics must be computed for
            the infomation dict self.input_info. This function must be called after every
            update_info, or everytime the info dic is updated.

            Returns None
        """
        return state.calculate_analytics(self)

    def _compute_sharpness(self):
        """ Measures how "sharp" (critical) the current position is, using the
            same definition as the cheat_detection human-likeness analyser:
            the spread in win-probability across the engine's top candidate
            moves. A position where the best move is winning and the next-best
            is losing is sharp (a lot is at stake); one where every candidate
            keeps roughly the same win-probability is not.

            Distinct from the Lucas analytics (which measure *structural*
            complexity -- number of good moves, mobility, activity): this keys
            off eval-stakes instead. Computed from a narrow, slightly deeper
            scan (multipv 5, depth 12) than the Lucas scan.

            Returns sharpness : float in [0, 1]. Falls back to 0.25 (the neutral
            / "critical" threshold) if the position can't be scanned, so a
            failed scan leaves the move-time pacing unchanged.
        """
        return state.compute_sharpness(self)

    def check_obvious_move(self):
        """ Given input information, check whether there is an obvious move in the
            position that we may play immediately.

            Returns [obvious_move: uci_str, obvious_move_found : bool]
            In the case that no obvious is found, obvious_move is None"""
        return simple_decisions.check_obvious_move(self)
    
    def _set_target_time(self, total_time):
        """ Given we are using human approaches to decide the move, we set and initial
            target time which we try to compute our human move. This is supposedly a
            reflection of how hurredly the player is before they've even thought
            about any moves. total_time is the time limit we cannot exceed.

            Returns target time, a non-negative float.
        """
        return decision_logic.set_target_time(self, total_time)

    def _get_time_taken(self, obvious:bool=False, human_filters:bool=True):
        """ Calculates the amount of time in total we should spend on a move.
            obvious is whether we made a quick obvious move.
            human_filters is whether we used human filters

            Returns time_taken : float
        """
        return decision_logic.get_time_taken(self, obvious=obvious, human_filters=human_filters)

    def _chosen_move_wc_loss(self, move_uci):
        """ Win-probability given up by the chosen move vs the engine's best.

            Prefers the deep narrow sharpness scan (multipv 5, no time cap):
            if the chosen move is one of its lines the loss is read directly;
            if it fell outside the top 5, the scan's spread is a lower bound
            on the loss (the move is worse than every scanned line). The
            full-width analysis is only a fallback estimate -- it is capped
            at 20ms, so its per-move evals are very noisy. Returns None if
            nothing can be read.
        """
        return decision_logic.chosen_move_wc_loss(self, move_uci)

    def _adjust_time_for_move_loss(self, move_uci, time_take):
        """ The hesitation before the mistake -- and its mirror, the decisive
            snap. Humans think longer in positions where they end up erring
            (difficulty consumes clock AND induces the error), giving a
            positive per-game correlation between move time and move loss.
            The engine's errors come from the human-probability sampling,
            independent of the decided think time, so the link is restored
            here: when the chosen move gives up real win probability, the
            already-decided think time is (sometimes) stretched; when it is
            clean, it is (sometimes) trimmed -- humans bang out moves they
            are sure of. The trim keeps the mean move time level despite the
            stretches. Not every mistake hesitates: snap blunders exist.

            Returns the adjusted time_take.
        """
        return decision_logic.adjust_time_for_move_loss(self, move_uci, time_take)

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
        return mood_manager.set_mood(self)


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
                    exists, takeback_move_uci = check_best_takeback_exists(board.copy(), move_obj.uci(), engine=self.stockfish_engine)
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
        analysis = self.stockfish_engine.analyse(board, limit=chess.engine.Limit(depth=8, time=0.02))
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
            next_analysis = self.stockfish_engine.analyse(dummy_board, limit=chess.engine.Limit(depth=8, time=0.02), multipv=PREMOVE_SCAN_MULTIPV)
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
            elif self._own_time_or_none() is not None and self._own_time_or_none() < FLAG_RACE_TIME \
                    and np.random.random() < self.scramble_veto_p:
                # In a flag race a queued premove fires on the server
                # unconditionally, so unsafe ones are a main source of
                # instant blunders (measured: bot emt-0 TP blunder rate
                # 0.136 vs human 0.072 -- humans' scramble premoves are
                # instinct-safe takebacks and king steps). Vet like the
                # opening branch, but gated on this game's scramble skill:
                # unconditional vetting collapsed the catastrophe tail
                # (endgame ACPL std ratio fell to 0.37x) -- a panicky game
                # must still queue the occasional howler. Takeback premoves
                # returned earlier are deliberately exempt.
                if check_safe_premove(board, candidate_premove) == True:
                    premove_uci = candidate_premove
                else:
                    self.log += "Scramble premove {} is not a safe premove, filtering out. \n".format(candidate_premove)
                    premove_uci = None
            else:
                premove_uci = candidate_premove
        return premove_uci

    def _own_time_or_none(self):
        """ Our latest clock reading, or None before the first update. """
        times = self.input_info.get("self_clock_times") if self.input_info else None
        return times[-1] if times else None
    
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
            analysis_object = self.ponder_stockfish_engine.analyse(board, limit=chess.engine.Limit(depth=8, time=time_allowed), multipv=ponder_width, root_moves=root_moves)
        else:
            analysis_object = self.stockfish_engine.analyse(board, limit=chess.engine.Limit(depth=8, time=time_allowed), multipv=ponder_width, root_moves=root_moves)
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
        
        max_depth = int(MAX_CALC_DEPTH_COEFF * search_width ** 0.5) # this is the maximum depth we will consider for ponder, as for too deep ponder we increase the chance of silly moves.

        if ponder_width is None:
            # As a maximum number of ponder moves (to prevent too quick responses),
            # per-game coverage draw when available (see _sample_game_character),
            # else by time control.
            initial_time = self.input_info["self_initial_time"]
            if self.game_ponder_width is not None:
                max_ponder_no = self.game_ponder_width
                if max_ponder_no <= 0:
                    # A no-ponder game (see GAME_PONDER_WIDTH_CLIP): this
                    # game never prepares replies at all.
                    self.log += "Ponder width 0 this game: skipping ponder. \n"
                    return None
            elif initial_time <= 180:
                max_ponder_no = 3
            else:
                max_ponder_no = 4
        
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
        
        ponder_depth = min(ponder_depth, max_depth)

        if use_ponder:
            
            analysis_object = self.ponder_stockfish_engine.analyse(board, limit=chess.engine.Limit(depth=8, time=0.02), multipv=ponder_width)
            
        else:
            analysis_object = self.stockfish_engine.analyse(board, limit=chess.engine.Limit(depth=8, time=0.02), multipv=ponder_width)
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
        
        # For noise dictionary. We implement different levels of noise depending on the stage of the game
        noise_dic = {"opening": 0.8,
                     "midgame": 1.2,
                     "endgame": 0.3,}
        
        
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
                # top_human_move_dic = self._alter_move_probabilties(top_human_move_dic, dummy_board, prev_board = board, prev_prev_board=prev_board, log= False)
                top_human_move_dic = self._alter_move_prob_nn(top_human_move_dic, dummy_board, prev_board = board, prev_prev_board=prev_board, log= False)
                top_human_moves = sorted(top_human_move_dic.keys(), key=lambda x: top_human_move_dic[x], reverse=True)[:search_width]
                
            re_evaluate_dic = self._re_evaluate(dummy_board, top_human_moves, search_width, depth=ponder_depth, prev_board = board.copy(), limit=[ponder_depth*search_width, time_allowed/2], use_ponder=use_ponder)
            # adding noise
            noise_phase = noise_dic[game_phase]
            for move_uci in re_evaluate_dic.keys():
                eval_, depth_considered = re_evaluate_dic[move_uci]
                if eval_ is  None:
                    # move never got considered
                    # Get stockfish evaluation of move, but penalise heavily
                    if use_ponder:
                        an_obj = self.ponder_stockfish_engine.analyse(dummy_board, limit=chess.engine.Limit(depth=8, time=0.02), root_moves=[chess.Move.from_uci(move_uci)])
                        if "score" in an_obj:
                            new_eval = extend_mate_score(an_obj["score"].pov(self.input_info["side"]).score(mate_score=2500))
                        else:
                            # something went wrong
                            self.log += "Something went wrong with analysis object with use_ponder: {}. Returning no ponder dic. \n".format(an_obj)
                            return None
                    else:
                        an_obj = self.stockfish_engine.analyse(dummy_board, limit=chess.engine.Limit(depth=8, time=0.02), root_moves=[chess.Move.from_uci(move_uci)])
                        if "score" in an_obj:
                            new_eval = extend_mate_score(an_obj["score"].pov(self.input_info["side"]).score(mate_score=2500))
                        else:
                            # something went wrong
                            self.log += "Something went wrong with analysis object: {}. Returning no ponder dic. \n".format(an_obj)
                            return None
                    eval_ = new_eval - 100
                    re_evaluate_dic[move_uci][0] = new_eval - 100 # penalty
                base_noise_sd = 40*(np.tanh(eval_/(self.playing_level*50)))**2 + 20
                noise_sd = HUMAN_EVAL_NOISE_SCALE*4*base_noise_sd/(time_allowed*(depth_considered+4))
                
                noise = np.random.randn()*noise_sd*noise_phase        
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
        return mood_manager.check_opp_blunder(self)
    
    def make_move(self, log:bool=True, seed:int=None):
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
        # First establish the a random seed for random, numpy and torch for reproducibility and write it to the log
        if seed is None:
            random_seed = np.random.randint(0, 1000000)
        else:
            random_seed = seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        self.log += "Random seed for numpy and torch set to {} \n".format(random_seed)
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
            # Now that the move is known, restore the human link between
            # think time and move quality (hesitation before the mistake).
            return_dic["time_take"] = self._adjust_time_for_move_loss(
                return_dic["move_made"], return_dic["time_take"])

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
            # Probability of searching for a *full* premove (anticipating the
            # opponent's reply), by situation; anything else falls back to the
            # takeback-only search. Both are scaled by this game's premove
            # propensity (see _sample_game_character): an eager game premoves
            # more of everything, a reluctant game sometimes doesn't even
            # queue the obvious takeback.
            premove_sf = self.game_premove_sf if self.game_premove_sf is not None else 1.0
            # Confidence against weaker opposition shows up in the snap/premove
            # channel, not the think-time channel (a decided think time can't
            # beat the engine-compute floor, so scaling it barely moves the
            # instant-move rate; humans facing a weaker opponent premove and
            # bang out moves more). ~+15% propensity at +200 Elo, symmetric.
            opp_rating = self.input_info["opp_rating"]
            self_rating = self.input_info["self_rating"]
            if opp_rating is not None and self_rating is not None:
                rating_ratio = (self_rating - opp_rating) / self_rating
                premove_sf *= 1 + np.tanh(2 * rating_ratio)
            if self.mood == "hurry" and own_time < 20:
                full_premove_prob = 0.3*self_initial_time/(own_time + 0.3*self_initial_time)
            elif self_initial_time <= 60 and phase_of_game(self.current_board) == "opening":
                full_premove_prob = 0.9
            else:
                full_premove_prob = 0.0
            if np.random.random() < min(full_premove_prob * premove_sf, 0.98):
                return_dic["premove"] = self.get_premove(after_board, self.input_info["side"])
            elif np.random.random() < premove_sf:
                # look for takeback premoves only
                return_dic["premove"] = self.get_premove(after_board, self.input_info["side"], takeback_only=True)
            else:
                return_dic["premove"] = None
                self.log += "Skipped premove search this move (premove propensity {:.2f}). \n".format(premove_sf)
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
        
        time_per_position = PONDER_TIME_PER_POSITION # nominal cost per ponder position (see search_constants)
        time_left = return_dic["time_take"] - time_spent
        search_width = self._decide_breadth(time_left) # this is slightly incorrect, but close enough
        if time_left > time_per_position*search_width * 1.15:
            self.log += "Have enough time to ponder for he next position. Time taken so far: {} \n".format(time_spent)
            ponder_start = time.time()
            ponder_dic = self.ponder(after_board, time_left/1.15, search_width, time_per_position=time_per_position, prev_board=self.current_board.copy())
            ponder_end = time.time()
            self.log += "Took {} seconds for pondering. \n".format(ponder_end - ponder_start)
        else:
            self.log += "Do not have enough time to ponder for he next position. Time taken so far: {} \n".format(time_spent)
            ponder_dic = None

        # If ponder dic is not None, for every position in ponder dic, also see if there is a premove we can make
        if ponder_dic is not None:
            return_ponder_dic = {}
            for board_fen, response_move_uci in ponder_dic.items():
                b = chess.Board()
                b.set_board_fen(board_fen)
                b.turn = self.input_info["side"]
                # push response move uci to the board, but first check if it is legal
                if chess.Move.from_uci(response_move_uci) in b.legal_moves:
                    b.push_uci(response_move_uci)
                    # get premove for the board -- gated by this game's premove
                    # propensity, like the direct premove searches above
                    if np.random.random() < (self.game_premove_sf if self.game_premove_sf is not None else 1.0):
                        premove = self.get_premove(b, self.input_info["side"], takeback_only=True)
                    else:
                        premove = None
                    if premove is not None:
                        return_ponder_dic[board_fen] = {"move":response_move_uci, "premove":premove}
                    else:
                        return_ponder_dic[board_fen] = {"move":response_move_uci, "premove":None}
                else:
                    # no premove for this position
                    return_ponder_dic[board_fen] = {"move":response_move_uci, "premove":None}                
        else:
            return_ponder_dic = None

        return_dic["ponder_dic"] = return_ponder_dic
        
        if log == True:
            self._write_log()
        self.log += "Returning return dic with all of our engine's calculations: \n"
        self.log += "{} \n".format(return_dic)
        # Targeted console output
        print(f"[ENGINE] Final output: {return_dic}")
        
        # log our calculating information for this move
        if log == True:
            self._write_log()
        
        return return_dic

    def close_engines(self):
        """ Close all the engines """
        self.stockfish_engine.quit()
        self.ponder_stockfish_engine.quit()
        self.stockfish_scorer.engine.quit()

if __name__ == "__main__":
    engine = Engine(playing_level=3)    
    # b = chess.Board("3r2k1/3r1p1p/PQ2p1p1/8/5q2/2P2N2/1P3PP1/R3K2R w KQ - 1 24")
    input_dic ={'fens': ['r2qk2r/pp3ppp/2p1pn2/4n3/1bPP4/P1N5/1P2QPPP/R1B2RK1 b kq - 1 12', 'r2qk2r/pp3ppp/2p1pn2/4n3/2PP4/P1b5/1P2QPPP/R1B2RK1 w kq - 0 13', 'r2qk2r/pp3ppp/2p1pn2/4P3/2P5/P1b5/1P2QPPP/R1B2RK1 b kq - 0 13', 'r2qk2r/pp3ppp/2p1pn2/4b3/2P5/P7/1P2QPPP/R1B2RK1 w kq - 0 14', 'r2qk2r/pp3ppp/2p1pn2/4Q3/2P5/P7/1P3PPP/R1B2RK1 b kq - 0 14', 'r2q1rk1/pp3ppp/2p1pn2/4Q3/2P5/P7/1P3PPP/R1B2RK1 w - - 1 15', 'r2q1rk1/pp3ppp/2p1pn2/4Q1B1/2P5/P7/1P3PPP/R4RK1 b - - 2 15', 'rq3rk1/pp3ppp/2p1pn2/4Q1B1/2P5/P7/1P3PPP/R4RK1 w - - 3 16'], 'self_clock_times': [55, 55, 54, 53, 50, 49, 41, 40], 'opp_clock_times': [57, 57, 56, 55, 54, 51, 50, 48], 'last_moves': ['b4c3', 'd4e5', 'c3e5', 'e2e5', 'e8g8', 'c1g5', 'd8b8'], 'side': True, 'self_initial_time': 60, 'opp_initial_time': 60, 'opp_rating': 2543, 'self_rating': 2560}
    start = time.time()
    engine.update_info(input_dic)
    # Set random seeds for reproducibility
    random_seed = 743825
    print(engine.make_move(log=False, seed=random_seed))
    end = time.time()
    
    print("Engine log contents:")
    print(engine.log)
    print("finished in {} seconds".format(end-start))
    
    # Clean up the engine processes
    print("Closing engine instances")
    engine.stockfish_scorer.engine.quit()
    engine.ponder_stockfish_engine.quit()
    engine.stockfish_engine.quit()
