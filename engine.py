# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:29:08 2024

@author: xusem
"""
import datetime
import os
import time
from typing import Any, Optional
import numpy as np
import random
import torch

import chess
import chess.engine
import chess.polyglot

from models.models import MoveScorer, StockFishSelector
from models.alter_move_prob_nn import AlterMoveProbNN

from common.constants import (PATH_TO_STOCKFISH, MOVE_FROM_WEIGHTS_OP_PTH, MOVE_FROM_WEIGHTS_MID_PTH,
                              MOVE_FROM_WEIGHTS_END_PTH, MOVE_TO_WEIGHTS_MID_PTH,
                              MOVE_TO_WEIGHTS_END_PTH, MOVE_TO_WEIGHTS_OP_PTH,
                              ALTER_MOVE_PROB_WEIGHTS_PTH,
                              QUICKNESS,
                              HUMAN_EVAL_NOISE_SCALE,
                              PATH_TO_PONDER_STOCKFISH, MOVE_FROM_WEIGHTS_TACTICS_PTH,
                              MOVE_TO_WEIGHTS_TACTICS_PTH,
                              DEPTH_PENALTY, ZERO_DEPTH_PENALTY, CAPTURE_BONUS,
                              OPENING_REPERTOIRE_PATH,
                              )

from common.board_information import (
    phase_of_game, calculate_threatened_levels,
            )
from common.search_constants import (
    MAX_CALC_DEPTH_COEFF, PONDER_TIME_PER_POSITION, MODERATE_SHARPNESS_BREADTH_BONUS,
    STOCKFISH_THREADS, STOCKFISH_HASH_MB, MIDGAME_PREMOVE_VETO_P,
    OPENING_BREADTH_STRENGTH_BONUS, MIDGAME_BREADTH_STRENGTH_BONUS,
)
from common.utils import (extend_mate_score)
from common.logging import get_logger, LogLevel, LegacyLoggerAdapter
from engine_components import simple_decisions, state, mood_manager, stockfish_move_logic, human_move_logic, decision_logic, game_character, premover, ponderer, opening_book

# TODO: Intelligent premoves
# TODO: 3-fold repetition logic

class Engine:
    """ Class for engine instance.
    
        The Engine is responsible for the following things ONLY
        receiving board information -> outputting move and premoves
        
        All other history related data to do with past moves etc are not handled
        in the Engine instance. They are handled in the client wrapper
    """
    def __init__(self, playing_level:int = 6, log_file: Optional[str] = None, opening_book_path:str = "assets/data/Opening_books/bullet.bin", repertoire_book_path:str = OPENING_REPERTOIRE_PATH, quickness: Optional[float] = None, eval_noise_scale: Optional[float] = None, moderate_sharpness_breadth_bonus: Optional[int] = None, midgame_premove_veto_p: Optional[float] = None, opening_breadth_strength_bonus: Optional[int] = None, midgame_breadth_strength_bonus: Optional[int] = None):
        # Per-instance move-time pacing; None keeps the global QUICKNESS, so
        # live behaviour is unchanged. The simulator sets it to give the two
        # bots of a self-play pair different pacing.
        self.quickness = QUICKNESS if quickness is None else quickness
        # Per-instance eval-noise scale (see the noise formula in
        # get_human_move / ponderer.py); None keeps the global
        # HUMAN_EVAL_NOISE_SCALE. Overridable per instance so the strength
        # calibration work (breadth vs. noise, independent of playing_level)
        # can vary it per simulated bot.
        self.eval_noise_scale = HUMAN_EVAL_NOISE_SCALE if eval_noise_scale is None else eval_noise_scale
        # Per-instance breadth bonus for the moderate-sharpness band (see
        # decide_breadth / search_constants.py); None keeps the global
        # MODERATE_SHARPNESS_BREADTH_BONUS. Overridable so an Elo-delta
        # experiment can A/B the fix (0 = pre-fix behaviour) against the
        # shipped default on an otherwise identical bot.
        self.moderate_sharpness_breadth_bonus = (
            MODERATE_SHARPNESS_BREADTH_BONUS if moderate_sharpness_breadth_bonus is None
            else moderate_sharpness_breadth_bonus)
        # Per-instance probability of vetting an ordinary premove with
        # check_safe_premove (see premover.py / MIDGAME_PREMOVE_VETO_P);
        # None keeps the global default. A strength/human-likeness lever to
        # sweep, not a fixed correctness knob -- see search_constants.py.
        self.midgame_premove_veto_p = (
            MIDGAME_PREMOVE_VETO_P if midgame_premove_veto_p is None
            else midgame_premove_veto_p)
        # Per-instance breadth strength dials for the opening/midgame phases
        # only (see decide_breadth / search_constants.py 2d); None keeps the
        # respective global default (0 = no behavioural change). Endgame has
        # no equivalent override -- the human elo-progression data showed no
        # rating-driven improvement there, so it's deliberately not a lever.
        self.opening_breadth_strength_bonus = (
            OPENING_BREADTH_STRENGTH_BONUS if opening_breadth_strength_bonus is None
            else opening_breadth_strength_bonus)
        self.midgame_breadth_strength_bonus = (
            MIDGAME_BREADTH_STRENGTH_BONUS if midgame_breadth_strength_bonus is None
            else midgame_breadth_strength_bonus)
        # Values are populated by update_info; typed loosely (Any) since
        # they're heterogeneous (bool, list, int, float) and set as a batch
        # via dict.update() rather than individual attribute assignment.
        self.input_info: dict[str, Any] = {
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
        # Every call against these two is wall-clock capped (not depth-capped)
        # to keep move latency low; that cap is unchanged. Threads/Hash above
        # UCI defaults just let more search happen inside the same wall time
        # -- free strength, zero effect on latency or modelled behaviour.
        stockfish_config = {"Threads": STOCKFISH_THREADS, "Hash": STOCKFISH_HASH_MB}
        self.stockfish_engine.configure(stockfish_config)
        self.ponder_stockfish_engine.configure(stockfish_config)
        self.stockfish_analysis: Optional[list] = None
        self._stockfish_path = PATH_TO_STOCKFISH  # Store for potential engine restart

        # initialise move prob altering model
        self.move_prob_altering_model = AlterMoveProbNN()
        self.move_prob_altering_model.load_state_dict(torch.load(ALTER_MOVE_PROB_WEIGHTS_PTH, weights_only=True))
        self.move_prob_altering_model.eval()
        self.move_prob_altering_model.load_params_dict()
        
        # Getting opening books: repertoire_book (small, hand-curated,
        # consulted first) and opening_book (broad fallback) -- see
        # engine_components/opening_book.py.
        self.opening_book = chess.polyglot.open_reader(opening_book_path)
        self.repertoire_book = chess.polyglot.open_reader(repertoire_book_path)
        
        # lucas statistics for the current position
        self.lucas_analytics: dict[str, Any] = {
            "complexity": None,
            "win_prob": None,
            "eff_mob": None,
            "narrowness": None,
            "activity": None,
            }
        # Position "sharpness": win-probability spread across the top candidate
        # moves (see _compute_sharpness). Drives the "complicated position"
        # time-scaling in _get_time_taken. 0.25 is the neutral / critical point.
        self.sharpness: Optional[float] = None
        # The sharpness scan itself ({move_uci: win_chance} from the deep
        # narrow scan), kept for _chosen_move_wc_loss. None if the scan failed.
        self.sharpness_scan: Optional[dict] = None
        # Per-game character (see _sample_game_character): one draw per game
        # each, resampled at every game boundary, None until the first
        # update_info. game_pace_sf scales every base think time in
        # _get_time_taken; game_premove_sf scales every premove-search
        # probability in make_move.
        self.game_pace_sf: Optional[float] = None
        self.game_premove_sf: Optional[float] = None
        self.game_snap_gate: Optional[float] = None
        self.game_ponder_snap_sf: Optional[float] = None
        self.game_scramble_skill: Optional[float] = None
        self.game_scramble_fire_sf: Optional[float] = None
        self.game_ponder_width: Optional[int] = None
        self._last_seen_ply: Optional[int] = None
        # A bool to track whether we have updated analytics following updating info
        self.analytics_updated = False

        self.playing_level = playing_level
        self.mood = "confident"
        self.just_blundered: Optional[bool] = None
        
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
    
    def get_stockfish_move(self, board:Optional[chess.Board] = None, analysis=None, last_move_uci:Optional[str] = None, log:bool = True, ):
        """ Uses board information to get a move strictly from stockfish with no human
            filters. Very fast, and only called for in necessary situations (when in
            super low time)

            Returns a move_uci string of move made
        """
        return stockfish_move_logic.get_stockfish_move(self, board=board, analysis=analysis, last_move_uci=last_move_uci, log=log)

    def adjust_human_prob(self, move_dic, board : Optional[chess.Board] = None):
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

    def _alter_move_prob_nn(self, move_dic : dict, board:chess.Board, prev_board:Optional[chess.Board] = None, prev_prev_board:Optional[chess.Board] = None, log:bool = True):
        """ Given a move dictionary with move uci as key and value as their unaltered
            probabilities, we alter the probabilties to make moves stick out more
            (for example hanging pieces more likely to be moved etc).

            Returns an altered move_dic.
        """
        return human_move_logic.alter_move_prob_nn(self, move_dic, board, prev_board=prev_board, prev_prev_board=prev_prev_board, log=log)

    def _alter_move_probabilties(self, move_dic : dict, board:chess.Board, prev_board:Optional[chess.Board] = None, prev_prev_board:Optional[chess.Board] = None, log:bool = True):
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
            self.log += "Detected game phase is opening, consulting opening books for matching positions. \n"
            played_move_obj = opening_book.consult_book(self, self.current_board)
            if played_move_obj is not None:
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
        assert self.stockfish_analysis is not None  # populated by calculate_analytics, called before get_human_move
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
            noise_sd = self.eval_noise_scale*4*base_noise_sd/(target_time*(depth_considered+4))
            
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
    
    def _ponder_moves(self, board:chess.Board, move_ucis: list, search_width:int, prev_board: Optional[chess.Board] = None, log:bool = True, use_ponder:bool=False):
        """ We ponder on the given board position, and consider the moves given by the list
            of move_ucis. We again use human probabilities to narrow our search width.

            Returns a dictionary with:
                key: the move uci from move_ucis
                value: [move uci of response, eval of response]
            eval of response is given from the perspective of ourselves.
        """
        return ponderer.ponder_moves(self, board, move_ucis, search_width, prev_board=prev_board, log=log, use_ponder=use_ponder)

    def _recursive_ponder(self, board: chess.Board, move_uci : str, no_root_moves, depth: int, prev_board: Optional[chess.Board] = None, limit = None, use_ponder:bool= False):
        """ Recursive function for getting evaluations during pondering.

            Returns [move_uci eval, depth_considered]
            If limit is None, then depth_considered is None. If there are time limit
            constraints, then depth_considered is the depth that move_uci has been considered.
        """
        return ponderer.recursive_ponder(self, board, move_uci, no_root_moves, depth, prev_board=prev_board, limit=limit, use_ponder=use_ponder)

    def _re_evaluate(self, board:chess.Board, re_evaluate_moves: list, no_root_moves: int, depth:int = 0, prev_board:Optional[chess.Board] = None, limit=None, use_ponder:bool=False):
        """ Given a list of move_ucis, apply them to the current board and re_evaluate
            them using top human_moves only. This gives a non_accurate evaluation
            and simulates human foresight not being exhaustive.

            Returns a dictionary with key move_uci and value the evaluation (from our pov)
        """
        return ponderer.re_evaluate(self, board, re_evaluate_moves, no_root_moves, depth=depth, prev_board=prev_board, limit=limit, use_ponder=use_ponder)

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
        return premover.get_premove(self, board, side, takeback_only=takeback_only)

    def _own_time_or_none(self):
        """ Our latest clock reading, or None before the first update. """
        return ponderer.own_time_or_none(self)

    def stockfish_ponder(self, board:chess.Board, time_allowed : float, ponder_width:int, use_ponder:bool = False, root_moves:Optional[list]= None):
        """ Given a board position that is not our side turn, we ponder moves using
            stockfish and return a dictionary with has key board_fen and value uci.
            This method is much faster than self.ponder()

        """
        return ponderer.stockfish_ponder(self, board, time_allowed, ponder_width, use_ponder=use_ponder, root_moves=root_moves)

    def ponder(self, board: chess.Board, time_allowed : float, search_width : int, time_per_position : float = 0.1, prev_board:Optional[chess.Board] = None, ponder_width: Optional[int] = None, use_ponder:bool=False):
        """ Given a board position that is not our (side) turn, we ponder possible moves
            and return a dictionary which has key board_fen (so position only), and value uci in response.
            Time allowed represents how much we may ponder. Time per position is roughly
            the amount of time to form probabilies and alter them.

            Returns a dictionary with key board_fen and value move_uci response
        """
        return ponderer.ponder(self, board, time_allowed, search_width, time_per_position=time_per_position, prev_board=prev_board, ponder_width=ponder_width, use_ponder=use_ponder)

    def _check_opp_blunder(self):
        """ Check from last couple of positions whether opponent has made a blunder on the
            last move, and did so by hanging a valuable piece.

            Returns None. We only update the self.opp_just_blundered variable
        """
        return mood_manager.check_opp_blunder(self)
    
    def make_move(self, log:bool=True, seed:Optional[int]=None):
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
        return_dic: dict[str, Any] = {}
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
            assert obvious_move is not None  # obvious_move_found guarantees a real move uci
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
