import time
import random
import numpy as np
import chess
import chess.engine
import atexit

from .logger import Logger
from .state_manager import StateManager
from .analyzer import Analyzer, STOCKFISH # Import shared STOCKFISH instance
from .human_move_logic import HumanMoveLogic
from .mood_manager import MoodManager # Needed for noise adjustments potentially
from common.constants import PATH_TO_PONDER_STOCKFISH
from common.utils import extend_mate_score
from common.board_information import phase_of_game, calculate_threatened_levels

# Initialize a separate Stockfish engine instance specifically for pondering
try:
    PONDER_STOCKFISH = chess.engine.SimpleEngine.popen_uci(PATH_TO_PONDER_STOCKFISH)
except Exception as e:
    print(f"CRITICAL: Failed to initialize PONDER_STOCKFISH engine at {PATH_TO_PONDER_STOCKFISH}: {e}")
    PONDER_STOCKFISH = None

class Ponderer:
    def __init__(self, logger: Logger, state_manager: StateManager, analyzer: Analyzer, human_move_logic: HumanMoveLogic, mood_manager: MoodManager):
        self.logger = logger
        self.state_manager = state_manager
        self.analyzer = analyzer # Main analyzer for comparison evals etc.
        self.human_move_logic = human_move_logic
        self.mood_manager = mood_manager # Access mood for potential adjustments

        if PONDER_STOCKFISH is None:
             self.logger.add_log("CRITICAL: Ponder Stockfish engine failed to initialize. Pondering will use main engine or fail.\n")


    def _get_engine(self, use_ponder_engine: bool) -> chess.engine.SimpleEngine | None:
        """Selects the appropriate Stockfish engine instance."""
        if use_ponder_engine:
            if PONDER_STOCKFISH:
                return PONDER_STOCKFISH
            else:
                self.logger.add_log("WARNING: Ponder engine requested but not available. Falling back to main engine.\n")
                return STOCKFISH # Fallback to main engine
        else:
            return STOCKFISH

    def _ponder_moves_internal(self, board: chess.Board, move_ucis: list, search_width: int, prev_board: chess.Board = None, use_ponder_engine: bool = False, log: bool = True) -> dict:
        """
        Internal helper for pondering. Evaluates opponent responses to given moves.
        Returns a dictionary {move_uci: [response_uci, eval_after_response]}
        """
        if log:
            self.logger.add_log(f"Internally pondering moves {move_ucis} for FEN {board.fen()}\n")
        return_dic = {}
        engine = self._get_engine(use_ponder_engine)
        side = self.state_manager.get_side() # Our side

        if engine is None or side is None:
            if log:
                self.logger.add_log("ERROR: Cannot ponder moves, engine or side not available.\n")
            return {uci: [None, 0] for uci in move_ucis} # Return default failure state

        for move_uci in move_ucis:
            try:
                dummy_board = board.copy()
                move_obj = chess.Move.from_uci(move_uci)
                if move_obj not in dummy_board.legal_moves:
                    if log:
                        self.logger.add_log(f"WARNING: Skipping illegal move {move_uci} during internal ponder.\n")
                    return_dic[move_uci] = [None, 0] # Indicate failure for this move
                    continue

                dummy_board.push(move_obj)

                # Check if game ended after our move
                outcome = dummy_board.outcome()
                if outcome is not None:
                    winner = outcome.winner
                    eval_ = 0
                    if winner == side: eval_ = 2500
                    elif winner == (not side): eval_ = -2500
                    if log:
                        self.logger.add_log(f"Game ends after {move_uci}. Outcome: {outcome.termination}, Winner: {winner}, Eval: {eval_}\n")
                    return_dic[move_uci] = [None, eval_]
                    continue

                # Get opponent's likely responses using human logic (opponent's perspective)
                game_phase = phase_of_game(dummy_board)
                # Note: Human probs are calculated from the perspective of the current turn (opponent)
                un_altered_move_dic = self.human_move_logic.get_human_probabilities(dummy_board, game_phase, log=False)

                if not un_altered_move_dic or len(un_altered_move_dic) <= 2:
                    if log:
                        self.logger.add_log("Too few human moves found for opponent response, using legal moves.\n")
                    root_moves = list(dummy_board.legal_moves)
                else:
                    # Alter probabilities from opponent's perspective
                    altered_move_dic = self.human_move_logic.alter_move_probabilties(
                        un_altered_move_dic, 
                        dummy_board, 
                        log=False,
                        prev_board=board.copy(),  # Pass the previous board explicitly
                        prev_prev_board=prev_board if prev_board else None    # We often don't have prev_prev_board in pondering, pass None
                    )
                    human_move_ucis = list(altered_move_dic.keys())
                    root_moves = [chess.Move.from_uci(x) for x in human_move_ucis[:search_width]]

                if not root_moves:
                    if log:
                        self.logger.add_log(f"WARNING: No legal moves for opponent after {move_uci}. Setting eval to 0.\n")
                    return_dic[move_uci] = [None, 0] # Should be draw/stalemate?
                    continue


                # Analyze opponent's best response from the filtered set
                # Use a small time limit for responsiveness
                single_analysis = engine.analyse(dummy_board, chess.engine.Limit(time=0.02), root_moves=root_moves)

                if "pv" in single_analysis and single_analysis["pv"]:
                    response_uci = single_analysis["pv"][0].uci()
                    # Eval is from OUR perspective after their response
                    eval_after_response = extend_mate_score(single_analysis['score'].pov(side).score(mate_score=2500))
                    return_dic[move_uci] = [response_uci, eval_after_response]
                else:
                    if log:
                        self.logger.add_log(f"ERROR: No PV found in analysis for opponent response after {move_uci}. Analysis: {single_analysis}. Using random response.\n")
                    response_uci = random.choice(root_moves).uci()
                    # Use current eval as fallback (less accurate)
                    current_analysis = self.analyzer.get_stockfish_analysis()
                    eval_fallback = 0
                    if current_analysis:
                         eval_fallback = extend_mate_score(current_analysis[0]['score'].pov(side).score(mate_score=2500))
                    return_dic[move_uci] = [response_uci, eval_fallback]

            except (ValueError, chess.engine.EngineError, Exception) as e:
                if log:
                    self.logger.add_log(f"ERROR during internal ponder for move {move_uci}: {e}\n")
                return_dic[move_uci] = [None, 0] # Indicate failure

        if log:
            self.logger.add_log(f"Internal ponder results: {return_dic}\n")
        return return_dic


    def _recursive_ponder(self, board: chess.Board, move_uci: str, no_root_moves: int, depth: int, prev_board: chess.Board = None, limit: list = None, use_ponder_engine: bool = False) -> list:
        """
        Recursive function for multi-depth pondering.

        Args:
            limit: [evaluations_left, time_left, depth_considered, total_depth, comparison_eval] (optional)

        Returns:
            [evaluation, depth_considered]
        """
        start_time = time.time()
        side = self.state_manager.get_side()
        if side is None: return [0, 0] # Cannot evaluate without side

        # Base case: Ponder the immediate response
        ponder_results = self._ponder_moves_internal(board, [move_uci], no_root_moves, prev_board=prev_board, use_ponder_engine=use_ponder_engine, log=False)
        response_uci, current_eval = ponder_results.get(move_uci, [None, 0])
        end_time = time.time()
        time_taken = end_time - start_time
        # self.logger.add_log(f"Ponder depth {depth} for {move_uci} took {time_taken:.4f}s. Result: {response_uci}, Eval: {current_eval}\n")


        if depth <= 1 or response_uci is None:
            depth_considered = limit[2] if limit else 1
            # self.logger.add_log(f"Base case reached (depth={depth}, response={response_uci}). Final Eval: {current_eval}, Depth Considered: {depth_considered}\n")
            return [current_eval, depth_considered]

        # Recursive step with time/evaluation limits if provided
        if limit:
            evaluations_left, time_left, depth_considered, total_depth, comparison_eval = limit
            new_time_left = time_left - time_taken

            # self.logger.add_log(f"Recursive check: Time Left={new_time_left:.3f}s, Evals Left={evaluations_left-1}\n")

            if new_time_left <= 0.07: # Not enough time for another level
                self.logger.add_log("Time limit exceeded during recursion.\n")
                return [current_eval, depth_considered]

            # Forecast if we can finish remaining evaluations
            # Estimate time per eval based on current level's time_taken
            time_per_eval_est = time_taken if time_taken > 0 else 0.02 # Avoid division by zero
            forecast_evals_possible = new_time_left / time_per_eval_est
            new_evaluations_left = evaluations_left - 1 # Decrement evals needed

            next_board = board.copy()
            next_board.push_uci(move_uci)
            next_board.push_uci(response_uci) # Board state after opponent's response

            # --- Adaptive Depth Control ---
            next_depth = depth - 1 # Default next depth
            reason = "Continuing recursion."

            # Condition 1: Prune if line is much worse than comparison
            if current_eval < comparison_eval - 250:
                 reason = f"Pruning recursion: Eval {current_eval} << comparison {comparison_eval}."
                 return [current_eval, depth_considered] # Stop recursion

            # Condition 2: Reduce depth if running out of time
            elif forecast_evals_possible < new_evaluations_left - 1:
                 if depth > 2: # Only reduce if depth allows
                      next_depth = depth - 2
                      reason = f"Reducing depth: Forecast ({forecast_evals_possible:.1f}) < Needed ({new_evaluations_left-1})."
                 else: # Cannot reduce further, stop here
                      reason = f"Stopping recursion: Low forecast ({forecast_evals_possible:.1f}) vs Needed ({new_evaluations_left-1}) at depth {depth}."
                      return [current_eval, depth_considered]

            # Condition 3: Continue normally if line is promising or on schedule
            elif current_eval > comparison_eval + 100:
                 reason = f"Continuing recursion: Promising line (Eval {current_eval} > comparison {comparison_eval})."
                 # Keep next_depth = depth - 1

            else: # On schedule, not particularly promising or bad
                 reason = "Continuing recursion: On schedule."
                 # Keep next_depth = depth - 1

            self.logger.add_log(reason + f" Next Depth: {next_depth}\n")
            return self._recursive_ponder(
                next_board, response_uci, no_root_moves, next_depth,
                prev_board=board.copy(), # Pass the board before opponent's response
                limit=[new_evaluations_left, new_time_left, depth_considered + 1, total_depth, comparison_eval],
                use_ponder_engine=use_ponder_engine
            )

        else: # No limits, simple recursion
            next_board = board.copy()
            next_board.push_uci(move_uci)
            next_board.push_uci(response_uci)
            return self._recursive_ponder(next_board, response_uci, no_root_moves, depth - 1, prev_board=board.copy(), use_ponder_engine=use_ponder_engine)


    def re_evaluate_moves(self, board: chess.Board, moves_to_re_evaluate: list, no_root_moves: int, depth: int = 1, prev_board: chess.Board = None, limit: list = None, use_ponder_engine: bool = False, log: bool = True) -> dict:
        """
        Re-evaluates a list of candidate moves using recursive pondering.

        Args:
            limit: [total_evaluations, time_limit] (optional)

        Returns:
            Dictionary {move_uci: [evaluation, depth_considered]}
        """
        if log:
            self.logger.add_log(f"Re-evaluating moves: {moves_to_re_evaluate} with depth {depth}, RootMovesWidth={no_root_moves}\n")
            if limit: self.logger.add_log(f"Limits: Evals={limit[0]}, Time={limit[1]:.3f}s\n")

        random.shuffle(moves_to_re_evaluate) # Avoid bias from move order
        return_dic = {}
        comparison_eval = -9999 # Track best eval found so far for pruning

        if limit:
            evaluations_budget, time_budget = limit
            start_time = time.time()
            time_left = time_budget
            evaluations_left = evaluations_budget

            for move_uci in moves_to_re_evaluate:
                if time_left <= 0.07 or evaluations_left <= 0:
                    if log:
                        self.logger.add_log(f"Stopping re-evaluation early for {move_uci}: Time Left={time_left:.3f}s, Evals Left={evaluations_left}\n")
                    return_dic[move_uci] = [None, 0] # Indicate not evaluated
                    continue

                # Pass limits to recursive function
                eval_result, depth_considered = self._recursive_ponder(
                    board, move_uci, no_root_moves, depth, prev_board=prev_board,
                    limit=[evaluations_left, time_left, 1, depth, comparison_eval],
                    use_ponder_engine=use_ponder_engine
                )
                return_dic[move_uci] = [eval_result, depth_considered]

                if eval_result is not None:
                    comparison_eval = max(eval_result, comparison_eval)

                # Update remaining resources (approximate for evaluations)
                evaluations_left -= depth_considered # Reduce by actual depth considered
                time_now = time.time()
                time_left = time_budget - (time_now - start_time)

        else: # No limits
            for move_uci in moves_to_re_evaluate:
                eval_result, _ = self._recursive_ponder(
                    board, move_uci, no_root_moves, depth, prev_board=prev_board, use_ponder_engine=use_ponder_engine, log=log
                )
                # Depth considered is just the requested depth when no limits
                return_dic[move_uci] = [eval_result, depth]

        if log:
            self.logger.add_log(f"Re-evaluation results: {return_dic}\n")
        return return_dic


    def stockfish_ponder(self, board: chess.Board, time_allowed: float, ponder_width: int, use_ponder_engine: bool = False, root_moves: list = None) -> dict | None:
        """ Ponders using only Stockfish analysis (faster). """
        self.logger.add_log(f"Stockfish pondering FEN {board.fen()} (Time: {time_allowed:.3f}s, Width: {ponder_width})\n")

        if board.outcome() is not None:
            self.logger.add_log("Board position is game over. Cannot ponder.\n")
            return None

        engine = self._get_engine(use_ponder_engine)
        if engine is None:
             self.logger.add_log("ERROR: Stockfish engine not available for stockfish_ponder.\n")
             return None

        if root_moves is None:
            root_moves = list(board.legal_moves)
            if not root_moves:
                 self.logger.add_log("No legal moves to ponder.\n")
                 return None

        try:
            analysis_object = engine.analyse(board, limit=chess.engine.Limit(time=time_allowed), multipv=ponder_width, root_moves=root_moves)
            if isinstance(analysis_object, dict): analysis_object = [analysis_object] # Ensure list

            return_dic = {}
            for line in analysis_object:
                if "pv" in line and len(line["pv"]) >= 2:
                    opp_move = line["pv"][0]
                    response_move_uci = line["pv"][1].uci()
                    # Key is the board state *after* opponent's move
                    dummy_board = board.copy()
                    dummy_board.push(opp_move)
                    board_fen_after_opp = dummy_board.board_fen() # Use board_fen for key
                    return_dic[board_fen_after_opp] = response_move_uci
                elif "pv" in line and len(line["pv"]) == 1:
                     self.logger.add_log(f"Ponder line for {line['pv'][0].uci()} has no response (depth 1 or game end).\n")
                else:
                     self.logger.add_log(f"Skipping invalid ponder line: {line}\n")


            if not return_dic:
                self.logger.add_log("Stockfish ponder did not yield any valid response lines.\n")
                return None

            self.logger.add_log(f"Stockfish ponder results: {return_dic}\n")
            return return_dic

        except (chess.engine.EngineError, Exception) as e:
             self.logger.add_log(f"ERROR during stockfish_ponder analysis: {e}\n")
             return None


    def human_ponder(self, board: chess.Board, time_allowed: float, search_width: int, time_per_position: float = 0.1, prev_board: chess.Board = None, ponder_width: int = None, use_ponder_engine: bool = False) -> dict | None:
        """ Ponders using human-like evaluation and recursive analysis. """
        self.logger.add_log(f"Human pondering FEN {board.fen()} (Time: {time_allowed:.3f}s, SearchWidth: {search_width})\n")

        if board.outcome() is not None:
            self.logger.add_log("Board position is game over. Cannot ponder.\n")
            return None

        side = self.state_manager.get_side()
        engine = self._get_engine(use_ponder_engine) # Engine for initial opponent move selection
        if engine is None or side is None:
             self.logger.add_log("ERROR: Engine or side not available for human_ponder.\n")
             return None


        # --- Determine Ponder Depth and Width ---
        variations_allowed = max(1, int(time_allowed / time_per_position))
        initial_time = self.state_manager.get_info("self_initial_time", 60)

        if ponder_width is None:
            # Dynamically determine ponder_width based on time control
            max_ponder_no = 3 if initial_time > 180 else 2
            ponder_width = 1 # Default width
            ponder_depth = 0
            for width_candidate in range(max_ponder_no, 0, -1):
                depth_candidate = round(variations_allowed / (width_candidate * search_width))
                if depth_candidate >= 2: # Require at least depth 2 for meaningful ponder
                    ponder_width = width_candidate
                    ponder_depth = depth_candidate
                    break
            if ponder_depth == 0: # If no combination yielded depth >= 2, use minimal settings
                 ponder_depth = 1
                 ponder_width = 1
                 self.logger.add_log("Could not achieve depth >= 2, using minimal ponder settings (Width=1, Depth=1).\n")
        else:
            # Use provided ponder_width, calculate depth
            ponder_depth = round(variations_allowed / (ponder_width * search_width))
            ponder_depth = max(ponder_depth, 1) # Ensure at least depth 1

        self.logger.add_log(f"Ponder settings: Width={ponder_width}, Depth={ponder_depth}, Variations Budget={variations_allowed}\n")


        # --- Get Candidate Opponent Moves ---
        try:
            # Get top N opponent moves according to the engine
            analysis_object = engine.analyse(board, limit=chess.engine.Limit(time=0.02), multipv=ponder_width)
            if isinstance(analysis_object, dict): analysis_object = [analysis_object]

            if not analysis_object or "pv" not in analysis_object[0] or not analysis_object[0]["pv"]:
                self.logger.add_log("ERROR: Could not get candidate opponent moves from engine.\n")
                return None

            opp_moves_to_consider = [entry["pv"][0].uci() for entry in analysis_object if "pv" in entry and entry["pv"]]
            san_opp_moves = [board.san(chess.Move.from_uci(m)) for m in opp_moves_to_consider]
            self.logger.add_log(f"Opponent moves to ponder: {san_opp_moves}\n")

        except (chess.engine.EngineError, Exception) as e:
             self.logger.add_log(f"ERROR getting opponent moves for ponder: {e}\n")
             return None


        # --- Re-evaluate Responses for Each Opponent Move ---
        return_dic = {}
        san_return_dic = {} # For logging

        # Noise configuration based on game phase (applied during re-evaluation)
        noise_config = {"opening": 0.8, "midgame": 1.2, "endgame": 0.3}
        playing_level = self.state_manager.get_info("playing_level", 6) # Get playing level if available

        # Calculate limits for re_evaluate_moves
        # Distribute time/evals somewhat evenly, but maybe give more to top move?
        # Simple even distribution for now:
        num_opp_moves = len(opp_moves_to_consider)
        if num_opp_moves == 0: return None

        eval_budget_per_move = max(1, ponder_depth * search_width) # Budget based on target depth/width
        time_budget_per_move = time_allowed / (num_opp_moves * 1.1) # Distribute time with slight buffer

        for opp_move_uci in opp_moves_to_consider:
            try:
                dummy_board_after_opp = board.copy()
                opp_move_obj = chess.Move.from_uci(opp_move_uci)
                if opp_move_obj not in dummy_board_after_opp.legal_moves: continue # Skip illegal
                dummy_board_after_opp.push(opp_move_obj)

                if dummy_board_after_opp.outcome() is not None: continue # Skip if game over

                board_fen_key = dummy_board_after_opp.board_fen()
                game_phase = phase_of_game(dummy_board_after_opp)

                # Get our likely responses using human logic
                top_human_move_dic = self.human_move_logic.get_human_probabilities(dummy_board_after_opp, game_phase, log=False)

                if not top_human_move_dic or len(top_human_move_dic) <= 2:
                    our_candidate_moves = [m.uci() for m in dummy_board_after_opp.legal_moves]
                else:
                    altered_move_dic = self.human_move_logic.alter_move_probabilties(
                        top_human_move_dic, 
                        dummy_board_after_opp, 
                        log=False,
                        prev_board=board.copy(),  # The board before the opponent's move
                        prev_prev_board=prev_board if prev_board else None  # The board before that if available
                    )
                    our_candidate_moves = list(altered_move_dic.keys())[:search_width] # Top N altered moves

                if not our_candidate_moves: continue # No moves for us to make

                # Re-evaluate these candidate moves
                re_eval_limit = [eval_budget_per_move, time_budget_per_move]
                re_evaluate_dic = self.re_evaluate_moves(
                    dummy_board_after_opp, our_candidate_moves, search_width,
                    depth=ponder_depth, prev_board=board.copy(), # Pass board before opp move
                    limit=re_eval_limit, use_ponder_engine=use_ponder_engine, log=False
                )

                # Apply noise and select best response based on noisy eval
                noisy_eval_dic = {}
                noise_phase_factor = noise_config.get(game_phase, 1.0)
                for move_uci, (eval_result, depth_considered) in re_evaluate_dic.items():
                    if eval_result is None: # Move wasn't evaluated due to limits
                         # Get fallback eval (penalized)
                         fallback_engine = self._get_engine(use_ponder_engine)
                         if fallback_engine:
                              try:
                                   fb_analysis = fallback_engine.analyse(dummy_board_after_opp, chess.engine.Limit(time=0.01), root_moves=[chess.Move.from_uci(move_uci)])
                                   eval_result = extend_mate_score(fb_analysis['score'].pov(side).score(mate_score=2500)) - 100 # Penalty
                              except: eval_result = -100 # Default penalty
                         else: eval_result = -100
                         depth_considered = 0 # Mark as not properly evaluated
                         self.logger.add_log(f"Move {move_uci} got fallback eval {eval_result} due to limits.\n")


                    # Apply noise based on eval, time, depth
                    base_noise_sd = 40 * (np.tanh(eval_result / (playing_level * 50)))**2 + 20
                    noise_sd = 4 * base_noise_sd / (max(time_allowed, 0.1) * (depth_considered + 4))
                    noise = np.random.randn() * noise_sd * noise_phase_factor
                    noisy_eval = eval_result + noise

                    # Bonus for captures (especially of hanging pieces)
                    move_obj = chess.Move.from_uci(move_uci)
                    if dummy_board_after_opp.is_capture(move_obj):
                        capture_bonus = 40 * calculate_threatened_levels(move_obj.to_square, dummy_board_after_opp)
                        noisy_eval += capture_bonus

                    noisy_eval_dic[move_uci] = noisy_eval

                if not noisy_eval_dic: continue # Skip if no moves could be evaluated

                best_response_uci = max(noisy_eval_dic, key=noisy_eval_dic.get)
                return_dic[board_fen_key] = best_response_uci
                san_return_dic[board_fen_key] = dummy_board_after_opp.san(chess.Move.from_uci(best_response_uci))

            except (ValueError, chess.engine.EngineError, Exception) as e:
                 self.logger.add_log(f"ERROR during human_ponder loop for opp_move {opp_move_uci}: {e}\n")


        self.logger.add_log(f"Human ponder results (FEN -> Response SAN): {san_return_dic}\n")
        return return_dic if return_dic else None


    def close_ponder_engine(self):
        """Safely closes the dedicated ponder Stockfish engine."""
        if PONDER_STOCKFISH:
            try:
                PONDER_STOCKFISH.quit()
                self.logger.add_log("Ponder Stockfish engine closed successfully.\n")
            except chess.engine.EngineTerminatedError:
                 self.logger.add_log("Ponder Stockfish engine already terminated.\n")
            except Exception as e:
                 self.logger.add_log(f"ERROR: Failed to close Ponder Stockfish engine: {e}\n")

# Safer atexit handler that checks for None logger
def safe_close_ponder_engine():
    try:
        if PONDER_STOCKFISH:
            PONDER_STOCKFISH.quit()
    except Exception:
        pass  # Silently handle errors during shutdown

atexit.register(safe_close_ponder_engine)