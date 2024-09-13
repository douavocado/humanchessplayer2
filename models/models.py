# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 13:30:31 2024

@author: xusem
"""
import chess

import torch
from torch import nn

from board_encodings import moveto_position_list_one_hot, position_list_one_hot

chess.BaseBoard.from_position_list_one_hot = position_list_one_hot
chess.BaseBoard.to_position_list_one_hot = moveto_position_list_one_hot

class PieceSelectorNN(nn.Module):
    def __init__(self, input_channels=14):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(input_channels,18,3, padding=1)
        self.conv2 = nn.Conv2d(18,28,3, padding=1)
        self.conv3 = nn.Conv2d(28,42,3, padding=1)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(8*8*42, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, logits=True):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        # print(x)
        # x = self.softmax(x)
        # print(x , self.softmax)
        # we return the logits
        if logits == False:
            x = self.softmax(x)
        return x

class MoveScorer:
    """ Main model for getting human moves and their probabilities """
    
    def __init__(self, move_from_weights_path, move_to_weights_path):
        self.model_from = PieceSelectorNN()
        self.model_from.load_state_dict(torch.load(move_from_weights_path))
        self.model_from.eval()
        
        self.model_to = PieceSelectorNN(input_channels=16)
        self.model_to.load_state_dict(torch.load(move_to_weights_path))
        self.model_to.eval()

    def get_moves(self, board, san=True, top=5):
        from_input_ = torch.tensor(board.from_position_list_one_hot()).reshape(1,8,8,14).permute(0,3,1,2).float()
        from_output_ = self.model_from(from_input_, logits=False)[0]
        
        move_dic = {}
        for piece_type in range(1,7):
            for square in board.pieces(piece_type, board.turn):
                to_input_ = torch.tensor(board.to_position_list_one_hot(square)).reshape(1,8,8,16).permute(0,3,1,2).float()
                to_output_ = self.model_to(to_input_, logits=False)[0]
                
                possible_to_squares = [move.to_square for move in board.legal_moves if move.from_square == square]
                for to_square_mr in possible_to_squares:
                    prob = from_output_[chess.square_mirror(square)].item()*to_output_[chess.square_mirror(to_square_mr)].item()
                    
                    move = chess.Move(square, to_square_mr)
                    if move in board.legal_moves:
                        if san == True: # give san notation of move
                            move_dic[board.san(move)] = prob
                        else: # use uci of move, which is not board dependent
                            move_dic[move.uci()] = prob

        sorted_dic = {k: v for k, v in sorted(move_dic.items(), key=lambda item: item[1], reverse=True)}
        top_keys = list(sorted_dic.keys())[:top]
        # top_dic = {k:sorted_dic[k] for k in top_keys}
        return top_keys
    
    def get_prob_from_moves(self, board, moves, san=True):
        ''' Given board and a set of chess.Moves, return a dictionary with their
            respective human probabilities. '''
        from_input_ = torch.tensor(board.from_position_list_one_hot()).reshape(1,8,8,14).permute(0,3,1,2).float()
        from_output_ = self.model_from(from_input_, logits=False)[0]
        
        return_dic = {}
        for move in moves:
            start_sq = move.from_square
            to_sq = move.to_square
            to_input_ = torch.tensor(board.to_position_list_one_hot(start_sq)).reshape(1,8,8,16).permute(0,3,1,2).float()
            to_output_ = self.model_to(to_input_, logits=False)[0]
            
            prob = from_output_[chess.square_mirror(start_sq)].item()*to_output_[chess.square_mirror(to_sq)].item()
            
            if san == True:
                return_dic[board.san(move)] = prob
            else:
                return_dic[move.uci()] = prob
        return return_dic
    
    def get_move_dic(self, board, san=True, top=5):
        from_input_ = torch.tensor(board.from_position_list_one_hot()).reshape(1,8,8,14).permute(0,3,1,2).float()
        from_output_ = self.model_from(from_input_, logits=False)[0]
        
        # get all starting squares
        starting_sqs = set([move.from_square for move in board.legal_moves])
        
        move_dic = {}
        for square in starting_sqs:
            to_input_ = torch.tensor(board.to_position_list_one_hot(square)).reshape(1,8,8,16).permute(0,3,1,2).float()
            to_output_ = self.model_to(to_input_, logits=False)[0]
            
            possible_to_squares = set([move.to_square for move in board.legal_moves if move.from_square == square])
            for to_square_mr in possible_to_squares:
                prob = from_output_[chess.square_mirror(square)].item()*to_output_[chess.square_mirror(to_square_mr)].item()
                
                move = chess.Move(square, to_square_mr)
                if move in board.legal_moves:
                    if san == True: # give san notation of move
                        move_dic[board.san(move)] = prob
                    else: # use uci of move, which is not board dependent
                        move_dic[move.uci()] = prob
                elif board.piece_type_at(square) == chess.PAWN and chess.square_rank(to_square_mr) in [0,7]:
                    # add all promotion types
                    for promotion_type in range(2,6):
                        promotion_move = chess.Move(square, to_square_mr, promotion=promotion_type)
                        if san == True: # give san notation of move
                            move_dic[board.san(promotion_move)] = prob
                        else: # use uci of move, which is not board dependent
                            move_dic[promotion_move.uci()] = prob
                else:
                    print('Filtered move', move.uci(),' not a legal move')

        sorted_dic = {k: v for k, v in sorted(move_dic.items(), key=lambda item: item[1], reverse=True)}
        top_keys = list(sorted_dic.keys())[:top]
        top_dic = {k:sorted_dic[k] for k in top_keys}
        return top_keys, top_dic

class StockFishSelector:
    def __init__(self, path_to_engine):
        self.engine = chess.engine.SimpleEngine.popen_uci(path_to_engine)
        
    def get_moves(self, board, san=True, top=5, time=0.1):
        top_moves = []
        analysed_variations = self.engine.analyse(board, limit=chess.engine.Limit(time=time), multipv=top)
        for move_dic in analysed_variations:
            move = move_dic['pv'][0]
            if san == True:
                top_moves.append(board.san(move))
            else:
                top_moves.append(move.uci())
                
        return top_moves
    
    def get_move_eval_dic(self, board, san=True, top=5, time=0.1, root_moves=[]):
        top_move_dic = {}
        if len(root_moves) == 0:
            analysed_variations = self.engine.analyse(board, limit=chess.engine.Limit(time=time), multipv=top)
        else:
            analysed_variations = self.engine.analyse(board, limit=chess.engine.Limit(time=time), multipv=len(root_moves), root_moves=root_moves)
       
        if isinstance(analysed_variations, dict):
            analysed_variations = [analysed_variations]
        for move_dic in analysed_variations:
            move = move_dic['pv'][0]
            eval_ = move_dic['score'].pov(board.turn).score(mate_score=2500)
            #clipping
            eval_ = min(eval_, 2500)
            eval_ = max(eval_, -2500)
            if san == True:
                top_move_dic[board.san(move)] = eval_
            else:
                top_move_dic[move.uci()] = eval_
                
        return top_move_dic