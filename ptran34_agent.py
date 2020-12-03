#!/usr/bin/env python3

"""
File Name:      training_agent.py
Authors:        Phong Tran
Date:           12/1/2020

Description:    My agent that just uses the PyTorch AlphaGo Zero implementation. No stockfish.
Source:         Adapted from recon-chess (https://pypi.org/project/reconchess/)
"""

# %%
from player import Player
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import entropy
import chess
import chess.engine
from typing import List, Dict, Tuple
import time
from scipy.special import softmax
import random

# %% Set PyTorch device
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu") 

# %% Helper functions
def game_is_over(board: chess.Board):
    """
    Check if the board state is terminal, and returns win or loss based on the current
    POV decided by board.turn. That means you have to be careful that board.turn is on the correct turn.
    """
    pov_color = board.turn
    enemy_color = not board.turn

    p1_king = board.king(pov_color)
    p2_king = board.king(enemy_color)

    if p1_king and p2_king:
        return False, 0
    if p1_king is None:
        return True, -1
    elif p2_king is None:
        return True, 1
    else:
        # p1 and p2 are both none? Something bizarre happened
        print('?????????')
        return False, 0

def board_to_torch(board: chess.Board, pov: bool, ply: int):
    """
    Convert a board to a bitboard representation.
    0-5: White's bitboards for pawn, knight, bishop, rook, queen king
    6-11: Black's bitboards for pawn, knight, bishop, rook, queen, king
    12: binary plane set to 1 or 0 based on whose turn it is.
    """
    torch_board = torch.zeros((6 + 6 + 2, 8, 8))
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    piece_colors = [chess.BLACK, chess.WHITE]
    for piece_color in piece_colors:
        for piece_type in piece_types:
            squares = list(board.pieces(piece_type, piece_color))
            
            plane_offset = 6 if piece_color == chess.BLACK else 0
            # Internally, chess.PAWN, knight, etc. are encoded as [1, 2, 3, 4, 5, 6]
            p = (piece_type - 1) + plane_offset # target plane
            for square in squares:
                col = chess.square_file(square)
                row = chess.square_rank(square)
                torch_board[p, row, col] = 1
    
    # Set player color channel
    if pov == chess.WHITE:
        torch_board[-2, :, :] = 1
    else:
        torch_board[-2, :, :] = 0
    
    # Set ply number channel
    torch_board[-1, :, :] = ply

    return torch_board.unsqueeze(0)

def board_to_numpy(board: chess.Board):
    """
    pieces = [
        None,
        chess.Piece(chess.PAWN, chess.WHITE),
        chess.Piece(chess.KNIGHT, chess.WHITE),
        chess.Piece(chess.BISHOP, chess.WHITE),
        chess.Piece(chess.ROOK, chess.WHITE),
        chess.Piece(chess.QUEEN, chess.WHITE),
        chess.Piece(chess.KING, chess.WHITE),
        chess.Piece(chess.PAWN, chess.BLACK),
        chess.Piece(chess.KNIGHT, chess.BLACK),
        chess.Piece(chess.BISHOP, chess.BLACK),
        chess.Piece(chess.ROOK, chess.BLACK),
        chess.Piece(chess.QUEEN, chess.BLACK),
        chess.Piece(chess.KING, chess.BLACK),
    ]"""
    numpy_board = np.zeros((13, 8, 8)) # None, white pieces, black pieces
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    piece_colors = [chess.BLACK, chess.WHITE]
    for piece_color in piece_colors:
        for piece_type in piece_types:
            squares = list(board.pieces(piece_type, piece_color))
            
            plane_offset = 7 if piece_color == chess.BLACK else 1
            # Internally, chess.PAWN, knight, etc. are encoded as [1, 2, 3, 4, 5, 6]
            p = (piece_type - 1) + plane_offset # target plane
            for square in squares:
                col = chess.square_file(square)
                row = chess.square_rank(square)
                numpy_board[p, row, col] = 1
    # Define empty squares
    filled_mask = np.clip(numpy_board.sum(axis=0), 0, 1)
    numpy_board[0, :, :] = 1 - filled_mask
    return numpy_board

def map_move_plane(piece: chess.Piece, move: chess.Move):
    # The knight is the most complex one. Here we go...
    x0, y0 = chess.square_file(move.from_square), chess.square_rank(move.from_square)
    x1, y1 = chess.square_file(move.to_square), chess.square_rank(move.to_square)
    if piece.piece_type == chess.KNIGHT:

        if x1 - x0 == 1 and y1 - y0 == 2:
            return 56
        elif x1 - x0 == 2 and y1 - y0 == 1:
            return 57
        elif x1 - x0 == 2 and y1 - y0 == -1:
            return 58
        elif x1 - x0 == 1 and y1 - y0 == -2:
            return 59
        elif x1 - x0 == -1 and y1 - y0 == -2:
            return 60
        elif x1 - x0 == -2 and y1 - y0 == -1:
            return 61
        elif x1 - x0 == -2 and y1 - y0 == 1:
            return 62
        elif x1 - x0 == -1 and y1 - y0 == 2:
            return 63
        else:
            return 999 # Did I mess up somewhere?
    else:
        dist = chess.square_distance(move.from_square, move.to_square)
        offset = 8 * (dist - 1)
        if x0 == x1 and y0 < y1:   # North
            return 0 + offset
        elif x0 < x1 and y0 < y1:  # NE
            return 1 + offset
        elif x0 < x1 and y0 == y1: # E
            return 2 + offset
        elif x0 < x1 and y0 > y1:  # SE
            return 3 + offset
        elif x0 == x1 and y0 > y1: # south
            return 4 + offset
        elif x0 > x1 and y0 > y1:  # SW
            return 5 + offset
        elif x0 > x1 and y0 == y1: # W
            return 6 + offset
        elif x0 > x1 and y0 < y1:  # NW
            return 7 + offset

def move_mask(board: chess.Board, color: bool):
    """
    The output of the neural network is an 64x8x8 tensor.
    (The paper had 9 additional dimensions for underpromotions, but I'm just going to assume
    everything promotes to a queen.)
    The cells of each 8x8 plane denote where to pick up a piece,
    the dim (of the 73) denote what kind of move to make.
    The first 56 planes encode a queen move, in each direction (8) and each distance (1 through 7 squares)
    0  - 7:  Queen move, 1 step,   {N, NE, E, SE, S, SW, W, NW}
    8  - 15: Queen move, 2 steps,  {N, NE, E, SE, S, SW, W, NW}
    16 - 23: Queen move, 3 steps,  {N, NE, E, SE, S, SW, W, NW}
    24 - 31: Queen move, 4 steps,  {N, NE, E, SE, S, SW, W, NW}
    32 - 39: Queen move, 5 steps,  {N, NE, E, SE, S, SW, W, NW}
    40 - 47: Queen move, 6 steps,  {N, NE, E, SE, S, SW, W, NW}
    48 - 55: Queen move, 7 steps,  {N, NE, E, SE, S, SW, W, NW}
    56: Knight move, 1 o'clock
    57: Knight move, 2 o'clock
    58: Knight move, 4 o'clock
    59: Knight move, 5 o'clock
    60: Knight move, 7 o'clock
    61: Knight move, 8 o'clock
    62: Knight move, 10 o'clock
    63: Knight move, 11 o'clock
    """
    # Make sure the board is set to the correct color
    board.turn = color
    # First, get the list of legal moves.
    moves = list(board.pseudo_legal_moves)
    # Get the piece type corresponding to each move
    pieces = [board.piece_at(move.from_square) for move in moves]
    # Get the corresponding plane for each move and piece combination
    piece_planes = [map_move_plane(piece, move) for piece, move in zip(pieces, moves)]

    mask = torch.zeros((64, 8, 8)) # 64x8x8
    for move, plane in zip(moves, piece_planes):
        x0, y0 = chess.square_file(move.from_square), chess.square_rank(move.from_square)
        mask[plane, y0, x0] = 1
    return mask

def dim_to_move(dim: int, start: chess.Square) -> chess.Move:
    x0, y0 = chess.square_file(start), chess.square_rank(start)
    x1, y1 = x0, y0
    if dim == 56:
        x1 = x0 + 1
        y1 = y0 + 2
    elif dim == 57:
        x1 = x0 + 2
        y1 = y0 + 1
    elif dim == 58:
        x1 = x0 + 2
        y1 = y0 - 1
    elif dim == 59:
        x1 = x0 + 1
        y1 = y0 - 2
    elif dim == 60:
        x1 = x0 - 1
        y1 = y0 - 2
    elif dim == 61:
        x1 = x0 - 2
        y1 = y0 - 1
    elif dim == 62:
        x1 = x0 - 2
        y1 = y0 + 1
    elif dim == 63:
        x1 = x0 - 1
        y1 = y0 + 2
    else:
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        dist = dim // 8 + 1
        direction = directions[dim % 8]

        if direction == 'N':
            x1 = x0
            y1 = y0 + dist
        elif direction == 'NE':
            x1 = x0 + dist
            y1 = y0 + dist
        elif direction == 'E':
            x1 = x0 + dist
            y1 = y0
        elif direction == 'SE':
            x1 = x0 + dist
            y1 = y0 - dist
        elif direction == 'S':
            x1 = x0
            y1 = y0 - dist
        elif direction == 'SW':
            x1 = x0 - dist
            y1 = y0 - dist
        elif direction == 'W':
            x1 = x0 - dist
            y1 = y0
        elif direction == 'NW':
            x1 = x0 - dist
            y1 = y0 + dist
    
    if x1 < 0 or x1 > 7 or y1 < 0 or y1 > 7:
        return None # Invalid move
    else:
        target_square = chess.square(x1, y1)
        return chess.Move(start, target_square)

def idx_to_move(idx: int) -> chess.Move:
    """The neural network has to be flattened to 4096 choose a move. Use this to get the action idx
    that corresponds to a chess.Move"""
    orig_dim = idx // 64 # 8 x 8
    norm_idx = idx - (idx // 64 * 64)
    row = norm_idx // 8
    col = norm_idx % 8
    start = chess.square(col, row)
    move = dim_to_move(orig_dim, start)
    return move


def numpy_boards_PMF(boards: List[chess.Board]):
    """
    Get the average board, assuming a uniform distribution.
    """
    avg_board = np.zeros((13, 8, 8))
    for board in boards:
        numpy_board = board_to_numpy(board)
        avg_board += numpy_board
    avg_board /= len(boards)
    return avg_board


def PMF_to_board(board_PMF: np.ndarray, enemy_count: int, pov: bool,
                 enemy_captured = None, enemy_captured_at = None):
    start_time = time.time()
    
    PMF = np.copy(board_PMF)

    if np.isnan(PMF).any():
        print("Function was given a PMF with Nans in it")
        print(PMF)
        # print(1 / 0)

    output_board = chess.Board(None)
    chosen_probs = np.ones((8, 8), dtype=np.float)
    enemy_color = not pov

    pieces = [
        None,
        chess.Piece(chess.PAWN, chess.WHITE),
        chess.Piece(chess.KNIGHT, chess.WHITE),
        chess.Piece(chess.BISHOP, chess.WHITE),
        chess.Piece(chess.ROOK, chess.WHITE),
        chess.Piece(chess.QUEEN, chess.WHITE),
        chess.Piece(chess.KING, chess.WHITE),
        chess.Piece(chess.PAWN, chess.BLACK),
        chess.Piece(chess.KNIGHT, chess.BLACK),
        chess.Piece(chess.BISHOP, chess.BLACK),
        chess.Piece(chess.ROOK, chess.BLACK),
        chess.Piece(chess.QUEEN, chess.BLACK),
        chess.Piece(chess.KING, chess.BLACK),
    ]

    # Check if King probabilities are valid.
    white_king_sum = PMF[6, :, :].sum()
    black_king_sum = PMF[12,:, :].sum()
    if white_king_sum < 0.1:
        # print("Had to randomly place the white king somewhere.")
        # Take some probability from the empty squares
        for y in range(8):
            for x in range(8):
                value = PMF[0, y, x]
                PMF[0, y, x] = value * 0.75
                PMF[6, y, x] = value * 0.25
    if black_king_sum < 0.1:
        # Take some probability from the empty squares
        # print("Had to randomly place the black king somewhere.")
        for y in range(8):
            for x in range(8):
                value = PMF[0, y, x]
                PMF[0, y, x] = value * 0.75
                PMF[12, y, x] = value * 0.25

    # First place the king, because that's always a constraint
    while output_board.king(chess.WHITE) is None:
        square = np.random.choice(64)
        y, x = chess.square_rank(square), chess.square_file(square)
        p_w = PMF[6, y, x]
        place_it = np.random.choice(2, p=[1-p_w, p_w])
        if place_it == 1:
            output_board.set_piece_at(square, chess.Piece(chess.KING, chess.WHITE))
            if chess.WHITE == enemy_color:
                chosen_probs[y, x] = p_w

    while output_board.king(chess.BLACK) is None:
        square = np.random.choice(64)
        y, x = chess.square_rank(square), chess.square_file(square)
        p_b = PMF[12, y, x]
        place_it = np.random.choice(2, p=[1-p_b, p_b])
        if place_it == 1:
            output_board.set_piece_at(square, chess.Piece(chess.KING, chess.BLACK))
            if chess.BLACK == enemy_color:
                chosen_probs[y, x] = p_b


    # If we captured an enemy piece, then set the probability that was there to always
    # select one of the enemy pieces.
    if enemy_captured:
        y, x = chess.square_rank(enemy_captured_at), chess.square_file(enemy_captured_at)
        PMF = np.nan_to_num(PMF)
        PMF[:, y, x] += 1e-6
        PMF[0, y, x] = 0 # can't be an empty square
        # Edge case: somehow the probability for any enemy piece here was 0...
        if PMF[:, y, x].sum() == 0:
            if pov == chess.WHITE:
                PMF[7:, y, x] = 1
                PMF[1:7, y, x] = 0
            else:
                PMF[1:7, y, x] = 1
                PMF[7:, y, x] = 0
        if np.isnan(PMF[:, y, x]).any():
            print("Encountered NaN in PMF at this square")
            print(y, x, PMF)
            # print(1/0)
        PMF[:, y, x] = PMF[:, y, x] / PMF[:, y, x].sum() # normalize


    # Place all our own pieces
    for square in np.arange(64):
        y, x = chess.square_rank(square), chess.square_file(square)
        if PMF[:, y, x].sum() == 0:
            PMF[:, y, x] = 1
        if np.isnan(PMF).any():
            print("Invalid PMF\n", PMF)
            # print( 1/ 0 /0)
        square_PMF = PMF[:, y, x] / PMF[:, y, x].sum()
        piece_ind = np.random.choice(13, p=square_PMF)
        if pov == chess.WHITE and piece_ind not in [0, 1, 2, 3, 4, 5, 6]:
            continue
        elif pov == chess.BLACK and piece_ind not in [0, 7, 8, 9, 10, 11, 12]:
            continue
        prob = square_PMF[piece_ind]
        chosen_probs[y, x] = prob
        piece = pieces[piece_ind]
        output_board.set_piece_at(square, piece)
    
    if pov == chess.WHITE:
        for square in np.arange(64):
            y, x = chess.square_rank(square), chess.square_file(square)

    enemies_placed = 1
    iter = 0
    while enemies_placed < enemy_count:
        iter += 1
        square = np.random.choice(64)
        y, x = chess.square_rank(square), chess.square_file(square)
        square_PMF = PMF[:, y, x] / PMF[:, y, x].sum()
        piece_ind = np.random.choice(13, p=square_PMF)
        prob = square_PMF[piece_ind]
        chosen_probs[y, x] = prob
        piece = pieces[piece_ind]
        output_board.set_piece_at(square, piece)

        enemies_placed = len(output_board.pieces(chess.PAWN, enemy_color)) + len(output_board.pieces(chess.KNIGHT, enemy_color)) + len(output_board.pieces(chess.BISHOP, enemy_color)) + len(output_board.pieces(chess.ROOK, enemy_color)) + len(output_board.pieces(chess.QUEEN, enemy_color)) + len(output_board.pieces(chess.KING, enemy_color))
        if iter >= 4096:
            break
    

    posterior = np.prod(chosen_probs)
    output_board.turn = pov
    return output_board, posterior


# %% MCTS
class StochasticMCTS:
    def __init__(self, nnet: torch.nn.Module, engine: chess.engine.EngineProtocol, exploration_constant=(1/np.sqrt(2))) -> None:
        # Parameters
        self.exploration_constant = exploration_constant
        self.engine = engine
        self.nnet = nnet
        
        # Set of fen values to track if we've visited the state before.
        self.visited = set()

        # P[s, a], initial estimate of taking an action from s according to the NNet policy.
        self.P = {} # type: Dict[str, torch.Tensor]

        # Q[s, a], expected rewards for each action at each state.
        # Used in computing the upper confidence bound (UCB).
        self.Q = {} # type: Dict[str, np.ndarray]

        # N[s, a], the visit counter. Once the iterations are over,
        # taking the distribution at N[s] gives us the improved policy pi.
        self.N = {} # type: Dict[str, np.ndarray]

        # Track what the root boards were and their weights so we can use them to find the average improved policy
        # self.root_boards = []
        # self.root_boards_weight = []
        self.root_boards = {} # type: Dict[str, float]

        # It's possible we ran out of time before MCTS could find Nsa, the precursor to the improved policy.
        # In that case, when we try to calculate the improved policy, we'll just have to rely on P[s] instead.
        self.updated_Nsa = False
        
        # Grab these later to create neural network training set
        self.potentials = None # type: List[Tuple[chess.Board, float]]

        return
    
    def timed_search(self, board_PMF, ply: int, enemy_count: int, color: bool, time_limit):
        start_time = time.time()
        
        iter = 0
        potentials = [PMF_to_board(board_PMF, enemy_count, color) for i in range(50)]
        potentials.sort(key=lambda x: x[1], reverse=True)
        self.potentials = potentials
        probs = np.array([p[1] for p in potentials])
        probs = probs / probs.sum()

        # for i in range(len(potentials)):
        #     board, prob = potentials[ind]
        #     self.root_boards[board.fen()] = prob
        #     # print('Turn?', board.turn)
        #     self.search(board.copy())

        for i in range(10):
            try:
                ind = np.random.choice(len(potentials), p=probs)
                board, prob = potentials[ind]
                self.root_boards[board.fen()] = prob
                # print('Turn?', board.turn)
                self.search(board.copy(), ply)
            except ValueError:
                print(np.array([p[1] for p in potentials]))
                print('enemy count', enemy_count)
                # print(1 / 0 / 0)

        while iter == 0 or time.time() - start_time < time_limit:
            iter += 1
            # print('Iteration...', iter)
            # Choose a board to use for the root of MCTS search
            try:
                ind = np.random.choice(len(potentials), p=probs)
                board, prob = potentials[ind]
                self.root_boards[board.fen()] = prob
                # print('Turn?', board.turn)
                self.search(board.copy(), ply)
            except ValueError:
                print(np.array([p[1] for p in potentials]))
                print('enemy count', enemy_count)
                # print(1 / 0 / 0)
    
    def improved_policy(self) -> np.ndarray:
        """Returns the improved policy generated after running MCTS"""
        avg_pi = np.zeros(4096)
        for s, prob in self.root_boards.items():
            if self.P.get(s) is None:
                self.P[s] = torch.ones(4096, dtype=torch.float) / 4096
                board = chess.Board(s)
                mask = move_mask(board.copy(), board.turn)
                mask = mask.view(-1)
                self.P[s] *= mask
                self.P[s] = self.P[s] / self.P[s].sum()
                if torch.isnan(self.P[s]).any():
                    print("The policy output has NaNs in it??")
                    print(board.fen())
                    # print(move_mask(board.copy(), board.turn))
            Ns = self.N.get(s, self.P[s].cpu().detach().numpy())
            # print(np.around(Ns, 2)[Ns > 0])
            pi = Ns / Ns.sum() * prob

            if np.isnan(Ns).any():
                print("The counts had NaNs in it??")
                print(Ns)
            if np.isnan(pi).any():
                print("Pi had NaNs in it??")
                print(pi)

            avg_pi += pi

        # avg_pi += 1e-6
        avg_pi = np.nan_to_num(avg_pi)
        avg_pi = avg_pi / avg_pi.sum()
        return avg_pi
            
    def predict_pv(self, board: chess.Board, ply: int) -> Tuple[torch.Tensor, float]:
        """Uses the NN and stockfish to generate a policy vector of length 4096"""
        # King capture heuristic
        # If there's an open king attack, we have to go for it.
        enemy_king_square = board.king(not board.turn)
        if enemy_king_square is not None:
            attackers = board.attackers(board.turn, enemy_king_square)
            if attackers:
                attacker_square = attackers.pop()
                move = chess.Move(attacker_square, enemy_king_square)
                value = 1
                piece = board.piece_at(move.from_square)
                plane = map_move_plane(piece, move)
                heur_policy = torch.zeros((64, 8, 8), dtype=torch.float)
                x0, y0 = chess.square_file(move.from_square), chess.square_rank(move.from_square)
                heur_policy[plane, y0, x0] = 1.0
                heur_policy = heur_policy.view(-1)
                return heur_policy, value

        torch_board = board_to_torch(board, board.turn, ply)
        torch_board = torch_board.to(device)
        
        policy, value = self.nnet(torch_board)

        # Mask out the illegal moves in the policy tensor.
        policy = policy.cpu()
        mask = move_mask(board, board.turn)
        policy = policy * mask

        # # Generate stockfish's policy and value predictions
        # lamb = 0.70 # Percentage of own policy to use
        # sf_policy, sf_value, error = stockfish_pv(board.copy(), board.turn, self.engine)

        # # Incorporate stockfish if it didn't error
        # if not error:
        #     policy = (lamb * policy) + (1 - lamb) * sf_policy
        #     value = lamb * value + (1 - lamb) * sf_value

        # Flatten policy and normalize probabilities
        policy = policy.view((-1)) # type: torch.Tensor
        policy = policy / policy.sum()

        # Pull the value out of the tensor
        value = value.item()
        return policy, value

    def search(self, board: chess.Board, ply: int):
        """Recursive MCTS algorithm"""
        s = board.fen() # Use the fen representation as a hash
        
        # Check if the board is terminal. If it is, backpropagate -1 or 1.
        finished, value = game_is_over(board.copy())
        if finished:
            return -1 * value
        
        # Selection: encountered leaf
        if s not in self.visited:
            self.visited.add(s)
            # The neural net simultaneously performs:
            # - expansion: implicitly generate the child nodes by finding a policy to follow on the next iteration
            # - simulation: the neural net returns the value of the current board state.
            self.P[s], value = self.predict_pv(board.copy(), ply)
            return -1 * value
        
        # Selection: Not at leaf, so we need to choose a child state using UCB
        max_ucb, best_action = -np.inf, np.random.choice(4096)
        for a in np.arange(4096): # policy tensor is 64 8x8 planes, flattened
            Qsa = self.Q.setdefault(s, np.zeros(4096))[a]
            Nsa = self.N.setdefault(s, np.zeros(4096))[a]
            total_Nsa = self.N[s].sum()
            c_puct = self.exploration_constant
            u = Qsa + c_puct * self.P[s][a] * np.sqrt(total_Nsa) / (1 + Nsa)
            if u > max_ucb:
                max_ucb = u
                best_action = a

        # Convert the action idx into a move usable by the chess interface.
        a = best_action
        # print('\nBest action', a, 'for', ['black', 'white'][board.turn])
        move = idx_to_move(a)

        # REALLY make sure that the move is in the set of pseudo legal moves
        real_moves = list(board.pseudo_legal_moves)
        while move not in real_moves:
            a = np.random.choice(4096)
            move = idx_to_move(a)

        # Perform search with the child as root
        s_prime = board.copy()
        s_prime.push(move)
        # print('First board turn', board.turn)
        # print('sprime turn', s_prime.turn)
        v = self.search(s_prime, ply + 1)

        # Update Q values and visited counts, N
        self.Q[s][a] = (self.N[s][a] * self.Q[s][a] + v) / (self.N[s][a] + 1)
        self.N[s][a] += 1
        self.updated_Nsa = True
        return -v

# %% Neural network
class Residual(nn.Module):
    def __init__(self) -> None:
        super(Residual, self).__init__()
        self.conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.conv(out)
        out = self.norm(out)
        out += x
        out = self.relu(out)
        return out

class ValueHead(nn.Module):
    def __init__(self) -> None:
        super(ValueHead, self).__init__()
        self.conv = nn.Conv2d(256, 1, kernel_size=1, stride=1)
        self.norm = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 1)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        out = out.view(-1, 64)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.tanh(out)
        return out

class PolicyHead(nn.Module):
    def __init__(self) -> None:
        super(PolicyHead, self).__init__()
        self.conv = nn.Conv2d(256, 100, kernel_size=1, stride=1)
        self.norm = nn.BatchNorm2d(100)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(100 * 64, 8*8*64)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        out = out.view(-1, 100 * 64)
        out = self.fc(out)
        out = self.softmax(out)
        return out


class AlphaZeroChess(nn.Module):
    def __init__(self) -> None:
        super(AlphaZeroChess, self).__init__()
        # self.conv = nn.Conv2d(13, 256, kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(14, 256, kernel_size=3, stride=1, padding=1)
        self.res = Residual()
        self.policy_head = PolicyHead()
        self.value_head = ValueHead()
    
    def forward(self, board):
        x = self.conv(board)
        # for _ in range(8):
        for _ in range(19):
            x = self.res(x)
        p = self.policy_head(x)
        p = p.view(-1, 64, 8, 8)
        v = self.value_head(x)
        return p, v

# %% Agent that can handle uncertainty
class PhongAgent(Player):    
   
    def handle_game_start(self, color, board):
        """
        This function is called at the start of the game.
        # https://piazza.com/class/kdt2jakcbyw3mb?cid=406
        # Initialize BoardState() and set the belief for where all the pieces are.
        # If it's the game start, we should know this with 100% certainty.

        :param color: chess.BLACK or chess.WHITE -- your color assignment for the game
        :param board: chess.Board -- initial board state
        :return:
        """
        self.engine = None
        checkpoint = torch.load('ptran34_model.pt', map_location=lambda device, loc: device)
        self.nnet = AlphaZeroChess()
        self.nnet.to(device)
        self.nnet.load_state_dict(checkpoint['model_state_dict'])
        self.nnet.eval()
        
        self.color = color
        self.enemy_color = not color
        self.belief = numpy_boards_PMF([chess.Board()])
        self.enemies_remaining = 16

        self.entropy = np.zeros((8, 8))
        self.opponent_captured_my_piece = False
        self.opponent_captured_my_square = None

        self.turn_time_max = 25
        self.turn_time_left = self.turn_time_max 

        self.training_data = None

        self.ply = 1 if color == chess.BLACK else 0
    
    def handle_opponent_move_result(self, captured_piece, captured_square):
        """
        This function is called at the start of your turn and gives you the chance to update your board.

        :param captured_piece: bool - true if your opponents captured your piece with their last move
        :param captured_square: chess.Square - position where your piece was captured
        """         
        start_time = time.time()
        
        # Perform an initial "blind" propagation to get a sense of the state's entropy
        all_boards = []
        probs = []
        # Generate boards from the PMF that fit the constraints
        iter = 0
        while len(all_boards) < 100 and time.time() - start_time < 5:
            iter += 1
            board, prob = PMF_to_board(self.belief, self.enemies_remaining, self.color)
            board.turn = not self.color
            moves = list(board.pseudo_legal_moves)
            valid_moves = []
            for move in moves:
                target_square = move.to_square
                if captured_piece:
                    if target_square == captured_square or board.is_en_passant(move):
                        valid_moves.append(move)
                else:
                    # If captured is false, then ignore the moves that result in a capture
                    if board.is_capture(move) or board.is_en_passant(move):
                        continue
                    else:
                        valid_moves.append(move)
            for move in valid_moves:
                board_copy = board.copy()
                board_copy.push(move)
                all_boards.append(board_copy)
                probs.append(prob / len(valid_moves))

        print("Generated boards:", len(all_boards))

        if len(all_boards) == 0:
            # Had a hard time collecting data, so maybe the belief diverged at some point.
            # Add some noise on the empty squares to summon new pieces on the next iterations.
            print("Adding noise to empty squares.")
            for y in range(8):
                for x in range(8):
                    if self.belief[0, y, x] > 0.8:
                        noise = np.ones(13) * 0.01
                        if self.color == chess.WHITE:
                            noise[1:7] = 0
                        else:
                            noise[7:] = 0
                        noise[0] = 0.6
                        noise = noise / noise.sum()
                        self.belief[:, y, x] = noise
            self.entropy = entropy(self.belief, axis=0)
            self.opponent_captured_my_piece = captured_piece
            self.opponent_captured_my_square = captured_square
            return self.belief
        
        # Normalize probs
        probs = np.array(probs)
        probs = probs / probs.sum()

        # Get numpy boards for all boards
        numpy_boards = [board_to_numpy(b) for b in all_boards]

        # Weight each numpy board by their probability, which was inherited from their parent.
        numpy_boards = [b * probs[i] for i, b in enumerate(numpy_boards)]
        total_board = np.zeros((13, 8, 8))
        for nb in numpy_boards:
            total_board += nb
        
        # Normalize across the 0th dim, which represents the PMF for different piece types
        final_PMF = total_board / total_board.sum(axis=0)
        self.entropy = entropy(final_PMF, axis=0)
        self.opponent_captured_my_piece = captured_piece
        self.opponent_captured_my_square = captured_square

        # Update self board
        # if captured_piece:
        #     self.self_board.remove_piece_at(captured_square)
        # Use updated self board to really ensure the probabilities are correct
        # my_pieces = []
        # for piece_type in chess.PIECE_TYPES:
        #     my_pieces += [(sq, chess.Piece(piece_type, self.color)) for sq in self.self_board.pieces(piece_type, self.color)]
        
        # Update time left
        time_consumed = time.time() - start_time
        self.turn_time_left -= time_consumed
        return

    def choose_sense(self, possible_sense, possible_moves, seconds_left):
        """
        This function is called to choose a square to perform a sense on.

        :param possible_sense: List(chess.SQUARES) -- list of squares to sense around
        :param possible_moves: List(chess.Moves) -- list of acceptable moves based on current board
        :param seconds_left: float -- seconds left in the game

        :return: chess.SQUARE -- the center of 3x3 section of the board you want to sense
        :example: choice = chess.A1
        """
        self.turn_time_max = min(self.turn_time_max, seconds_left / 40)
        print("Time remaining:", seconds_left, "adjusted time:", self.turn_time_max)
        
        start_time = time.time()

        # Calculate the window with max total entropy
        # best_y, best_x = np.unravel_index(ent.argmax(), ent.shape)
        # best_y = np.clip(best_y, 1, 6) # No reason to center the sensor along the walls.
        # best_x = np.clip(best_x, 1, 6)
        best_x, best_y = np.random.randint(1, 7), np.random.randint(1, 7)
        best_total = 0
        for y in range(1, 7):
            for x in range(1, 7):
                x0, x1 = x-1, x+2
                y0, y1 = y-1, y+2
                total = self.entropy[y0:y1, x0:x1].sum()
                if total > best_total:
                    best_x, best_y = x, y
                    best_total = total
        
        # Update time left
        time_consumed = time.time() - start_time
        self.turn_time_left -= time_consumed

        print('Suggested sense at ', best_x, best_y)
        return chess.square(best_x, best_y)

    def handle_sense_result(self, sense_result):
        """
        This is a function called after your picked your 3x3 square to sense and gives you the chance to update your
        board.

        :param sense_result: A list of tuples, where each tuple contains a :class:`Square` in the sense, and if there
                             was a piece on the square, then the corresponding :class:`chess.Piece`, otherwise `None`.
        :example:
        [
            (A8, Piece(ROOK, BLACK)), (B8, Piece(KNIGHT, BLACK)), (C8, Piece(BISHOP, BLACK)),
            (A7, Piece(PAWN, BLACK)), (B7, Piece(PAWN, BLACK)), (C7, Piece(PAWN, BLACK)),
            (A6, None), (B6, None), (C8, None)
        ]
        """
        start_time = time.time()

        # Redo the propagation, but this time we also need to make it consistent with the sense result
        all_boards = []
        probs = []
        # Generate boards from the PMF that fit the constraints
        while len(all_boards) < 100 and time.time() - start_time < 5:  
            board, prob = PMF_to_board(self.belief, self.enemies_remaining, self.color)
            board.turn = not self.color
            moves = list(board.pseudo_legal_moves)
            valid_moves = []
            for move in moves:
                target_square = move.to_square
                if self.opponent_captured_my_piece:
                    if target_square == self.opponent_captured_my_square or board.is_en_passant(move):
                        valid_moves.append(move)
                else:
                    # If captured is false, then ignore the moves that result in a capture
                    if board.is_capture(move) or board.is_en_passant(move):
                        continue
                    else:
                        valid_moves.append(move)
            
            # Now check if the moves result in boards that are consistent with the sense result
            valid_boards = []
            for move in valid_moves:
                board_copy = board.copy()
                board_copy.push(move)

                consistent = True
                for sense in sense_result:
                    square = sense[0] # type: int
                    piece = sense[1] # type: chess.piece

                    if (board_copy.color_at(square) == None and piece == None):
                        continue # OK, both empty
                    elif (piece != None) and (board_copy.color_at(square) == piece.color) and (board_copy.piece_at(square) == piece):
                        continue # OK, both sense and board have the same piece and color.
                    else:
                        consistent = False
                        break

                if consistent:
                    valid_boards.append(board_copy)

            for valid_board in valid_boards:
                all_boards.append(valid_board)
                probs.append(prob / len(valid_boards))

        
        if len(all_boards) == 0:
            # Had a hard time collecting data, so maybe the belief diverged at some point.
            # Add some noise on the empty squares to summon new pieces on the next iterations.
            print("Adding noise to empty squares.")
            for y in range(8):
                for x in range(8):
                    if self.belief[0, y, x] > 0.8:
                        noise = np.ones(13) * 0.01
                        if self.color == chess.WHITE:
                            noise[1:7] = 0
                        else:
                            noise[7:] = 0
                        noise[0] = 0.6
                        noise = noise / noise.sum()
                        self.belief[:, y, x] = noise

            # Reapply sense result
            for sense in sense_result:
                square = sense[0] # type: int
                piece = sense[1] # type: chess.Piece

                # Based on piece type and color,
                # calculate the target dim to update in self.belief, a 13x8x8 array.
                dim = 0
                offset = 0
                if piece is not None:
                    offset = 7 if piece.color == chess.BLACK else 1
                    dim = (piece.piece_type - 1) + offset
                
                rank = chess.square_rank(square)
                file = chess.square_file(square)

                # Set the cell to 0 across all dims...
                self.belief[:, rank, file] = 0
                # Then set the piece dim to 1 since we know with certainty what it is.
                self.belief[dim, rank, file] = 1
            
            # Update time left
            time_consumed = time.time() - start_time
            self.turn_time_left -= time_consumed
            return self.belief

        # Reapply sense result
        for sense in sense_result:
            square = sense[0] # type: int
            piece = sense[1] # type: chess.Piece

            # Based on piece type and color,
            # calculate the target dim to update in self.belief, a 13x8x8 array.
            dim = 0
            offset = 0
            if piece is not None:
                offset = 7 if piece.color == chess.BLACK else 1
                dim = (piece.piece_type - 1) + offset
            
            rank = chess.square_rank(square)
            file = chess.square_file(square)

            # Set the cell to 0 across all dims...
            self.belief[:, rank, file] = 0
            # Then set the piece dim to 1 since we know with certainty what it is.
            self.belief[dim, rank, file] = 1

        # Normalize probs
        probs = np.array(probs)
        if np.isnan(probs).any() or probs.sum() == 0:
            print("In handle sense result, the probs had a NaN or all zeros")
            print(probs, sense_result)
            # print(1 / 0 / 0)
        probs = probs / probs.sum()

        # Get numpy boards for all boards
        numpy_boards = [board_to_numpy(b) for b in all_boards]

        # Weight each numpy board by their probability, which was inherited from their parent.
        numpy_boards = [b * probs[i] for i, b in enumerate(numpy_boards)]
        total_board = np.zeros((13, 8, 8))
        for nb in numpy_boards:
            total_board += nb
        
        # Normalize across the 0th dim, which represents the PMF for different piece types
        final_PMF = total_board / total_board.sum(axis=0)
        if np.isnan(final_PMF).any():
            print("Final PMF after sense calculations had NaN")
            print(final_PMF, sense_result)
            # print(1 /  0 / 0)
        self.belief = final_PMF
        
        # Update time left
        time_consumed = time.time() - start_time
        self.turn_time_left -= time_consumed
        
        return self.belief

    def choose_move(self, possible_moves, seconds_left):
        """
        Choose a move to enact from a list of possible moves.

        :param possible_moves: List(chess.Moves) -- list of acceptable moves based only on pieces
        :param seconds_left: float -- seconds left to make a move
        
        :return: chess.Move -- object that includes the square you're moving from to the square you're moving to
        :example: choice = chess.Move(chess.F2, chess.F4)
        
        :condition: If you intend to move a pawn for promotion other than Queen, please specify the promotion parameter
        :example: choice = chess.Move(chess.G7, chess.G8, promotion=chess.KNIGHT) *default is Queen
        """
        start_time = time.time()
        possible_moves = list(possible_moves)

        mcts = StochasticMCTS(self.nnet, self.engine)

        if np.isnan(self.belief).any():
            print("Found NaN in belief before running MCTS")
            print(self.belief)
            # print(1 / 0 / 0)

        mcts.timed_search(self.belief, self.ply, self.enemies_remaining, self.color, self.turn_time_left * 0.8)
        pi = mcts.improved_policy()
        if np.isnan(pi).any():
            print("ERRONEOUS POLICY", pi)
        pi = np.nan_to_num(pi) # Rare case where a board with zero legal moves was passed to MCTS... so we get a policy of all 0 -> NaN
        if pi.sum() == 0:
            pi = np.ones(4096) / 4096

        # Create some boards to save for the neural net training input
        # possibilities = [PMF_to_board(self.belief, self.enemies_remaining, self.color) for _ in range(100)]
        # possibilities.sort(key=lambda x: x[1], reverse=True)
        # possibilities = possibilities[:50]
        possibilities = mcts.potentials[:1]
        boards = [board_to_torch(p[0], self.color, self.ply).cpu().detach().numpy() for p in possibilities]
        self.training_data = [(board, pi) for board in boards]
        print('Acted on believed board:')
        print(possibilities[0][0], possibilities[0][1])

        # Override policy if likely board has a king attack.
        likely_board = possibilities[0][0]
        enemy_king_square = likely_board.king(not self.color)
        if enemy_king_square is not None:
            attackers = likely_board.attackers(self.color, enemy_king_square)
            if attackers:
                attacker_square = attackers.pop()
                print('OVERRIDING POLICY - SPOTTED KING ATTACK.', attacker_square, enemy_king_square)
                move = chess.Move(attacker_square, enemy_king_square)
                value = 1
                piece = likely_board.piece_at(move.from_square)
                plane = map_move_plane(piece, move)
                heur_policy = torch.zeros((64, 8, 8), dtype=torch.float)
                x0, y0 = chess.square_file(move.from_square), chess.square_rank(move.from_square)
                heur_policy[plane, y0, x0] = 1.0
                heur_policy = heur_policy.view(-1)
                pi = heur_policy
                self.training_data = [(board_to_torch(likely_board, self.color, self.ply), pi)]

        move_idx = np.random.choice(4096, p=pi)
        move = idx_to_move(move_idx)
        if move not in possible_moves:
            for _ in range(100): # Try a bunch of times to choose a move that's valid
                move_idx = np.random.choice(4096, p=pi)
                move = idx_to_move(move_idx)
                if move in possible_moves:
                    break
        # If we STILL don't have a valid move... choose one randomly from the available ones.
        had_to_choose_random = False
        if move not in possible_moves:
            had_to_choose_random = True
            move = random.choice(possible_moves)

        notification = "(had to choose randomly)" if had_to_choose_random else None
        print(["Black", "White"][self.color], 'chose move:', move, notification)

        # Update time left
        time_consumed = time.time() - start_time
        self.turn_time_left -= time_consumed
        return move

    def handle_move_result(self, requested_move, taken_move, reason, captured_piece, captured_square):
        """
        This is a function called at the end of your turn/after your move was made and gives you the chance to update
        your board.

        :param requested_move: chess.Move -- the move you intended to make
        :param taken_move: chess.Move -- the move that was actually made
        :param reason: String -- description of the result from trying to make requested_move
        :param captured_piece: bool - true if you captured your opponents piece
        :param captured_square: chess.Square - position where you captured the piece
        """
        start_time = time.time()

        # update move count
        self.ply += 2

        print('Move that', ["Black", "White"][self.color], "requested:", requested_move)
        print('Move that', ["Black", "White"][self.color], "actually made:", taken_move)
        requested_move = requested_move # type: chess.Move
        taken_move = taken_move # type: chess.Move
        captured_piece = captured_piece # type: bool
        captured_square = captured_square # type: int

        # If both requested_move and taken_move are None, then that probably means we ran out of time on our move
        # and didn't choose anything (we passed).
        if taken_move is None and requested_move is None:
            return # keep self.belief the same

        all_boards = [] # type: List[chess.Board]
        probs = []
        # Sample a bunch of boards from the PMF that fits the move result constraints
        iter = 0
        choice_type = None
        while len(all_boards) < 100 and time.time() - start_time < 5:
            iter += 1
            board, prob = PMF_to_board(self.belief, self.enemies_remaining, self.color, enemy_captured=captured_piece, enemy_captured_at=captured_square)
            board.turn = self.color
            
            # If the taken_move is None, but requested_move is not None, then that means we requested an illegal move.
            # We requested that move under the assumption it was legal in one of our boards, so filter those.
            if taken_move is None and requested_move is not None:
                # print("Requested an illegal move.")
                choice_type = "Requested an illegal move"
                if not board.is_pseudo_legal(requested_move):
                    all_boards.append(board)
                    probs.append(prob)

            else:
                # If...
                # (1) requested_move != taken_move
                # (2) captured_enemy == False
                # (3) The moving piece was a pawn
                # then that means our pawn ran into an obstruction during its move and could only move one space.
                # Therefore, the only valid_boards are the ones where the move was invalid.
                if requested_move != taken_move and captured_piece == False:
                    if board.is_pseudo_legal(taken_move) and board.piece_at(taken_move.from_square).piece_type == chess.PAWN and not board.is_pseudo_legal(requested_move):
                        # print("Pawn obstructed.")
                        choice_type = "Pawn obstructed"
                        all_boards.append(board)
                        probs.append(prob)
                
                # If...
                # (1) requested_move != taken_move
                # (2) captured_enemy == True
                # Then that means our piece ran into a different enemy piece while moving and captured that one instead.
                # Therefore, the only valid_boards are the ones where taken_move is valid and there's an enemy piece at the target square
                elif requested_move != taken_move and captured_piece == True:
                    if board.is_pseudo_legal(taken_move) and board.color_at(taken_move.to_square) == (not self.color):
                        # print("Took a different piece.")
                        choice_type = "Took a different piece."
                        all_boards.append(board)
                        probs.append(prob)

                # If...
                # (1) requested_move == taken_move
                # (2) captured_enemy == True
                elif requested_move == taken_move and captured_piece == True:
                    # (3) moving piece is a pawn
                    # (4) captured_enemy_at is different than the target square
                    # We did an en passant capture, so only boards that pawn moving with a valid en passant are valid
                    if board.is_pseudo_legal(taken_move) and taken_move.to_square != captured_square and board.piece_type_at(taken_move.from_square) == chess.PAWN:
                        choice_type = "en passant capture"
                        all_boards.append(board)
                        probs.append(prob)

                    # (3) captured_enemy_at has an enemy piece
                    # (4) captured_enemy_at == taken_move.to_square
                    # we successfuly captured an enemy piece at the location. Only add boards that had a capture there.
                    # Ignore the boards that don't have a capture here.
                    elif board.is_pseudo_legal(taken_move) and captured_square == taken_move.to_square and board.color_at(captured_square) == self.enemy_color:
                        choice_type = "Captured what we wanted"
                        all_boards.append(board)
                        probs.append(prob)

                # Final case
                # We made a valid move that didn't capture anything.
                elif requested_move == taken_move and captured_piece == False:
                    if not board.is_capture(taken_move):
                        choice_type = "Successfully made a move without capturing anything."
                        all_boards.append(board)
                        probs.append(prob)

        # Update the number of enemies remaining
        if captured_piece:
            self.enemies_remaining -= 1
            print('ENEMIES REMAINING:', self.enemies_remaining)

        print("Move result type", choice_type)
        print("Generated boards:", len(all_boards))
        self.ally_move_boards = all_boards
        if len(all_boards) == 0:
            # Had a hard time collecting data, so maybe the belief diverged at some point.
            # Add some noise on the empty squares to summon new pieces on the next iterations.
            print("Adding noise to empty squares.")
            for y in range(8):
                for x in range(8):
                    if self.belief[0, y, x] > 0.8:
                        noise = np.ones(13) * 0.01
                        if self.color == chess.WHITE:
                            noise[1:7] = 0
                        else:
                            noise[7:] = 0
                        noise[0] = 0.6
                        noise = noise / noise.sum()
                        self.belief[:, y, x] = noise

            # reset variables
            self.turn_time_left = self.turn_time_max
            return self.belief
        
        # in all the boards, push the taken_move
        really_valid_boards = []
        for board in all_boards:
            if taken_move is not None and board.is_pseudo_legal(taken_move):
                board_copy = board.copy()
                board_copy.push(taken_move)
                really_valid_boards.append(board_copy)
                board_copy.turn = not self.color
        if len(really_valid_boards) > 0:
            all_boards = really_valid_boards

        # Confirm that all moves r

        # Normalize probs
        probs = np.array(probs)
        probs = probs / probs.sum()

        # Get numpy boards for all boards
        numpy_boards = [board_to_numpy(b) for b in all_boards]

        # Weight each numpy board by their probability, which was inherited from their parent.
        numpy_boards = [b * probs[i] for i, b in enumerate(numpy_boards)]
        total_board = np.zeros((13, 8, 8))
        for nb in numpy_boards:
            total_board += nb
        
        # Normalize across the 0th dim, which represents the PMF for different piece types
        final_PMF = total_board / total_board.sum(axis=0)
        self.belief = final_PMF
        
        # Update time left
        time_consumed = time.time() - start_time
        self.turn_time_left -= time_consumed

        # reset variables
        self.turn_time_left = self.turn_time_max
        # self.turn_time_left = min(self.turn_time_max, )

        return self.belief

    def handle_game_end(self, winner_color, win_reason):  # possible GameHistory object...
        """
        This function is called at the end of the game to declare a winner.

        :param winner_color: Chess.BLACK/chess.WHITE -- the winning color
        :param win_reason: String -- the reason for the game ending
        """
        pass
