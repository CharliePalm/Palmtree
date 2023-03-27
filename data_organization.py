import os
import pickle
import chess
from chess.pgn import read_game, Game
from copy import deepcopy
import numpy as np
import torch
piece_map = {
    'p' : [1,0,0,0,0,0,0,0,0,0,0,0,0],
    'P' : [0,0,0,0,0,0,1,0,0,0,0,0,0],
    'n' : [0,1,0,0,0,0,0,0,0,0,0,0,0],
    'N' : [0,0,0,0,0,0,0,1,0,0,0,0,0],
    'b' : [0,0,1,0,0,0,0,0,0,0,0,0,0],
    'B' : [0,0,0,0,0,0,0,0,1,0,0,0,0],
    'r' : [0,0,0,1,0,0,0,0,0,0,0,0,0],
    'R' : [0,0,0,0,0,0,0,0,0,1,0,0,0],
    'q' : [0,0,0,0,1,0,0,0,0,0,0,0,0],
    'Q' : [0,0,0,0,0,0,0,0,0,0,1,0,0],
    'k' : [0,0,0,0,0,1,0,0,0,0,0,0,0],
    'K' : [0,0,0,0,0,0,0,0,0,0,0,1,0],
    '.' : [0,0,0,0,0,0,0,0,0,0,0,0,1],
}

column_map = {
    'a' : [1,0,0,0,0,0,0,0,0,0,0,0,0],
    'b' : [0,1,0,0,0,0,0,0,0,0,0,0,0],
    'c' : [0,0,1,0,0,0,0,0,0,0,0,0,0],
    'd' : [0,0,0,1,0,0,0,0,0,0,0,0,0],
    'e' : [0,0,0,0,1,0,0,0,0,0,0,0,0],
    'f' : [0,0,0,0,0,1,0,0,0,0,0,0,0],
    'g' : [0,0,0,0,0,0,1,0,0,0,0,0,0],
    'h' : [0,0,0,0,0,0,0,1,0,0,0,0,0],
}

row_map = {
    '1' : [1,0,0,0,0,0,0,0,0,0,0,0,0],
    '2' : [0,1,0,0,0,0,0,0,0,0,0,0,0],
    '3' : [0,0,1,0,0,0,0,0,0,0,0,0,0],
    '4' : [0,0,0,1,0,0,0,0,0,0,0,0,0],
    '5' : [0,0,0,0,1,0,0,0,0,0,0,0,0],
    '6' : [0,0,0,0,0,1,0,0,0,0,0,0,0],
    '7' : [0,0,0,0,0,0,1,0,0,0,0,0,0],
    '8' : [0,0,0,0,0,0,0,1,0,0,0,0,0],
}

turn_map = {
    chess.WHITE: [1,1,1,1,1,1,0,0,0,0,0,0,0],
    chess.BLACK: [0,0,0,0,0,0,1,1,1,1,1,1,0],
    'b': [0,0,0,0,0,0,1,1,1,1,1,1,0],
    'w': [1,1,1,1,1,1,0,0,0,0,0,0,0],
}

classification_map = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7
}

class DataOrganization:
    MAX_MOVES = 100
    GAME_COUNT = 20
    def __init__(self, path=''):
        self.path = path
        self.x = []
        self.y_c = []
        self.y_f = []
        self.y_t = []


    def create_pickle(self, name, obj):
        out = open(os.path.join(os.getcwd(), name), 'wb')
        pickle.dump(obj, out)
        out.close()
        return obj

    def depickle(self, name):
        with open(os.path.join(os.getcwd(), name), 'rb') as pick:
            tree = pickle.load(pick)
        return tree

    def gather_data(self):
        for file in os.listdir(self.path):
            print('reading ' + file)
            games = 0
            if (file.endswith('pgn')):
                pgn = open(os.path.join(self.path, file))
                game = read_game(pgn)
                while game is not None and games < self.GAME_COUNT:
                    games += 1
                    result = game.headers['Result'].strip()
                    if result == '1-0':
                        self.get_game_data(game, True, games)
                    elif result == '0-1':
                        self.get_game_data(game, False, games)
                    else:
                        self.get_game_data(game, False, games, False)
                        self.get_game_data(game, True, games, False)
                    # the chess package can sometimes be finicky about which games it wants to read. This prevents any unnecessary breakdowns
                    try:
                        game = read_game(pgn)
                    except:
                        game = read_game(pgn)
        self.create_pickle('data/gm_data', self)

    def check(self):
        for i in range(len(self.x1)):
            print(self.x1.shape)
            print(self.get_board_from_binary(self.x1[i]))
            print(self.convert_binary_to_move(self.x1[i]))
            print('white' if self.find_key(self.x1[i][7][8], turn_map) == chess.WHITE else 'black')
            print(self.y[i])
    
    def check_input(self, x):
        print(self.get_board_from_binary(x[0]))
        print(self.convert_binary_to_move(x[0]))
        print('white' if self.find_key(x[0][7][8], turn_map) == chess.WHITE else 'black')


    def get_game_data(self, game: Game, is_white, games, is_win=True):
        board = chess.Board()
        l = len(game.mainline_moves())
        for iterations, move in enumerate(game.mainline_moves()):
            # we dont want white's moves if we're black and vice versa and we dont want to overload the model with opening knowledge
            if (games % 10 != 0 and iterations < 15):
                board.push(move)
                continue
            move_uci = chess.Move.uci(move)
            y_f_i, y_t_i = np.zeros(shape=(64), dtype=np.float32), np.zeros(shape=(64), dtype=np.float32)
            f_index, t_index = self.get_index(move_uci)
            y_f_i[f_index], y_t_i[t_index] = 1.0, 1.0
            x_i = self.get_binary_board(board)
            self.x.append(torch.tensor(x_i, dtype=torch.float32).permute((2, 0, 1)))
            self.y_f.append(torch.tensor(y_f_i, dtype=torch.float32))
            self.y_t.append(torch.tensor(y_t_i, dtype=torch.float32))
            self.y_c.append(torch.tensor([((1.0 if iterations % 2 == (0 if is_white else 1) else -1.0) if is_win else 0) / (l - iterations)], dtype=torch.float32))
            board.push(move)
    
    def get_binary_board(self, board: chess.Board, is_single_batch = False):
        map = board.piece_map()
        output_board = np.zeros(shape=(8, 9, 13), dtype=np.float32)
        iter = 0
        for i in range(8):
            row = np.zeros(shape=(9, 13), dtype=np.float32)
            for j in range(8):
                row[j] = piece_map['.'] if iter not in map else piece_map[map[iter].symbol()]
                iter += 1
            output_board[i] = row
        output_board[7][8] = turn_map[board.turn]
        return output_board if not is_single_batch else np.asarray([output_board])
    
    '''
    this somewhat non pythonic method does what the above does without the middle man of the chess package.
    much faster when caller only has fen string to work with
    '''
    def get_binary_board_from_fen(self, fen: str):
        output_board = np.zeros(shape=(8, 9, 13))
        fen_idx = 0
        for i in range(8):
            row = np.zeros(shape=(9, 13))
            j = 0
            while j < 8:
                if fen[fen_idx] in piece_map:
                    row[j] = piece_map[fen[fen_idx]]
                    j += 1
                else:
                    for k in range(int(fen[fen_idx])):
                        row[j] = piece_map['.']
                        j += 1
                fen_idx += 1
            fen_idx += 1 # for / character
            output_board[7-i] = row
        output_board[7][8] = turn_map[fen[fen_idx]]
        return torch.tensor(output_board, dtype=torch.float32).permute(2, 0, 1)
     
    @staticmethod   
    def get_index(move: str) -> tuple:
        return classification_map[move[0]] + (int(move[1]) - 1) * 8, classification_map[move[2]] + (int(move[3]) - 1) * 8


    '''
    Mostly for debug, used to ensure boards look the way they're supposed to by converting binary boards into human readable character representations
    This is very slow, I would not recommend using this for large batches of data
    '''
    def get_board_from_binary(self, board):
        to_ret = np.zeros(shape=(8, 8), dtype=str)
        for i in range(8):
            for j in range(8):
                for piece in piece_map:
                    if np.allclose(np.asarray(piece_map[piece]), np.asarray(board[i][j])):
                        to_ret[i][j] = piece
                        break
        return to_ret

    def find_key(self, val, dict):
        for i in dict.keys():
            if np.allclose(np.asarray(dict[i]), val):
                return i

    '''
    it's necessary to 'clean' the fen because we don't want the same position represented multiple times through the move counters at the end of the fen
    '''
    @staticmethod
    def clean_fen(fen: str, start=''):
        if fen is not None and ' ' in fen:
            for i in (fen.split(' ')[0:4]):
                start += i
            return start
        return fen

    @staticmethod
    def dirty_fen(clean_fen: str):
        out_fen = ''
        board_idx = 0
        for idx, c in enumerate(clean_fen):
            if board_idx == 64:
                out_fen += ' ' + c + ' '
            elif board_idx > 64 and c not in ['K', 'Q', 'k', 'q']:
                if c == '-':
                    out_fen += c + ' ' if out_fen[len(out_fen)-1] == ' ' else ' ' + c + ' ' 
                else:
                    out_fen += ' ' + clean_fen[idx:len(clean_fen)] + ' '
                    break
            else:
                out_fen += c
            board_idx += int(c) if c.isnumeric() else 1 if c != '/' else 0
        
        return out_fen + '20 21'