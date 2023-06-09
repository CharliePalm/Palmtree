import os
import pickle
import chess
from chess.pgn import read_game
from copy import deepcopy
import numpy as np

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

# this character is appended to the end of the move if a piece is being promoted to the piece denoted by the letter
promotion_map = {
    'n' : [0,0,0,0,0,0,0,0,1,0,0,0,0],
    'b' : [0,0,0,0,0,0,0,0,0,1,0,0,0],
    'r' : [0,0,0,0,0,0,0,0,0,0,1,0,0],
    'q' : [0,0,0,0,0,0,0,0,0,0,0,1,0],
    '.' : [0,0,0,0,0,0,0,0,0,0,0,0,1],
}

turn_map = {
    chess.WHITE: [1,1,1,1,1,1,0,0,0,0,0,0,0],
    chess.BLACK: [0,0,0,0,0,0,1,1,1,1,1,1,0],
    'b': [0,0,0,0,0,0,1,1,1,1,1,1,0],
    'w': [1,1,1,1,1,1,0,0,0,0,0,0,0],
}

class DataOrganization:
    MAX_MOVES = 100
    GAME_COUNT = 200
    def __init__(self, path=''):
        self.path = path
        self.x1 = []
        self.x2 = []
        self.y = []


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
                        self.get_game_data_v2(game, True, games)
                    elif result == '0-1':
                        self.get_game_data_v2(game, False, games)
                    else:
                        self.get_game_data_v2(game, False, games)
                        self.get_game_data_v2(game, True, games)
                    # the chess package can sometimes be finicky about which games it wants to read. This prevents any unnecessary breakdowns
                    try:
                        game = read_game(pgn)
                    except:
                        game = read_game(pgn)
        self.x1 = np.asarray(self.x1)
        self.x2 = np.asarray(self.x2)
        self.y = np.asarray(self.y)
        print(self.x2.shape)
        print(self.x1.shape)
        print(self.y.shape)

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

    def get_game_data(self, game, is_white, games):
        iter = 0
        board = chess.Board()
        for move in game.mainline_moves():
            # we dont want white's moves if we're black and vice versa and we dont want to overload the model with opening knowledge
            if (iter % 2 != (0 if is_white else 1)) or (games % 5 != 0 and iter < 15):
                iter += 1
                board.push(move)
                continue
            move_uci = chess.Move.uci(move)
            input_board = self.get_binary_board(board, is_white)
            legal_moves = board.legal_moves
            num_moves_taken = np.random.randint(1, 5)

            if legal_moves.count() < num_moves_taken:
                board.push(move)
                iter += 1
                continue
            # we want the model to see more bad moves than good moves but we also dont want it to adapt to a certain probability
            # (i.e. just guessing that 3/4 of moves will be 0 and 1/4 will be one and outputting the same value for everything)
            # even if this skews its binary accuracy to be low
            # in practice, it will see all legal moves and only a handful will be good
            legal_moves = np.random.choice(list(legal_moves), num_moves_taken)
            legal_moves[num_moves_taken - 1] = move
            for legal_move in legal_moves:
                legal_move_uci = chess.Move.uci(legal_move)
                board_copy = self.append_binary_move(deepcopy(input_board), legal_move_uci)
                self.y.append(1 if move_uci == legal_move_uci else 0)
                self.x1.append(board_copy)
            board.push(move)
            iter += 1

    def append_binary_move(self, board, move):
        binary_move = self.get_binary_move(move)
        for i in range(5):
            board[i][8] = binary_move[i]
        return board
        
    def get_game_data_v2(self, game, is_white, games):
        board = chess.Board()
        for iterations, move in enumerate(game.mainline_moves()):
            # we dont want white's moves if we're black and vice versa and we dont want to overload the model with opening knowledge
            if (iterations % 2 != (0 if is_white else 1)) or (games % 10 != 0 and iterations < 15):
                board.push(move)
                continue
            move_uci = chess.Move.uci(move)
            x1_i = self.get_binary_board_v2(board)


            x2_i, y_i = self.get_x2_y(board, move_uci)
            x2_i[self.MAX_MOVES - 1] = np.ones(shape=(5, 13)) if is_white else  x2_i[self.MAX_MOVES - 1]
            self.x1.append(x1_i)
            self.x2.append(x2_i)
            self.y.append(y_i)
            board.push(move)
    
    def get_binary_board(self, board: chess.Board, is_single_batch = False):
        map = board.piece_map()
        output_board = np.zeros(shape=(8, 9, 13))
        iter = 0
        for i in range(8):
            row = np.zeros(shape=(9, 13))
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
        return output_board
    
    def get_binary_board_v2(self, board):
        map = board.piece_map()
        output_board = np.zeros(shape=(8, 8, 13))
        iter = 0
        for i in range(8):
            row = np.zeros(shape=(8, 13))
            for j in range(8):
                row[j] = piece_map['.'] if iter not in map else piece_map[map[iter].symbol()]
                iter += 1
            output_board[7-i] = row
        return output_board
    
    def get_x2_y(self, board: chess.Board, move):
        y_i = np.zeros(shape=(self.MAX_MOVES))
        x2_i = np.zeros(shape=(self.MAX_MOVES, 5, 13))
        for idx, legal_move in enumerate(board.legal_moves()):
            legal_move_uci = chess.Move.uci(legal_move)
            y_i[idx] = 1 if move == legal_move_uci else 0
            x2_i[idx] = self.get_binary_move(legal_move_uci)
        return x2_i, y_i
        
    '''
    Converts moves to binary representations in constant time using the global maps
    This can either be done to create probability vector training outputs or inputs when assigning probabilities to legal moves
    '''
    def get_binary_move(self, move):
        arr = np.zeros(shape=(5, 13))
        arr[0] = column_map[move[0]]
        arr[1] = row_map[move[1]]
        arr[2] = column_map[move[2]]
        arr[3] = row_map[move[3]]
        
        arr[4] = promotion_map[move[4]] if len(move) == 5 else promotion_map['.']
        return arr
    
    def prepare_input(self, board: chess.Board, move: str, is_batch=False):
        board = self.get_binary_board(board, board.turn)
        a = self.append_binary_move(board, move)
        return np.asarray([self.append_binary_move(board, move)] if not is_batch else self.append_binary_move(board, move))
    
    def prepare_input_v2(self, board: chess.Board, is_batch=False):
        binary_moves, uci_moves = self.get_move_input_v2(board)
        return self.get_binary_board_v2(board), binary_moves, uci_moves

    def prepare_graph_train_data(self, board):
        return self.get_binary_board_v2(board), 

    def get_move_input_v2(self, board: chess.Board):
        binary_moves = np.zeros(shape=(self.MAX_MOVES, 5, 13))
        uci_moves = []
        idx = 0
        for idx, move in enumerate(board.generate_legal_moves()):
            uci = chess.Move.uci(move)
            binary_moves[idx] = self.get_binary_move(uci)
            uci_moves.append(uci)

        if board.turn:
            binary_moves[idx + 1] = np.ones(shape=(5, 13))
        return binary_moves, uci_moves
    '''
    Mostly for debug, used to ensure binary representaitons of moves look the way they're supposed to
    '''
    def convert_binary_to_move(self, move):
        try:
            toRet = self.find_key(move[0], column_map)
            toRet += self.find_key(move[1], row_map)
            toRet += self.find_key(move[2], column_map)
            toRet += self.find_key(move[3], row_map)
            toRet += self.find_key(move[4], promotion_map)
        except:
            print(np.zeros(shape=(5, 13)))
            print(move)

        return toRet

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