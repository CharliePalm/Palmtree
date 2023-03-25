import numpy as np
from os import environ
from data_organization import DataOrganization
from keras.models import Model
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, Add, LeakyReLU, ZeroPadding2D, BatchNormalization, AveragePooling2D, MaxPooling2D
from keras.optimizers import Adam
import chess
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

class Palmtree():
    model: Model = None
    max_nodes_per_iteration = 100
    def __init__(self, data_org=None):
        self.model = self.create_model_v2()
        self.data_org = data_org if data_org else DataOrganization()
    
    '''
    finds the best move with no searching
    '''
    def generate_best_move(self, board: chess.Board) -> str:
        _, max_move = self.get_all_moves(board)
        # if we return an empty string that means the board is terminal. Caller's error, we'll print the board for their debug benefit
        if max_move == '':
            print('ERROR: No moves found for position:')
            print(board)
        return max_move

    '''
    generates move probabilities for the position
    returns a dictionary of key value pairs of moves and probabilities
    '''
    def get_all_moves(self, board: chess.Board):
        move_probs = {}
        x = []
        moves = []
        for move in board.generate_legal_moves():
            board.push(move)
            x.append(self.data_org.get_binary_board(board))
            moves.append(move.uci())
            board.pop()
        x = np.asarray(x)
        if x.shape == (0,):
            return x, None
        prediction = self.model.predict_on_batch(x)
        max = np.NINF
        max_move = ''
        for x, y in zip(moves, prediction):
            move_probs[x] = y[0]
            if y[0] > max:
                max = y[0]
                max_move = x
        return move_probs, max_move

    def print_best_moves(self, moves):
        to_print = {}
        m = np.NINF
        for move in moves:
            if len(to_print) < 3:
                m = moves[move]
                to_print[move] = m
            else:
                to_remove = None
                for best in to_print:
                    if moves[move] > to_print[best]:
                        to_remove = best
                if to_remove:
                    del to_print[to_remove] 
                    to_print[move] = moves[move]
        print(to_print)
                
    def make_move(self, board: chess.Board):
        return self.explore_moves(board)

    def make_searchless_move(self, board):
        moves, max_move = self.get_all_moves(board)
        vals = np.array(list(moves.values()))
        p = vals / sum(vals)
        choice = np.random.choice(list(moves.keys()), p=p)
        return choice, moves[choice]

    def make_random_move(self, board: chess.Board):
        r = np.random.randint(0, board.legal_moves.count())
        for idx, move in enumerate(board.legal_moves):
            if idx == r:
                return move
    
    '''
    given a board, return a vector of moves->probabilities and a policy rating
    '''
    def get_moves(self, board: chess.Board()):
        [[from_squares], [to_squares], [[rating]]] = self.model.predict_on_batch(np.array([self.data_org.get_binary_board(board)]))
        return from_squares, to_squares, rating
    
    def predict(self, board: chess.Board):
        from_squares, to_squares, rating = self.get_moves(board)
        return from_squares, to_squares, rating
    
    def predict_on_batch(self, batch):
        return self.model.__call__(batch)

    @staticmethod
    def interpret_p(from_squares, to_squares, board: chess.Board):
        moves = []
        p = []
        p_total = 0
        for move in board.legal_moves:
            m = move.uci()
            from_index, to_index = Palmtree.get_index(m)
            p1 = from_squares[from_index]
            p2 = to_squares[to_index]
            val = p1 * p2
            if not np.isreal(val):
                val = 0
            p.append(val)
            moves.append(m)
            p_total += val
        m = np.NINF
        max_move = None
        if not moves or not p_total:
            return None, None, None
        for idx, move in enumerate(moves):
            p[idx] /= p_total
            if p[idx] > m:
                m = p[idx]
                max_move = moves[idx]
        return max_move, moves, p

    @staticmethod
    def get_proportional_prediction(moves, move_probs):
        return np.random.choice(moves, p=move_probs)

    @staticmethod
    def get_index(move: str):
        return classification_map[move[0]] + (int(move[1]) - 1) * 8, classification_map[move[2]] + (int(move[3]) - 1) * 8

    def create_model_v2(self, num_residuals=10):
        board_input = x = Input((8, 9, 13))
        x = ZeroPadding2D()(x)
        x = (Conv2D(filters=256, kernel_size=2))(x)
        x = (LeakyReLU(alpha=.01))(x)
        for i in range(num_residuals):
            x = self.build_residual_layer(x, i % 3 == 2)
        
        from_output = self.build_state_output(x, 'from_output')
        to_output = self.build_state_output(x, 'to_output')
        classification_output = self.build_classification_output(x)
        model = Model(board_input, [from_output, to_output, classification_output])
        print('done creating model')
        # Compile the model
        model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(.00003))
        print(model.summary())
        return model

    def build_residual_layer(self, x_in, normalize):
        x = x_res = x_in
        x = (Conv2D(filters=256, kernel_size=2, padding="same"))(x)
        if normalize: x = BatchNormalization(axis=3)(x)
        x = (LeakyReLU(alpha=.01))(x)
        x = (Add())([x, x_res])
        return x

    def build_state_output(self, x, name='state_output'):
        x = (Conv2D(filters=256, kernel_size=2, activation='relu'))(x)
        x = BatchNormalization(axis=3)(x)
        x = (Conv2D(filters=128, kernel_size=2, activation='relu'))(x)
        x = BatchNormalization(axis=3)(x)
        x = (Conv2D(filters=64, kernel_size=2, activation='relu'))(x)
        x = (AveragePooling2D(pool_size=(2, 2)))(x)
        x = BatchNormalization(axis=3)(x)
        x = Flatten()(x)
        x = Dense(64, activation='softmax', name=name)(x)
        return x

    def build_classification_output(self, x, name='class_output'):
        x = (Conv2D(filters=256, kernel_size=2))(x)
        x = (LeakyReLU(alpha=.01))(x)
        x = (Conv2D(filters=128, kernel_size=2))(x)
        x = BatchNormalization(axis=3)(x)
        x = (Conv2D(filters=64, kernel_size=2))(x)
        x = (LeakyReLU(alpha=.01))(x)
        x = (AveragePooling2D(pool_size=(2, 2)))(x)
        x = (Flatten())(x)
        x = (Dense(128, activation='softsign'))(x)
        x = (Dense(1, activation='tanh', name=name))(x)
        return x