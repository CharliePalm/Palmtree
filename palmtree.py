import numpy as np
from data_organization import DataOrganization
from torch.nn import Conv2d, MaxPool2d, Linear, ReLU, Softmax, Softsign, Tanh, ZeroPad2d, Module, LeakyReLU, BatchNorm1d, Flatten, CrossEntropyLoss, MSELoss
from torch.optim import SGD, Adam
from torch import from_numpy, device
from torch.cuda import FloatTensor, is_available
import torch
device = device('cuda:0' if is_available() else 'cpu')
import chess
from copy import deepcopy

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

class Palmtree(Module):
    max_nodes_per_iteration = 100
    intro_layers = []
    residual_layers = []
    classification_layers = []
    from_policy_layers = []
    to_policy_layers = []
    
    def __init__(self, data_org=None, num_residuals=24):
        super(Palmtree, self).__init__()
        self.create_model(10)
        self = self.to(device=device)
        self.data_org = data_org if data_org else DataOrganization()
        self.optimizer = SGD(self.parameters(), .00005)
        self.policy_loss = CrossEntropyLoss()
        self.classification_loss = MSELoss()
        
    def create_model(self, num_residuals: int):
        self.intro_layers = [
            ZeroPad2d(padding=2),
            Conv2d(13, 256, 2, 1, 0, device=device),
            LeakyReLU()
        ]
        self.zero_pad_1 = self.intro_layers[0]
        self.conv_1 = self.intro_layers[1]
        self.lr_1 = self.intro_layers[2]
        
        for idx in range(num_residuals):
            layer = []
            layer.append(Conv2d(256, 256, 2, 1, padding='same', device=device))
            layer.append(LeakyReLU())
            layer.append(BatchNorm1d(11))
            setattr(self, 'conv_'+str(idx+2), layer[0])
            setattr(self, 'lr_'+str(idx+2), layer[1])
            setattr(self, 'bn_'+str(idx+1), layer[2])
            self.residual_layers.append(layer)
        # define classification layers
        self.classification_layers = [
            ('c_conv_1', Conv2d(256, 256, 2, 1, padding=0, device=device)),
            ('c_lr_1', LeakyReLU()),
            ('c_bn_1', BatchNorm1d(10)),
            ('c_conv_2', Conv2d(256, 128, 2, 1, padding=0, device=device)),
            ('c_lr_2', LeakyReLU()),
            ('c_bn_2', BatchNorm1d(9)),
            ('c_conv_3', Conv2d(128, 64, 2, 1, padding=0, device=device)),
            ('c_lr_3', LeakyReLU()),
            ('c_mp', MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c_f', Flatten()),
            ('c_l_1', Linear(1234, 128)),
            ('c_ss', Softsign()),
            ('c_l_2', Linear(128, 1)),
            ('c_out', Tanh())
        ]
        for entry in self.from_policy_layers:
            setattr(self, entry[0], entry[1])
        # define policy layers
        self.from_policy_layers = [
            ('conv_1', Conv2d(256, 256, 2, 1, padding=0, device=device)),
            ('relu_1', ReLU()),
            ('bn_1', BatchNorm1d(10)),
            ('conv_2', Conv2d(256, 128, 2, 1, padding=0, device=device)),
            ('relu_2', ReLU()),
            ('bn_2', BatchNorm1d(9)),
            ('c_3', Conv2d(128, 64, 2, 1, padding=0, device=device)),
            ('relu_2', ReLU()),
            ('bn_3', BatchNorm1d(8)),
            ('mp', MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('flatten', Flatten()),
            ('lin', Linear(5 * 7 * 32, 64)),
            ('out', Softmax())
        ]
        
        self.to_policy_layers = deepcopy(self.from_policy_layers)
        for entry in self.from_policy_layers:
            setattr(self, 'f_' + entry[0], entry[1])
        for entry in self.to_policy_layers:
            setattr(self, 't_' + entry[0], entry[1])


    
    def forward(self, x: torch.Tensor):
        print(x.shape)
        x = x.to(device=device)
        for layer in self.intro_layers:
            print(layer)
            x = layer(x)
        for residual_layer in self.residual_layers:
            x_in = x
            for layer in residual_layer:
                print(layer)
                x = layer(x)
            x += x_in
        
        classification_out = policy_from_out = policy_to_out = x
        for layer in self.classification_layers:
            classification_out = layer[1](classification_out)
        for layer in self.from_policy_layers:
            policy_from_out = layer[1](policy_from_out)
        for layer in self.to_policy_layers:
            policy_to_out = layer[1](policy_to_out)

        return classification_out, policy_from_out, policy_to_out
        
    def train(self, x, y_class, y_fr, y_to, epochs, verbose=True) -> dict:
        hist_to = []
        hist_from = []
        hist_class = []
        for epoch in range(epochs):
            running_loss = 0.0
            for idx, pos in enumerate(x):
                self.optimizer.zero_grad()
                classification, policy_from, policy_to = self(pos)
                
                class_loss = self.classification_loss(classification, y_class[idx])
                from_loss = self.policy_loss(policy_from, y_fr[idx])
                to_loss = self.policy_loss(policy_to, y_to[idx])
                class_loss.backward()
                from_loss.backward()
                to_loss.backward()
                
                self.optimizer.step()
                
                running_loss += to_loss.item() + from_loss.item() + class_loss.item()
                hist_to.append(to_loss.item())
                hist_to.append(to_loss.item())

                if verbose and idx % 30 == 29:
                    print(f'[{epoch + 1}, {idx + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
        return {
            'class': hist_class, 'from': hist_from, 'to': hist_to
        }


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
    def get_moves(self, board: chess.Board):
        [[from_squares], [to_squares], [[rating]]] = self(np.array([self.data_org.get_binary_board(board)]))
        return from_squares, to_squares, rating
    
    def predict_on_batch(self, batch):
        return self(batch)

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
    def get_index(move: str) -> tuple:
        return classification_map[move[0]] + (int(move[1]) - 1) * 8, classification_map[move[2]] + (int(move[3]) - 1) * 8