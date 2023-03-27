import numpy as np
from data_organization import DataOrganization
from torch.nn import Conv2d, MaxPool2d, Linear, ReLU, Softmax, Softsign, Tanh, ZeroPad2d, Module, LeakyReLU, BatchNorm2d, Flatten, CrossEntropyLoss, MSELoss
from torch.optim import SGD, Adam
from torch import from_numpy, device
from torch.utils.data import DataLoader
from torch.cuda import is_available
from batching import Data
import torch
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
    device = device('cuda:0' if is_available() else 'cpu')

    def __init__(self, data_org=None, num_residuals=5):
        super(Palmtree, self).__init__()
        self.create_model(10)
        self = self.to(device=self.device)
        self.data_org = data_org if data_org else DataOrganization()
        self.optimizer = SGD(self.parameters(), .001)
        self.policy_loss = CrossEntropyLoss()
        self.classification_loss = MSELoss()
        
    def create_model(self, num_residuals: int):
        self.intro_layers = [
            ZeroPad2d(padding=2),
            Conv2d(13, 256, 2, 1, 0, device=self.device),
            LeakyReLU()
        ]
        self.zero_pad_1 = self.intro_layers[0]
        self.conv_1 = self.intro_layers[1]
        self.lr_1 = self.intro_layers[2]
        
        for idx in range(num_residuals):
            layer = []
            layer.append(Conv2d(256, 256, 3, 1, padding='same', device=self.device))
            layer.append(LeakyReLU())
            layer.append(BatchNorm2d(256, device=self.device))
            setattr(self, 'conv_'+str(idx+2), layer[0])
            setattr(self, 'lr_'+str(idx+2), layer[1])
            setattr(self, 'bn_'+str(idx+1), layer[2])
            self.residual_layers.append(layer)
        # define classification layers
        self.classification_layers = [
            ('c_conv_1', Conv2d(256, 256, 2, 1, padding=0, device=self.device)),
            ('c_lr_1', LeakyReLU()),
            ('c_bn_1', BatchNorm2d(256, device=self.device)),
            ('c_conv_2', Conv2d(256, 128, 2, 1, padding=0, device=self.device)),
            ('c_lr_2', LeakyReLU()),
            ('c_bn_2', BatchNorm2d(128, device=self.device)),
            ('c_conv_3', Conv2d(128, 64, 2, 1, padding=0, device=self.device)),
            ('c_lr_3', LeakyReLU()),
            ('c_mp', MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c_f', Flatten()),
            ('c_l_1', Linear(1024, 256, device=self.device)),
            ('c_ss', Softsign()),
            ('c_l_2', Linear(256, 1, device=self.device)),
            ('c_out', Tanh())
        ]
        for entry in self.from_policy_layers:
            setattr(self, entry[0], entry[1])
        # define policy layers
        self.from_policy_layers = [
            ('conv_1', Conv2d(256, 256, 2, 1, padding=0, device=self.device)),
            ('relu_1', ReLU()),
            ('bn_1', BatchNorm2d(256, device=self.device)),
            ('conv_2', Conv2d(256, 128, 2, 1, padding=0, device=self.device)),
            ('relu_2', ReLU()),
            ('bn_2', BatchNorm2d(128, device=self.device)),
            ('c_3', Conv2d(128, 64, 2, 1, padding=0, device=self.device)),
            ('relu_3', ReLU()),
            ('bn_3', BatchNorm2d(64, device=self.device)),
            ('mp', MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('f', Flatten()),
            ('lin_1', Linear(1024, 256, device=self.device)),
            ('relu_4', ReLU()),
            ('lin_2', Linear(256, 64, device=self.device)),
            ('out', Softmax(dim=1))
        ]
        
        self.to_policy_layers = deepcopy(self.from_policy_layers)
        for entry in self.from_policy_layers:
            setattr(self, 'f_' + entry[0], entry[1])
        for entry in self.to_policy_layers:
            setattr(self, 't_' + entry[0], entry[1])


    
    def forward(self, x):
        for layer in self.intro_layers:
            x = layer(x)
        for residual_layer in self.residual_layers:
            x_in = x
            for layer in residual_layer:
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
        verbosity = 100
        hist_to = []
        hist_from = []
        hist_class = []
        hist_total = []
        data = DataLoader(Data(x, device, y_class, y_fr, y_to), 8, False)
        for epoch in range(epochs):
            running_loss = 0.0
            h_t = h_f = h_c = h_total = 0
            for idx, (x_i, y_c, y_f, y_t) in enumerate(data):
                self.optimizer.zero_grad()
                classification, policy_from, policy_to = self(x_i)
                class_loss = self.classification_loss(classification, y_c)
                from_loss = self.policy_loss(policy_from, y_f)
                to_loss = self.policy_loss(policy_to, y_t)
                
                loss = to_loss + from_loss + class_loss
                running_loss += loss.item()
                
                h_t += to_loss.item()
                h_f += from_loss.item()
                h_c += class_loss.item()
                loss.backward()
                self.optimizer.step()
            if verbose:
                #print(f'[{epoch + 1}, {idx + 1:5d}] loss: {running_loss / verbosity:.3f}')
                h_total += running_loss
                running_loss = 0.0
                print(f'epoch {epoch + 1} complete. loss:\ntotal: {h_total / len(data):.3f}, from: {h_f / len(data):.3f}, to: {h_t / len(data):.3f}, class: {h_c / len(data):.3f}')
            hist_total.append(h_total / len(data))
            hist_from.append(h_f / len(data))
            hist_class.append(h_c / len(data))
            hist_to.append(h_t / len(data))
        return {
            'class': hist_class, 'from': hist_from, 'to': hist_to, 'total': hist_total
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
        return self(batch.x)

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