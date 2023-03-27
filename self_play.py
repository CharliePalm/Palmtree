import chess
import numpy as np
from data_organization import DataOrganization
from graph import MCTS, Node
from copy import deepcopy
from threading import Lock, Thread
import matplotlib.pyplot as plt
from player import Player
from time import sleep
import torch
from random import choice
import multiprocessing as mp
from isolate import train_model
MAX_QUEUE_SIZE = 10e3
TRAIN_EPOCHS = 50
MAX_DATA_LEN = 2 ** 16
MAX_SIZE = 20e6
CHECKPOINT = 20
class SelfPlay:
    BEST = 1
    CURRENT = 0 
    DRAW = -1
    wins = { chess.WHITE: 0, chess.BLACK: 0, None: 0 }
    data_lock = Lock()
    data = {}
    training_started = False

    def __init__(self, initial_model, epochs=10, simulations=2, data_org: DataOrganization=None, games_per_epoch=100):
        self.epochs = epochs
        self.simulations = simulations
        self.current_model = initial_model
        self.best_model = deepcopy(initial_model)
        self.data_org = data_org if data_org else DataOrganization()
        self.losses = []
        self.mcts = MCTS(self.data_org, model=initial_model)
        self.games_per_epoch = games_per_epoch

    async def play_epoch(self):
        for i in range(self.games_per_epoch):
            self.iterate()
            if self.mcts.size > MAX_SIZE:
                self.mcts = MCTS(self.data_org, chess.Board(), True, self.current_model)

        print('starting play')
        wins = {self.BEST: 0, self.CURRENT: 0}
        for i in range(10):
            winner = self.play_game()
            wins[self.BEST] += int(winner == self.BEST)
            wins[self.CURRENT] += int(winner == self.BEST)
        print(wins)
        self.best_model = self.best_model if wins[self.BEST] > wins[self.CURRENT] else self.current_model
        self.current_model = deepcopy(self.best_model)
        return wins

    # this tests the new model against the old one. If it improved, we swap the two
    def play_game(self, verbose=False):
        board = chess.Board()

        # the white variable refers to who's playing as white. For example: white = 1 indicates that the best model is playing as white (as defined by the globals). white = 0 implies the opposite.
        white = np.random.randint(0, 2)
        players = [Player(self.current_model), Player(self.best_model)] if white == 0 else [Player(self.best_model), Player(self.current_model)]
        if white == 0:
            print('current plays white')
        else:
            print('current plays black')
        idx = 0
        outcome = None
        while outcome is None:
            move = players[idx % 2].make_move(board)
            board.push_uci(move)
            outcome = board.outcome()
            if verbose:
                print(board)
                print(board.fen())
                print()
            idx += 1
        winner = outcome.winner
        print(outcome.result())
        print(outcome.termination)
        print(outcome.winner)
        if winner == None:
            winner = self.DRAW
        else:
            winner = self.BEST if (winner == chess.WHITE and white == self.BEST) or (winner == chess.BLACK and white != self.BEST) else self.CURRENT
            if winner == self.BEST:
                print('the best model won!')
            else:
                print('current model won!')
        return winner
    
    def iterate(self):
        path = []
        board = chess.Board()
        node = self.mcts.root
        while True:
            print(board)
            for _ in range(self.simulations):
                self.mcts.select(node)
                if len(self.mcts.propagation_queue) > 1000:
                    self.mcts.flush()
            self.mcts.flush()

            path.append(node)
            (a, b) = node.calc_pi()
            _, moves, move_probs = self.current_model.interpret_p(a, b, chess.Board(node.dirty_fen))
            selected_move = self.current_model.get_proportional_prediction(moves, move_probs)
            board.push_uci(selected_move)
            fen = board.fen()
            o = board.outcome()
            if o is not None:
                print(o.result())
                print(o.termination)
                print('graph size: ' + str(self.mcts.size))
                self.mcts.flush()

                terminal_node = self.mcts.get(self.data_org.clean_fen(fen))
                if terminal_node is None:
                    print('terminal node not found')
                    terminal_node = self.mcts.add_node(fen, float(o.winner is not None), node.dirty_fen, not node.color, node.depth + 1, selected_move)
                    terminal_node.mark_as_explored(o)
                    terminal_node.update_value(terminal_node.base_value, True, set(), True)
                path.append(terminal_node)
                Thread(target=self.gather_data, args=(path,), daemon=True).start()
                sleep(20)
                return
            node = self.mcts.get(self.data_org.clean_fen(fen))

    def gather_data(self, path):
        with self.data_lock:
            d = len(self.data)
            if d > MAX_DATA_LEN:
                choices = np.random.choice(list(self.data.keys()), size=d-MAX_DATA_LEN, replace=False)
                for choice in choices:
                    del self.data[choice]
            for node in path:
                pi = node.calc_pi()
                self.data[node.fen] = (self.data_org.get_binary_board_from_fen(node.dirty_fen), torch.tensor(pi[0], dtype=torch.float32), torch.tensor(pi[1], dtype=torch.float32), node.base_value)
        if not self.training_started:
            self.training_started = True
            self.start_training()

    def start_training(self):
        while 1:
            with self.data_lock:
                if not self.data:
                    continue
                l = len(self.data)
                x = []
                y_f = []
                y_t = []
                y_c = []
                for idx, fen in enumerate(self.data):
                    x.append(self.data[fen][0])
                    y_f.append(self.data[fen][1])
                    y_t.append(self.data[fen][2])
                    y_c.append(torch.tensor([self.data[fen][3]], dtype=torch.float32))
            self.train(x, y_c, y_f, y_t)
            sleep(240)

    def train(self, x, y_c, y_f, y_t):
        self.data_org.create_pickle('models/self_play_v11', self.current_model)
        with self.data_lock:
            self.data_org.create_pickle('data/data', self.data)
        with mp.Pool() as pool:
            model, batch_loss = pool.apply(func=train_model, args=(self.current_model, x, y_c, y_f, y_t, TRAIN_EPOCHS))
            pool.close()
        self.save_model(model, batch_loss)

    def save_model(self, model, losses):
        v = sum(losses) / len(losses)
        if np.isnan(v):
            print('nan loss encountered')
            return
        self.mcts.update_model(model)
        self.current_model = model
        self.losses += losses
        print('batch loss: ' + str(v))
        
    def shutdown(self):
        self.data_org.create_pickle('models/self_play_v11', self.current_model)
        self.data_org.create_pickle('data/data', self.data)
        plt.plot(self.losses)
        plt.show()

    async def commence(self):
        try:
            self.data = self.data_org.depickle('data/data')
            if not self.data:
                self.data = {}
        except:
            pass
        try:
            for i in range(self.epochs):
                print('starting epoch ' + str(i + 1))
                await self.play_epoch()
        except KeyboardInterrupt:
            self.shutdown()