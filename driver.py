import faulthandler
faulthandler.enable()
import numpy as np
from os import getcwd, environ
import plaidml.keras
environ["KERAS_BACKEND"] = "plaidml.keras.backend"
environ["RUNFILES_DIR"] = "/Library/Frameworks/Python.framework/Versions/3.8/shared/plaidml"
environ["PLAIDML_NATIVE_PATH"] = "/Library/Frameworks/Python.framework/Versions/3.8/lib/libplaidml.dylib"
plaidml.keras.install_backend()
from data_organization import DataOrganization
from palmtree import Palmtree
from self_play import SelfPlay
from lichess_integration import LichessIntegration
from player import Player
from asyncio import run
#print('using ' + keras.backend.backend() + ' as backend')
import chess
import matplotlib.pyplot as plt
import multiprocessing as mp

# this class handles all model training and data creation
class Driver:
    data = None
    tree = None
    def __init__(self):
        self.data_org = DataOrganization(getcwd() + '/Games')

    def train(self, use_existing_data=True, epochs=50, batch_size=200, data_len=None):
        if not self.data and use_existing_data:
            self.load_data()
        l = len(self.data)
        x = np.zeros(shape=(l, 8, 9, 13))
        y1 = np.zeros(shape=(l, 64))
        y2 = np.zeros(shape=(l, 64))
        y3 = np.zeros(shape=(l))
        for idx, fen in enumerate(self.data):
            x[idx] = self.data[fen][0]
            y1[idx] = self.data[fen][1]
            y2[idx] = self.data[fen][2]
            y3[idx] = self.data[fen][3]
        try:
            print(len(x))
            print(len(y2))
            print(len(y1))
            print(len(y3))

            hist = self.tree.model.fit(x=x, y=[y1, y2, y3], epochs=epochs, verbose=1, batch_size=batch_size, shuffle=True)
        except KeyboardInterrupt:
            pass
        self.pickle(self.tree, 'models/ALBERT')
        plt.plot(hist.history['loss'])
        print(hist.history)
        plt.show()

    def pickle(self, data, path):
        self.data_org.create_pickle(path, data)

    def load_data(self):
        self.data = self.data_org.depickle('data/data')

    def start_self_play(self, epochs=100):
        self.self_play = SelfPlay(self.tree, data_org=self.data_org)
        run(self.self_play.commence())

    def create_data(self):
        self.data = self.data_org.gather_data()
        self.pickle(self.data_org, 'data/data')

    def start_play(self):
        response = input('would you like to start playing through the lichess API or locally? y=lichess, n=local\n')
        if response == 'y':
            lichess = LichessIntegration(self.tree)
            lichess.listen()
        else:
            self.play_console()
    
    def check_data(self):
        self.data = self.data_org.depickle('data/data')
        s = 0
        for entry in self.data:
            '''
            print(entry)
            print(self.data_org.get_board_from_binary(self.data[entry][0]))
            print(Palmtree.interpret_p(self.data[entry][1], self.data[entry][2], chess.Board(self.data_org.dirty_fen(entry))))'''
            print(self.data_org.dirty_fen(entry))
            print(self.data[entry][3])
            s += self.data[entry][3]
        print(s / len(self.data))
        
    def clean_data(self):
        self.data = self.data_org.depickle('data/data')
        print(len(self.data))
        to_del = []
        for entry in self.data:
            '''
            print(entry)
            print(self.data_org.get_board_from_binary(self.data[entry][0]))
            print(Palmtree.interpret_p(self.data[entry][1], self.data[entry][2], chess.Board(self.data_org.dirty_fen(entry))))'''
            b = chess.Board(self.data_org.dirty_fen(entry))
            for m in b.legal_moves:
                b.push(m)
                if b.outcome() == None and self.data[entry][3] == -1:
                    to_del.append(entry)
                    break
                b.pop()

        for entry in to_del:
            del self.data[entry]
        self.data_org.create_pickle('data/data', self.data)

    def start_best_v_random(self, times=10, verbose=True):
        self.tree = self.data_org.depickle('models/self_play_v11')
        player = Player(self.tree)
        for i in range(times):
            board = chess.Board()
            wins = {'random': 0, 'model': 0}
            white = np.random.randint(0, 2)
            if white:
                print('random plays white')
            else:
                print('random plays black')
            idx = 0
            outcome = None
            while outcome is None:
                if verbose:
                    print(board)
                    print()
                move = player.make_move(board) if idx % 2 == white else self.tree.make_random_move(board).uci()
                board.push_uci(move)
                outcome = board.outcome()
                idx += 1
            winner = outcome.winner
            if verbose:
                print(board)
                print()
                print(outcome.result())
            if white and outcome.winner == chess.WHITE or not white and outcome.winner == chess.BLACK:
                wins['random'] += 1
                print('random wins')
            elif outcome.winner is not None:
                wins['model'] += 1
                print('model wins')
            else:
                print('draw')
        print(wins)
        return winner
        
    def play_console(self):
        board = chess.Board()
        whiteOrBlack = 1
        player = Player(self.tree)
        print('created player')
        if (whiteOrBlack):
            move = player.make_move(board)
            board.push(chess.Move.from_uci(move))

        userInput = ''
        while (userInput != 'q' or userInput != 'Q') and not board.is_game_over():
            print(board)
            while 1:
                print('waiting')
                userInput = input("\n")
                print('got input: ' + userInput)
                try:
                    board.push(chess.Move.from_uci(userInput))
                    if (board.is_game_over()):
                        print('darn, ya got me! Good game!')
                        return
                    break
                except:
                    print('invalid move')
                    continue
            print(board)
            move = player.make_move(board)
            print('I think I\'ll play', move)
            board.push(chess.Move.from_uci(move))
        print('Good game! I\'ll getcha next time!')


if __name__ == "__main__":
    mp.set_start_method('spawn')
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-t', '--train', action="store_true")
    parser.add_argument('-p', '--play', action="store_true")
    parser.add_argument('-sp', '--self', action="store_true")
    parser.add_argument('-l', '--load', action="store_true")
    parser.add_argument('-g', '--gather', action="store_true")
    parser.add_argument('-e', '--epochs', type=int)
    parser.add_argument('-d', '--data')
    parser.add_argument('-m', '--model', type=str)
    parser.add_argument('-bvr', '--best_v_random', action="store_true")
    parser.add_argument('-cd', '--check_data', action='store_true')
    parser.add_argument('-cl', '--clean_data', action='store_true')
    mp.set_forkserver_preload(['keras', 'numpy', 'plaidml-keras', 'chess', 'Palmtree', 'os'])
    mp.freeze_support()

    driver = Driver()
    args = parser.parse_args()
    
    if args.model:
        driver.tree = driver.data_org.depickle('models/' + args.model)
    if not driver.tree:
        Driver.tree = Palmtree(driver.data_org)
    if args.train:
        driver.train(epochs=args.epochs)
    elif args.gather:
        driver.create_data()
    elif args.self:
        driver.start_self_play(epochs=args.epochs)
    elif args.load:
        driver.load_data()
    elif args.play:
        driver.start_play()
    elif args.best_v_random:
        driver.start_best_v_random()
    elif args.check_data:
        driver.check_data()
    elif args.clean_data:
        driver.clean_data()