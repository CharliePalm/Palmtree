from palmtree import Palmtree
from graph import MCTS
from threading import Thread, Lock
from chess import Board
from data_organization import DataOrganization
class Player:
    mcts: MCTS = None
    discovery_lock = Lock()
    def __init__(self, model: Palmtree, simulations=500):
        self.model = model
        self.simulations = simulations
        self.data_org = DataOrganization()
        self.mcts = MCTS(self.data_org, Board(), False, self.model)
        #Thread(target=self.discover, args=(), daemon=True).start()

    '''
    given a board, return a uci move
    '''
    def make_move(self, board: Board) -> str:
        new_fen = board.fen()
        new_root = self.mcts.get(new_fen)
        if new_root is None:
            new_root = self.mcts.add_node(new_fen, 0, None, board.turn, 0, None)
        else:
            self.mcts.update_root(new_root)
        node = self.mcts.root
        with self.discovery_lock:
            for i in range(self.simulations):
                self.mcts.select(node)
            self.mcts.flush()
            (f_a, t_a) = node.calc_pi()
            _, moves, probs = self.model.interpret_p(f_a, t_a, Board(node.dirty_fen))
            move = self.model.get_proportional_prediction(moves, probs)
            board.push_uci(move)
            node = self.mcts.get(board.fen())
            self.mcts.update_root(node)
        board.pop()
        return move

    '''
    think about game states until interrupted
    '''
    def discover(self):
        while 1:
            with self.discovery_lock:
                self.mcts.select(self.mcts.root)