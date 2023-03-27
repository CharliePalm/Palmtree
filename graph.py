from typing import List
import numpy as np
from dataclasses import dataclass
import chess
import sys
from palmtree import Palmtree
from threading import Lock, Thread, active_count
import multiprocess as mp
from isolate import select
from time import sleep
from batching import Data
from torch.utils.data import DataLoader
sys.setrecursionlimit(100000)
from isolate import flush_queue
C_EXPLORE = 10 # ideally this would be a learned constant. We need to prevent the model from frequenting the same states to prevent it from drawing early on
C_EXPLOIT = 50
UNCALCULATED = -2
MAX_PROPAGATION_SIZE = 10e3
BATCH_SIZE = 4
VISIT_REDUCTION_SCALE = 1.1
'''
node class that represents a position. 
'''
@dataclass(order=True)
class Node:
    value: np.float64
    explored = 0
    policy_t = None
    policy_f = None
    policy = 1 / 64 ** 2 # ~somewhat~ average policy a completely random selection from the model
    queuing = False

    def __init__(self, fen, value, color, depth, dirty_fen, parent, move, learning=True):
        self.fen = fen
        self.value = self.base_value = value
        self.color = color
        self.parents = {move: parent}
        self.depth = depth
        self.dirty_fen = dirty_fen
        self.move = move
        self.learning = learning
        self.children = {}
        self.lock = Lock()
        self.visits = 1

    def __str__(self) -> str:
        return 'value: ' + str(self.value) + '\n' + 'position: ' + str(self.fen)
    
    def visit(self):
        if not self.explored:
            self.visits += 1
            self.calc_value()
            with self.lock:
                keys = list(self.children.keys())
            for child in keys:
                self.children[child].calc_value()
        return self
    
    '''
    during training, when we run a simulation from this node, we update it's value based on the simulation result
    '''
    def update_value(self, new_value, propagate=False, current_propagation: set = None, skip=False):
        if not skip and not self.explored:
            self.base_value = (self.visits * self.base_value + new_value) / (VISIT_REDUCTION_SCALE * (self.visits + 1)) if self.base_value == UNCALCULATED else new_value
            self.calc_value()
        elif skip and not self.explored:
            skip = False
            if new_value != 0:
                self.explored = True
                self.base_value = new_value
        with self.lock:
            keys = list(self.parents.keys() if self.parents else [])
        if propagate and current_propagation is not None and keys:
            for move in keys:
                p = self.parents[move]
                if p.fen not in current_propagation:
                    current_propagation.add(p.fen)
                    p.update_value(-1 * self.base_value, propagate, current_propagation, skip)

    def calc_value(self):
        if not self.explored and self.parents:
            self.value = self.base_value + (C_EXPLORE if self.learning else C_EXPLOIT) * (self.policy) * np.sqrt((np.log2(self.get_parent_visits())) / (self.visits + 1))
        elif self.explored and self.base_value == 1:
            self.value = np.inf
        else:
            self.value = self.base_value

    def get_parent_visits(self):
        s = 0
        with self.lock:
            keys = list(self.parents.keys())
        for p in keys:
            s += self.parents[p].visits
        return s

    def set_pi(self, f, t):
        self.policy_t = t
        self.policy_f = f

    def set_child_policy(self):
        if self.children:
            _, moves, p = Palmtree.interpret_p(self.policy_f, self.policy_t, chess.Board(self.dirty_fen))
            if moves:
                for idx, move in enumerate(moves):
                    new_val = np.log2(p[idx] / 1.25 + 1.2)
                    self.children[move].policy = new_val if new_val > self.children[move].policy else self.children[move].policy

    def mark_as_explored(self, outcome: chess.Outcome):
        self.explored = 1 if outcome.winner is not None else -1
        self.value = self.base_value = 1 if outcome.winner is not None else 0

    def add_child(self, child, move):
        if move not in self.children:
            self.children[move] = child

    def add_parent(self, parent, move):
        if self.parents and move not in self.parents:
            self.parents[move] = parent

    def set_as_root(self):
        self.value = self.base_value = 0
        self.parents = None

    def calc_pi(self):
        from_squares, to_squares = np.zeros(shape=(64)), np.zeros(shape=(64))
        if not self.children:
            from_squares.fill(1 / 64)
            to_squares.fill(1 / 64)
            s = 1
        else:
            total_children_visits = 0
            with self.lock:
                keys = list(self.children.keys())
            for child in keys:
                total_children_visits += self.children[child].visits
            s = 0
            for child in keys:
                (a, b) = Palmtree.get_index(child)
                p = self.children[child].visits / total_children_visits
                from_squares[a] += p
                to_squares[b] += p
                s += p
        return (from_squares / s, to_squares / s)

class MCTS:
    queue_lock = Lock()
    model_lock = Lock()
    propagating = False
    def __init__(self, data_org, parent_board: chess.Board = None, learning=True, model: Palmtree = None):
        self.positions = {}
        self.size = 0
        self.learning = learning
        self.data_org = data_org
        self.model = model if model is not None else Palmtree(self.data_org)
        self.terminal_nodes = []
        self.propagation_queue = {}
        self.root = None
        if not parent_board:
            parent_board = chess.Board()
        self.add_node(parent_board.fen(), 0, None, parent_board.turn, 0, None)
        self.add_to_propagation_queue([self.root])
        #Thread(target=self.propagate, args=(), daemon=True).start()
        
    def add_node(self, fen: str, value: float, parent_fen: str, color: bool, depth: int, parent_move) -> Node:
        n = None
        clean_fen = self.data_org.clean_fen(fen)
        clean_parent_fen = self.data_org.clean_fen(parent_fen)
        if clean_fen not in self.positions:
            if parent_fen is not None:
                parent = self.get(clean_parent_fen)
                n = Node(clean_fen, value, color, depth, fen, parent, parent_move, self.learning)
                parent.add_child(n, parent_move)
            else:
                # root case
                n = Node(clean_fen, value, color, depth, fen, None, parent_move, self.learning)
                self.update_root(n)
            self.positions[clean_fen] = n
            self.size += 1
        else:
            n = self.positions[clean_fen].visit()
            if parent_fen is not None:
                p = self.get(parent_fen)
                p.add_child(n, parent_move)
                n.add_parent(p, parent_move)
            else:
                self.update_root(n)
        return n

    def update_root(self, new_root: Node):
        new_root.set_as_root()
        self.root = new_root

    def select(self, node: Node) -> Node:
        path = set()
        while 1:
            node.visit()
            path.add(node.fen)
            # if this is the first time the node was visited, we return it
            if not node.children:
                self.traverse(node)
                return
            new_node = node.children[max(node.children, key=lambda child: node.children[child].value if node.children[child].fen not in path else np.NINF)]
            # if we haven't explored this far yet just pick a random node
            if node.base_value == UNCALCULATED:
                self.flush()
                new_node = node.children[max(node.children, key=lambda child: node.children[child].value if node.children[child].fen not in path else np.NINF)]
            node = new_node
            
    def traverse(self, node: Node):
        board = chess.Board(node.dirty_fen)
        nodes = []
        data = []
        for m in board.legal_moves:
            board.push(m)
            n = self.add_node(board.fen(), UNCALCULATED, node.dirty_fen, board.turn, node.depth + 1, m.uci())
            nodes.append(n)
            data.append(self.data_org.get_binary_board_from_fen(n.dirty_fen))
            o = board.outcome()
            if o is not None:
                n.mark_as_explored(o)
            board.pop()
        self.add_to_propagation_queue(nodes)
        
    def propagate(self, x, y):
        node_idx = 0
        with self.model_lock:
            for values, fs, ts  in self.generate_propagation_data(x):
                for idx, value in enumerate(values):
                    node = self.get(y[node_idx])
                    propagation = set()
                    if not node.explored:
                        node.update_value(value.item(), True, propagation)
                        node.set_pi(fs[idx].detach().cpu().numpy(), ts[idx].detach().cpu().numpy())
                    else:
                        node.update_value(node.base_value, True, propagation, True)
                    node_idx += 1

    def flush(self):
        if self.propagation_queue:
            x = []
            y = []
            with self.queue_lock:
                x = list(self.propagation_queue.values())
                y = list(self.propagation_queue.keys())
                self.propagation_queue.clear()
            self.propagate(x, y)

    def generate_propagation_data(self, x):
        i = 0
        l = len(x)
        while i < l:
            ds = Data(x[i:i+BATCH_SIZE if i + BATCH_SIZE < l else l], device=Palmtree.device)
            yield self.model.predict_on_batch(ds)
            i += BATCH_SIZE

    def visit_nodes(self, positions: set):
        for position in positions:
            self.get(position).visit()

    async def add_positions_to_propagation_queue(self, parent: Node):
        self.add_to_propagation_queue(parent.children.values())

    def add_to_propagation_queue(self, nodes: List[Node]):
        with self.queue_lock:
            for node in nodes:
                # both operations are (in the average case) O(1) so not a big deal to do this with a held lock
                if node.fen not in self.propagation_queue: 
                    self.propagation_queue[node.fen] = self.data_org.get_binary_board_from_fen(node.dirty_fen) 

    def update_model(self, model):
        with self.model_lock:
            self.model = model
    '''
    returns the node associated with the position
    '''
    def get(self, fen: str) -> Node:
        if fen in self.positions: return self.positions[fen]
        clean_fen = self.data_org.clean_fen(fen)
        return self.positions[clean_fen] if clean_fen in self.positions else None
    
    '''
    async def select(self, node, iterations: int):
        flush_promise = None
        for _ in range(int(iterations)):
            if not flush_promise: flush_promise = self.flush_queue()
            pool = mp.Pool()
            select_result = pool.map_async(select, [node for _ in range(self.num_processes)])
            pool.close()
            results = select_result.get()
            for result in results:
                if result:
                    parent = self.get(result[0][0])
                    if parent.policy_f is None:
                        await flush_promise
                        flush_promise = None
                    #print(parent)
                    #print(result[0][1])
                    for child in result[0][1]:
                        self.add_node(result[0][1][child], UNCALCULATED, parent.dirty_fen, not parent.color, parent.depth + 1, child)
                    void = self.add_positions_to_propagation_queue(parent)
                    self.visit_nodes(result[1])
                    await void
        if flush_promise is not None: await flush_promise
        return
        '''