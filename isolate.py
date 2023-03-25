'''
This file contains all the lightweight functions performed via multiprocessing.

Using spawn is incredibly slow as all scripts are recompiled, which means we need to reimport pretty much everything.
With keras this takes a loooooong long time.

Using fork is much quicker - at least double the efficiency, sometimes triple, but unstable: long story short, MacOS has 
a tendency to create zombie processes this way which can indifenitely hold semaphores.

Our middle ground is using forkserver, which is a little bit of both. The python multiprocessing module allows us to cache
certain things (such as module imports) which roughly doubles efficiency.
'''
from chess import Board
import numpy as np
def flush_queue(x, model):
    f, t, values = model.predict_on_batch(x)
    return f, t, values

def train_model(model, x, y1, y2, y3, epochs):
    l = len(x)
    print('data length: ' + str(l))
    if l:
        hist = model.model.fit(x, [y1, y2, y3], epochs=epochs, batch_size=32, verbose=0)
    return model, hist.history['loss']

def predict_on_batch(model, batch):
    return model.predict_on_batch(batch)

def select(node) -> tuple[str, list[str], set[str], bool]:
    path = set()
    while 1:
        path.add(node.fen)
        # if this is the first time the node was visited, we return it
        if not node.children:
            return traverse(node.dirty_fen), path
        p = np.array([node.children[child].value + 1 if node.children[child].fen not in path else 0 for child in node.children])
        p /= sum(p)
        key = np.random.choice(list(node.children.keys()), p=p)
        node = node.children[key]

def traverse(position):
    board = Board(position)
    new_positions = {}
    for move in board.legal_moves:
        board.push(move)
        new_positions[move.uci()] = board.fen()
        board.pop()
    return position, new_positions