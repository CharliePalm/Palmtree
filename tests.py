import unittest
import chess
from os import environ
import plaidml.keras
environ["KERAS_BACKEND"] = "plaidml.keras.backend"
environ["RUNFILES_DIR"] = "/Library/Frameworks/Python.framework/Versions/3.8/shared/plaidml"
environ["PLAIDML_NATIVE_PATH"] = "/Library/Frameworks/Python.framework/Versions/3.8/lib/libplaidml.dylib"
plaidml.keras.install_backend()
import numpy as np
from data_organization import DataOrganization
from palmtree import Palmtree
import datetime
from graph import MCTS
from copy import copy
class TestDataOrganization(unittest.TestCase):
    data_org = DataOrganization()

    def test_binary_board_from_fen(self):
        board = chess.Board()
        binary_board = self.data_org.get_binary_board(board)
        fen_board = self.data_org.get_binary_board_from_fen(board.fen())
        self.assertTrue(np.all(binary_board == fen_board))
        

        binary_board = self.data_org.get_binary_board(chess.Board('4Q3/3p4/pRPP4/8/3k1NP1/1nq3P1/5K2/4rnB1 b - - 0 1'))
        fen_board = self.data_org.get_binary_board_from_fen('4Q3/3p4/pRPP4/8/3k1NP1/1nq3P1/5K2/4rnB1 b - - 0 1')
        self.assertTrue(np.all(binary_board == fen_board))

    def test_fen_vs_board_speed(self):
        fen = '4Q3/3p4/pRPP4/8/3k1NP1/1nq3P1/5K2/4rnB1 b - - 0 1'
        now = datetime.datetime.now()
        board = chess.Board('4Q3/3p4/pRPP4/8/3k1NP1/1nq3P1/5K2/4rnB1 b - - 0 1')
        binary_board = self.data_org.get_binary_board(board)

        from_board_time = datetime.datetime.now() - now

        now = datetime.datetime.now()
        binary_board = self.data_org.get_binary_board_from_fen('4Q3/3p4/pRPP4/8/3k1NP1/1nq3P1/5K2/4rnB1 b - - 0 1')
        from_fen_time = datetime.datetime.now() - now
        self.assertGreater(from_board_time, from_fen_time)

        board = chess.Board('4Q3/3p4/pRPP4/8/3k1NP1/1nq3P1/5K2/4rnB1 b - - 0 1')
        now = datetime.datetime.now()
        binary_board = self.data_org.get_binary_board(board)
        from_board_time = datetime.datetime.now() - now
        self.assertGreater(from_board_time, from_fen_time)

    def test_get_index(self):
        move = 'a1b1'
        indices = (0, 1)
        self.assertEqual(Palmtree.get_index(move), indices)
        move = 'e2f4'
        indices = (12, 29)
        self.assertEqual(Palmtree.get_index(move), indices)

    def test_fen_clean_dirty(self):
        fens = [
            '3Q4/4bB2/p6p/6k1/p1rq4/pP6/3P2PP/n6K w - - 0 1', 
            '7N/KR4bP/p1k1P2r/8/2pPR3/1p1Q1P2/8/7q w - - 0 1',
            chess.Board().fen(),
            'r1bqkbnr/1pp1pppp/p1n5/3pP3/8/2N5/PPPP1PPP/R1BQKBNR w KQkq d6 0 4',
            '1rb2bnr/2p1pk1p/p1nP2p1/1p1p1p2/PP4P1/2NB1N2/2PP1P1P/R1BQ1RK1 b - - 2 10',
            '7N/KR4bP/p1k1P2r/8/2pPR3/1p1Q1P2/8/7q b - - 0 1',
            'r1bq2kr/pp1pp1pN/n6n/2p2pP1/1P6/2P5/P2PPP1P/RNBQKBR1 w Q f6 20 30'
        ]
        for fen in fens:
            clean_fen = DataOrganization.clean_fen(copy(fen))
            self.assertEqual(DataOrganization.dirty_fen(clean_fen)[0 : len(fen) - 4], fen[0 : len(fen) - 4])
        
    def test_propagation(self):
        b = chess.Board()
        mcts = MCTS(self.data_org)
        moves = ['e2e4', 'e7e5', 'd1f3', 'b7b5', 'f1c4', 'g7g6', 'f3f7']
        parent_fen = None
        depth = 0
        fens = []
        parent_move = None
        for move in moves:
            fens.append(b.fen())
            mcts.add_node(b.fen(), 0, parent_fen, b.turn, depth, parent_move)
            parent_fen = b.fen()
            depth += 1
            b.push_uci(move)
            parent_move = move
        mcts.add_node(b.fen(), 0, parent_fen, b.turn, depth, parent_move)
        mcts.get(fens[-1]).mark_as_explored(b.outcome())
        mcts.get(fens[-1]).update_value(1, True, set(), True)
        self.assertEqual(mcts.get(fens[-1]).base_value, -1 * mcts.get(fens[-2]).base_value)
        self.assertEqual(mcts.get(fens[-1]).base_value, 1)

if __name__ == '__main__':
    unittest.main()