import requests
import json
import chess
from palmtree import Palmtree
from player import Player
PORT = 4200
base_path = 'https://lichess.org/api/'
KEY = 'lip_D2bQfLRCYFUJVqYEHKh8' #TODO : DELETE ME LATER

class LichessIntegration:
    model: Palmtree
    def __init__(self, model):
        self.games = {}
        self.model = Player(model)
        self._r = TokenSession(KEY)

    def listen(self):
        '''
        the main gameplay loop 
        '''
        full_response = ''
        for event in self.stream_incoming_events():
            chunk = event.decode()
            if chunk != '\n':
                full_response += chunk
                continue
            if full_response == '' or full_response == '\n':
                full_response = ''
                continue
            full_response = full_response.split('\n')
            for line in full_response:
                if line == '' or line == '\n':
                    continue
                event = json.loads(line)
                full_response = ''
                if event['type'] == 'challenge':
                    self.accept_challenge(event['challenge']['id'])
                elif event['type'] ==  'gameStart':
                    self.games[event['game']['gameId']] = [chess.Board(), None]
                    self.play_game(event['game']['gameId'], True if event['game']['color'] == 'white' else False)
                elif event['type'] == 'gameFinish':
                    del self.games[event['game']['gameId']]
    
    def play_game(self, game_id, isWhite):
        if isWhite:
            self.make_move(game_id)
        full_response = ''
        for event in self.stream_game_state(game_id):
            chunk = event.decode()
            if chunk != '\n':
                full_response += chunk
                continue
            if full_response == '' or full_response == '\n':
                full_response = ''
                continue
            full_response = full_response.split('\n')
            for line in full_response:
                try:
                    event = json.loads(line)
                except:
                    continue
                if event['type'] != 'gameState':
                    if event['type'] == 'gameFull':
                        event = event['state']
                    else:
                        continue
                if 'winner' in event:
                    return
                moves = event['moves']
                if not self.games[game_id][0]:
                    return
                last_move: str = moves[len(moves) - 4: len(moves)]
                if last_move[0].isdigit():
                    last_move = moves[len(moves) - 5: len(moves)]
                if last_move != self.games[game_id][1]:
                    self.games[game_id][0].push(chess.Move.from_uci(last_move))
                    self.games[game_id][1] = last_move
                    self.make_move(game_id)
            full_response = ''

    # everything below is taken from the provided lichess python client, with some minor exceptions
    def stream_incoming_events(self):
        '''Get your realtime stream of incoming events.
        :return: stream of incoming events
        :rtype: iterator over the stream of events
        '''
        path = 'stream/event'
        yield from self._r.get(base_path + path, stream=True, headers=self._r.headers)
    
    def make_move(self, game_id):
        '''
            Make a move in a bot game.
            :param str game_id: ID of a game
            :param str move: move to make
            :return: success
            :rtype: bool
        '''
        move = self.model.make_move(self.games[game_id][0])
        self.games[game_id][0].push(chess.Move.from_uci(move))
        self.games[game_id][1] = move
        path = f'bot/game/{game_id}/move/{move}'
        return self._r.post(base_path + path, headers=self._r.headers)
    
    def accept_challenge(self, challenge_id):
        '''Accept an incoming challenge.
        :param str challenge_id: ID of a challenge
        :return: success
        :rtype: bool
        '''
        path = f'challenge/{challenge_id}/accept'
        return self._r.post(base_path + path, headers=self._r.headers)

    def stream_game_state(self, game_id):
        '''Get the stream of events for a board game.

        :param str game_id: ID of a game
        :return: iterator over game states
        '''
        print(game_id)
        path = f'bot/game/stream/{game_id}'
        yield from self._r.get(base_path + path, stream=True, headers=self._r.headers)

# from the lichess provided api documentation
class TokenSession(requests.Session):
    '''Session capable of personal API token authentication.
    :param str token: personal API token
    '''

    def __init__(self, token):
        super().__init__()
        self.token = token
        self.headers = {'Authorization': 'Bearer ' + token}