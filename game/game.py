import numpy as np
import random
from game.game_field import GameField
from abc import abstractmethod

import torch


class Player:
    def __init__(self):
        self.points = 0

    @abstractmethod
    def get_move(self, game_field: GameField):
        pass


class AiPlayer(Player):
    def __init__(self, is_student, model):
        super().__init__()
        self.is_student = is_student
        self.model = model
        self.eps_threshold = None

    def get_move(self, game_field: GameField, eps_threshold: float, device):
        sample = random.random()
        valid_moves = game_field.valid_moves()
        self.model.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the largest expected reward.
                state = torch.FloatTensor(game_field.field, device=device)
                action = self.model(state.unsqueeze(dim=0).unsqueeze(dim=0)).max(1)[1].view(1, 1)
                move = game_field.convert_lineidx_to_move(action.item())
                if move in valid_moves:
                    return move

        # If exploration or Ai choose unvalid move pick random move
        move = random.choice(valid_moves)
        return move


class HumanPlayer(Player):
    def get_move(self, game_field: GameField):
        valid_moves = game_field.valid_moves()
        move = (-1, -1)
        while move not in valid_moves:
            input_move = input('Please give Row,Column').split(',')
            move = int(input_move[0]), int(input_move[1])

        return move


class RandomPlayer(Player):
    def get_move(self, game_field: GameField):
        valid_moves = game_field.valid_moves()
        move = random.choice(valid_moves)

        print('Making random move')

        return move


class Game:
    # TODO HOW many new fields?
    # TODO is move valid?
    # TODO is game over?
    # TODO get_player_score?
    # TODO Random move Ai move and User move instead of make move

    def __init__(self, player_1: Player, player_2: Player):
        self.game_field = GameField()
        self.player1 = player_1
        self.player2 = player_2
        self.active_player: Player = random.choice([self.player1, self.player2])

    @property
    def winner(self):
        if self.player1.points == self.player2.points:
            return None
        else:
            return self.player1 if self.player1.points > self.player2.points else self.player2

    def change_player(self):
        self.active_player = self.player1 if self.active_player == self.player2 else self.player2

    def game_over(self):
        return -1 not in self.game_field.field
