import numpy as np
import random
from game.game_field import GameField
from abc import abstractmethod


class Player:
    def __init__(self):
        self.points = 0

    @abstractmethod
    def get_move(self, game_field:GameField):
        pass

class AiPlayer(Player):
    def __init__(self, is_student, model):
        super().__init__()
        self.is_student = is_student
        self.model = model

    def get_move(self, game_field:GameField):
        valid_moves = game_field.valid_moves()
        move = random.choice(valid_moves)
        #TODO exploration

        return move

class HumanPlayer(Player):
    def get_move(self, game_field:GameField):
        valid_moves = game_field.valid_moves()
        move = (-1,-1)
        while move not in valid_moves:
            input_move = input('Please give Row,Column').split(',')
            move = int(input_move[0]),int(input_move[1])

        return move

class RandomPlayer(Player):
    def get_move(self, game_field:GameField):
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

    def __init__(self,player_1:Player,player_2:Player):
        self.game_field = GameField()
        self.player1 = player_1
        self.player2 = player_2
        self.active_player:Player = random.choice([self.player1, self.player2])

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
