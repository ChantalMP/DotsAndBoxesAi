import tensorflow
from gameLogic import *
from gamePlay import Game
import random


class ValidAi:
    def result(self, move):
        if validate_move(move[0], move[1], move[2]):
            return 1
        else:
            return 0

    def create_train_data(self,count):
        for i in range(count):
            game = Game()
            random_move_count = random.randint(0,game.free_edge_count())
            game.n_random_moves(random_move_count)

