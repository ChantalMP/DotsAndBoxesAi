import random
from gameView import init_Field, field_to_str, test_field_full, width, height
from gameLogic import validate_move, new_full_fields, game_over
import numpy as np


# give random field and check if valid
class Game:

    def __init__(self):
        self.rows, self.columns = init_Field()
        self.outstr = field_to_str(rows=self.rows, columns=self.columns)
        self.player1 = {"Name": "Player1", "Points": 0}
        self.player2 = {"Name": "Player2", "Points": 0}
        self.whose_turn = random.randint(0, 1)
        self.field_arrays = [self.rows, self.columns]
        self.obstacle_count = self._obstacle_count()

    def get_player_score(self, player_nr):
        if player_nr == 1:
            return self.player1['Points'] - self.player2['Points']
        elif player_nr == 2:
            return self.player2['Points'] - self.player1['Points']
        else:
            return 0

    def calculate_active_player(self, whose_turn):

        if whose_turn == 1:
            return self.player1
        else:
            return self.player2

    def make_move(self, array_i, height, width):
        if validate_move(self.field_arrays, array_i, height, width) == True:
            self.field_arrays[array_i][height][width] = 1

            return True
        else:
            return False

    def random_move(self):
        success = False
        while not success and self.free_edge_count() > 0:
            w = random.randint(0, width + 1)
            h = random.randint(0, height + 1)
            i = random.randint(0, 2)
            success = self.make_move(i, h, w)
        return i, h, w

#not used
    def n_random_moves(self , n):
        for i in range(0,n):
            self.random_move()

#not used
    def convert_user_move_to_array(self, move):
        try:
            move = move.split(" ")
            move = list(map(int, move))
            if len(move) != 3:
                return False
            else:
                return move
        except:
            return False

    def free_edge_count(self):
        counter = 0
        for i in self.rows:
            for y in i:
                if y == 0:
                    counter+=1
                else:
                    continue
        for i in self.columns:
            for y in i:
                if y == 0:
                    counter+=1
                else:
                    continue
        return counter

    def _obstacle_count(self):
        c = 0
        for h in range(height):
            for w in range(width):
                if test_field_full(self.field_arrays[0], self.field_arrays[1], h, w):
                    c += 1
        return c