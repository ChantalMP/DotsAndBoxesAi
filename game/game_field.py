import numpy as np
import random

class GameField:

    def __init__(self, max_obstacles = 2, max_obstacle_width = 3):
        self.square_size = 8
        self.num_actions = self.square_size * (self.square_size + 1) * 2
        self.max_obstacles = max_obstacles
        self.max_obstacle_width = max_obstacle_width
        self.representation_size = self.square_size*2+1
        self.game_field = self.init_field()
        self.create_obstacles()

    def init_field(self):
        field = -np.ones(shape=(self.representation_size, self.representation_size))
        #set borders
        border = np.array([0,1]*self.square_size+[0])
        field[0] = field[-1] = field[:,0] = field[:,-1] = border
        #set inner
        inner_line_1 = np.array([0, -1] * (self.square_size-1) + [0])
        inner_line_2 = np.array([-1, 0] * (self.square_size - 1) + [-1])
        ind1 = [i for i in range(1, self.representation_size-1, 2)]
        ind2 = [i for i in range(2, self.representation_size - 1, 2)]
        field[ind1,1:-1]=inner_line_1
        field[ind2,1:-1] = inner_line_2

        return field

    def create_obstacles(self):
        obstacle_number = random.randrange(1, self.max_obstacles + 1)
        obstacle_lengths = np.random.randint(1, self.max_obstacle_width+1, obstacle_number)
        obstacle_starts = np.random.randint(0, self.square_size*self.square_size, obstacle_number)
        obstacle_ends = obstacle_starts+obstacle_lengths
        for start, end in zip(obstacle_starts, obstacle_ends):
            horizontal = random.randint(0,1)
            for i in range(start, min(end+1, self.square_size*self.square_size-1)):
                if horizontal:
                    self.paint_square(i)
                else:
                    self.game_field = self.game_field.T
                    self.paint_square(i)
                    self.game_field = self.game_field.T

    def paint_square(self, square):
        y = int(square/self.square_size) * 2
        x = (square%self.square_size) * 2 + 1
        self.game_field[y,x] = 1
        self.game_field[y+1, x-1] = 1
        self.game_field[y+1, x+1] = 1
        self.game_field[y+2, x] = 1

    def valid_moves(self):
        return list(zip(*np.where(self.game_field == -1)))

    # checks if field to right of vertical edge is full (for printing)
    def is_square_full(self, y, x):
        try:
            return (self.game_field[y,x] == 1 and self.game_field[y-1, x+1] == 1 and self.game_field[y, x+2] == 1 and self.game_field[y+1, x+1] == 1)
        except:
            return False

    def __str__(self):
        out = ""
        for y in range(self.representation_size):
            for x in range(self.representation_size):
                if y%2 == 0:
                    if self.game_field[y,x] == 1:
                        out+=" ---"
                    elif self.game_field[y,x] == -1:
                        out+= "    "
                else:
                    if self.game_field[y,x] == 1:
                        if self.is_square_full(y=y,x=x):
                            out += "| x "
                        else:
                            out += "|   "
                    elif self.game_field[y,x] == -1:
                        out += "    "
            out += "\n"
        return out

if __name__ == '__main__':
    gf = GameField()
    print(gf.max_obstacles, gf.max_obstacle_width)
    print(gf)