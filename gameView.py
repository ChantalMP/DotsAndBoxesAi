import random
import numpy as np

square_size = 8
width = square_size
height = square_size
num_actions = square_size*(square_size+1)*2
max_obstacles = 2
max_obstacle_width = 3

class MyException(Exception):
    pass


def init_Field():
    # rows
    rows = np.zeros(shape=(width + 1, height), dtype=int)
    # columns
    columns = np.zeros(shape=(width, height + 1), dtype=int)
    for i in range(0, width):
        rows[0][i] = 1
        rows[width][i] = 1
        columns[i][0] = 1
        columns[i][width] = 1

    rows, columns = create_obstacles(rows, columns)
    return rows, columns


def create_obstacles(rows, columns):
    obstacle_number = random.randrange(1, max_obstacles+1)

    for i in range(obstacle_number):
        obstacle_length = random.randrange(1, max_obstacle_width+1)
        #find random but free place -> place = column border on the right (like full fields)
        fields = []
        obstacle_length_intern = obstacle_length
        while(len(fields) != obstacle_length):
            fields = []
            h = random.randrange(0, height)
            w = random.randrange(0, width)
            if not test_field_full(rows, columns, h, w):
                fields.append((h, w))
                obstacle_length_intern -= 1
                while (obstacle_length_intern != 0):
                    dir = random.randrange(0, 2)
                    if dir == 0:  # vertical
                        h = h + 1 if h == 0 else h - 1
                        w = w
                    else:  # horizontal
                        w = w + 1 if w == 0 else w - 1
                        h = h
                    if not test_field_full(rows, columns, h, w):
                        if (h, w) not in fields:
                            fields.append((h, w))
                            obstacle_length_intern -= 1
                        else:
                            break  # try again
                    else:
                        break  # try again

        for elem in fields:
            setField(rows, columns, elem[0], elem[1])

    return rows, columns


def setField(rows, columns, h, w):
    global height, width
    if h >= height or w >= width:
        raise MyException("Invalid width or height")
    columns[h][w] = 1
    columns[h][w + 1] = 1
    rows[h][w] = 1
    rows[h + 1][w] = 1
    return rows, columns


#test field right to given edgen
#called when painting the vertical border
def test_field_full(rows, columns, height, width):
    if width < len(columns):
        if columns[height][width] and columns[height][width+1] and rows[height][width] == 1 and rows[height+1][width] == 1:
                    return True
        else:
            return False
    else:
        return False

def field_to_str(rows, columns):
    out = "  "
    for i in range(0,width):
        if i < 10:
            out += '  '+str(i)
        else:
            out+=' ' + str(i)
    out += "\n"
    # Because the array sizes are different, a one size fits all approach requires expection handling
    for h in range(height+1):
        # one for columns, one for rows
        for i in range(2):
            for w in range(width+1):
                # catch if too big
                if i == 0 and w == 0:
                        out += '  '
                if i == 0 and w < len(rows[0]):
                    if rows[h][w] == 1 :
                        out += " --"
                    else:
                        out += "   "
                elif i == 1 and h < len(columns):
                    if w == 0:
                        if h < 10:
                            out += ' ' +str(h)
                        else:
                            out += str(h)

                    if columns[h][w] == 1:
                        if test_field_full(rows, columns, h, w):
                            out += "| x"
                        else:
                            out += "|  "
                    else:
                        out += "   "
            out += "\n"

    return out