import numpy as np
width = 4
height = 4
rows = np.zeros(shape=(width + 1, height), dtype=int)
columns = np.zeros(shape=(width, height + 1), dtype=int)


def convert_action_to_move(action):
    array_i = 0
    w = 0
    h = 0
    for i in range(action):
        w += 1
        if w >= width + array_i:
            w = 0
            if array_i == 1:
                h += 1
            array_i = 1 - array_i

    return array_i, h, w


def convert_input_array_to_field(input):
    a = rows
    b = columns
    field = [a, b]
    for i in range(len(input)):
        array_i, h, w = convert_action_to_move(i)
        field[array_i][h][w] = input[i]
    return field


def convert_field_to_inputarray(field):
    # field = [rows, colomns]
    input = np.zeros(40)
    index = 0
    for h in range(height + 1):
        # one for columns, one for rows
        for i in range(2):
            for w in range(width + 1):
                # catch if too big
                if i == 0 and w < len(field[0][0]):
                    if field[0][h][w] == 1:
                        input[index] = 1
                elif i == 1 and h < len(field[1]):
                    if field[1][h][w] == 1:
                        input[index] = 1
                index += 1
    return input


array_i , w,h = convert_action_to_move(8)
# print(array_i, ", " , h, ", " , w)

input = np.zeros(40)
#print(input)

f = convert_input_array_to_field(input)
#print(f)
a = convert_field_to_inputarray(f)
#print(a)

a = [[1,2,3],[4,5,6]]
print(a[1,2])