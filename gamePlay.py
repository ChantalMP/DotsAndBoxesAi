import random
from gameView import init_Field , print_Field
from gameLogic import validate_move , newFullField

rows, columns = init_Field()
outstr = print_Field(rows=rows, columns=columns)
player1 = {"Name": "Player1", "Points": 0}
player2 = {"Name": "Player2", "Points": 0}
whose_turn = random.randint(0, 1)
field_arrays = [rows, columns]




def calculate_active_player(whose_turn):
    if whose_turn == 0:
        return player1
    else:
        return player2


def make_move(array_i, height, width):
    if validate_move(field_arrays, array_i, height, width) == True:
        print("making move")
        field_arrays[array_i][height][width] = 1
        return True
    else:
        return False


def convert_user_move_to_array(move):
    try:
        move = move.split(" ")
        move = list(map(int, move))
        print(move)
        if len(move) != 3:
            return False
        else:
            return move
    except:
        return False

print(print_Field(field_arrays[0], field_arrays[1]))

while (True):
    active_player = calculate_active_player(whose_turn)
    move = input("{} make your move!".format(active_player["Name"]))
    move = convert_user_move_to_array(move)
    # then move invalid
    if move == False:
        print("move false")
        continue

    if make_move(move[0], move[1], move[2]) == True:
        print("success")
        if newFullField(field_arrays, move[0], move[1], move[2]):
            active_player["Points"]+=1
        else:
            whose_turn = 1 - whose_turn

    else:
        print("failed")

    print(print_Field(field_arrays[0], field_arrays[1]))
    print(player1)
    print(player2)

