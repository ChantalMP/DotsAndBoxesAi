from gameView import width, height

def new_full_fields(field, which, h, w):
    ret = 0
    if which == 0:#horizontal
        #field above
        if h != 0:
            if field[1][h-1][w] == 1 and field[0][h-1][w] == 1 and field[1][h-1][w+1] == 1:
                ret +=1
        #beyond
        if h != height:
            if field[1][h][w] == 1 and field[0][h+1][w] == 1 and field[1][h][w+1] == 1:
                ret += 1
    else:#vertical
        #left side
        if w != 0:
            if field[0][h][w-1] == 1 and field[1][h][w-1]==1 and field[0][h+1][w-1] == 1:
                ret += 1
        # right side
        if w != width:
            if field[0][h][w] == 1 and field[1][h][w+1]==1 and field[0][h+1][w] == 1:
                ret += 1
    return ret

def validate_move(field_arrays, array_i, height, width):
    try:
        if field_arrays[array_i][height][width] == 1:
            return False
        else:
            return True
    except:
        return False

def game_over(game):
    obstacle_count = game.obstacle_count
    p1 = game.player1["Points"]
    p2 = game.player2["Points"]
    if obstacle_count + p1 + p2 == (width * height):
        return True
    else:
        return False


