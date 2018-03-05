width = 4
height = 4

def newFullField(field, which, h, w):
    if which == 0:#horizontal
        #field above
        if h != 0:
            if field[1][h-1][w] == 1 and field[0][h-1][w] == 1 and field[1][h-1][w+1] == 1:
                print(field[1][h-1][w], field[0][h-1][w], field[1][h-1][w+1])
                return True
        #beyond
        if h != height:
            if field[1][h][w] == 1 and field[0][h+1][w] == 1 and field[1][h][w+1] == 1:
                return True
    else:#vertical
        #left side
        if w != 0:
            if field[0][h][w-1] == 1 and field[1][h][w-1]==1 and field[0][h+1][w-1] == 1:
                return True
        # right side
        if w != width:
            if field[0][h][w] == 1 and field[1][h][w+1]==1 and field[0][h+1][w] == 1:
                return True
    return False
