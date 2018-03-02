import random
import numpy as np
width = 4
height = 4
# obstacle_length = random.randrange(1,3)
# obstacle_number = random.randrange(1,3)

def init_Field():
    # zeilen
    zeilen = np.zeros(shape=(width+1 ,height) , dtype= int)
    # spalten
    spalten = np.zeros(shape=(width , height+1) , dtype= int)
    for i in range(0 , width):
        zeilen[0][i] = 1
        zeilen[width][i] = 1
        spalten[i][0] = 1
        spalten[i][width] = 1

    return  zeilen,spalten

def print_Field(zeilen , spalten):
    out = ""
    # Because the array sizes are different, a one size fits all approach requires expection handling
    for h in range(height+1):
        # one for spalten one for zeilen
        for i in range(2):
            for w in range(width+1):
                # catch if too big
                if i == 0 and w < len(zeilen[0]):
                    if zeilen[h][w] == 1 :
                        out += "--"
                    else:
                        out += "  "
                elif i == 1 and h < len(spalten):
                    if spalten[h][w] == 1 and i == 1 :
                        out += "|"
                    else:
                        out += "  "
            out += "\n"

    return out

zeilen,spalten = init_Field()
outstr = print_Field(zeilen=zeilen , spalten= spalten)
print(outstr)