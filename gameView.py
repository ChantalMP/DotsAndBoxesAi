import random
import numpy as np
width = 4
height = 4
# obstacle_length = random.randrange(1,3)
# obstacle_number = random.randrange(1,3)

def init_Field():
    # rows
    rows = np.zeros(shape=(width+1 ,height) , dtype= int)
    # columns
    columns = np.zeros(shape=(width , height+1) , dtype= int)
    for i in range(0 , width):
        rows[0][i] = 1
        rows[width][i] = 1
        columns[i][0] = 1
        columns[i][width] = 1

    return  rows,columns

#kann man später wieder löschen
def test_config():
    # rows
    rows, columns = init_Field()

    columns[0][1] = 1
    rows[1][0] = 1

    columns[2][2] = 1
    columns[2][3] = 1
    rows[2][2] = 1
    rows[3][2] = 1

    #full
    rows = np.ones(shape=(width + 1, height), dtype=int)
    columns = np.ones(shape=(width, height + 1), dtype=int)

    return  rows,columns

#does field belong to one user or obstacle?
#test field left to possible edge
#called when painting the vertical border
def test_field_full(rows, columns, height, weight):
    if weight < len(columns):
        if columns[height][weight+1] ==  1:
            if rows[height][weight] == 1:
                if rows[height+1][weight] == 1:
                    return True
        else:
            return False

def print_Field(rows, columns):
    out = ""
    # Because the array sizes are different, a one size fits all approach requires expection handling
    for h in range(height+1):
        # one for columns, one for rows
        for i in range(2):
            for w in range(width+1):
                # catch if too big
                if i == 0 and w < len(rows[0]):
                    if rows[h][w] == 1 :
                        out += " --"
                    else:
                        out += "   "
                elif i == 1 and h < len(columns):
                    if columns[h][w] == 1 and i == 1 : #warum fragst du nochmal ob i == 1 ist?
                        if test_field_full(rows, columns, h, w):
                            out += "| x"
                        else:
                            out += "|  "
                    else:
                        out += "   "
            out += "\n"

    return out

'''
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
'''

zeilen,spalten = init_Field()
zeilen,spalten = test_config()
outstr = print_Field(rows=zeilen, columns= spalten)
print(outstr)