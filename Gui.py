import pygame
import time
from gameView import width, height, test_field_full
from gameLogic import new_full_fields, game_over,validate_move
import os, keras
from keras.models import load_model
from gameAiPlayAlwaysValidAivsAI import find_best, GameExtended


pygame.init()
display_width = width*100
display_height = height*100

black = (0, 0, 0)
white = (255, 255, 255)
grey = (235, 235, 235)
dark_grey = (100,100,100)
red = (255, 0, 0)
green = (0, 235, 20)
dark_green = (0,155,0)
blue = (0, 20, 235)
dark_blue = (0, 0, 155)

model_name = "temp_mm500_hsmin1728_hsmax3456_lr1.0_d0.5_hl3_na144_tiFalse_1.h5"
model = load_model(model_name)

global lines
# img =  pygame.image.load('racecar.png)
# drawing: gameDisplay.blit(img, (x,y))

gameDisplay = pygame.display.set_mode((display_width, display_height))  # width and height
pygame.display.set_caption("Dots and Boxes AI")
clock = pygame.time.Clock()

line_length = 80
line_thickness = 10
vertical_space = 100
horizontal_space = 80
lines = []

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

def draw_full_field(h, w, color = dark_grey):
    rect = pygame.Rect(w*line_length+horizontal_space+line_thickness, h*line_length+line_thickness+vertical_space, line_length-line_thickness, line_length-line_thickness)
    pygame.draw.rect(gameDisplay, color, rect)

def newFullField(field, which, h, w, color = dark_blue):
    if which == 0:#horizontal
        #field above
        if h != 0:
            if field[1][h-1][w] == 1 and field[0][h-1][w] == 1 and field[1][h-1][w+1] == 1:
                draw_full_field(h-1, w, color)
        #beyond
        if h != height:
            if field[1][h][w] == 1 and field[0][h+1][w] == 1 and field[1][h][w+1] == 1:
                draw_full_field(h, w, color)
    else:#vertical
        #left side
        if w != 0:
            if field[0][h][w-1] == 1 and field[1][h][w-1]==1 and field[0][h+1][w-1] == 1:
                draw_full_field(h, w-1, color)
        # right side
        if w != width:
            if field[0][h][w] == 1 and field[1][h][w+1]==1 and field[0][h+1][w] == 1:
                draw_full_field(h, w, color)

def draw_full_fields(action_num, field, color):
    array_i, h, w = convert_action_to_move(action_num)
    newFullField(field, array_i, h, w, color)

def draw_move(action, field, color):
    global lines
    new_line = lines[action]
    array_i_, h_, w_ = convert_action_to_move(action)
    field[array_i_][h_][w_] = 1
    pygame.draw.rect(gameDisplay, red, new_line)
    pygame.display.update()
    pygame.time.wait(500)
    pygame.draw.rect(gameDisplay, color, new_line)
    field_color = green if color == dark_green else blue
    draw_full_fields(action, field, field_color)
    pygame.display.update()

    return field


def define_lines(rows, columns):
    my_line_array = []
    for h in range(height + 1):
        # one for columns, one for rows
        for i in range(2):
            for w in range(width + 1):
                # catch if too big
                if i == 0 and w < len(rows[0]):
                    #horizontal
                    l = pygame.Rect(w*line_length + horizontal_space+line_thickness, h*line_length+vertical_space, line_length-line_thickness, line_thickness)
                    my_line_array.append(l)
                    if rows[h][w] == 1:
                        pygame.draw.rect(gameDisplay, black, l)
                    else:
                        pygame.draw.rect(gameDisplay, grey, l)
                elif i == 1 and h < len(columns):
                    #vertical
                    l = pygame.Rect(w * line_length + horizontal_space, h * line_length + vertical_space+line_thickness, line_thickness, line_length-line_thickness)
                    my_line_array.append(l)
                    if columns[h][w] == 1:
                        pygame.draw.rect(gameDisplay, black, l)
                        if test_field_full(rows, columns, h, w):
                            draw_full_field(h, w)
                    else:
                        pygame.draw.rect(gameDisplay, grey, l)
    return my_line_array

def text_objects(text, font):
    textSurface = font.render(text, True, black)
    return textSurface, textSurface.get_rect()


def message_display(text):
    rect = pygame.Rect(80, 40, 400, 30)
    pygame.draw.rect(gameDisplay, white, rect)
    largeText = pygame.font.Font('freesansbold.ttf', 30)
    TextSuf, TextRect = text_objects(text, largeText)
    TextRect.topleft = (80, 40)
    gameDisplay.blit(TextSuf, TextRect)
    pygame.display.update()


def game_over_show(env, user_nr, ai_nr):
    points_user = env.calculate_active_player(user_nr)["Points"]
    points_ai = env.calculate_active_player(ai_nr)["Points"]
    if points_user > points_ai:
        message_display('You win with {} points! Ai points: {}'.format(points_user, points_ai))
    else:
        message_display('You lose with {} points! Ai points: {}'.format(points_user, points_ai))



def ai_move(field, env, ai_number):
    ais_turn = True
    while ais_turn:
        ais_turn = False
        input = env.convert_and_reshape_field_to_inputarray(field)
        action = find_best(model.predict(input)[0], env)
        array_i, h, w = convert_action_to_move(action)
        field = draw_move(action, field, dark_green)
        new_fields = new_full_fields(field, array_i, h, w)
        env.calculate_active_player(ai_number)["Points"] += new_fields
        user_nr = 1 if ai_number == 2 else 2
        print_points(env.calculate_active_player(user_nr)["Points"], env.calculate_active_player(ai_number)["Points"])
        if game_over(env):
            return field, True
        if new_fields != 0:
            ais_turn = True
            pygame.time.wait(500)

    return field, False

def print_points(points_user, points_ai):
    rect = pygame.Rect(80,40, 400, 30)
    pygame.draw.rect(gameDisplay, white, rect)
    myfont = pygame.font.SysFont(None, 40)
    points = myfont.render("Points User: {}, Points Ai: {}".format(points_user, points_ai), 1, black)
    gameDisplay.blit(points, (80,40))

def print_time(time):
    rect = pygame.Rect(720, 40, 400, 30)
    pygame.draw.rect(gameDisplay, white, rect)
    myfont = pygame.font.SysFont(None, 40)
    points = myfont.render("{}".format(time), 1, black)
    gameDisplay.blit(points, (720, 40))

def game_loop_ai_vs_user():
    gameDisplay.fill(white)
    gameexit = False
    gameover = False
    env = GameExtended()
    print_points(0,0)
    rows, columns = env.rows, env.columns
    global lines
    lines = define_lines(rows, columns)
    field = [rows, columns]
    user_number = 1
    ai_number = 2
    pygame.display.update()
    timer = time.time()
    while not gameexit:
        time_left = 5 - int(time.time() - timer)
        # print_time(time_left)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.MOUSEBUTTONUP and not gameover:

                pos = pygame.mouse.get_pos()
                for idx, line in enumerate(lines):
                    array_i, h, w = convert_action_to_move(idx)
                    if line.collidepoint(pos) and field[array_i][h][w] != 1:
                        #user move
                        draw_move(idx, field, dark_blue)
                        array_i, h, w = convert_action_to_move(idx)
                        new_fields = new_full_fields(field, array_i, h, w)
                        env.calculate_active_player(user_number)["Points"] += new_fields
                        print_points(env.calculate_active_player(user_number)["Points"], env.calculate_active_player(ai_number)["Points"])
                        if game_over(env):
                            gameover = True
                            break
                        if new_fields == 0:
                            pygame.event.set_blocked(pygame.MOUSEBUTTONUP)
                            field, gameover = ai_move(field, env, ai_number)
                            pygame.event.set_allowed(pygame.MOUSEBUTTONUP)
                        timer = time.time()
                        break


        if gameover:
            game_over_show(env, user_number, ai_number)
            pygame.time.wait(5000)
            game_loop_ai_vs_user()


        pygame.display.update()
        clock.tick(10)  # frames per second

game_loop_ai_vs_user()