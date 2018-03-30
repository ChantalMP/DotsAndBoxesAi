import pygame
import time
from gameView import width, height, test_field_full
from gamePlay import init_Field

pygame.init()
display_width = width*100
display_height = height*100

black = (0, 0, 0)
white = (255, 255, 255)
grey = (235, 235, 235)
dark_grey = (100,100,100)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)

# img =  pygame.image.load('racecar.png)
# drawing: gameDisplay.blit(img, (x,y))

gameDisplay = pygame.display.set_mode((display_width, display_height))  # width and height
pygame.display.set_caption("Dots and Boxes AI")
clock = pygame.time.Clock()

gameDisplay.fill(white)

rows,  columns = init_Field()
line_length = 80
line_thickness = 10
vertical_space = 100
horizontal_space = 80
lines = []

def draw_full_field(h, w, color = dark_grey):
    print(h, w)
    rect = pygame.Rect(w*line_length+horizontal_space+line_thickness, h*line_length+line_thickness+vertical_space, line_length-line_thickness, line_length-line_thickness)
    pygame.draw.rect(gameDisplay, color, rect)

def define_lines(rows, columns):

    for i in range(0, width):
        for h in range(height + 1):
            # one for columns, one for rows
            for i in range(2):
                for w in range(width + 1):
                    # catch if too big
                    if i == 0 and w < len(rows[0]):
                        #horizontal
                        l = pygame.Rect(w*line_length + horizontal_space+line_thickness, h*line_length+vertical_space, line_length-line_thickness, line_thickness)
                        lines.append(l)
                        if rows[h][w] == 1:
                            pygame.draw.rect(gameDisplay, black, l)
                        else:
                            pygame.draw.rect(gameDisplay, grey, l)
                    elif i == 1 and h < len(columns):
                        #vertical
                        l = pygame.Rect(w * line_length + horizontal_space, h * line_length + vertical_space+line_thickness, line_thickness, line_length-line_thickness)
                        lines.append(l)
                        if columns[h][w] == 1:
                            pygame.draw.rect(gameDisplay, black, l)
                            if test_field_full(rows, columns, h, w):
                                draw_full_field(h, w)
                        else:
                            pygame.draw.rect(gameDisplay, grey, l)

define_lines(rows, columns)


def text_objects(text, font):
    textSurface = font.render(text, True, black)
    return textSurface, textSurface.get_rect()


def message_display(text):
    largeText = pygame.font.Font('freesansbold.ttf', 115)
    TextSuf, TextRect = text_objects(text, largeText)
    TextRect.center = ((display_width / 2), (display_height / 2))
    gameDisplay.blit(TextSuf, TextRect)
    pygame.display.update()
    time.sleep(3)

    game_loop()


def game_over():
    message_display('You loose!')


def game_loop():
    gameexit = False
    gameover = False
    while not gameexit:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.MOUSEBUTTONUP:
                print("CLICKED!!!")
                pos = pygame.mouse.get_pos()
                if my_rect.collidepoint(pos):
                    print("hey")
                    pygame.draw.rect(gameDisplay, red, my_rect)
                    pygame.display.update()
                    time.sleep(0.5)
                    pygame.draw.rect(gameDisplay, black, my_rect)


                gameover = True

        if gameover:
            print("over")
            game_over()

            # print(event)

        pygame.display.update()
        clock.tick(60)  # frames per second


game_loop()
