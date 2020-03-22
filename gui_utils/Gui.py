import pygame
import time
# from game.gameView import width, height, test_field_full
# from game.gameLogic import new_full_fields, game_over,validate_move
from game.game import Game, HumanPlayer, AiPlayer, RandomPlayer, Player

# from model.gameAiPlayAlwaysValidAivsAI import find_best, GameExtended

black = (0, 0, 0)
white = (255, 255, 255)
grey = (235, 235, 235)
dark_grey = (100, 100, 100)
red = (255, 0, 0)
green = (0, 235, 20)
dark_green = (0, 155, 0)
blue = (0, 20, 235)
dark_blue = (0, 0, 155)


class GameGui:
    def __init__(self, player_1: Player, player_2: Player):
        self.line_length = 80
        self.line_thickness = 10
        self.vertical_space = 100
        self.horizontal_space = 80
        self.lines = []

        self.player_1 = player_1
        self.player_2 = player_2
        self.game = Game(player_1=player_1, player_2=player_2)

        self.display_width = self.game.game_field.field_size * 100
        self.display_height = self.game.game_field.field_size * 100
        self.game_display = None

    def text_objects(self, text, font):
        textSurface = font.render(text, True, black)
        return textSurface, textSurface.get_rect()

    def message_display(self, text):
        rect = pygame.Rect(80, 40, 400, 30)
        pygame.draw.rect(self.game_display, white, rect)
        largeText = pygame.font.Font('freesansbold.ttf', 30)
        TextSuf, TextRect = self.text_objects(text, largeText)
        TextRect.topleft = (80, 40)
        self.game_display.blit(TextSuf, TextRect)
        pygame.display.update()

    def game_over_show(self, env, user_nr, ai_nr):
        points_user = env.calculate_active_player(user_nr)["Points"]
        points_ai = env.calculate_active_player(ai_nr)["Points"]
        if points_user > points_ai:
            self.message_display('You win with {} points! Ai points: {}'.format(points_user, points_ai))
        else:
            self.message_display('You lose with {} points! Ai points: {}'.format(points_user, points_ai))

    def print_points(self, points_user, points_ai):
        rect = pygame.Rect(80, 40, 400, 30)
        pygame.draw.rect(self.game_display, white, rect)
        myfont = pygame.font.SysFont(None, 40)
        points = myfont.render("Points User: {}, Points Ai: {}".format(points_user, points_ai), 1, black)
        self.game_display.blit(points, (80, 40))

    def create_lines(self):
        for y in range(self.game.game_field.representation_size):
            for x in range(self.game.game_field.representation_size):
                if y % 2 == 0:
                    line = pygame.Rect(x//2 * self.line_length + self.horizontal_space + self.line_thickness,
                                       y // 2 * self.line_length + self.vertical_space,
                                       self.line_length - self.line_thickness, self.line_thickness)
                    self.lines.append(line)
                    if self.game.game_field.field[y, x] == 1:
                        pygame.draw.rect(game_gui.game_display, black, line)
                    elif self.game.game_field.field[y, x] == -1:
                        pygame.draw.rect(game_gui.game_display, grey, line)

                else:
                    line = pygame.Rect(x // 2 * self.line_length + self.horizontal_space,
                                       y // 2 * self.line_length + self.vertical_space + self.line_thickness, self.line_thickness,
                                       self.line_length - self.line_thickness)
                    self.lines.append(line)
                    if self.game.game_field.field[y, x] == 1:
                        pygame.draw.rect(game_gui.game_display, black, line)
                        if self.game.game_field.is_square_full(y=y, x=x):
                            pass
                            #self.draw_full_field(y, x)  # field on the right
                    elif self.game.game_field.field[y, x] == -1:
                        pygame.draw.rect(game_gui.game_display, grey, line)


if __name__ == '__main__':
    player_1 = RandomPlayer()
    player_2 = HumanPlayer()
    game_gui = GameGui(player_1=player_1, player_2=player_2)
    pygame.init()

    game_gui.game_display = pygame.display.set_mode((game_gui.display_width, game_gui.display_height))
    game_gui.game_display.fill(white)
    game_gui.create_lines()
    pygame.display.set_caption("Dots and Boxes AI")
    clock = pygame.time.Clock()

    pygame.display.update()

    while not game_gui.game.game_over():

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.MOUSEBUTTONUP and not game_gui.game.game_over():
                pos = pygame.mouse.get_pos()
