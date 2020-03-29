import pygame
from game.game import Game, HumanPlayer, AiPlayer, RandomPlayer, Player

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
                if self.game.game_field.field[y, x] != 0:
                    if y % 2 == 0:
                        line = pygame.Rect(x // 2 * self.line_length + self.horizontal_space + self.line_thickness,
                                           y // 2 * self.line_length + self.vertical_space,
                                           self.line_length - self.line_thickness, self.line_thickness)
                        self.lines.append(line)
                        if self.game.game_field.field[y, x] == 1:
                            pygame.draw.rect(game_gui.game_display, black, line)
                        elif self.game.game_field.field[y, x] == -1:
                            pygame.draw.rect(game_gui.game_display, grey, line)

                    else:
                        line = pygame.Rect(x // 2 * self.line_length + self.horizontal_space,
                                           y // 2 * self.line_length + self.vertical_space + self.line_thickness,
                                           self.line_thickness,
                                           self.line_length - self.line_thickness)
                        self.lines.append(line)
                        if self.game.game_field.field[y, x] == 1:
                            pygame.draw.rect(game_gui.game_display, black, line)
                            if self.game.game_field.is_square_full(y=y, x=x):
                                self.draw_full_field(y, x, color=dark_grey)  # field on the right
                        elif self.game.game_field.field[y, x] == -1:
                            pygame.draw.rect(game_gui.game_display, grey, line)


    def draw_full_field(self, y, x, color=dark_grey):
        rect = pygame.Rect(x // 2 * self.line_length + self.horizontal_space + self.line_thickness,
                           y // 2 * self.line_length + self.line_thickness + self.vertical_space,
                           self.line_length - self.line_thickness,
                           self.line_length - self.line_thickness)
        pygame.draw.rect(self.game_display, color, rect)

    def draw_new_full_fields(self, y, x, color):
        horizontal = (y % 2) == 0

        if horizontal:
            # Above
            if y != 0:
                if self.game.game_field.check_square(y, x, direction='Above'):
                    self.draw_full_field(y - 2, x, color)
            # Below
            if y != self.game.game_field.representation_size:
                if self.game.game_field.check_square(y, x, direction='Below'):
                    self.draw_full_field(y, x, color)
        else:
            # Left
            if x != 0:
                if self.game.game_field.check_square(y, x, direction='Left'):
                    self.draw_full_field(y, x - 2, color)
            # Right
            if x != self.game.game_field.representation_size:
                if self.game.game_field.check_square(y, x, direction='Right'):
                    self.draw_full_field(y, x, color)

        pygame.display.update()

    def draw_move(self, idx, y, x, color):
        line = self.lines[idx]
        self.game.game_field.make_move((y, x))
        pygame.draw.rect(self.game_display, red, line)
        pygame.display.update()
        pygame.event.pump()
        pygame.time.wait(500)
        pygame.draw.rect(self.game_display, color, line)
        pygame.display.update()
        field_color = green if color == dark_green else blue
        self.draw_new_full_fields(y, x, field_color)

    def random_move(self):
        while not game_gui.game.game_over():
            y, x = player_1.get_move(game_gui.game.game_field)  # Always valid
            idx = game_gui.game.game_field.convert_move_to_lineidx(y, x)
            game_gui.draw_move(idx, y, x, color=dark_green)
            new_full_fields = game_gui.game.game_field.new_full_fields((y, x))
            game_gui.game.active_player.points += new_full_fields

            game_gui.print_points(game_gui.game.player2.points, game_gui.game.player1.points)
            if new_full_fields == 0:
                game_gui.game.change_player()
                break


if __name__ == '__main__':
    player_1 = RandomPlayer()
    player_2 = HumanPlayer()
    game_gui = GameGui(player_1=player_1, player_2=player_2)
    game_gui.game.active_player = player_2
    pygame.init()

    game_gui.game_display = pygame.display.set_mode((game_gui.display_width, game_gui.display_height))
    game_gui.game_display.fill(white)
    game_gui.create_lines()
    pygame.display.set_caption("Dots and Boxes AI")
    clock = pygame.time.Clock()

    pygame.display.update()
    print(len(game_gui.lines))

    while not game_gui.game.game_over():

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.MOUSEBUTTONUP and not game_gui.game.game_over():
                pos = pygame.mouse.get_pos()
                for idx, line in enumerate(game_gui.lines):
                    if line.collidepoint(pos):
                        y, x = game_gui.game.game_field.convert_lineidx_to_move(idx)
                        if game_gui.game.game_field.field[y, x] == -1:
                            game_gui.draw_move(idx, y, x, color=dark_blue)

                        new_full_fields = game_gui.game.game_field.new_full_fields((y, x))
                        game_gui.game.active_player.points += new_full_fields

                        game_gui.print_points(game_gui.game.player2.points, game_gui.game.player1.points)
                        pygame.event.pump()

                        if game_gui.game.game_over():
                            break
                        if new_full_fields == 0:
                            pygame.event.set_blocked(pygame.MOUSEBUTTONUP)
                            game_gui.game.change_player()
                            game_gui.random_move()
                            pygame.event.set_allowed(pygame.MOUSEBUTTONUP)

                        break

    print(f'Winner is: {game_gui.game.winner}')
