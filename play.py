from game.game import Game,RandomPlayer,HumanPlayer

if __name__ == '__main__':
    player_1 = RandomPlayer()
    player_2 = HumanPlayer()
    game = Game(player_1=player_1,player_2=player_2)
    print('Starting Game')
    while not game.game_over():
        print(game.game_field)
        # This is always valid
        move = game.active_player.get_move(game.game_field) # move is y,x
        game.game_field.make_move(move)
        new_full_fields = game.game_field.new_full_fields(move)
        game.active_player.points += new_full_fields

        if new_full_fields == 0:
            game.change_player()

    print(f'Winner is: {game.winner}')
    print(game.game_field)