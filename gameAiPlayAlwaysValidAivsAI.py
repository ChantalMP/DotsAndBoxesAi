import tensorflow
import numpy as np
import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd
from gameLogic import *
from gamePlay import Game
import random
from gameView import width, height, field_to_str, num_actions
from keras.models import load_model
import os.path
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras import losses
from keras import optimizers

# similar result but faster then True
train_mode_immediate = False
# random moves


epsilon = 0.05


class GameExtended(Game):
    def __init__(self):
        super().__init__()
        self.random_plays = 0

    def convert_action_to_move(self, action):
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

#TODO
    def convert_input_array_to_field(self, input):
        a = self.rows
        b = self.columns
        field = [a, b]
        for i in range(len(input)):
            array_i, h, w = self.convert_action_to_move(i)
            field[array_i][h][w] = input[i]
        return field

#TODO
    def convert_field_to_inputarray(self, field):
        # field = [rows, colomns]
        input = np.zeros(num_actions)
        index = 0
        for h in range(height + 1):
            # one for columns, one for rows
            for i in range(2):
                for w in range(width + 1):
                    # catch if too big
                    if i == 0 and w < len(field[0][0]):
                        if field[0][h][w] == 1:
                            input[index] = 1
                        index += 1
                    elif i == 1 and h < len(field[1]):
                        if field[1][h][w] == 1:
                            input[index] = 1
                        index += 1

        return input

#TODO
    def convert_and_reshape_field_to_inputarray(self, field):
        input_array = self.convert_field_to_inputarray(field)
        r_input_array = input_array.reshape((1, -1))
        return r_input_array

    # action = move
    #TODO
    def _update_state(self, action, playernr):
        array_i, height, width = self.convert_action_to_move(action)
        # delete this not used
        old_field = [self.rows, self.columns]
        self.success = self.make_move(array_i, height, width)
        if not self.success:
            self.random_plays += 1
            array_i, height, width = self.random_move()
        new_fields = new_full_fields([self.rows, self.columns], array_i, height, width)
        self.calculate_active_player(playernr)["Points"] += new_fields

    def _get_reward(self, playernr, old_score):
        return (self.get_player_score(playernr) - old_score)

#TODO
    def act(self, action, playernr):
        old_score = self.get_player_score(playernr)
        self._update_state(action, playernr)
        # reward = self._get_reward(playernr, old_score)
        gameover = game_over(self)
        return self.convert_and_reshape_field_to_inputarray([self.rows, self.columns]), old_score, gameover

#TODO
    def random_act(self, playernr):
        success = False
        while not success and self.free_edge_count() > 0:
            action = random.randint(0, num_actions)
            array_i, h, w = self.convert_action_to_move(action)
            success = validate_move([self.rows, self.columns], array_i, h, w)
            if success:
                self._update_state(action, playernr)
                gameover = game_over(self)
                return self.convert_and_reshape_field_to_inputarray([self.rows, self.columns]), gameover


class Ai:
    def __init__(self, playernr, model_name, max_memory=100, discount=.9):
        self.playernr = playernr
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount
        self.model_name = model_name
        self.model = self.init_model(self.model_name)

    def init_model(self, model_name):
        #     keras
        model = Sequential()
        model.add(Dense(hidden_size_0, input_shape=(num_actions,), activation='relu'))
        model.add(Dense(hidden_size_1, activation='relu'))
        model.add(Dense(hidden_size_0, activation='relu'))
        model.add(Dense(num_actions))  # output layer
        model.compile(optimizer=optimizers.adadelta(lr=learning_rate), loss=losses.mse)
        if os.path.isfile(temp_model(self.model_name)):
            model = load_model(temp_model(self.model_name))
            print("model_loaded")

        return model

#TODO
    def remember(self, states, gameover):
        self.memory.append([states, gameover])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

#TODO

    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)
        env_dim = num_actions
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((min(len_memory, batch_size), num_actions))

        arr = np.random.randint(0, len_memory, size=(min(len_memory, batch_size)))
        for i, idx in enumerate(arr):
            state_t, action_t, reward_t, state_next = self.memory[idx][0]
            gameover = self.memory[idx][1]
            inputs[i:i + 1] = state_t
            targets[i] = model.predict(state_t)[0]
            if gameover:
                targets[i, action_t] = reward_t
            else:
                Q_sa = find_best_for_state(model.predict(state_next)[0], state_next)
                targets[i, action_t] = reward_t + self.discount * Q_sa

        return inputs, targets


# tensorboard logging method simplified for our project
def write_log(callback, train_loss, ai_wins, ai_fields,latest_wins, batch_no):
    global epsilon
    summary = tf.Summary()
    # add train_loss
    summary_value = summary.value.add()
    summary_value.simple_value = train_loss
    summary_value.tag = "train_loss"
    # add ai_wins
    summary_value = summary.value.add()
    summary_value.simple_value = ai_wins
    summary_value.tag = "ai_wins"
    # add random_moves
    summary_value = summary.value.add()
    summary_value.simple_value = ai_fields
    summary_value.tag = "ai_fields"
    summary_value = summary.value.add()
    summary_value.simple_value = latest_wins
    summary_value.tag = "latest_wins_in_100_turns"
    callback.writer.add_summary(summary, batch_no)
    callback.writer.flush()

#TODO state?
def find_best_for_state(q, state):
    index = np.argmax(q)
    prediction = np.max(q)
    tmp = np.copy(q)
    while state[0][index] != 0:
        tmp[index] = -100000
        index = np.argmax(tmp)
        prediction = np.max(tmp)
    return prediction

#TODO
def find_best(q, env):
    action = np.argmax(q)
    array_i, h, w = env.convert_action_to_move(action)
    tmp = np.copy(q)
    while not validate_move([env.rows, env.columns], array_i, h, w):
        tmp[action] = -100000
        action = np.argmax(tmp)
        array_i, h, w = env.convert_action_to_move(action)
    return action

#not used in this version
def random_player_move(gameover, playernr):
    input = False

    random_should_play = True
    while not gameover and random_should_play:
        random_should_play = False
        playernr = playernr
        old_points = env.player2["Points"]
        input, gameover = env.random_act(playernr)
        new_points = env.player2["Points"]
        if new_points > old_points:
            random_should_play = True
        if verbose:
            pass
            # print("Random PLAYED")
            # print(field_to_str(env.rows, env.columns))

    return input, gameover


# here we should define a taker_player_move
# return an action(maybefalse) and if it took
#TODO
def taker_player_move():
    did_take = False
    for i in range(0,num_actions):
        array_i, h, w = env.convert_action_to_move(i)
        if validate_move([env.rows,env.columns], array_i, h, w):
            did_take = True if new_full_fields([env.rows,env.columns],array_i,h,w) > 0 else False
            if did_take:
                return i,did_take

    return False, did_take

#TODO
def ai_player_move(input, gameover, ai: Ai, loss, use_taker_player:bool):
    action = False
    old_score = False
    input_old = False
    active_player = env.player1 if ai.playernr == 1 else env.player2

    ai_should_play = True
    while ai_should_play and not gameover:
        ai_should_play = False

        playernr = ai.playernr
        input_old = input
        # sometimes  guessing is better than predicting
        # get next action
        if np.random.rand() <= epsilon:
            valid = False
            if verbose:
                pass
                #print("THIS WAS JUST A GUESS")
            while not valid:
                action = random.randint(0, num_actions - 1)
                array_i, h, w = env.convert_action_to_move(action)
                valid = validate_move([env.rows, env.columns], array_i, h, w)
        else:
            did_take = False
            if use_taker_player:
                action,did_take = taker_player_move()
            if not did_take:
                q = ai.model.predict(input_old)
                action = find_best(q[0], env)

            predicted = True

        # apply action, get rewards and new state
        old_points = active_player["Points"]
        input, old_score, gameover = env.act(action, playernr)
        new_points = active_player["Points"]
        if new_points > old_points:
            ai_should_play = True
        if verbose:
            pass
            # print("AI {} PLAYED".format(ai.playernr))
            # print(field_to_str(env.rows, env.columns))
        if ai_should_play:
            winner = None
            if gameover:
                winnernr = 1 if env.player1["Points"] > env.player2["Points"] else 2
                if ai.playernr == winnernr:
                    winner = True
                else:
                    winner = False
            if champion != ai.playernr:
                loss = evaluate_ai(loss, ai, old_score, input_old, action, input, gameover, batch_size, game_count,winner=winner)

    return input, gameover, old_score, input_old, action, loss

#TODO
def evaluate_ai(loss, ai: Ai, old_score, input_old, action, input, gameover, batch_size,game_count, winner=None):
    global epsilon, epsilon_min, epsilon_decay

    reward = env._get_reward(playernr=ai.playernr, old_score=old_score)
    if winner == True:
        reward += 0
    elif winner == False:
        reward -= 0
    # store experience
    ai.remember([input_old, action, reward, input], gameover)
    # adapt model
    if train_mode_immediate:
        inputs, targets = ai.get_batch(ai.model, batch_size=batch_size)
        loss += ai.model.train_on_batch(inputs, targets)
    else:
        if gameover and game_count % 4 == 0:
            inputs, targets = ai.get_batch(ai.model, batch_size=batch_size)
            loss = ai.model.train_on_batch(inputs, targets)

    return loss

def temp_model(model_name):
    return "temp_{}".format(model_name)

def learning_ai(ai_1, ai_2, champion):
    if champion == 1:
        return ai_2
    else:
        return ai_1

#TODO
if __name__ == "__main__":
    # TODO
    # randomly choose best move from best 5
    epoch = 80000000
    max_memory = 1 if train_mode_immediate else 500
    hidden_size_0 = num_actions * 12
    hidden_size_1 = num_actions * 24
    batch_size = 1 if train_mode_immediate else 200
    learning_rate = 1.0
    # learning_rate 1.0 for adadelta
    # only needed for sgd
    # decay_rate = learning_rate/epoch
    discount = 0.5
    champion = 1
    if not os.path.isfile('champion.txt'):
        champion_file = open('champion.txt', 'w')
        champion_file.write("1")
        champion_file.close()
    else:
        champion_file = open('champion.txt', 'r')
        for line in champion_file:
            champion = int(line)
        champion_file.close()

    model_name = "mm{}_hsmin{}_hsmax{}_lr{}_d{}_hl{}_na{}_ti{}".format(max_memory, hidden_size_0, hidden_size_1,
                                                                       learning_rate, discount, "3", num_actions,
                                                                       train_mode_immediate)
    model_name_1 = model_name + "_1.h5"
    model_name_2 = model_name + "_2.h5"
    ai_player_1 = Ai(max_memory=max_memory, playernr=1, discount=discount, model_name=model_name_1)
    ai_player_2 = Ai(max_memory=max_memory, playernr=2, discount=discount, model_name=model_name_2)
    model_name_1 = "temp_" + model_name_1
    model_name_2 = "temp_" + model_name_2
    model_epochs_trained = 0

    if not os.path.isfile('{}.txt'.format(learning_ai(ai_player_1, ai_player_2, champion).model_name)):
        training_file = open('{}.txt'.format(learning_ai(ai_player_1, ai_player_2, champion).model_name), 'w')
    training_file = open('{}.txt'.format(learning_ai(ai_player_1, ai_player_2, champion).model_name), 'r')
    model_save_found = False
    epsilon_found = False

    for line in training_file:
        try:
            key, value = line.split(" ")
        except ValueError:
            continue
        if key == temp_model(learning_ai(ai_player_1, ai_player_2, champion).model_name):
            model_epochs_trained = value
            model_save_found = True

    training_file.close()
    if model_save_found == False:
        print("epoch save not found defaulting to 0")
        cham_ai = ai_player_1 if champion==1 else ai_player_2
        cham_file = open('{}.txt'.format(cham_ai.model_name), 'a')
        cham_file.write(
            "\n" + temp_model(cham_ai.model_name) + " " + str(0))
        cham_file.close()

        training_file = open('{}.txt'.format(learning_ai(ai_player_1, ai_player_2, champion).model_name), 'a')
        training_file.write(
            "\n" + temp_model(learning_ai(ai_player_1, ai_player_2, champion).model_name) + " " + str(0))
        training_file.close()

    # logging----- tensorboard --host 127.0.0.1 --logdir=./logs ---- logs are saved on the project directory
    log_path = './logs/' + learning_ai(ai_player_1, ai_player_2, champion).model_name
    callback = TensorBoard(log_path)
    callback.set_model(learning_ai(ai_1=ai_player_1, ai_2=ai_player_2, champion=champion))

    #     Train
    game_count = 0
    old_total_learning_wins = 0
    total_learning_wins = 0
    loss = 0.
    print(model_epochs_trained)
    for e in range(int(model_epochs_trained), epoch):
        if e % 100 == 0 and e != model_epochs_trained:
            verbose = True
            print(learning_ai(ai_player_1, ai_player_2, champion).model_name)
        else:
            verbose = False
        env = GameExtended()
        loss = 0. if train_mode_immediate else loss
        gameover = False
        predicted = False
        old_score_1 = False
        input_old_1 = False
        action_1 = False
        input_1 = env.convert_and_reshape_field_to_inputarray([env.rows, env.columns])
        old_score_2 = False
        input_old_2 = False
        action_2 = False
        input_2 = False

        # printing fields don't really help a lot right now i think
        if verbose:
            print("starting game")
            # print(field_to_str(env.rows, env.columns))

        ai_2_played = False

        # input_2 = output_1 and other way round
        while not gameover:
            # AIMOVE
            # input_2, gameover, old_score_1, input_old_1, action_1, loss = ai_player_move(input=input_1, gameover=gameover,
            #                                                                              ai=ai_player_1, loss=loss, use_taker_player=champion==1)
            input_2, gameover, old_score_1, input_old_1, action_1, loss = ai_player_move(input=input_1,
                                                                                         gameover=gameover,
                                                                                         ai=ai_player_1, loss=loss,
                                                                                         use_taker_player=False)

            if ai_2_played and champion != 2:
                winner = None
                if gameover:
                    winnernr = 1 if env.player1["Points"] > env.player2["Points"] else 2
                    if 2 == winnernr:
                        winner = True
                    else:
                        winner = False
                loss = evaluate_ai(loss, ai_player_2, old_score_2, input_old_2, action_2, input_2, gameover,
                                   batch_size, game_count,winner=winner)

            if not gameover:
                # input_1, gameover, old_score_2, input_old_2, action_2, loss = ai_player_move(input=input_2, gameover=gameover,
                #                                                                              ai=ai_player_2,
                #                                                                              loss=loss,use_taker_player=champion==2)
                input_1, gameover, old_score_2, input_old_2, action_2, loss = ai_player_move(input=input_2,
                                                                                             gameover=gameover,
                                                                                             ai=ai_player_2,
                                                                                             loss=loss,
                                                                                             use_taker_player=False)
                ai_2_played = True
                if champion != 1:
                    winner = None
                    if gameover:
                        winnernr = 1 if env.player1["Points"] > env.player2["Points"] else 2
                        if 1 == winnernr:
                            winner = True
                        else:
                            winner = False
                    loss = evaluate_ai(loss, ai_player_1, old_score_1, input_old_1, action_1, input_1, gameover,
                                       batch_size, game_count,winner=winner)

        # logging after each game saving with the epoch number.
        # play it against random

        # logging after each game saving with the epoch number.
        champion_field = env.player1["Points"] if champion == 1 else env.player2["Points"]
        learning_field = env.player1["Points"] if champion == 2 else env.player2["Points"]

        learning_wins = 0

        if learning_field > champion_field:
            learning_wins = 1
            total_learning_wins += 1

        game_count += 1
        if game_count == 100:
            old_total_learning_wins = total_learning_wins
        write_log(callback, train_loss=loss, ai_wins=learning_wins, ai_fields=learning_field,latest_wins=old_total_learning_wins, batch_no=e)

        if e % 50 == 0 and e != model_epochs_trained:
            l_ai = learning_ai(ai_1=ai_player_1, ai_2=ai_player_2, champion=champion)
            l_ai.model.save(temp_model(l_ai.model_name), overwrite=True)
            training_file = open('{}.txt'.format(l_ai.model_name), 'r')
            out = ""
            for line in training_file:
                try:
                    key, value = line.split(" ")
                except ValueError:
                    continue
                if key == temp_model(l_ai.model_name):
                    out += temp_model(l_ai.model_name) + " " + str(e)
                else:
                    out += line
                out += "\n"
            training_file.close()
            new_file = open('{}.txt'.format(l_ai.model_name), 'w')
            new_file.write(out)
            new_file.close()

        if game_count == 100:
            if total_learning_wins >= 95:
                champion = 1 if champion == 2 else 2
                champion_file = open('champion.txt', 'w')
                champion_file.write("{}".format(champion))
                champion_file.close()
                log_path = './logs/' + learning_ai(ai_player_1, ai_player_2, champion).model_name
                callback = TensorBoard(log_path)
                callback.set_model(learning_ai(ai_1=ai_player_1, ai_2=ai_player_2, champion=champion))
            game_count = 0
            total_learning_wins = 0

    l_ai = learning_ai(ai_1=ai_player_1, ai_2=ai_player_2, champion=champion)
    l_ai.model.save(l_ai.model_name, overwrite=False)
