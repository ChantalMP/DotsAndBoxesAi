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

epsilon_max = 1.0
epsilon_min = 0.01
epsilon = epsilon_max
# epsilon_decay_down = 0.999999
# epsilon_decay_up = 1 + (1 - epsilon_decay_down)
epsilon_decay = 0.999999

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

    def convert_input_array_to_field(self, input):
        a = self.rows
        b = self.columns
        field = [a, b]
        for i in range(len(input)):
            array_i, h, w = self.convert_action_to_move(i)
            field[array_i][h][w] = input[i]
        return field

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

    def convert_and_reshape_field_to_inputarray(self, field):
        input_array = self.convert_field_to_inputarray(field)
        r_input_array = input_array.reshape((1, -1))
        return r_input_array

    # action = move
    def _update_state(self, action, playernr):
        array_i, height, width = self.convert_action_to_move(action)
        old_field = [self.rows, self.columns]
        self.success = self.make_move(array_i, height, width)
        if not self.success:
            self.random_plays += 1
            array_i, height, width = self.random_move()
        new_fields = new_full_fields([self.rows, self.columns], array_i, height, width)
        self.calculate_active_player(playernr)["Points"] += new_fields

    def _get_reward(self, playernr, old_score):
        return (self.get_player_score(playernr) - old_score)

    def act(self, action, playernr):
        old_score = self.get_player_score(playernr)
        self._update_state(action, playernr)
        # reward = self._get_reward(playernr, old_score)
        gameover = game_over(self)
        return self.convert_and_reshape_field_to_inputarray([self.rows, self.columns]), old_score, gameover

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
    def __init__(self, playernr, max_memory=100, discount=.9):
        self.playernr = playernr
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, gameover):
        self.memory.append([states, gameover])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

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
def write_log(callback, train_loss, ai_wins, ai_fields, batch_no):
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
    # current_epsilon
    summary_value = summary.value.add()
    summary_value.simple_value = epsilon
    summary_value.tag = "epsilon"
    callback.writer.add_summary(summary, batch_no)
    callback.writer.flush()


def find_best_for_state(q, state):
    index = np.argmax(q)
    prediction = np.max(q)
    tmp = np.copy(q)
    while state[0][index] != 0:
        tmp[index] = -100000
        index = np.argmax(tmp)
        prediction = np.max(tmp)
    return prediction


def find_best(q, env):
    action = np.argmax(q)
    array_i, h, w = env.convert_action_to_move(action)
    tmp = np.copy(q)
    while not validate_move([env.rows, env.columns], array_i, h, w):
        tmp[action] = -100000
        action = np.argmax(tmp)
        array_i, h, w = env.convert_action_to_move(action)
    return action


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
            print("Random PLAYED")
            # print(field_to_str(env.rows, env.columns))

    return input, gameover


def ai_player_move(input, gameover, ai: Ai, model, loss):
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
                print("THIS WAS JUST A GUESS")
            while not valid:
                action = random.randint(0, num_actions - 1)
                array_i, h, w = env.convert_action_to_move(action)
                valid = validate_move([env.rows, env.columns], array_i, h, w)
        else:
            q = model.predict(input_old)
            action = find_best(q[0], env)
            predicted = True
        # apply action, get rewards and new state
        old_points = active_player["Points"]
        input, old_score, gameover = env.act(action, playernr)
        new_points = active_player["Points"]
        if new_points > old_points:
            ai_should_play = True
        if verbose:
            print("AI {} PLAYED".format(ai.playernr))
            # print(field_to_str(env.rows, env.columns))
        if ai_should_play:
            loss = evaluate_ai(loss, ai, model, old_score, input_old, action, input, gameover, batch_size)

    return input, gameover, old_score, input_old, action, loss


def evaluate_ai(loss, ai: Ai, model, old_score, input_old, action, input, gameover, batch_size):
    global epsilon, epsilon_min, epsilon_decay

    reward = env._get_reward(playernr=ai.playernr, old_score=old_score)
    # store experience
    ai.remember([input_old, action, reward, input], gameover)
    # adapt model
    if train_mode_immediate:
        inputs, targets = ai.get_batch(model, batch_size=batch_size)
        loss += model.train_on_batch(inputs, targets)
    else:
        if gameover:
            inputs, targets = ai.get_batch(model, batch_size=batch_size)
            loss = model.train_on_batch(inputs, targets)

    if gameover:
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay


    return loss


if __name__ == "__main__":

    epoch = 8000000
    max_memory = 1 if train_mode_immediate else 500
    hidden_size_0 = num_actions * 12
    hidden_size_1 = num_actions * 24
    batch_size = 1 if train_mode_immediate else 50
    learning_rate = 1.0
    # learning_rate 1.0 for adadelta
    # only needed for sgd
    # decay_rate = learning_rate/epoch
    discount = 0.5
    model_name = "mm{}_hsmin{}_hsmax{}_lr{}_d{}_hl{}_na{}_ti{}_smoothepsi.h5".format(max_memory, hidden_size_0, hidden_size_1,
                                                                          learning_rate, discount, "3", num_actions,
                                                                          train_mode_immediate)
    print(model_name)
    model_temp_name = "temp_" + model_name
    model_epochs_trained = 0

    #     keras
    model = Sequential()

    model.add(Dense(hidden_size_0, input_shape=(num_actions,), activation='relu'))
    model.add(Dense(hidden_size_1, activation='relu'))
    model.add(Dense(hidden_size_0, activation='relu'))
    model.add(Dense(num_actions))  # output layer
    model.compile(optimizer=optimizers.adadelta(lr=learning_rate), loss=losses.mse)
    if os.path.isfile(model_temp_name):
        model = load_model(model_temp_name)
        print("model_loaded")

    if not os.path.isfile('{}.txt'.format(model_name)):
        training_file = open('{}.txt'.format(model_name),'w')
    training_file = open('{}.txt'.format(model_name), 'r')
    model_save_found = False
    epsilon_found = False

    for line in training_file:
        try:
            key, value = line.split(" ")
        except ValueError:
            continue
        if key == model_temp_name:
            model_epochs_trained = value
            model_save_found = True
        if key == "epsilon":
            epsilon = float(value)
            epsilon_found = True
        if key == "epsilondecay":
            epsilon_decay = float(value)
    training_file.close()
    if model_save_found == False:
        print("epoch save not found defaulting to 0")
        training_file = open('{}.txt'.format(model_name), 'a')
        training_file.write("\n" + model_temp_name + " " + str(0))
        training_file.close()

    if epsilon_found == False:
        print("epsilon not found defaulting given start values")
        training_file = open('{}.txt'.format(model_name), 'a')
        training_file.write("\n" + "epsilon" + " " + str(epsilon))
        training_file.write("\n" + "epsilondecay" + " " + str(epsilon_decay))
        training_file.close()

    print("epsilon: {}, decay: {}".format(epsilon, epsilon_decay))
    # logging----- tensorboard --host 127.0.0.1 --logdir=./logs ---- logs are saved on the project directory
    log_path = './logs/' + model_name
    callback = TensorBoard(log_path)
    callback.set_model(model)

    ai_player_1 = Ai(max_memory=max_memory, playernr=1, discount=discount)
    ai_player_2 = Ai(max_memory=max_memory, playernr=2, discount=discount)

    #     Train
    game_count = 0
    loss = 0.
    print(model_epochs_trained)
    for e in range(int(model_epochs_trained), epoch):
        if e % 100 == 0 and e != model_epochs_trained:
            verbose = True
            print(model_name)
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
            input_2, gameover, old_score_1, input_old_1, action_1, loss = ai_player_move(input_1, gameover,
                                                                                         ai_player_1, model, loss)
            if ai_2_played:
                loss = evaluate_ai(loss, ai_player_2, model, old_score_2, input_old_2, action_2, input_2, gameover,
                                   batch_size)

            if not gameover:
                input_1, gameover, old_score_2, input_old_2, action_2, loss = ai_player_move(input_2, gameover,
                                                                                             ai_player_2, model,
                                                                                             loss)
                ai_2_played = True
                loss = evaluate_ai(loss, ai_player_1, model, old_score_1, input_old_1, action_1, input_1, gameover,
                                   batch_size)

        # logging after each game saving with the epoch number.
        if e % 10 == 0 and e != model_epochs_trained: #play 1 4th of games against random
            # play it against random
            env = GameExtended()
            input = env.convert_and_reshape_field_to_inputarray([env.rows, env.columns])
            gameover = False
            predicted = False
            old_score = False
            input_old = False
            action = False
            ai_wins = 0
            random_wins = 0
            ai_fields = 0
            random_fields = 0

            if verbose:
                print("starting RANDOM game")
                # print(field_to_str(env.rows, env.columns))
            while not gameover:
                # AIMOVE
                input_old = input
                input, gameover, old_score, input_old, action, loss = ai_player_move(input_1, gameover, ai_player_1,
                                                                                     model, loss)
                # RANDOMMOVE
                if not gameover:
                    input, gameover = random_player_move(gameover, 2)
                    evaluate_ai(loss, ai_player_1, model, old_score, input_old, action, input, gameover, batch_size)

            # logging after each game saving with the epoch number.
            current_ai_field = env.player1["Points"]
            current_random_field = env.player2["Points"]
            if current_ai_field > current_random_field:
                ai_wins = 1
            elif current_random_field > current_ai_field:
                random_wins = 1
            ai_fields = current_ai_field
            random_fields = current_random_field

            # final evolution
            if verbose:
                print("Ai Wins: {}, with {} fields \n Random Wins: {} with {} fields".format(ai_wins, ai_fields,
                                                                                         random_wins,

                                                                                         random_fields))

            write_log(callback, train_loss=loss, ai_wins=ai_wins, ai_fields=ai_fields, batch_no=e)
            if verbose:
                print("Epoch {:03d} | Loss {:.4f}".format(e, loss))
                print("Epsilon is {} with Epsilon Decay {}".format(epsilon, epsilon_decay))


        if e % 50 == 0 and e != model_epochs_trained:
            model.save(model_temp_name, overwrite=True)
            training_file = open('{}.txt'.format(model_name), 'r')
            out = ""
            for line in training_file:
                try:
                    key, value = line.split(" ")
                except ValueError:
                    continue
                if key == model_temp_name:
                    out += model_temp_name + " " + str(e)
                elif key == "epsilon":
                    out += "\nepsilon " + str(epsilon)
                elif key == "epsilondecay":
                    out += "\nepsilondecay " + str(epsilon_decay)
                else:
                    out += line
                out += "\n"
            training_file.close()
            new_file = open('{}.txt'.format(model_name), 'w')
            new_file.write(out)
            new_file.close()

    model.save(model_name, overwrite=False)
