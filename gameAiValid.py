import tensorflow
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd
from gameLogic import *
from gamePlay import Game
import random
from gameView import width, height, field_to_str
from keras.models import load_model


class GameExtended(Game):
    def __init__(self):
        super().__init__()

    def create_train_data(self):
        game = GameExtended()

        random_move_count = random.randint(0, game.free_edge_count()-1)
        game.n_random_moves(random_move_count)
        input_array = self.convert_field_to_inputarray([game.rows, game.columns])
        r_input_array = input_array.reshape((1,-1))
        return game, r_input_array

    def convert_action_to_move(self,action):
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

    def convert_input_array_to_field(self,input):
        a = self.rows
        b = self.columns
        field = [a, b]
        for i in range(len(input)):
            array_i, h, w = self.convert_action_to_move(i)
            field[array_i][h][w] = input[i]
        return field

    def convert_field_to_inputarray(self,field):
        # field = [rows, colomns]
        input = np.zeros(40)
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

    # action = move
    def _update_state(self, action):
        array_i, height, width = self.convert_action_to_move(action)
        self.success = self.make_move(array_i, height, width)

    def _get_reward(self):
        return 1 if self.success else -1

    def act(self, action):
        self._update_state(action)
        reward = self._get_reward()
        return self.create_train_data(), reward


class ValidAi:
    def __init__(self, max_memory=100, discount=.9):
        self.max_memory = max_memory
        self.memory = list()
        # self.discount = discount

    def remember(self, states):
        self.memory.append([states])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        env_dim = self.memory[0][0][0].shape[1]
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))

        for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
            state_t, action_t, reward_t = self.memory[idx][0]
            inputs[i:i + 1] = state_t
            targets[i] = model.predict(state_t)[0]
            targets[i, action_t] = reward_t

        return inputs, targets


if __name__ == "__main__":

    epsilon = .1  # random moves
    num_actions = 40
    epoch = 25000
    max_memory = 500
    hidden_size = 100
    batch_size = 50

    #     keras
    model = Sequential()

    model.add(Dense(hidden_size, input_shape=(40,), activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(optimizer=sgd(lr=.2), loss='mse')
    model = load_model("model_{}_{}_{}_{}_{}.h5".format(num_actions, epoch, max_memory, hidden_size, batch_size))

    testing_model = False

    if not testing_model:
        exp_replay = ValidAi(max_memory=max_memory)

        #     Train
        env = GameExtended()
        env, input = env.create_train_data()
        general_success = 0
        game_count = 0
        for e in range(epoch):
            loss = 0.
            predicted = False
            # sometimes  guessing is better than predicting
            if np.random.rand() <= epsilon:
                action = random.randint(0, num_actions-1)
            else:
                q = model.predict(input)
                action = np.argmax(q[0])
                game_count += 1
                predicted = True

            (env_new, input_new), reward = env.act(action)
            if reward == 1 and predicted:
                general_success += 1
            else:
                if e > int(epoch/2) and predicted:
                    print(field_to_str(env.rows,env.columns))

            # store experience
            exp_replay.remember([input, action , reward])
            input = input_new
            env = env_new
            # adapt model
            inputs,targets = exp_replay.get_batch(model,batch_size=batch_size)

            loss += model.train_on_batch(inputs,targets)
            #print("Epoch {:03d}/99999 | Loss {:.4f} | Win count {}".format(e, loss, success_count))

            if (e % 1000 == 0):
                print("Epoch {:03d}/{} | Loss {:.4f}".format(e,epoch, loss))
                print(float(general_success) / game_count, '% winning rate')
                general_success = 0
                game_count = 0
                model.save("model_temp_{}_{}_{}_{}_{}.h5".format(num_actions, epoch, max_memory, hidden_size, batch_size),
                           overwrite=True)

        model.save("model_{}_{}_{}_{}_{}.h5".format(num_actions, epoch, max_memory, hidden_size, batch_size), overwrite=False)
    else:
        model = load_model("model_{}_{}_{}_{}_{}.h5".format(num_actions, epoch, max_memory, hidden_size, batch_size))
        env = GameExtended()
        success_count = 0
        for i in range(1000):

            env , input_array = env.create_train_data()
            field = env.convert_input_array_to_field(input_array.reshape(-1,1))
            print(field_to_str(field[0], field[1]))

            prediction = model.predict(input_array)
            action = np.argmax(prediction[0])

            env._update_state(action)

            print(env.convert_action_to_move(action))
            print(field_to_str(env.rows,env.columns))
            print(env.success)
            success_count += 1

        print("successcount: ", success_count)





