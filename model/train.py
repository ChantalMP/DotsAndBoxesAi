import random
import math
from itertools import count, chain
from multiprocessing import Pool
import numpy as np
from collections import namedtuple

import torch
from torch.nn import functional as F

from game.game import Game
from game.game import AiPlayer

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class TrainWrapper:
    def __init__(self,batch_size=128,gamma=0.999,eps_start=0.9,eps_end=0.05,eps_decay=200,target_update=10,num_episodes=50):
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_episodes = num_episodes

        self.n_actions = 144
        self.active_student = 1
        self.models = [1,2]

        # TODO init net
        self.model = None

        # TODO maybe call eval where target_net is used
        # TODO teacher learner

        # TODO choose optimizer
        # optimizer = optim.RMSprop(policy_net.parameters())

        self.steps_done = 0

        # TODO tensorboard integration

    def select_action(self,state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the largest expected reward.
                return self.model(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def change_student(self):
        self.active_student = 1 if self.active_student == 2 else 2

    # return ('state', 'action', 'next_state', 'reward') tuples for each move of student
    def play_AI_game(self, _):
        student_is_1 = True if self.active_student == 1 else False
        player1 = AiPlayer(is_student = student_is_1, model = self.models[0])
        player2 = AiPlayer(is_student = not student_is_1, model = self.models[1])
        game = Game(player1, player2)

        transitions = []
        while not game.game_over():
            state = game.game_field.field.copy()
            move = game.active_player.get_move(game.game_field)  # move is y,x
            game.game_field.make_move(move)
            new_full_fields = game.game_field.new_full_fields(move)
            game.active_player.points += new_full_fields

            if game.active_player.is_student:
                transition = Transition(state=state, action=move, next_state=game.game_field.field, reward=new_full_fields, done = game.game_over())
                transitions.append(transition)

            if new_full_fields == 0:
                game.change_player()

        return transitions

    def reshape_batches(self, batches):
        reshaped_batches = []
        for batch in batches:
            batch = Transition(*zip(*batch))
            states = np.array(batch.state,dtype=np.int16)
            next_states = np.array(batch.next_state,dtype=np.int16)
            actions = np.array(batch.action,dtype=np.int16)
            rewards = np.array(batch.reward,dtype=np.int16)
            dones = np.array(batch.done,dtype=np.int16)
            reshaped_batches.append((states,next_states,actions,rewards,dones))

        return reshaped_batches

    def convert_transitions_to_batches(self, transitions):
        n_batches = len(transitions)//self.batch_size
        transitions_array = np.array(transitions)
        np.random.shuffle(transitions_array)
        batches = np.split(transitions_array[:n_batches*self.batch_size], n_batches)
        batches = self.reshape_batches(batches)

        return batches


    def train(self):

        model = self.models[self.active_student-1]
        for i_episode in range(self.num_episodes):
            # play games
            with Pool(4) as p:
                transition_lists = p.map(self.play_AI_game, range(4))
            transitions = list(chain.from_iterable(transition_lists)) # concat lists
            batches = self.convert_transitions_to_batches(transitions)

            for states,next_states,actions,rewards,dones in batches:
                model.train()

                # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
                # columns of actions taken. These are the actions which would've been taken
                # for each batch state according to policy_net
                state_action_values = self.model(states).gather(1, actions)

                # Compute V(s_{t+1}) for all next states.
                # TODO this was target net but we changed it to the same model
                self.model.eval()
                next_state_values = torch.zeros(self.batch_size, device=self.device)
                # Only for next state which are not game end
                next_state_values[dones == 0] = self.model(states[dones == 0]).max(1)[0].detach()
                # Compute the expected Q values
                expected_state_action_values = (next_state_values * self.gamma) + rewards

                # Compute Huber loss
                loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

                # Optimize the model
                self.optimizer.zero_grad()
                loss.backward()
                for param in self.model.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.optimizer.step()

        # TODO switch student and corresponding model depending on wins
        # TODO calculate calculate wins
        print('Complete')



if __name__ == '__main__':

    train_wrapper = TrainWrapper()
    train_wrapper.train()