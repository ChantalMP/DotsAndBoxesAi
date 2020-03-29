import random
import math
from itertools import count, chain
from multiprocessing import Pool
import numpy as np
from collections import namedtuple

import torch
from torch.nn import functional as F
from torch import optim

from game.game import Game
from game.game import AiPlayer
from model.models import Net

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class TrainWrapper:
    def __init__(self, batch_size=128, gamma=0.999, eps_start=0.9, eps_end=0.05, eps_decay=200, target_update=10,
                 num_episodes=50,eval_steps=25,win_ratio=0.9,games_per_episode=4):
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_episodes = num_episodes

        self.n_actions = 144
        self.eval_steps = eval_steps
        self.win_ratio =  win_ratio
        self.games_per_episode = games_per_episode
        self.active_student = 1
        self.models = [Net(), Net()]
        self.models[0].eval()
        self.models[1].eval()


        self.optims = [optim.AdamW(model.parameters()) for model in self.models]

        # total_params = sum(p.numel() for p in self.models[0].parameters())
        # trainable_params = sum(p.numel() for p in self.models[0].parameters() if p.requires_grad)
        # print(f'\nTotal Paramaters: {total_params:,}, Trainable Parameters: {trainable_params:,}\n')
        # TODO tensorboard integration


    def change_student(self):
        self.active_student = 1 if self.active_student == 2 else 2

    # return ('state', 'action', 'next_state', 'reward') tuples for each move of student
    def play_AI_game(self, _):
        assert self.models[0].training == False and self.models[1].training == False

        student_is_1 = True if self.active_student == 1 else False
        player1 = AiPlayer(is_student=student_is_1, model=self.models[0])
        player2 = AiPlayer(is_student=not student_is_1, model=self.models[1])
        game = Game(player1, player2)
        # TODO change place
        player1.eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * player1.model.steps_done / self.eps_decay)
        player2.eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                          math.exp(-1. * player2.model.steps_done / self.eps_decay)

        transitions = []
        while not game.game_over():
            state = game.game_field.field.copy()
            move = game.active_player.get_move(game.game_field,game.active_player.eps_threshold,self.device)  # move is y,x
            game.game_field.make_move(move)
            new_full_fields = game.game_field.new_full_fields(move)
            game.active_player.points += new_full_fields

            if game.active_player.is_student:
                y, x = move
                action = game.game_field.convert_move_to_lineidx(y,x)
                transition = Transition(state=state, action=action, next_state=game.game_field.field,
                                        reward=new_full_fields, done=game.game_over())
                transitions.append(transition)

            if new_full_fields == 0:
                game.change_player()

        winner = game.winner
        if winner is not None:
            game.winner.model.wins += 1

        return transitions

    def reshape_batches(self, batches):
        reshaped_batches = []
        for batch in batches:
            batch = Transition(*zip(*batch))
            states = torch.FloatTensor(batch.state).unsqueeze(dim=1)
            next_states = torch.FloatTensor(batch.next_state)
            actions = torch.LongTensor(batch.action)
            rewards = torch.FloatTensor(batch.reward)
            dones = torch.FloatTensor(batch.done)
            reshaped_batches.append((states, next_states, actions, rewards, dones))

        return reshaped_batches

    def convert_transitions_to_batches(self, transitions):
        n_batches = len(transitions) // self.batch_size
        transitions_array = np.array(transitions)
        np.random.shuffle(transitions_array)
        batches = np.split(transitions_array[:n_batches * self.batch_size], n_batches)
        batches = self.reshape_batches(batches)

        return batches

    def train_model(self):

        model = self.models[self.active_student - 1]
        optimizer = self.optims[self.active_student - 1]

        for i_episode in range(self.num_episodes):
            # play games
            from time import time
            # start = time()
            # transition_lists = [self.play_AI_game(a) for a in range(4)]
            with Pool(4) as p:
                transition_lists = p.map(self.play_AI_game, range(self.games_per_episode))

            # print(time()-start)
            transitions = list(chain.from_iterable(transition_lists))  # concat lists
            batches = self.convert_transitions_to_batches(transitions)

            for states, next_states, actions, rewards, dones in batches:
                model.train()

                # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
                # columns of actions taken. These are the actions which would've been taken
                # for each batch state according to policy_net
                model_output = model(states)
                state_action_values = model_output.gather(1, actions.unsqueeze(dim=-1))

                # Compute V(s_{t+1}) for all next states.
                # This was target_net but we changed it to the same model
                model.eval()
                next_state_values = torch.zeros(self.batch_size, device=self.device)
                # Only for next state which are not game end
                next_state_values[dones == 0] = model(states[dones == 0]).max(1)[0].detach()
                # Compute the expected Q values
                expected_state_action_values = (next_state_values * self.gamma) + rewards

                # Compute Huber loss
                loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                # TODO scheduler?
                for param in model.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()

            print('HI')
            if i_episode % self.eval_steps == 0 and i_episode > 0:
                student_model = self.models[self.active_student-1]
                teacher_model = self.models[1- (self.active_student-1)]
                total_wins = student_model.wins + teacher_model.wins

                print(f'Teacher Won: {teacher_model.wins / total_wins}\n Student Won: {student_model.wins/total_wins}\n')

                if student_model.wins > total_wins * self.win_ratio:
                    print('Switching student and teacher')
                    self.change_student()

                student_model.wins = 0
                teacher_model.wins = 0

        print('Complete')


if __name__ == '__main__':
    train_wrapper = TrainWrapper()
    train_wrapper.train_model()
