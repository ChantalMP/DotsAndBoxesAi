import torch
import sys
from random import shuffle
from typing import List
from torch.utils.data import Dataset
from game.game import Game
from game.game import AiPlayer
import math
from collections import namedtuple
import torch

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

from torch import multiprocessing as mp

class CustomDataset(Dataset):
    def __init__(self, models, active_student,length,eps_start=0.9, eps_end=0.05, eps_decay=200,n_workers=4):
        self.models = models
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.active_student = active_student
        self.length = length
        self.memory:List[Transition] = []
        self.games_per_worker = 1000
        self.n_workers = n_workers

    def transition_to_tensors(self, transition:Transition):
        state = torch.tensor(transition.state, dtype=torch.float32)
        next_state = torch.tensor(transition.next_state, dtype=torch.float32)
        action = torch.tensor(transition.action, dtype=torch.int64)
        reward = torch.tensor(transition.reward, dtype=torch.float32)
        done = torch.tensor(transition.done, dtype=torch.float32)
        return state,next_state,action,reward,done

    def __getitem__(self, _):
        if len(self.memory) <= 0:
            with mp.Pool(processes=self.n_workers) as p:
                transitions = p.map(self.play_AI_game,range(self.n_workers))
            transitions = [item for sublist in transitions for item in sublist]

            self.memory.extend(transitions)
            shuffle(self.memory)
            print(f'Filled mem with {len(self.memory)}')

        transition = self.memory.pop()
        state, next_state, action, reward, done = self.transition_to_tensors(transition)
        return state, next_state, action, reward, done

    def __len__(self):
        return self.length

    def update_model(self, models):
        self.models = models

    # return ('state', 'action', 'next_state', 'reward') tuples for each move of student
    def play_AI_game(self,_):
        transitions = []
        for i in range(self.games_per_worker):
            assert self.models[0].training == False and self.models[1].training == False
            student_is_1 = True if self.active_student == 1 else False
            player1 = AiPlayer(is_student=student_is_1, model=self.models[0])
            player2 = AiPlayer(is_student=not student_is_1, model=self.models[1])
            game = Game(player1, player2)

            player1.eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                                    math.exp(-1. * player1.model.steps_done / self.eps_decay)
            player2.eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                                    math.exp(-1. * player2.model.steps_done / self.eps_decay)

            while not game.game_over():
                state = game.game_field.field.copy()
                move = game.active_player.get_move(game.game_field, game.active_player.eps_threshold)  # move is y,x
                game.game_field.make_move(move)
                new_full_fields = game.game_field.new_full_fields(move)
                game.active_player.points += new_full_fields

                if game.active_player.is_student:
                    y, x = move
                    action = game.game_field.convert_move_to_lineidx(y, x)
                    transition = Transition(state=state, action=action, next_state=game.game_field.field,
                                            reward=new_full_fields, done=game.game_over())
                    transitions.append(transition)

                if new_full_fields == 0:
                    game.change_player()

            winner = game.winner
            if winner is not None:
                game.winner.model.wins += 1


        return transitions