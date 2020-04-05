import random
import math
from itertools import count, chain
import numpy as np
from collections import namedtuple
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch import optim
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.models import Net
from model.dataset import CustomDataset

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class TrainWrapper:
    def __init__(self, batch_size=256, gamma=0.999, target_update=10,
                 num_episodes=50000, eval_steps=1000, win_ratio=0.9, games_per_episode=4):
        self.batch_size = batch_size
        self.gamma = gamma

        self.target_update = target_update
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_episodes = num_episodes

        self.n_actions = 144
        self.eval_steps = eval_steps
        self.win_ratio = win_ratio
        self.games_per_episode = games_per_episode
        self.active_student = 1
        self.training_models = [Net(), Net()]
        self.training_models[0].to(self.device)
        self.training_models[1].to(self.device)
        self.training_models[0].eval()
        self.training_models[1].eval()

        self.playing_models = [Net(), Net()] #TODO these have to be updated regularlry
        self.playing_models[0].eval()
        self.playing_models[1].eval()

        # TODO LR
        lr = 1e-5
        self.optims = [AdamW(model.parameters(),lr=lr) for model in self.training_models]

        total_params = sum(p.numel() for p in self.training_models[0].parameters())
        trainable_params = sum(p.numel() for p in self.training_models[0].parameters() if p.requires_grad)
        print(f'\nTotal Paramaters: {total_params:,}, Trainable Parameters: {trainable_params:,}\n')
        self.writer = SummaryWriter()

    def change_student(self):
        self.active_student = 1 if self.active_student == 2 else 2

    def reshape_batches(self, batches):
        reshaped_batches = []
        for batch in batches:
            batch = Transition(*zip(*batch))
            states = torch.tensor(batch.state, dtype=torch.float32, device=self.device).unsqueeze(dim=1)
            next_states = torch.tensor(batch.next_state, dtype=torch.float32, device=self.device)
            actions = torch.tensor(batch.action, dtype=torch.int64, device=self.device)
            rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
            dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device)
            reshaped_batches.append((states, next_states, actions, rewards, dones))

        return reshaped_batches

    def convert_transitions_to_batches(self, transitions):
        n_batches = len(transitions) // self.batch_size
        transitions_array = np.array(transitions)
        np.random.shuffle(transitions_array)
        batches = np.split(transitions_array[:n_batches * self.batch_size], n_batches)
        batches = self.reshape_batches(batches)

        return batches

    def save_models(self, name=""):
        for idx,model in enumerate(self.training_models):
            torch.save(model, f"trained_models/model_{idx}_{name}.pth")

    def train_model(self):
        dataset = CustomDataset(models=self.playing_models,active_student=self.active_student,length=self.num_episodes*self.batch_size) # TODO make sure active student is changed
        dataloader = DataLoader(dataset=dataset,batch_size=self.batch_size,shuffle=False,num_workers=0,pin_memory=False) # TODO maybe just do MP in dataloader and num_workers =0

        training_model = self.training_models[self.active_student - 1]
        optimizer = self.optims[self.active_student - 1]

        eval_step = 0
        total_loss = 0.

        for i_episode,batch in enumerate(tqdm(dataloader,total=self.num_episodes)):
            for idx in range(len(batch)):
                batch[idx] = batch[idx].to(self.device)

            states, next_states, actions, rewards, dones = batch
            states = states.unsqueeze(dim=1)

            training_model.train()

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            model_output = training_model(states)
            state_action_values = model_output.gather(1, actions.unsqueeze(dim=-1))
            #
            #     # Compute V(s_{t+1}) for all next states.
            #     # This was target_net but we changed it to the same model
            training_model.eval()
            next_state_values = torch.zeros(self.batch_size, device=self.device)
            # Only for next state which are not game end
            next_state_values[dones == 0] = training_model(states[dones == 0]).max(1)[0].detach()
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * self.gamma) + rewards

            # Compute Huber loss
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            total_loss += loss.item()
            # TODO scheduler?
            for param in training_model.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()

            # TODO update playing models (they are just training models that always live on the cpu)

        if i_episode % self.eval_steps == 0 and i_episode > 0:
            eval_step +=1
            student_model = self.training_models[self.active_student - 1]
            teacher_model = self.training_models[1 - (self.active_student - 1)]
            total_wins = student_model.wins + teacher_model.wins

            print(
                f'\nTeacher Won: {teacher_model.wins / total_wins:.3f}\n Student Won: {student_model.wins / total_wins:.3f}\n')

            #Tensorboard
            self.writer.add_scalar(f'Loss/Player{self.active_student}', total_loss, eval_step)
            self.writer.add_scalar(f'Wins/Player{self.active_student}', student_model.wins, eval_step)


            if student_model.wins > total_wins * self.win_ratio:
                print('Switching student and teacher')
                self.change_student()

            student_model.wins = 0
            teacher_model.wins = 0
            total_loss = 0.
            self.save_models(name="eval")


        self.writer.close()
        self.save_models(name="version1")
        print('Complete')

# TODO parallelize training (get_move)
# TODO evaluate and work on better model
if __name__ == '__main__':
    train_wrapper = TrainWrapper()
    train_wrapper.train_model()
