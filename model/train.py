import random
import math
from itertools import count

import torch
from torch.nn import functional as F

from model.utils import ReplayMemory,Transition



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

        # TODO init game here
        # TODO get number of actions
        self.n_actions = None

        # TODO init net
        self.model = None

        # TODO maybe call eval where target_net is used


        # TODO choose optimizer
        # optimizer = optim.RMSprop(policy_net.parameters())

        self.memory = ReplayMemory(10000)

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

    def optimize_model(self):
        self.model.train()
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # This converts batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        # TODO not sure if we want this or if this would work correct for us
        # TODO maybe just don't save these at all?
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        # TODO this was target net but we changed it to the same model
        self.model.eval()
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train(self):

        for i_episode in range(self.num_episodes):
            # TODO reset game
            # TODO get current state
            for t in count():
                # Select and perform an action
                action = self.select_action(state)
                _, reward, done, _ = None # TODO get reward and if done or not
                reward = torch.tensor([reward], device=self.device)

                # TODO Observe new state
                if not done:
                    next_state = None # TODO change this
                else:
                    next_state = None

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state # TODO not sure?

                # Perform one step of the optimization (on the target network)
                self.optimize_model()
                if done:
                    break

        print('Complete')



if __name__ == '__main__':

    train_wrapper = TrainWrapper()