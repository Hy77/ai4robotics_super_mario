import torch
from torch import nn
import copy
import numpy as np
import random
from collections import deque

# Define HYPERPARAMETERS
BATCH_SIZE = 32
LEARNING_RATE = 0.00025
DISCOUNT_FACTOR = 0.9
TARGET_UPDATE_FREQ = 1e4
EXPLORATION_DECAY_RATE = 0.99999975
MIN_EXPLORATION_RATE = 0.1
INITIAL_MEMORY_THRESHOLD = 1e5
LEARNING_UPDATE_FREQ = 3


class MarioCNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        channels, height, width = input_dim

        self.online_network = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

        self.target_network = copy.deepcopy(self.online_network)

        for param in self.target_network.parameters():
            param.requires_grad = False

    def forward(self, input, network):
        if network == 'online':
            return self.online_network(input)
        elif network == 'target':
            return self.target_network(input)


class MarioAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory_buffer = deque(maxlen=100000)
        self.exploration_rate = 1
        self.current_step = 0

        self.use_cuda = torch.cuda.is_available()

        self.mario_model = MarioCNN(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.mario_model = self.mario_model.to(device='cuda')

        self.optimizer = torch.optim.Adam(self.mario_model.parameters(), lr=LEARNING_RATE)
        self.loss_function = torch.nn.SmoothL1Loss()

    def select_action(self, state):
        if np.random.rand() < self.exploration_rate:
            action_index = np.random.randint(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
            state_tensor = state_tensor.unsqueeze(0)
            action_values = self.mario_model(state_tensor, network='online')
            action_index = torch.argmax(action_values).item()

        self.exploration_rate *= EXPLORATION_DECAY_RATE
        self.exploration_rate = max(MIN_EXPLORATION_RATE, self.exploration_rate)

        self.current_step += 1
        return action_index

    def store_experience(self, state, next_state, action, reward, done):
        state = np.array(state)
        next_state = np.array(next_state)

        state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state).cuda() if self.use_cuda else torch.FloatTensor(next_state)
        action = torch.LongTensor([action]).cuda() if self.use_cuda else torch.LongTensor([action])
        reward = torch.DoubleTensor([reward]).cuda() if self.use_cuda else torch.DoubleTensor([reward])
        done = torch.BoolTensor([done]).cuda() if self.use_cuda else torch.BoolTensor([done])

        self.memory_buffer.append((state, next_state, action, reward, done,))

    def sample_experiences(self):
        batch = random.sample(self.memory_buffer, BATCH_SIZE)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def compute_td_estimate(self, state, action):
        current_Q_values = self.mario_model(state, network='online')[np.arange(0, BATCH_SIZE), action]
        return current_Q_values

    @torch.no_grad()
    def compute_td_target(self, reward, next_state, done):
        next_state_Q_values = self.mario_model(next_state, network='online')
        best_action = torch.argmax(next_state_Q_values, axis=1)
        next_Q_values = self.mario_model(next_state, network='target')
        next_Q_values = next_Q_values[np.arange(0, BATCH_SIZE), best_action]
        return (reward + (1 - done.float()) * DISCOUNT_FACTOR * next_Q_values).float()

    def update_online_network(self, td_estimate, td_target):
        loss = self.loss_function(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def synchronize_target_network(self):
        self.mario_model.target_network.load_state_dict(self.mario_model.online_network.state_dict())

    def learn(self):
        if self.current_step % TARGET_UPDATE_FREQ == 0:
            self.synchronize_target_network()

        if self.current_step < INITIAL_MEMORY_THRESHOLD:
            return None, None

        if self.current_step % LEARNING_UPDATE_FREQ != 0:
            return None, None

        state, next_state, action, reward, done = self.sample_experiences()
        td_estimate = self.compute_td_estimate(state, action)
        td_target = self.compute_td_target(reward, next_state, done)
        loss = self.update_online_network(td_estimate, td_target)

        return td_estimate.mean().item(), loss

