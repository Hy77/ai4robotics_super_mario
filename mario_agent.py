import torch
from torch import nn
import copy
import numpy as np
import random
from collections import deque

# Define HYPERPARAMETERS
BATCH_SIZE = 32
LEARNING_RATE = 0.00025
GAMMA = 0.9
EXPLORATION_DECAY_RATE = 0.99999975
MIN_EXPLORATION_RATE = 0.1

class MarioCNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.online_network = nn.Sequential(
            nn.Conv2d(in_channels=input_dim[0], out_channels=32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 512),
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

        self.mario_model = MarioCNN(self.state_dim, self.action_dim).float()
        if torch.cuda.is_available():
            self.mario_model = self.mario_model.to(device='cuda')

        self.optimizer = torch.optim.Adam(self.mario_model.parameters(), lr=LEARNING_RATE)
        self.loss_function = torch.nn.SmoothL1Loss()

    # Select action based on exploration rate or model prediction
    def select_action(self, state):
        if np.random.rand() < self.exploration_rate:
            action_index = np.random.randint(self.action_dim)
        else:
            state_tensor = self._convert_to_tensor(state, unsqueeze=True)
            action_values = self.mario_model(state_tensor, network='online')
            action_index = torch.argmax(action_values).item()

        self._update_exploration_rate()
        self.current_step += 1
        return action_index

    # Store experience in memory buffer
    def store_experience(self, state, next_state, action, reward, done):
        experience = self._prepare_experience(state, next_state, action, reward, done)
        self.memory_buffer.append(experience)

    # Learning process
    def learn(self):
        if not self._should_learn():
            return None, None

        state, next_state, action, reward, done = self.sample_experiences()
        td_estimate = self.compute_td_estimate(state, action)
        td_target = self.compute_td_target(reward, next_state, done)
        loss = self.update_online_network(td_estimate, td_target)

        if self.current_step % 1e4 == 0:
            self.synchronize_target_network()

        return td_estimate.mean().item(), loss

    # Sample experiences from memory buffer
    def sample_experiences(self):
        batch = random.sample(self.memory_buffer, BATCH_SIZE)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def compute_td_estimate(self, state, action):
        current_q_values = self.mario_model(state, network='online')[np.arange(0, BATCH_SIZE), action]
        return current_q_values

    @torch.no_grad()
    def compute_td_target(self, reward, next_state, done):
        next_state_q_values = self.mario_model(next_state, network='online')
        best_action = torch.argmax(next_state_q_values, axis=1)
        next_q_values = self.mario_model(next_state, network='target')
        next_q_values = next_q_values[np.arange(0, BATCH_SIZE), best_action]
        return (reward + (1 - done.float()) * GAMMA * next_q_values).float()

    # Prepare experience for storage
    def update_online_network(self, td_estimate, td_target):
        loss = self.loss_function(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def synchronize_target_network(self):
        self.mario_model.target_network.load_state_dict(self.mario_model.online_network.state_dict())

    def _convert_to_tensor(self, data, unsqueeze=False):
        tensor = torch.FloatTensor(data).cuda() if torch.cuda.is_available() else torch.FloatTensor(data)
        return tensor.unsqueeze(0) if unsqueeze else tensor

    def _prepare_experience(self, state, next_state, action, reward, done):
        state = self._convert_to_tensor(state)
        next_state = self._convert_to_tensor(next_state)
        action = torch.LongTensor([action]).cuda() if torch.cuda.is_available() else torch.LongTensor([action])
        reward = torch.DoubleTensor([reward]).cuda() if torch.cuda.is_available() else torch.DoubleTensor([reward])
        done = torch.BoolTensor([done]).cuda() if torch.cuda.is_available() else torch.BoolTensor([done])
        return state, next_state, action, reward, done

    def _update_exploration_rate(self):
        self.exploration_rate *= EXPLORATION_DECAY_RATE
        self.exploration_rate = max(MIN_EXPLORATION_RATE, self.exploration_rate)

    # Check if the agent should learn
    def _should_learn(self):
        if self.current_step < 1e5:
            return False
        if self.current_step % 3 != 0:
            return False
        return True
