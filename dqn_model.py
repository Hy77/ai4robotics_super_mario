# dqn.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque


class MarioDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(MarioDQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32).reshape(-1, 1)
        next_states = np.array(next_states)
        dones = np.array(dones, dtype=np.uint8).reshape(-1, 1)

        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.uint8)
        )

    def __len__(self):
        return len(self.buffer)


class Agent:
    def __init__(self, state_size, action_size, replay_buffer_size=400000, batch_size=32, gamma=0.99, tau=1e-3,
                 lr=0.00025,
                 update_freq=4):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.update_freq = update_freq
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_local = MarioDQN(state_size, action_size).to(self.device)
        self.model_target = MarioDQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model_local.parameters(), lr=lr)

        self.t_step = 0

        self.prev_x_pos = 0
        self.stuck_counter = 0
        self.stuck_time_threshold = 10
        self.prev_coins = 0
        self.prev_life = 2

    def step(self, state, action, reward, next_state, done, info):
        self.replay_buffer.push(state, action, reward, next_state, done)

        current_x_pos = info['x_pos']
        current_y_pos = info['y_pos']
        current_life = info['life']
        coins_collected = info['coins'] - self.prev_coins  # Calculate the number of coins collected in this step

        # Check if Mario is stuck in the same position
        if current_x_pos == self.prev_x_pos:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0  # Reset the counter

        # Encourage big jumps when stuck for a long time
        if self.stuck_counter >= self.stuck_time_threshold and action == 4:
            reward += 200  # Give extra reward for big jumps when stuck

        # Encourage moving forward
        if current_x_pos > self.prev_x_pos:
            reward += (current_x_pos - self.prev_x_pos) * 0.1  # Give extra reward proportional to the distance moved forward
        self.prev_x_pos = current_x_pos

        # If died -500
        if current_life < self.prev_life:
            reward -= 500

        # 如果Mario掉下去并且执行跳跃动作,给予额外奖励
        if current_y_pos < 79 and action == 2 or 3 or 4:  # 假设跳跃动作的索引为4
            reward += 300

        # Encourage collecting coins
        if coins_collected > 0:
            reward += coins_collected * 1  # Give extra reward for each coin collected

        self.prev_coins = info['coins']  # Update the previous coin count

        self.t_step = (self.t_step + 1) % self.update_freq
        if self.t_step == 0:
            if len(self.replay_buffer) > self.batch_size:
                experiences = self.replay_buffer.sample(self.batch_size)
                dqn_loss = self.learn(experiences, self.gamma)
                return dqn_loss

        return 0.0  # 如果没有学习发生,返回0作为默认损失值

    def act(self, state, eps):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.model_local.eval()
        with torch.no_grad():
            action_values = self.model_local(state)
        self.model_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        actions = actions.unsqueeze(1)

        q_targets_next = self.model_target(next_states).detach().max(1)[0]
        q_targets = rewards.squeeze(1) + (gamma * q_targets_next * (1 - dones.squeeze(1)))

        q_expected = self.model_local(states).gather(1, actions).squeeze(1)

        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.model_local, self.model_target, self.tau)

        return loss.item()

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
