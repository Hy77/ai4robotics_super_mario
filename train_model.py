from mario_env import MarioEnv
from cnn_model import MarioCNN
from dqn_model import MarioDQN
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import math
from torch.utils.tensorboard import SummaryWriter
import random

# Setup TensorBoard
writer = SummaryWriter('runs/mario_experiment')

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the environment and models
env = MarioEnv('SuperMarioBros-1-1-v0')
cnn = MarioCNN().to(device)
dqn = MarioDQN(input_dim=256, output_dim=env.action_space.n).to(device)

# Check and create directories for saving models
model_dirs = ['cnn_models', 'dqn_models']
for model_dir in model_dirs:
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
# Load model weights if available
try:
    cnn.load_state_dict(torch.load('cnn_models/cnn_model_ver1.pth'))
    dqn.load_state_dict(torch.load('dqn_models/dqn_model_ver1.pth'))
except FileNotFoundError:
    print("Model files not found, starting training from scratch.")

cnn.to(device)
dqn.to(device)

# Define epsilon decay parameters
# epsilon = 0.1  # Exploration rate
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 50000
epsilon_by_episode = lambda episode: epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * episode / epsilon_decay)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(list(cnn.parameters()) + list(dqn.parameters()), lr=0.001)

# Image preprocessing
transform = T.Compose([
    T.ToPILImage(),  # Convert numpy array to PIL image
    T.Resize((84, 84)),  # Resize image to match CNN input
    T.ToTensor(),  # Convert PIL image to Tensor
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Training loop
num_episodes = 2500  # Total number of episodes
for episode in range(num_episodes):
    state = env.reset()
    state = transform(state).unsqueeze(0).to(device)
    epsilon = epsilon_by_episode(episode)
    total_loss = 0
    steps = 0
    done = False
    while not done:
        # Use CNN to extract features
        state_features = cnn(state)

        # TODO: Use DQN to decide action
        
        if random.random() > epsilon:
            action = dqn(state_features).max(1)[1].item()  # Greedy action
        else:
            action = env.action_space.sample()  # Random action

        # Execute action
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = transform(next_state).unsqueeze(0).to(device)

        # Use CNN to get features for the next state
        next_state_features = cnn(next_state)

        # TODO: Compute target and loss
        with torch.no_grad():
            next_q_values = dqn(next_state_features)
            max_next_q_values = next_q_values.max(1)[0]
            expected_q_values = reward + 0.99 * max_next_q_values * (1 - terminated)

        current_q_values = dqn(state_features).gather(1, torch.tensor([[action]], device=device))
        loss = criterion(current_q_values, expected_q_values.unsqueeze(1))

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1
        state = next_state

        done = terminated or truncated

    # Save the model every 100 episodes and overwrite the previous save
    if episode % 100 == 0:
        # Save both CNN and DQN models' state_dict to the same files each time
        torch.save(cnn.state_dict(), 'cnn_models/cnn_model_ver1.pth')
        # TODO: Save DQN model state dict
        torch.save(dqn.state_dict(), 'dqn_models/dqn_model_ver1.pth')

        print(f'Episode {episode}, Total Loss: {total_loss}, Steps: {steps}')
        writer.add_scalar('Total Loss', total_loss, episode)
        writer.add_scalar('Steps', steps, episode)

env.close()
writer.close()
