from mario_env import MarioEnv
from cnn_model import MarioCNN
# TODO: from dqn_model import DQN
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
import random

# Setup TensorBoard
writer = SummaryWriter('runs/mario_experiment')

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the environment and models
env = MarioEnv('SuperMarioBros-1-1-v0')
cnn = MarioCNN().to(device)
# TODO: dqn = DQN(input_dim=256, output_dim=env.action_space.n).to(device)
epsilon = 0.1  # Exploration rate

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

    total_loss = 0
    steps = 0
    while True:
        # Use CNN to extract features
        state_features = cnn(state)

        # TODO: Use DQN to decide action
        # if random.random() > epsilon:
        #     action = dqn(state_features).max(1)[1].item()  # Greedy action
        # else:
        #     action = env.action_space.sample()  # Random action

        # Execute action
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = transform(next_state).unsqueeze(0).to(device)

        # Use CNN to get features for the next state
        next_state_features = cnn(next_state)

        # TODO: Compute target and loss
        # next_q_values = dqn(next_state_features)
        # expected_q_values = reward + 0.99 * next_q_values.max(1)[0] * (1 - terminated)
        # loss = criterion(dqn(state_features).gather(1, torch.tensor([[action]], device=device)), expected_q_values.unsqueeze(1))

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1
        state = next_state

        if terminated or truncated:
            break

    # Save the model every 100 episodes and overwrite the previous save
    if episode % 100 == 0:
        # Save both CNN and DQN models' state_dict to the same files each time
        torch.save(cnn.state_dict(), 'cnn_models/cnn_model_ver1.pth')
        # TODO: Save DQN model state dict
        # torch.save(dqn.state_dict(), 'dqn_models/dqn_model_ver1.pth')

        print(f'Episode {episode}, Total Loss: {total_loss}, Steps: {steps}')
        writer.add_scalar('Total Loss', total_loss, episode)
        writer.add_scalar('Steps', steps, episode)

env.close()
writer.close()
