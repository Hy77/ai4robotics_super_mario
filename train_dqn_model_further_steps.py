# train_dqn.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from mario_env import make_env
from cnn_model import MarioCNN
from dqn_model import Agent
import torchvision.transforms as T
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = T.Compose([
    T.ToTensor(),
    T.Resize((84, 84)),
    T.Normalize(mean=[0.5], std=[0.5])
])

def process_state_image(state):
    state = np.array(state)
    state = np.squeeze(state)
    state = np.transpose(state, (1, 2, 0))
    state = transform(state)
    state = state.unsqueeze(0)
    return state.float().to(device)

env = make_env(skip_frames=4)
action_size = env.action_space.n

# 加载预训练的CNN模型
cnn_model = MarioCNN(output_size=256).to(device)
cnn_model.load_state_dict(torch.load('cnn_models/cnn_model_ver10.pth'))
cnn_model.eval()

agent = Agent(state_size=256, action_size=action_size)

num_episodes = 500
max_t = 10000
eps_start = 1.0
eps_end = 0.01
eps_decay = 0.995

# 记录损失
losses = []

for i_episode in range(1, num_episodes + 1):
    state = env.reset()
    state = process_state_image(state)
    with torch.no_grad():
        state_features = cnn_model(state).squeeze()
    score = 0
    eps = max(eps_end, eps_start * eps_decay ** i_episode)

    for t in range(max_t):
        action = agent.act(state_features.cpu().numpy(), eps)
        next_state, reward, done, info = env.step(action)
        score += reward

        next_state = process_state_image(next_state)
        with torch.no_grad():
            next_state_features = cnn_model(next_state).squeeze()

        dqn_loss = agent.step(state_features.cpu().numpy(), action, reward, next_state_features.cpu().numpy(), done, info)

        state = next_state
        state_features = next_state_features

        if done:
            print(f"Episode {i_episode} - Score: {score} - Eps: {eps:.2f} - DQN Loss: {dqn_loss:.4f}")
            break

    losses.append(dqn_loss)
    if i_episode % 100 == 0:
        torch.save(agent.model_local, f'dqn_models/dqn_model_episode_{i_episode}.pth')

# 绘制损失图像
plt.plot(range(num_episodes), losses)
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.title('DQN Training Loss')
plt.show()
