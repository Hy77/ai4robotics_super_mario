import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from mario_env import make_env
from cnn_model import MarioCNN
from dqn_model import Agent
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = T.Compose([
    T.ToTensor(),
    T.Resize((84, 84)),
    T.Normalize(mean=[0.5], std=[0.5])
])

def process_state_image(state):
    state = np.array(state)
    state = np.squeeze(state)  # 去除多余的维度
    state = np.transpose(state, (1, 2, 0))  # 将通道维度移到最后
    state = transform(state)
    state = state.unsqueeze(0)  # 添加批次维度
    return state.float().to(device)

env = make_env(skip_frames=4)
action_size = env.action_space.n
mario_cnn = MarioCNN(output_size=256).to(device)
agent = Agent(state_size=256, action_size=action_size)

# Load pre-trained models if available
mario_cnn = mario_cnn.to(device)
agent.model_local = agent.model_local.to(device)  # 将DQN模型移到GPU上
agent.model_target = agent.model_target.to(device)

# 定义CNN模型的损失函数和优化器
cnn_criterion = nn.MSELoss()
cnn_optimizer = optim.Adam(mario_cnn.parameters(), lr=1e-3)

num_episodes = 500
max_t = 10000
eps_start = 1.0
eps_end = 0.01
eps_decay = 0.995

for i_episode in range(1, num_episodes + 1):
    state = env.reset()
    state = process_state_image(state)
    state_features = mario_cnn(state).squeeze().detach()
    score = 0
    eps = max(eps_end, eps_start * (eps_decay ** i_episode))

    for t in range(max_t):
        action = agent.act(state_features.cpu().numpy(), eps)
        next_state, reward, done, info = env.step(action)
        max_x_pos = info['x_pos']
        score += reward
        env.render()
        next_state = process_state_image(next_state)
        next_state_features = mario_cnn(next_state).squeeze().detach()
        predicted_features = mario_cnn(next_state).squeeze()

        cnn_loss = cnn_criterion(predicted_features, next_state_features.detach())

        # 反向传播并更新CNN模型的参数
        cnn_optimizer.zero_grad()
        cnn_loss.backward()
        cnn_optimizer.step()

        dqn_loss = agent.step(state_features, action, reward, next_state_features, done, info)
        state = next_state
        state_features = next_state_features

        if done:
            print(f"Episode {i_episode} - Score: {score} - Max X Position: {max_x_pos} - Eps: {eps:.2f} - CNN Loss: {cnn_loss:.4f} - DQN Loss: {dqn_loss:.4f}")
            break

    if i_episode % 500 == 0:
        torch.save(mario_cnn, f'cnn_models/cnn_model_episode_{i_episode}.pth')
        torch.save(agent.model_local, f'dqn_models/dqn_model_episode_{i_episode}.pth')
