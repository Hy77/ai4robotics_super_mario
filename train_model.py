import torch
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
    state = np.squeeze(state)  # 去除多余的维度
    state = np.transpose(state, (1, 2, 0))  # 将通道维度移到最后
    state = transform(state)
    state = state.unsqueeze(0)  # 添加批次维度
    return state.float().to(device)

env = make_env(skip_frames=4)
action_size = env.action_space.n

mario_cnn = MarioCNN(output_size=256).to(device)
agent = Agent(state_size=256, action_size=action_size)

num_episodes = 500
max_t = 10000
eps_start = 1.0
eps_end = 0.00001
eps_decay = 0.995

mean_rewards = []
scores = []

for i_episode in range(1, num_episodes + 1):
    state = env.reset()
    state = process_state_image(state)
    state_features = mario_cnn(state).squeeze().detach()
    score = 0
    total_reward = 0
    steps = 0

    eps = eps_end + (eps_start - eps_end) * np.exp(-1. * i_episode / (num_episodes * 0.5))

    while True:
        action = agent.act(state_features.cpu().numpy(), eps)
        next_state, reward, done, info = env.step(action)
        env.render()
        score += reward
        total_reward += reward
        steps += 1

        next_state = process_state_image(next_state)
        next_state_features = mario_cnn(next_state).squeeze().detach()
        cnn_loss = torch.mean((next_state_features - mario_cnn(next_state).squeeze().detach()) ** 2)

        dqn_loss = agent.step(state_features.cpu().numpy(), action, reward, next_state_features.cpu().numpy(), done,
                              info)

        state = next_state
        state_features = next_state_features

        if done:
            mean_reward = total_reward / steps
            mean_rewards.append(mean_reward)
            scores.append(score)
            print(
                f"Episode {i_episode} - Score: {score} - Mean Reward: {mean_reward:.2f} - Eps: {eps:.2f} - CNN Loss: {cnn_loss:.4f} - DQN Loss: {dqn_loss:.4f}")
            break

    if i_episode % 100 == 0:
        torch.save(mario_cnn, f'cnn_models/cnn_model_episode_{i_episode}.pth')
        torch.save(agent.model_local, f'dqn_models/dqn_model_episode_{i_episode}.pth')

# 绘制平均奖励和分数的曲线图
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(mean_rewards) + 1), mean_rewards, label='Mean Reward')
plt.plot(range(1, len(scores) + 1), scores, label='Score')
plt.xlabel('Episode')
plt.ylabel('Value')
plt.title('Mean Reward and Score vs Episode')
plt.legend()
plt.savefig('reward_score_plot.png')
plt.show()