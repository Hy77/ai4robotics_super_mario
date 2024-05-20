import torch
import numpy as np
from mario_env import make_env
from cnn_model import MarioCNN
from dqn_model import Agent
import torchvision.transforms as T
import matplotlib.pyplot as plt
import gym
import imageio

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

env = make_env(skip_frames=1)

action_size = env.action_space.n

mario_cnn = torch.load('cnn_models/cnn_model_ver9.pth')
mario_cnn = mario_cnn.to(device)  # 将 CNN 模型移到 GPU
mario_cnn.eval()

agent = Agent(state_size=256, action_size=action_size)
agent.model_local = torch.load('dqn_models/dqn_model_ver9.pth')
agent.model_local = agent.model_local.to(device)  # 将 DQN 模型移到 GPU
agent.model_local.eval()

num_episodes = 50
scores = []
dist = 0
max_x_pos = 0
best_episode_frames = []
best_episode_id = 0
flag_get_frames = []
flag_get_id = 0

for i_episode in range(1, num_episodes + 1):
    state = env.reset()
    state = process_state_image(state)
    state_features = mario_cnn(state).squeeze().detach()
    score = 0
    max_score = 0
    done = False
    episode_frames = []

    while not done:
        action = agent.act(state_features.cpu().numpy(), 0.1)
        next_state, reward, done, info = env.step(action)
        env.render()
        frame = env.render(mode='rgb_array')
        episode_frames.append(frame.copy())  # 存储当前episode的帧
        score += reward
        next_state = process_state_image(next_state)
        next_state_features = mario_cnn(next_state).squeeze().detach()
        state_features = next_state_features
        dist = info['x_pos']

    scores.append(score)
    print(f"Episode {i_episode}, Score: {score}, Distance: {dist}")

    if dist > max_x_pos:
        max_x_pos = dist
        best_episode_frames = episode_frames  # 更新最佳episode的帧
        best_episode_id = i_episode

    if info['flag_get']:
        flag_get_frames = episode_frames  # 更新得分最高的episode的帧
        flag_get_id = i_episode

env.close()

# 保存跑得最远的那个episode的视频
video_path = 'videos'
writer = imageio.get_writer(f'{video_path}/best_mario_run_distance.mp4', fps=30)
for frame in best_episode_frames:
    writer.append_data(frame)
writer.close()

# 保存得分最高的episode的视频
writer = imageio.get_writer(f'{video_path}/flag_get_mario_run.mp4', fps=30)
for frame in flag_get_frames:
    writer.append_data(frame)
writer.close()

print(f"Best score: {max(scores)}; Most far episode: {i_episode}")
print(f"Max distance: {max_x_pos}; Farthest episode: {best_episode_id}")
