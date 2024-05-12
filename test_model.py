import torch
import numpy as np
from mario_env import make_env
from cnn_model import MarioCNN
from dqn_model import Agent
import torchvision.transforms as T
import matplotlib.pyplot as plt
import gym
import imageio
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()

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

mario_cnn = torch.load('cnn_models/cnn_model_ver8.pth')
mario_cnn.eval()

agent = Agent(state_size=256, action_size=action_size)
agent.model_local = torch.load('dqn_models/dqn_model_ver8.pth')
agent.model_local.eval()

num_episodes = 3
scores = []
max_x_pos = 0
best_episode_frames = []

for i_episode in range(1, num_episodes + 1):
    state = env.reset()
    state = process_state_image(state)
    state_features = mario_cnn(state).squeeze().detach()
    score = 0
    done = False
    episode_frames = []

    while not done:
        action = agent.act(state_features.cpu().numpy(), 0.1)
        next_state, reward, done, info = env.step(action)
        frame = env.render(mode='rgb_array')
        episode_frames.append(frame.copy())  # 存储当前episode的帧
        score += reward
        next_state = process_state_image(next_state)
        next_state_features = mario_cnn(next_state).squeeze().detach()
        state_features = next_state_features

    scores.append(score)
    print(f"Episode {i_episode}, Score: {score}")

    if info['x_pos'] > max_x_pos:
        max_x_pos = info['x_pos']
        best_episode_frames = episode_frames  # 更新最佳episode的帧

env.close()

# 保存跑得最远的那个episode的视频
video_path = 'videos'
writer = imageio.get_writer(f'{video_path}/best_mario_run.mp4', fps=30)
for frame in best_episode_frames:
    writer.append_data(frame)
writer.close()

print(f"Best score: {max(scores)}")