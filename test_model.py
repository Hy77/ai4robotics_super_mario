# test_model.py
import torch
import numpy as np
from mario_env import make_env
from cnn_model import MarioCNN
from dqn_model import Agent
import torchvision.transforms as T
import matplotlib.pyplot as plt
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

def test_model(cnn_model_path, dqn_model_path, num_episodes=50, video_path='videos'):
    env = make_env(skip_frames=1)
    action_size = env.action_space.n

    # 加载预训练的CNN模型
    cnn_model = torch.load(cnn_model_path)
    cnn_model.eval()

    # 加载预训练的DQN模型
    agent = Agent(state_size=256, action_size=action_size)
    agent.model_local = torch.load(dqn_model_path)
    agent.model_local.eval()

    scores = []
    max_x_pos = 0
    best_episode_frames = []
    best_episode_id = 0

    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        state = process_state_image(state)
        with torch.no_grad():
            state_features = cnn_model(state).squeeze()
        score = 0
        episode_frames = []

        while True:
            action = agent.act(state_features.cpu().numpy(), eps=0.1)  # 测试时不需要探索
            next_state, reward, done, info = env.step(action)
            score += reward

            frame = env.render(mode='rgb_array')
            episode_frames.append(frame.copy())  # 存储当前episode的帧

            next_state = process_state_image(next_state)
            with torch.no_grad():
                next_state_features = cnn_model(next_state).squeeze()

            state = next_state
            state_features = next_state_features

            if done:
                print(f"Episode {i_episode} - Score: {score}")
                break

        scores.append(score)

        if info['x_pos'] > max_x_pos:
            max_x_pos = info['x_pos']
            best_episode_frames = episode_frames  # 更新最佳episode的帧
            best_episode_id = i_episode

    env.close()

    # 保存跑得最远的那个episode的视频
    writer = imageio.get_writer(f'{video_path}/best_mario_run_new.mp4', fps=30)
    for frame in best_episode_frames:
        writer.append_data(frame)
    writer.close()

    print(f"Best score: {max(scores)}; Most far episode: {best_episode_id}")

    return scores

if __name__ == "__main__":
    cnn_model_path = 'cnn_models/cnn_model_ver10.pth'  # 替换为你的CNN模型路径
    dqn_model_path = 'dqn_models/dqn_model_ver10.pth'  # 替换为你的DQN模型路径

    scores = test_model(cnn_model_path, dqn_model_path, num_episodes=50)

    plt.plot(range(1, len(scores) + 1), scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Test Scores')
    plt.show()
