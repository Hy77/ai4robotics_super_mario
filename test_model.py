import torch
from pathlib import Path
from mario_agent import MarioAgent
from mario_env import make_env
import imageio
import matplotlib.pyplot as plt
import numpy as np

env = make_env(4)
env.reset()

pre_trained_model = Path('comb_mario_models/mario_model_50000.pth')  # Path to the pre-trained model
mario = MarioAgent(state_dim=(4, 84, 84), action_dim=env.action_space.n)

if pre_trained_model.exists():
    pre_trained = torch.load(pre_trained_model, map_location=('cuda' if torch.cuda.is_available() else 'cpu'))
    mario.mario_model.load_state_dict(pre_trained.get('model'))
    mario.exploration_rate = pre_trained.get('exploration_rate')

episodes = 100
best_episode_frames = []

rewards = []
max_distances = []
flag_gets = []

for episode in range(1, episodes + 1):
    state = env.reset()

    episode_reward = 0
    episode_frames = []
    max_distance = 0

    # Update epsilon
    if episode <= 50:
        mario.exploration_rate = 1
    else:
        mario.exploration_rate = 0.1

    while True:
        env.render()

        action = mario.select_action(state)
        next_state, reward, done, info = env.step(action)
        mario.store_experience(state, next_state, action, reward, done)
        frame = env.render(mode='rgb_array')
        episode_frames.append(frame.copy())  # 存储当前episode的帧
        episode_reward += reward
        max_distance = max(max_distance, info['x_pos'])

        state = next_state

        if info['flag_get']:
            best_episode_frames = episode_frames
            break
        elif done:
            break

    rewards.append(episode_reward)
    max_distances.append(max_distance)
    flag_gets.append(info['flag_get'])

    print(
        f"Episode {episode} - "
        f"Step {mario.current_step} - "
        f"Epsilon {mario.exploration_rate:.3f} - "
        f"Mean Reward {episode_reward} - "
        f"Max Distance {max_distance} - "
        f"Flag Get {info['flag_get']}"
    )

video_path = 'videos'
writer = imageio.get_writer(f'{video_path}/best_mario_run_flag_get_test_new.mp4', fps=30)
for frame in best_episode_frames:
    writer.append_data(frame)
writer.close()

# Plotting the results
plt.figure(figsize=(12, 5))
plt.plot(rewards, label='Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Episode Rewards')
plt.legend()

plt.show()
