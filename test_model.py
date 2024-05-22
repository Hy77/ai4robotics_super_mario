import torch
from pathlib import Path
from mario_agent import MarioAgent
from mario_env import make_env
import imageio

env = make_env(4)
env.reset()

pre_trained_model = Path('comb_mario_models/mario_model_50000.pth')  # Path to the pre-trained model
mario = MarioAgent(state_dim=(4, 84, 84), action_dim=env.action_space.n)

if pre_trained_model.exists():
    pre_trained = torch.load(pre_trained_model, map_location=('cuda' if torch.cuda.is_available() else 'cpu'))
    mario.mario_model.load_state_dict(pre_trained.get('model'))
    mario.exploration_rate = pre_trained.get('exploration_rate')

episodes = 50
best_episode_frames = []

for episode in range(1, episodes + 1):
    state = env.reset()

    episode_reward = 0
    episode_frames = []

    while True:
        env.render()

        action = mario.select_action(state)
        next_state, reward, done, info = env.step(action)
        mario.store_experience(state, next_state, action, reward, done)
        frame = env.render(mode='rgb_array')
        episode_frames.append(frame.copy())  # 存储当前episode的帧
        episode_reward += reward

        state = next_state

        if info['flag_get']:
            best_episode_frames = episode_frames
            break
        elif done:
            break

    print(
        f"Episode {episode} - "
        f"Step {mario.current_step} - "
        f"Epsilon {mario.exploration_rate} - "
        f"Mean Reward {episode_reward} - "
        f"Flag Get {info['flag_get']}"
    )

video_path = 'videos'
writer = imageio.get_writer(f'{video_path}/best_mario_run_flag_get.mp4', fps=30)
for frame in best_episode_frames:
    writer.append_data(frame)
writer.close()
