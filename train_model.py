import torch
from mario_agent import MarioAgent, MarioCNN
from mario_env import make_env

env = make_env(4)
env.reset()

pre_trained_model = None  # Path of model
mario = MarioAgent(state_dim=(4, 84, 84), action_dim=env.action_space.n)

if pre_trained_model is not None:
    pre_trained = torch.load(pre_trained_model, map_location=('cuda' if torch.cuda.is_available() else 'cpu'))
    mario.mario_model.load_state_dict(pre_trained.get('model'))
    mario.exploration_rate = pre_trained.get('exploration_rate')

episodes = 50000

for episode in range(1, episodes + 1):

    state = env.reset()

    episode_reward = 0
    episode_loss = 0
    episode_loss_length = 0

    while True:
        action = mario.select_action(state)
        next_state, reward, done, info = env.step(action)
        mario.store_experience(state, next_state, action, reward, done)

        q, loss = mario.learn()

        episode_reward += reward
        if loss:
            episode_loss += loss
            episode_loss_length += 1

        state = next_state

        if done or info['flag_get']:
            break

    if episode_loss_length == 0:
        episode_loss = 0
    else:
        episode_loss /= episode_loss_length

    print(
        f"Episode {episode} - "
        f"Step {mario.current_step} - "
        f"Epsilon {mario.exploration_rate} - "
        f"Reward {episode_reward} - "
        f"Loss {episode_loss:.3f}"
    )

    # Save model every 1000 episodes
    if episode % 1000 == 0:
        torch.save(dict(model=mario.mario_model.state_dict(), exploration_rate=mario.exploration_rate),
                   f"comb_mario_models/mario_model_{episode}.pth")
