from mario_env import MarioEnv

# 初始化超级马里奥游戏环境，确保使用最新版本的游戏环境
env = MarioEnv('SuperMarioBros-1-1-v0')
state = env.reset()

done = False
env.reset()
for step in range(5000):
    action = env.action_space.sample()  # 现在可以正确访问 action_space
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    if done:
       env.reset()

env.close()
