import torch
import torch.optim as optim
import numpy as np
from mario_env import make_env
from cnn_model import MarioCNN
from dqn_model import Agent

# 创建游戏环境
env = make_env(skip_frames=4)
action_size = env.action_space.n

# 加载预训练的CNN模型
pretrained_cnn = MarioCNN(num_frames=4)
pretrained_cnn.load_state_dict(torch.load("mario_cnn_selfsupervised.pth"))

# 创建DQN智能体
agent = Agent(state_size=256, action_size=action_size)

# 设置训练参数
num_episodes = 1000
max_steps = 10000
batch_size = 32
gamma = 0.99
learning_rate = 0.00025
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995

# 定义优化器
optimizer = optim.Adam(agent.model_local.parameters(), lr=learning_rate)

# 训练循环
for episode in range(num_episodes):
    state = env.reset()
    state = np.transpose(state, (1, 2, 0))  # 调整状态的维度顺序
    state = torch.FloatTensor(state).unsqueeze(0)  # 添加批次维度

    for step in range(max_steps):
        # 使用预训练的CNN模型提取特征
        with torch.no_grad():
            features = pretrained_cnn(state).squeeze()

        # 根据特征选择动作
        action = agent.act(features.numpy(), epsilon_start)

        # 执行动作并观察结果
        next_state, reward, done, _ = env.step(action)
        next_state = np.transpose(next_state, (1, 2, 0))  # 调整状态的维度顺序
        next_state = torch.FloatTensor(next_state).unsqueeze(0)  # 添加批次维度

        # 使用预训练的CNN模型提取下一状态的特征
        with torch.no_grad():
            next_features = pretrained_cnn(next_state).squeeze()

        # 存储经验
        agent.memory.push(features.numpy(), action, reward, next_features.numpy(), done)

        # 更新状态
        state = next_state

        # 进行学习
        if len(agent.memory) > batch_size:
            experiences = agent.memory.sample(batch_size)
            agent.learn(experiences, gamma, optimizer)

        if done:
            break

    # 更新epsilon值
    epsilon_start = max(epsilon_end, epsilon_decay * epsilon_start)

    # 打印训练进度
    if episode % 100 == 0:
        print(f"Episode: {episode}, Epsilon: {epsilon_start:.2f}")

# 保存训练好的DQN模型
torch.save(agent.model_local.state_dict(), "mario_dqn_pretrained.pth")