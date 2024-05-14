# train_cnn.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from mario_env import make_env
from cnn_model import MarioCNN
import torchvision.transforms as T
from collections import deque
import random
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置超参数
batch_size = 32
num_epochs = 500
learning_rate = 0.001
num_frames = 4

# 数据变换
transform = T.Compose([
    T.ToTensor(),
    T.Resize((84, 84)),
    T.Normalize(mean=[0.5], std=[0.5])
])

# 处理状态图像
def process_state_image(state):
    state = np.array(state)
    state = np.squeeze(state)
    state = np.transpose(state, (1, 2, 0))
    state = transform(state)
    state = state.unsqueeze(0)
    return state.float().to(device)

# 创建环境和模型
env = make_env(skip_frames=4)
cnn_model = MarioCNN(output_size=256).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate)

# 经验回放缓冲区
replay_buffer = deque(maxlen=10000)

# 预定义动作序列
action_sequences = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 向右移动
    [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1],  # 向右并跳跃
    [1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1],  # 向右并长跳
]

# 收集初始数据
for _ in range(1000):
    state = env.reset()
    action_sequence = random.choice(action_sequences)  # 使用预定义动作序列
    for action in action_sequence:
        next_state, reward, done, info = env.step(action)
        replay_buffer.append((state, next_state))
        state = next_state
        if done:
            break
    while not done:  # 使用随机动作
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        replay_buffer.append((state, next_state))
        state = next_state
        if done:
            break

# 训练模型并记录损失
losses = []

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for _ in range(len(replay_buffer) // batch_size):
        batch = random.sample(replay_buffer, batch_size)
        states, next_states = zip(*batch)

        states = torch.cat([process_state_image(s) for s in states])
        next_states = torch.cat([process_state_image(ns) for ns in next_states])

        state_features = cnn_model(states)
        next_state_features = cnn_model(next_states).detach()

        loss = criterion(state_features, next_state_features)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    losses.append(epoch_loss / (len(replay_buffer) // batch_size))
    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss / (len(replay_buffer) // batch_size):.4f}")

    # 每100个epoch保存一次模型
    if (epoch + 1) % 100 == 0:
        torch.save(cnn_model, f'cnn_models/cnn_model_ver_further_steps.pth')

# 绘制损失图像
plt.plot(range(num_epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('CNN Training Loss')
plt.show()
