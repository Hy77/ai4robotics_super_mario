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
learning_rate = 0.0001  # 降低学习率
num_frames = 4
grad_clip = 1.0  # 梯度裁剪阈值

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
# cnn_model = torch.load(f'cnn_models/cnn_model_ver10.pth')

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate)

# 权重初始化
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

cnn_model.apply(init_weights)

# 经验回放缓冲区
replay_buffer = deque(maxlen=10000)

# 训练模型并记录损失
losses = []

# 收集初始数据
for _ in range(1000):
    state = env.reset()
    for _ in range(num_frames):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        replay_buffer.append((state, next_state))
        state = next_state
        if done:
            break

# 训练模型
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
        nn.utils.clip_grad_norm_(cnn_model.parameters(), grad_clip)  # 梯度裁剪
        optimizer.step()

        epoch_loss += loss.item()

    losses.append(epoch_loss / (len(replay_buffer) // batch_size))
    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss / (len(replay_buffer) // batch_size):.4f}")

    # 每10个epoch保存一次模型
    if (epoch + 1) % 100 == 0:
        torch.save(cnn_model, f'cnn_models/cnn_model_epoch_{epoch + 1}.pth')

plt.plot(range(num_epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('CNN Training Loss')
plt.show()
