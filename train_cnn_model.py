import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from mario_env import make_env
from cnn_model import MarioCNN

# 设置超参数
num_epochs = 100
learning_rate = 0.001
num_frames = 4

# 创建游戏环境
env = make_env(skip_frames=num_frames)

# 创建CNN模型
model = MarioCNN(num_frames)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(num_epochs):
    state = env.reset()
    state = np.stack(state, axis=0)  # 将numpy数组列表合并为一个单一的numpy数组
    state = np.squeeze(state, axis=-1)  # 去除最后一个维度
    state = torch.FloatTensor(state).unsqueeze(0)  # 添加批次维度

    while True:
        # 前向传播
        features = model(state)

        # 计算重构损失
        loss = criterion(features, state)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 获取下一帧
        next_state, _, done, _ = env.step(env.action_space.sample())  # 随机选择动作
        next_state = np.stack(next_state, axis=0)  # 将numpy数组列表合并为一个单一的numpy数组
        next_state = np.squeeze(next_state, axis=-1)  # 去除最后一个维度
        next_state = torch.FloatTensor(next_state).unsqueeze(0)  # 添加批次维度

        # 更新状态
        state = next_state

        if done:
            break

    # 打印训练进度
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 保存训练好的模型
torch.save(model, "mario_cnn_selfsupervised.pth")