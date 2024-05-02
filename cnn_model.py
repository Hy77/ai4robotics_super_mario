import torch
import torch.nn as nn
import torch.nn.functional as F


class MarioCNN(nn.Module):
    def __init__(self):
        super(MarioCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2)  # 适应输入图像的大小
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(6400, 512)  # 适应从卷积层输出的维度
        self.fc2 = nn.Linear(512, 256)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)  # 展平卷积层输出
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # 这一层的输出将用于DQN决策层
        return x


# # 确保使用正确的设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = MarioCNN().to(device)
# # 假设输入是一个批量的游戏屏幕捕获，每个屏幕的大小是 84x84，并且有3个颜色通道
# test_input = torch.rand(5, 3, 84, 84).to(device)  # 5是批量大小
# test_output = model(test_input)
# print("Output shape:", test_output.shape)  # 应该输出：(5, 256)
