import torch
import torch.nn as nn
import torch.nn.functional as F

class MarioCNN(nn.Module):
    def __init__(self, output_size):
        super(MarioCNN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(64, output_size)
        self.fc = nn.Linear(64 * 7 * 7, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x