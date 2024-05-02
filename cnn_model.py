import torch
import torch.nn as nn
import torch.nn.functional as F


class MarioCNN(nn.Module):
    def __init__(self):
        super(MarioCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(6400, 512)  # Adapting the dimensions of the output from the convolutional layer
        self.fc2 = nn.Linear(512, 256)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)  # Spreading Convolutional Layer Output
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # The output of this layer will be used in the DQN decision layer
        return x
