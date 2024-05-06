import torch
import torch.nn as nn
import torch.nn.functional as F

class MarioDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MarioDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)  # First fully connected layer
        self.fc2 = nn.Linear(256, output_dim)  # Output layer with one output per action

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Activation function for hidden layer
        x = self.fc2(x)  # No activation function on output layer
        return x