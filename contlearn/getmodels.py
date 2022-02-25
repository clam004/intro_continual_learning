import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    """
    basic multi-layer perceptron
    """
    def __init__(self, hidden_size=400):
        super(MLP, self).__init__()
        self.flat = Flatten()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 10)

    def forward(self, input):
        x = self.flat(input)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)