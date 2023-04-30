import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 5, 7)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(5, 10, 7)
            self.fc1 = nn.Linear(10 * 381 * 381, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 2)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)  # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x