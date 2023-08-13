import torch
import torch.nn as nn
import torch.nn.functional as F
class discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(1, 3, 5)
        self.bn1_1 = nn.BatchNorm2d(3)
        self.fc1_2 = nn.Linear(10, 1728)
        self.bn1_2 = nn.BatchNorm1d(1728)
        self.fc2 = nn.Linear(1728 * 2, 800)
        self.bn2 = nn.BatchNorm1d(800)
        self.fc3 = nn.Linear(800, 1)

    def forward(self, x, y):     
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = torch.flatten(x, 1)
        y = F.relu(self.bn1_2(self.fc1_2(y)))        
        c = torch.cat([x, y], 1)
        c = F.relu(self.bn2(self.fc2(c)))
        c = torch.sigmoid(self.fc3(c))    
        return c


class generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1_1 = nn.Linear(128, 500)
        self.bn1_1 = nn.BatchNorm1d(500)
        self.fc1_2 = nn.Linear(10, 500)
        self.bn1_2 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(1000, 3 * 24 * 24)
        self.bn2 = nn.BatchNorm1d(3 * 24 * 24)
        self.conv2 = nn.ConvTranspose2d(3, 1, 5)

    def forward(self, x, y):

        x = F.relu(self.bn1_1(self.fc1_1(x)))
        y = F.relu(self.bn1_2(self.fc1_2(y)))
        c = torch.cat([x, y], 1)
        c = F.relu(self.bn2(self.fc2(c)))
        c = torch.reshape(c, (c.shape[0], 3, 24, 24))
        c = torch.tanh(self.conv2(c))
        return c
