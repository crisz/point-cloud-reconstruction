import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.stn3d import STN3d


class Naive_AutoEncoder(nn.Module):
    def __init__(self, num_points=1024):
        super(Naive_AutoEncoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 3, (1,))
        self.conv2 = torch.nn.Conv1d(3, 3, (1,))
        self.conv3 = torch.nn.Conv1d(3, 3, (1,))

        self.fc1 = nn.Linear(num_points, num_points)
        self.fc2 = nn.Linear(num_points, num_points)
        self.fc3 = nn.Linear(num_points, num_points)
        self.stn = STN3d()

    def forward(self, x):
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        return Variable(x, requires_grad=True), None
        x = x.transpose(2, 1)

        trans = self.stn(x)

        x = x.transpose(2, 1)

        x = torch.bmm(x, trans)

        x = x.transpose(2, 1)

        # x = x.view(-1, 1024)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = x.view(-1, 1024, 3)
        x = x.transpose(2, 1)

        return x, None
