
from torch import nn, max
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
import torch.nn.functional as F


class PointNetModel(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, num_points=1024):
        super(PointNetModel, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(1024)

        ###############

        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_points * 3)

        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(1024)

    def forward(self, input):
        input = input.transpose(2, 1)
        batchsize, dim, npoints = input.shape
        xb = F.relu(self.bn1(self.conv1(input)))
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = F.relu(self.bn3(self.conv3(xb)))
        xb = self.bn4(self.conv4(xb))
        xb = nn.MaxPool1d(xb.size(-1))(xb)

        ######################
        embedding = nn.Flatten(1)(xb)
        # can also be written as (xb.view(-1, 1024))
        ######################

        xb = F.relu(self.bn5(self.fc1(embedding)))
        xb = F.relu(self.bn6(self.fc2(xb)))
        output = self.fc3(xb)
        output = output.view(batchsize, dim, npoints)
        output = output.transpose(2, 1)
        return output, embedding
