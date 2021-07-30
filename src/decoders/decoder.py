from torch import nn
import torch.nn.functional as F
from config import config as cfg


class Decoder(nn.Module):
    ''' Just a lightweight Fully Connected decoder:
    '''

    def __init__(self, num_points=2048):
        super(Decoder, self).__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(cfg.code_size, cfg.code_size)
        self.fc2 = nn.Linear(cfg.code_size, cfg.code_size)
        self.fc3 = nn.Linear(cfg.code_size, cfg.code_size)
        self.fc4 = nn.Linear(512, 1024)
        self.fc5 = nn.Linear(1024, self.num_points * 3)
        self.th = nn.Tanh()
        self.do1 = nn.Dropout(0.4)
        self.do2 = nn.Dropout(0.4)
        self.do3 = nn.Dropout(0.4)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.do1(self.fc1(x)))
        x = F.relu(self.do2(self.fc2(x)))
        x = F.relu(self.do3(self.fc3(x)))
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.th(self.fc5(x))
        x = x.view(batchsize, 3, self.num_points)
        return x