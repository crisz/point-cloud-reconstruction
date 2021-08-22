import torch
from torch import nn

from utils.stn3d import STN3d


class Naive_AutoEncoder2(nn.Module):
    def __init__(self):
        super(Naive_AutoEncoder2, self).__init__()
        self.stn = STN3d()

    def forward(self, x):
        x = x.transpose(2, 1)
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        # x = x.transpose(2, 1)

        return x, None
