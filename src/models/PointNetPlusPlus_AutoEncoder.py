import numpy as np
import torch
from torch import nn

from encoders.pointnet_pp_encoder import PointNetSetAbstraction, PointNetSetAbstractionMsg
from decoders.decoder import Decoder
from config import config as cfg


class PointNetPlusPlus_AutoEncoder(nn.Module):
    def __init__(self, num_points=1024):
        super(PointNetPlusPlus_AutoEncoder, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], 3,
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,
                                             [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)

        self.lin1 = nn.Linear(1024, 512)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(512, cfg.code_size)

        self.decoder = Decoder(num_points=num_points)

    def forward(self, x):
        batch_size, num_points, dim = x.size()
        assert dim == 3, "Fail: expecting 3 (x-y-z) as last tensor dimension!"

        # Refactoring batch for 'PointNetfeat' processing
        x = x.permute(0, 2, 1)  # [BS, N, 3] => [BS, 3, N]

        # SA 1
        empty = torch.from_numpy(np.empty((batch_size, 0, num_points))).cuda()

        # Encoding
        l1_xyz, l1_points = self.sa1(x, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = l3_points.view(batch_size, 1024)
        embedding = self.lin2(self.relu(self.lin1(l3_points)))
        embedding = embedding.view(batch_size, cfg.code_size)
        # Decoding
        decoded = self.decoder(embedding)

        # Reshaping decoded output before returning..
        decoded = decoded.permute(0, 2, 1)  # [BS, 3, num_points] => [BS, num_points, 3]

        return decoded
