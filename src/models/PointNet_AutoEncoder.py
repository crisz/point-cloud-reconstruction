import torch
from torch import nn
from config import config as cfg
from decoders.decoder import Decoder
from encoders.pointnet_encoder import PointNetfeat


class PointNet_AutoEncoder(nn.Module):
    '''
    Complete AutoEncoder Model:
    Given an input point cloud X:
        - Step 1: encode the point cloud X into a latent low-dimensional code
        - Step 2: Starting from the code generate a representation Y as close as possible to the original input X

    Details:
    1. the 'code' size is hardocoded to 100 at line 45 - could be detrimental such a small code size
    1.1. If you change the code size you must modify accordingly also the decoder
    2. 'num_points' is the parameter controlling the number of points to be generated. In general we want to generate a number of points equal to the number of input points.
    '''

    def __init__(self, num_points=1024, feature_transform=False):
        super(PointNet_AutoEncoder, self).__init__()
        print("PointNet AE Init - num_points (# generated): %d" % num_points)

        # Encoder Definition
        self.encoder = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, int(cfg.code_size*2/3))
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(int(cfg.code_size*2/3), cfg.code_size)

        # Decoder Definition
        self.decoder = Decoder(num_points=num_points)

    def forward(self, x, add_noise=False):
        batch_size, num_points, dim = x.size()
        assert dim == 3, "Fail: expecting 3 (x-y-z) as last tensor dimension!"

        # Refactoring batch for 'PointNetfeat' processing
        x = x.permute(0, 2, 1)  # [BS, N, 3] => [BS, 3, N]

        # Encoding# [BS, 3, N] => [BS, 100]
        code, trans_points = self.encoder(x)
        # print("1", code.shape)
        code = self.fc2(self.relu(self.fc1(code)))
        # print("2", code.shape)
        code = code.view(batch_size, -1)
        if add_noise:
            noise = torch.rand(code.shape).cuda()
            code = code + noise

        # Decoding
        decoded = self.decoder(code)  # [BS, 3, num_points]

        # Reshaping decoded output before returning..
        decoded = decoded.permute(0, 2, 1)  # [BS, 3, num_points] => [BS, num_points, 3]

        return decoded, trans_points
