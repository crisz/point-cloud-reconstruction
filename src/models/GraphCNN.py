import torch
from torch import nn
from config import config as cfg
from decoders.decoder import Decoder
from encoders.graph_cnn_encoder import DGCNN
from encoders.pointnet_encoder import PointNetfeat


class GraphCNN(nn.Module):

    def __init__(self, num_points=1024, feature_transform=False):
        super(GraphCNN, self).__init__()

        # Encoder Definition
        args = {
            'k': 20,
            'emb_dims': 1024
        }
        self.encoder = DGCNN(args)
        self.fc1 = nn.Linear(1024, cfg.code_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(cfg.code_size, cfg.code_size)

        # Decoder Definition
        self.decoder = Decoder(num_points=num_points)

    def forward(self, x, add_noise=False, multi_resolution=None, use_max=None):
        batch_size, num_points, dim = x.size()
        assert dim == 3, "Fail: expecting 3 (x-y-z) as last tensor dimension!"

        # Refactoring batch for 'PointNetfeat' processing
        x = x.permute(0, 2, 1)  # [BS, N, 3] => [BS, 3, N]

        # Encoding# [BS, 3, N] => [BS, 100]
        code, trans_points = self.encoder(x)
        code = self.fc2(self.relu(self.fc1(code)))
        code = code.view(batch_size, -1)
        if add_noise:
            noise = torch.rand(code.shape).cuda()
            code = code + noise

        # Decoding
        decoded = self.decoder(code)  # [BS, 3, num_points]

        # Reshaping decoded output before returning..
        decoded = decoded.permute(0, 2, 1)  # [BS, 3, num_points] => [BS, num_points, 3]
        # print("Returning shape ", decoded.shape)
        return decoded, trans_points
