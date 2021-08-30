import torch
from torch import nn
from config import config as cfg
from decoders.decoder import Decoder
from decoders.pfnet_decoder import PFNetDecoder
from encoders.graph_cnn_encoder import DGCNN
from encoders.pointnet_encoder import PointNetfeat


class GraphCNN3(nn.Module):

    def __init__(self, resolutions=None, crop_point_num=1024):
        super(GraphCNN3, self).__init__()
        if resolutions is None:
            resolutions = [256, 512, 1024]

        # Encoder Definition
        embedded = 1024
        args = {
            'k': 20,
            'emb_dims': embedded
        }
        self.encoder1 = DGCNN(args)
        self.encoder2 = DGCNN(args)
        self.encoder3 = DGCNN(args)
        self.encoder1_fc1 = nn.Linear(embedded, cfg.code_size)
        self.encoder1_fc2 = nn.Linear(cfg.code_size, cfg.code_size)
        self.encoder2_fc1 = nn.Linear(embedded, cfg.code_size)
        self.encoder2_fc2 = nn.Linear(cfg.code_size, cfg.code_size)
        self.encoder3_fc1 = nn.Linear(embedded, cfg.code_size)
        self.encoder3_fc2 = nn.Linear(cfg.code_size, cfg.code_size)
        self.resize_fc1 = nn.Linear(3*cfg.code_size, cfg.code_size)
        self.resize_fc2 = nn.Linear(cfg.code_size, cfg.code_size)

        self.relu = nn.ReLU()

        # Decoder Definition
        self.decoder = PFNetDecoder(resolutions=resolutions, crop_point_num=crop_point_num)

    def forward(self, x, add_noise=False, multi_resolution=True, use_max=False):
        if not multi_resolution:
            x = [x]
        batch_size, num_points, dim = x[0].size()

        assert dim == 3, "Fail: expecting 3 (x-y-z) as last tensor dimension!"

        # Refactoring batch for 'PointNetfeat' processing

        x[0] = x[0].permute(0, 2, 1)  # [BS, N, 3] => [BS, 3, N]
        if multi_resolution:
            x[1] = x[1].permute(0, 2, 1)
            x[2] = x[2].permute(0, 2, 1)
        code1, _ = self.encoder1(x[0])
        code1 = self.encoder1_fc2(self.relu(self.encoder1_fc1(code1)))
        code1 = code1.view(1, batch_size, -1)

        if multi_resolution:
            code2, _ = self.encoder2(x[1])
            code2 = self.encoder2_fc2(self.relu(self.encoder2_fc1(code2)))
            code2 = code2.view(1, batch_size, -1)

            code3, _ = self.encoder3(x[2])
            code3 = self.encoder3_fc2(self.relu(self.encoder3_fc1(code1)))
            code3 = code3.view(1, batch_size, -1)

            code = torch.cat((code1, code2, code3), dim=0).view(-1, cfg.code_size*3)

            if use_max:
                code = torch.max(code, dim=0).values
            else:
                code = self.resize_fc2(self.relu(self.resize_fc1(code)))

        else:
            code = code1

        if add_noise:
            noise = torch.rand(code.shape).cuda()
            code = code + noise

        # Decoding
        decoded = self.decoder(code)  # [BS, 3, num_points]

        # Reshaping decoded output before returning..
        # decoded = decoded.permute(0, 2, 1)  # [BS, 3, num_points] => [BS, num_points, 3]
        x[0] = x[0].permute(0, 2, 1)  # [BS, N, 3] => [BS, 3, N]
        if multi_resolution:
            x[1] = x[1].permute(0, 2, 1)
            x[2] = x[2].permute(0, 2, 1)
        return decoded
