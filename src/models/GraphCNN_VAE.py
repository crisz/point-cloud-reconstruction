import torch
from torch import nn
from config import config as cfg
from decoders.decoder import Decoder
from encoders.graph_cnn_encoder import DGCNN
import torch.nn.functional as F


class GraphCNN_VAE(nn.Module):
    def __init__(self, num_points=1024, feature_transform=False, alpha=1):
        super(GraphCNN_VAE, self).__init__()
        print("PointNet AE Init - num_points (# generated): %d" % num_points)

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
        self.decoder = Decoder(num_points=num_points, code_size=cfg.code_size)

        self.hidden2mu = nn.Linear(cfg.code_size, cfg.code_size)
        self.hidden2log_var = nn.Linear(cfg.code_size, cfg.code_size)
        self.alpha = alpha

    @staticmethod
    def reparametrize(mu, log_var):
        # Reparametrization Trick to allow gradients to backpropagate from the
        # stochastic part of the model
        sigma = torch.exp(0.5 * log_var)
        z = torch.randn(size=(mu.size(0), mu.size(1)))
        z = z.type_as(mu)  # Setting z to be .cuda when using GPU training
        return mu + sigma * z

    def encode(self, hidden):
        mu = self.hidden2mu(hidden)
        log_var = self.hidden2log_var(hidden)
        return mu, log_var

    def forward(self, x, add_noise=False):
        batch_size, num_points, dim = x.size()

        # Refactoring batch for 'PointNetfeat' processing
        x = x.permute(0, 2, 1)  # [BS, N, 3] => [BS, 3, N]

        # Encoding# [BS, 3, N] => [BS, 100]
        code, trans_points = self.encoder(x)
        code = F.sigmoid(self.fc2(self.relu(self.fc1(code))))
        code = code.view(batch_size, -1)
        mu, log_var = self.encode(code)
        code = self.reparametrize(mu, log_var)
        if add_noise:
            noise = torch.rand(code.shape).cuda()
            code = code + noise

        # Decoding
        decoded = self.decoder(code)  # [BS, 3, num_points]

        # Reshaping decoded output before returning..
        decoded = decoded.permute(0, 2, 1)  # [BS, 3, num_points] => [BS, num_points, 3]

        kl_loss = self.alpha * (-0.5 * (1 + log_var - mu ** 2 -
                                        torch.exp(log_var)).sum(dim=1)).mean(dim=0)

        return kl_loss, decoded
