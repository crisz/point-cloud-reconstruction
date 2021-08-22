from torch import nn
import torch

from decoders.decoder import Decoder
from encoders.graph_cnn_encoder import DGCNN
from config import config as cfg
from math import log, pi
import numpy as np


def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * log(2 * pi)
    return log_z - z.pow(2) / 2


def sample_gaussian(size):
    y = torch.randn(*size).float()
    y = y.cuda()
    return y


def reparameterize_gaussian(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(std.size()).to(mean)
    return mean + std * eps


def gaussian_entropy(logvar):
    const = 0.5 * float(logvar.size(1)) * (1. + np.log(np.pi * 2))
    ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
    return ent


class PointFlow(nn.Module):
    def __init__(self, args, num_points=1024):
        super(PointFlow, self).__init__()
        self.input_dim = args.input_dim
        self.zdim = args.zdim
        self.use_latent_flow = args.use_latent_flow
        self.use_deterministic_encoder = args.use_deterministic_encoder
        self.prior_weight = args.prior_weight
        self.recon_weight = args.recon_weight
        self.entropy_weight = args.entropy_weight
        self.distributed = args.distributed
        self.truncate_std = None
        args = {
            'k': 20,
            'emb_dims': 1024
        }
        self.encoder = DGCNN(args)
        self.point_cnf = Decoder(num_points=num_points, code_size=cfg.code_size)
        self.latent_cnf = nn.Sequential()

    def multi_gpu_wrapper(self, f):
        self.encoder = f(self.encoder)
        self.point_cnf = f(self.point_cnf)
        self.latent_cnf = f(self.latent_cnf)

    def forward(self, x, opt, step, writer=None):
        opt.zero_grad()
        batch_size = x.size(0)
        num_points = x.size(1)
        z_mu, z_sigma = self.encoder(x)
        if self.use_deterministic_encoder:
            z = z_mu + 0 * z_sigma
        else:
            z = self.reparameterize_gaussian(z_mu, z_sigma)

        # Compute H[Q(z|X)]
        if self.use_deterministic_encoder:
            entropy = torch.zeros(batch_size).to(z)
        else:
            entropy = self.gaussian_entropy(z_sigma)

        # Compute the prior probability P(z)
        if self.use_latent_flow:
            w, delta_log_pw = self.latent_cnf(z, None, torch.zeros(batch_size, 1).to(z))
            log_pw = standard_normal_logprob(w).view(batch_size, -1).sum(1, keepdim=True)
            delta_log_pw = delta_log_pw.view(batch_size, 1)
            log_pz = log_pw - delta_log_pw
        else:
            log_pz = torch.zeros(batch_size, 1).to(z)

        # Compute the reconstruction likelihood P(X|z)
        z_new = z.view(*z.size())
        z_new = z_new + (log_pz * 0.).mean()
        y, delta_log_py = self.point_cnf(x, z_new, torch.zeros(batch_size, num_points, 1).to(x))
        log_py = standard_normal_logprob(y).view(batch_size, -1).sum(1, keepdim=True)
        delta_log_py = delta_log_py.view(batch_size, num_points, 1).sum(1)
        log_px = log_py - delta_log_py

        # Loss
        entropy_loss = -entropy.mean() * self.entropy_weight
        recon_loss = -log_px.mean() * self.recon_weight
        prior_loss = -log_pz.mean() * self.prior_weight
        return entropy_loss, recon_loss, prior_loss


    def encode(self, x):
        z_mu, z_sigma = self.encoder(x)
        if self.use_deterministic_encoder:
            return z_mu
        else:
            return self.reparameterize_gaussian(z_mu, z_sigma)

    def decode(self, z, num_points, truncate_std=None):
        # transform points from the prior to a point cloud, conditioned on a shape code
        y = self.sample_gaussian((z.size(0), num_points, self.input_dim), truncate_std)
        x = self.point_cnf(y, z, reverse=True).view(*y.size())
        return y, x

    def sample(self, batch_size, num_points, truncate_std=None, truncate_std_latent=None, gpu=None):
        assert self.use_latent_flow, "Sampling requires `self.use_latent_flow` to be True."
        # Generate the shape code from the prior
        w = self.sample_gaussian((batch_size, self.zdim), truncate_std_latent, gpu=gpu)
        z = self.latent_cnf(w, None, reverse=True).view(*w.size())
        # Sample points conditioned on the shape code
        y = self.sample_gaussian((batch_size, num_points, self.input_dim), truncate_std, gpu=gpu)
        x = self.point_cnf(y, z, reverse=True).view(*y.size())
        return z, x

    def reconstruct(self, x, num_points=None, truncate_std=None):
        num_points = x.size(1) if num_points is None else num_points
        z = self.encode(x)
        _, x = self.decode(z, num_points, truncate_std)
        return x