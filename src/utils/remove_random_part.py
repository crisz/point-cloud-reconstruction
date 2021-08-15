import sys
sys.path.append('<project directory>')

import torch
import math
import numpy as np
from config import config as cfg


def remove_random_part(batch, radius):
    # batch_size, points, dim = batch.shape
    batch = batch.cpu().numpy()
    for i in range(batch.shape[0]):
        deleted = 0
        x, y, z = torch.empty(3).normal_(mean=0, std=1)
        for j in range(batch.shape[1]):
            x0, y0, z0 = batch[i, j, :]
            x_sq_diff = (x - x0) ** 2
            y_sq_diff = (y - y0) ** 2
            z_sq_diff = (z - z0) ** 2
            if math.sqrt(x_sq_diff + y_sq_diff + z_sq_diff) < radius:
                batch[i, j, :] = (0, 0, 0)
                deleted += 1

    return torch.from_numpy(batch).cuda()


if __name__ == '__main__':
    print("** Remove random parts **")
