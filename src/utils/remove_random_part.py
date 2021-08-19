import random
import sys
sys.path.append('<project directory>')

import torch
import math
import numpy as np
from config import config as cfg


def remove_random_part(batch, radius):
    # batch_size, points, dim = batch.shape
    batch = batch.cpu().numpy() # [BS, N, 3]
    batch_size = batch.shape[0]
    num_points = batch.shape[1]
    batch_remaining = np.empty((batch_size, num_points, 3))

    for i in range(batch.shape[0]):
        while True:
            distances = []
            removed = np.zeros((batch.shape[0], batch.shape[1]))
            deleted = 0
            total = batch.shape[1]
            random_index = np.random.choice(total, size=1, replace=False)[0]
            ran_x, ran_y, ran_z = batch[i, random_index]
            # x, y, z = torch.empty(3).normal_(mean=mean, std=std)
            # x = torch.empty(1).normal_(mean=ran_x, std=1)
            # y = torch.empty(1).normal_(mean=ran_y, std=1)
            # z = torch.empty(1).normal_(mean=ran_z, std=1)
            x = ran_x
            y = ran_y
            z = ran_z
            # print("(x,y,z)=({},{},{})".format(x,y,z))
            # print(batch[i])
            for j in range(batch.shape[1]):
                x0, y0, z0 = batch[i, j, :]
                x_sq_diff = (x - x0) ** 2
                y_sq_diff = (y - y0) ** 2
                z_sq_diff = (z - z0) ** 2
                # print("Distance is {} while radius is {}".format(math.sqrt(x_sq_diff + y_sq_diff + z_sq_diff), radius))
                distances.append(math.sqrt(x_sq_diff + y_sq_diff + z_sq_diff))
                if math.sqrt(x_sq_diff + y_sq_diff + z_sq_diff) < radius:
                    # print("Removing")
                    # batch[i, j, :] = (0, 0, 0)
                    removed[i, j] = 1
                    deleted += 1
            # print("Deleted: {}; Total: {}".format(deleted, total))
            # print("Number of ones: {}".format(removed[i].sum()))
            # print("Percentage removed: {}".format(deleted/total))
            current_batch_remaining = np.empty((0, 3))
            if 0.2 < deleted/total < 0.4 or True:
                pred_removed = (0,0,0)
                pred_not_removed = (0,0,0)
                for j, is_removed in enumerate(removed[i]):
                    if is_removed == 1:
                        pred_removed = batch[i, j, :]
                        batch[i, j, :] = pred_not_removed
                        current_point = batch[i, j]
                        current_point = np.expand_dims(current_point, axis=0)
                        current_batch_remaining = np.concatenate((current_batch_remaining, current_point), axis=0)
                    else:
                        pred_not_removed = batch[i, j, :]
                        current_batch_remaining = np.concatenate((current_batch_remaining, np.array([pred_removed])), axis=0)
                # current_batch_remaining = np.expand_dims(current_batch_remaining, axis=0)
                batch_remaining[i, :, :] = current_batch_remaining
                break
            else:
                radius = np.array(distances).sum() / len(distances)

    return torch.from_numpy(batch).cuda(), torch.from_numpy(batch_remaining).cuda(), radius


if __name__ == '__main__':
    print("** Remove random parts **")
