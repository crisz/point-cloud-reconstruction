import sys
from pathlib import Path

from torch import Tensor
from tqdm import tqdm

import numpy as np
import torch

from models.GraphCNN import GraphCNN
from models.GraphCNN2 import GraphCNN2
from models.GraphCNN_VAE import GraphCNN_VAE
from utils.chamfer_loss import PointLoss
from utils.farthest_point_sample import random_occlude_pointcloud_v2, random_occlude_batch
from utils.load_data import load_data
from config import config as cfg
from utils.remove_random_part import remove_random_part
from utils.save_model import save_model, load_model

torch.manual_seed(42)


def get_input_tensor(mode="train"):
    data = load_data(mode=mode)
    data = np.delete(data, 3, axis=2)
    # data = data[:1]
    # data = np.repeat(data, 10, axis=0)

    np.save(str(Path(".") / "original.npy"), data)
    # data[:, 1:] = (0, 0, 0)

    data = torch.from_numpy(data).float()
    return data


def train(radius):
    model = GraphCNN()
    model.cuda()
    input_tensor_cpu = get_input_tensor(mode="train")
    input_tensor = input_tensor_cpu.cuda()
    val_tensor_cpu = get_input_tensor(mode="val")

    val_tensor = val_tensor_cpu.cuda()
    print(input_tensor.shape)

    epochs = 50
    criterion = PointLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    i = 0
    loss = None
    epochs_done = 0
    if len(sys.argv) > 2:
        print("Loading model...")
        model_name = sys.argv[2]
        model_state, optimizer_state, epochs_done = load_model(model_name)
        print("Loaded: ", model_state, optimizer_state, epochs_done)
        if model_state is not None:
            print("Loaded {} epochs".format(epochs_done))
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
            epochs = epochs - epochs_done - 1

        print("Loaded model, remaining epochs: ", epochs)
    for i in range(epochs):
        print("Epoch {}".format(i+1))
        for batch in tqdm(torch.split(input_tensor, 16)):
            optimizer.zero_grad()
            # batch_partially_removed, batch_remaining, radius = remove_random_part(batch, radius)
            batch_partially_removed, batch_remaining = random_occlude_batch(batch)

            y_pred, _ = model.forward(batch_partially_removed, add_noise=False)

            loss = criterion(batch_remaining, y_pred)
            # loss = recon_loss + 10*auto_loss
            loss.backward()

            with torch.no_grad():
                optimizer.step()
        print("Recon loss is: ", loss)
        if i%10 == 0 and len(sys.argv) > 2:
            model_name = sys.argv[2]+str(i)
            print(">> Saving the model")
            save_model(model_name, model=model, optimizer=optimizer, epoch=i+epochs_done)
    i += 1
    if len(sys.argv) > 2:
        print(">> Saving the model")
        save_model(sys.argv[2], model=model, optimizer=optimizer, epoch=i+epochs_done)
    print(">> Done! Saving the result on {}".format(str(cfg.y_pred_path)))

    with torch.no_grad():
        result = np.empty((0, 1024, 3))
        for batch in tqdm(torch.split(val_tensor, 16)):
            # batch_partially_removed, batch_remaining, radius = remove_random_part(batch, radius)
            batch_partially_removed, batch_remaining = random_occlude_batch(batch)
            y_pred, _ = model.forward(batch_partially_removed, add_noise=False)
            y_pred_npy = y_pred.detach().cpu().numpy()
            original = batch.detach().cpu().numpy()
            val_error = criterion(y_pred, batch_remaining)
            complete_cloud = torch.cat([y_pred, batch_partially_removed], dim=1).detach().cpu().numpy()
            rnd_indices = np.random.choice(complete_cloud.shape[1], size=1024, replace=True)
            complete_cloud = complete_cloud[:, rnd_indices, :]
            print(result.shape, original.shape, y_pred_npy.shape, complete_cloud.shape)
            print("Val error is: ", val_error)
            # result = np.concatenate([result, y_pred_npy], axis=0)
            # y_pred, trans_points = model.forward(batch, add_noise=True)
            # y_pred_npy = y_pred.detach().cpu().numpy()
            result = np.concatenate([result, original, y_pred_npy, complete_cloud], axis=0)


        # result = np.concatenate(result, axis=0)

    print("Result is ", result[:10])
    np.save(cfg.y_pred_path, result)
    print("Saved!")
    # Decomment in case you have open3d installed
    # out = y_pred[0].detach().numpy()
    # show_point_cloud(out)


if __name__ == '__main__':
    print("recon v1.4.29")
    radius = float(sys.argv[1])
    train(radius)
