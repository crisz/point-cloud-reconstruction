import sys
from pathlib import Path

from torch import Tensor
from tqdm import tqdm

import numpy as np
import torch

from models.GraphCNN import GraphCNN
from models.GraphCNN_VAE import GraphCNN_VAE
from utils.chamfer_loss import PointLoss
from utils.load_data import load_data
from config import config as cfg
from utils.remove_random_part import remove_random_part
from utils.save_model import save_model


def get_input_tensor(mode="train"):
    data = load_data(mode=mode, category="03790512")
    data = np.delete(data, 3, axis=2)
    # data = data[:1]
    # data = np.repeat(data, 10, axis=0)

    np.save(str(Path(".") / "original.npy"), data)
    # data[:, 1:] = (0, 0, 0)

    data = torch.from_numpy(data).float()
    return data


def train():
    model = GraphCNN_VAE()
    model.cuda()
    input_tensor_cpu = get_input_tensor(mode="train")
    input_tensor = input_tensor_cpu.cuda()
    val_tensor_cpu = get_input_tensor(mode="val")
    val_tensor = val_tensor_cpu.cuda()
    print(input_tensor.shape)

    epochs = 100
    criterion = PointLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    i = None
    loss = None
    for i in range(epochs):
        print("Epoch {}".format(i+1))
        for batch in tqdm(torch.split(input_tensor, 16)):
            optimizer.zero_grad()
            batch_partially_removed = remove_random_part(batch, 0.9)
            mu, log_var, x_out = model.forward(batch_partially_removed)
            kl_loss = (-0.5 * (1 + log_var - mu ** 2 -
                               torch.exp(log_var)).sum(dim=1)).mean(dim=0)
            print("B1 shape: ", batch.shape)
            print("B2 shape ", x_out.shape)
            recon_loss = criterion(batch, x_out)
            print("Loss 1 is ", kl_loss, kl_loss.shape)
            print("Loss 2 is ", recon_loss, recon_loss.shape)
            loss = recon_loss + kl_loss

            loss.backward()

            with torch.no_grad():
                optimizer.step()
        print("Recon loss is: ", loss)

    if len(sys.argv) > 2:
        print(">> Saving the model")
        save_model(sys.argv[2], model=model, optimizer=optimizer, epoch=i)
    print(">> Done! Saving the result on {}".format(str(cfg.y_pred_path)))

    with torch.no_grad():
        result = np.empty((0, 1024, 3))
        for batch in tqdm(torch.split(val_tensor, 32)):
            y_pred, trans_points = model.forward(batch)
            y_pred_npy = y_pred.detach().cpu().numpy()
            val_error = criterion(y_pred, batch)
            print("Val error is: ", val_error)
            result = np.concatenate([result, y_pred_npy], axis=0)
            y_pred, trans_points = model.forward(batch, add_noise=True)
            y_pred_npy = y_pred.detach().cpu().numpy()
            result = np.concatenate([result, y_pred_npy], axis=0)


        # result = np.concatenate(result, axis=0)

    print("Result is ", result[:10])
    np.save(cfg.y_pred_path, result)
    np.save("abc.npy", result[:10])
    np.save("def.npy", np.array((1, 2, 3)))
    print("Saved!")
    # Decomment in case you have open3d installed
    # out = y_pred[0].detach().numpy()
    # show_point_cloud(out)


if __name__ == '__main__':
    print("vae v2.15")
    train()
