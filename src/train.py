from pathlib import Path

from torch import Tensor
from tqdm import tqdm

import numpy as np
import torch
import sys

from models.GraphCNN import GraphCNN
from utils.chamfer_loss import PointLoss
from utils.load_data import load_data
from config import config as cfg
from models.PointNet_AutoEncoder import PointNet_AutoEncoder
from utils.remove_random_part import remove_random_part
from utils.save_model import save_model
import matplotlib.pyplot as plt


def get_input_tensor(mode="train", folder=cfg.dataset_base):
    data = load_data(mode=mode, category="03797390", folder=folder)
    data = np.delete(data, 3, axis=2)
    # data = data[:1]
    # data = np.repeat(data, 10, axis=0)

    np.save(str(Path(".") / "original.npy"), data)
    # data[:, 1:] = (0, 0, 0)

    data = torch.from_numpy(data).float()
    return data


def train():
    model = PointNet_AutoEncoder(feature_transform=True)
    model = GraphCNN()
    # model = Naive_AutoEncoder()
    # model = PointNetPlusPlus_AutoEncoder()
    # model = Naive_AutoEncoder2()
    model.cuda()
    input_tensor_cpu = get_input_tensor(mode="train")
    input_tensor = input_tensor_cpu.cuda()
    val_tensor_cpu = get_input_tensor(mode="val")
    val_tensor = val_tensor_cpu.cuda()
    print(input_tensor.shape)

    epochs = 5
    criterion = PointLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    y_pred = None
    y_preds = []

    i = None
    loss_vals = []
    for i in range(epochs):
        print("Epoch {}".format(i+1))
        err = None
        epoch_loss = []
        for batch in tqdm(torch.split(input_tensor, 16)):
            optimizer.zero_grad()
            y_pred, _ = model.forward(batch, add_noise=False)
            err = criterion(y_pred, batch)
            err.backward()
            epoch_loss.append(err.item())
            with torch.no_grad():
                optimizer.step()
        print("Error is: ", err)
        loss_vals.append(sum(epoch_loss)/len(epoch_loss))

    if len(sys.argv) > 2:
        print(">> Saving the model")
        save_model(sys.argv[2], model=model, optimizer=optimizer, epoch=i)
        print(">> Plotting...")
        print(loss_vals)
        plt.plot(np.array(loss_vals), 'r')
        plt.show()
        filename = "{}_loss_plot.png".format(sys.argv[2])
        plt.savefig(fname=Path(".")/filename)
    else:
        print("Model name not provided, skipping save")
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
    print("Saved!")
    # Decomment in case you have open3d installed
    # out = y_pred[0].detach().numpy()
    # show_point_cloud(out)


if __name__ == '__main__':
    print("v3.0.11")
    train()
    # print("** Remove random parts **")
    # batch = get_input_tensor(mode="train")
    # print("Using value {}".format(float(sys.argv[1])))
    # remove_random_part(batch, float(sys.argv[1]))
