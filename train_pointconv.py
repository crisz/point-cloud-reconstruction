from pathlib import Path

from tqdm import tqdm

from autoencoder import PointNet_AutoEncoder
import numpy as np
import torch

from chamfer_loss import PointLoss
from feature_extractor import PointNetfeat
from load_data import load_data
import config as cfg
# from utils.open3d_utils import show_point_cloud
from pointconv_model import PointNetModel


def get_input_tensor():
    data = load_data(category="03624134")
    data = np.delete(data, 3, axis=2)
    data = torch.from_numpy(data).float()
    return data


def train():
    model = PointNetModel()
    model.cuda()
    input_tensor_cpu = get_input_tensor()
    input_tensor = input_tensor_cpu.cuda()
    print(input_tensor.shape)

    epochs = 200
    criterion = PointLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    y_pred = None

    for i in range(epochs):
        print("Epoch {}".format(i+1))
        err = None
        for batch in tqdm(torch.split(input_tensor, 32)):
            y_pred, embedding = model.forward(batch)
            err = criterion(y_pred, batch)
            err.backward()

            with torch.no_grad():
                optimizer.step()
        print("Error is: ", err)

    print(">> Done! Saving the result on {}".format(str(cfg.y_pred_path)))

    with torch.no_grad():
        result = np.empty((0, 1024, 3))
        for batch in tqdm(torch.split(input_tensor, 32)):
            y_pred, embedding = model.forward(batch)
            y_pred_npy = y_pred.detach().cpu().numpy()
            result = np.concatenate([result, y_pred_npy], axis=0)

        # result = np.concatenate(result, axis=0)

    print("Result is ", result[:2])
    np.save(cfg.y_pred_path, result)
    print("Saved!")
    # Decomment in case you have open3d installed
    # out = y_pred[0].detach().numpy()
    # show_point_cloud(out)


if __name__ == '__main__':
    train()
