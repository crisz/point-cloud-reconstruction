from pathlib import Path

from tqdm import tqdm

import numpy as np
import torch

from models.Naive_AutoEncoder import Naive_AutoEncoder
from utils.chamfer_loss import PointLoss
from utils.load_data import load_data
from config import config as cfg
# from utils.open3d_utils import show_point_cloud
from models.PointNet_AutoEncoder import PointNet_AutoEncoder
from models.PointNetPlusPlus_AutoEncoder import PointNetPlusPlus_AutoEncoder


def get_input_tensor():
    data = load_data(category="02958343")
    data = np.delete(data, 3, axis=2)
    # data = data[:1]
    # data = np.repeat(data, 10, axis=0)

    np.save(str(Path(".") / "original.npy"), data)
    # data[:, 1:] = (0, 0, 0)

    data = torch.from_numpy(data).float()
    return data


def train():
    # model = PointNet_AutoEncoder(feature_transform=True)
    # model = Naive_AutoEncoder()
    model = PointNetPlusPlus_AutoEncoder()
    model.cuda()
    input_tensor_cpu = get_input_tensor()
    input_tensor = input_tensor_cpu.cuda()
    print(input_tensor.shape)

    epochs = 100
    criterion = PointLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    y_pred = None
    y_preds = []

    for i in range(epochs):
        print("Epoch {}".format(i+1))
        err = None
        for batch in tqdm(torch.split(input_tensor, 32)):
            optimizer.zero_grad()
            y_pred, trans_points = model.forward(batch)

            if err is None:
                y_preds.append(y_pred.detach().cpu().numpy()[0])

            err = criterion(y_pred, batch)
            err.backward()
            optimizer.step()
            # err.backward()
            #
            # with torch.no_grad():
            #     optimizer.step()
        print("Error is: ", err)

    result = np.stack(y_preds, axis=0)
    np.save(str(Path(".") / "y0evol.npy"), result)
    print(">> Done! Saving the result on {}".format(str(cfg.y_pred_path)))

    with torch.no_grad():
        result = np.empty((0, 1024, 3))
        for batch in tqdm(torch.split(input_tensor, 32)):
            y_pred, trans_points = model.forward(batch)
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
    train()
