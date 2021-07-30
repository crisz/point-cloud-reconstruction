from pathlib import Path

from torch import Tensor
from tqdm import tqdm

import numpy as np
import torch

from models.Naive_AutoEncoder import Naive_AutoEncoder
from models.Naive_AutoEncoder2 import Naive_AutoEncoder2
from utils.chamfer_loss import PointLoss
from utils.chamfer_loss2 import ChamferLoss
from utils.load_data import load_data
from config import config as cfg
# from utils.open3d_utils import show_point_cloud
from models.PointNet_AutoEncoder import PointNet_AutoEncoder
from models.PointNetPlusPlus_AutoEncoder import PointNetPlusPlus_AutoEncoder
from utils.stn3d import STN3d


def get_input_tensor(mode="train"):
    data = load_data(mode=mode)
    data = np.delete(data, 3, axis=2)
    # data = data[:1]
    # data = np.repeat(data, 10, axis=0)

    np.save(str(Path(".") / "original.npy"), data)
    # data[:, 1:] = (0, 0, 0)

    data = torch.from_numpy(data).float()
    return data


def train():
    model = PointNet_AutoEncoder(feature_transform=True)
    # model = Naive_AutoEncoder()
    # model = PointNetPlusPlus_AutoEncoder()
    # model = Naive_AutoEncoder2()
    model.cuda()
    input_tensor_cpu = get_input_tensor(mode="train")
    input_tensor = input_tensor_cpu.cuda()
    val_tensor_cpu = get_input_tensor(mode="val")
    val_tensor = val_tensor_cpu.cuda()
    print(input_tensor.shape)

    epochs = 100
    criterion = PointLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    y_pred = None
    y_preds = []

    for i in range(epochs):
        print("Epoch {}".format(i+1))
        err = None
        for batch in tqdm(torch.split(input_tensor, 16)):
            optimizer.zero_grad()
            y_pred, _ = model.forward(batch)
            err = criterion(y_pred, batch)
            err.backward()

            with torch.no_grad():
                optimizer.step()
        print("Error is: ", err)
        if i % 10 == 0:
            checkpoint_name = "cp_{}.pt".format(i)
            checkpoint_path = Path(".") / checkpoint_name
            torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': 0.0,  # TODO: solve
            }, checkpoint_path)
    print(">> Done! Saving the result on {}".format(str(cfg.y_pred_path)))

    with torch.no_grad():
        result = np.empty((0, 1024, 3))
        for batch in tqdm(torch.split(val_tensor, 32)):
            y_pred, trans_points = model.forward(batch)
            y_pred_npy = y_pred.detach().cpu().numpy()
            val_error = criterion(y_pred, batch)
            print("Val error is: ", val_error)
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
    print("v1.9")
    train()
