from tqdm import tqdm

from autoencoder import PointNet_AutoEncoder
import numpy as np
import torch

from chamfer_loss import PointLoss
from feature_extractor import PointNetfeat
from load_data import load_data
import config as cfg
# from utils.open3d_utils import show_point_cloud


def get_input_tensor():
    data = load_data()
    data = np.delete(data, 3, axis=2)
    data = torch.from_numpy(data).float()
    return data


def train():
    model = PointNet_AutoEncoder()
    # input_tensor = torch.from_numpy(np.array([[[1,0,0],[0,1,0],[0,0,1]]]).repeat(64, axis=0)).float()
    input_tensor = get_input_tensor()
    print(input_tensor.shape)
    # show_point_cloud(input_tensor[0])

    criterion = PointLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    y_pred = None

    for i in range(20):
        print("Epoch {}".format(i+1))
        err = None
        for batch in tqdm(torch.split(input_tensor, 32)):
            y_pred = model.forward(batch)
            #print(y_pred)
            err = criterion(y_pred, batch)
            # print(err)
            err.backward()

            with torch.no_grad():
                optimizer.step()
        print("Error is: ", err)

    np.save(cfg.y_pred_path, y_pred)
    # Decomment in case you have open3d installed
    # out = y_pred[0].detach().numpy()
    # show_point_cloud(out)


if __name__ == '__main__':
    train()
