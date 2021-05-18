from autoencoder import PointNet_AutoEncoder
import numpy as np
import torch

from chamfer_loss import PointLoss
from feature_extractor import PointNetfeat
from load_data import load_data


def get_input_tensor():
    data = load_data()
    data = np.delete(data, 3, axis=2)
    data = torch.from_numpy(data).float()
    return data


def train():
    model = PointNet_AutoEncoder()
    input_tensor = torch.from_numpy(np.array([[[1,0,0],[0,1,0],[0,0,1]]]).repeat(64, axis=0)).float()# get_input_tensor()
    criterion = PointLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for i in range(40):
        y_pred = model.forward(input_tensor)
        #print(y_pred)
        err = criterion(y_pred, input_tensor)
        print(err)
        err.backward()

        with torch.no_grad():
            optimizer.step()

    print(y_pred)


if __name__ == '__main__':
    train()
