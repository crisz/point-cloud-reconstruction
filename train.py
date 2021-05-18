from feature_extractor import PointNetfeat
import numpy as np
import torch
from load_data import load_data


def get_input_tensor():
    data = load_data()
    data = np.delete(data, 3, axis=1)
    data = torch.from_numpy(data).float()
    return data


def train():
    model = PointNetfeat(global_feat=True, feature_transform=False)
    input_tensor = get_input_tensor()
    codes = model.forward(input_tensor)
    print(codes.shape)


if __name__ == '__main__':
    train()