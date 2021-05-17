from feature_extractor import PointNetfeat
from load_data import load_data


def train():
    PointNetfeat(global_feat=True, feature_transform=False)
    load_data()


if __name__ == '__main__':
    train()