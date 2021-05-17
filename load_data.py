import config as cfg
import json
from pathlib import Path
import pandas as pd
import numpy as np

def load_data(mode="train"):
    file = None
    if mode == "train":
        file = cfg.train_split_info
    elif mode == "val":
        file = cfg.val_split_info
    elif mode == "test":
        file = cfg.test_split_info

    if file is None:
        raise RuntimeError("mode = {} not found".format(mode))

    with open(file) as f:
        data = json.load(f)
        data = [Path(item.replace('shape_data', str(cfg.dataset_base))) for item in data[:5]]
        data = [load_pts(item) for item in data]
        print(np.concatenate(data))



def load_pts(path):
    str_path = str(path)
    base, folder, file_name = str_path.split("/")
    data_path = Path(base) / folder / "points" / (file_name+".pts")
    labels_path = Path(base) / folder / "points_label" / (file_name+".seg")

    data = pd.read_csv(data_path, sep=" ")
    labels = pd.read_csv(labels_path)
    return pd.concat([data, labels], axis=1)
