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

    label_codes = load_label_codes()

    with open(file) as f:
        data = json.load(f)
        out = []
        for path in data[:5]:
            fixed_path = Path(path.replace('shape_data', str(cfg.dataset_base)))
            pts = load_pts(fixed_path, 1024)
            # transposed_points = pts.transpose()
            out.append(pts)
        
        out = np.stack(out, axis=0)
        print(out.shape)
        return out


def load_label_codes():
    label_codes = pd.read_csv(cfg.label_codes, sep="\t").values
    return {label: code for label, code in list(label_codes)}


def load_pts(path, size):
    str_path = str(path)
    base, folder, file_name = str_path.split("/")
    data_path = Path(base) / folder / "points" / (file_name+".pts")
    labels_path = Path(base) / folder / "points_label" / (file_name+".seg")

    data = pd.read_csv(data_path, sep=" ")
    labels = pd.read_csv(labels_path)

    df = pd.concat([data, labels], axis=1)
    to_be_removed = max(df.shape[0] - size, 0)
    drop_indices = np.random.choice(df.index, to_be_removed, replace=False)
    df_subset = df.drop(drop_indices)

    return df_subset.values


def drop_points_to(data, max):
    if data.shape[0] < max:
        return data

    return np.random.choice(data, max, 4)
