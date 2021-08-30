from tqdm import tqdm

from config import config as cfg
import json
from pathlib import Path
import pandas as pd
import numpy as np
import os
import h5py


def load_novel_categories(similar=True):
    dir = None
    if similar:
        dir = cfg.novel_categories_sim
    else:
        dir = cfg.novel_categories_dissim

    out = np.empty((0, 1024, 3))

    for file in os.listdir(dir):
        with h5py.File(dir / file, "r") as f:
            key = list(f.keys())[0]
            arr = np.array(f[key])
            tot, n_points, dim = arr.shape
            random_indices = np.random.choice(n_points, size=1024, replace=False)
            arr = arr[:, random_indices, :]
            out = np.concatenate((out, arr), axis=0)

    return out


def load_data(mode="train", category="all", folder=cfg.dataset_base): # TODO: when category == "all" don't train on all the categories but just the ones asked
    # You can use the "label_codes" variable to do this job
    file = None
    if mode == "train":
        file = cfg.train_split_info
    elif mode == "val":
        file = cfg.val_split_info
    elif mode == "test":
        file = cfg.test_split_info

    print("Loading file {}".format(str(file)))

    if file is None:
        raise RuntimeError("mode = {} not found".format(mode))

    # label_codes = load_label_codes()  # this is the synsetoffset file parsed, but it's not used

    with open(file) as f:
        data = json.load(f)
        out = []
        for path in tqdm(data):
            fixed_path = Path(path.replace('shape_data', str(folder)))
            pts = load_pts(fixed_path, 1024, category=category)
            # TODO: the "if" below is useless since, re-reading the document, the sampling has to be done with replacement
            if pts.shape[0] == 1024:  # We are dropping files with less than 1024 points. Is this correct? Worth a check
                # I (cris) tried to replace the 1024-shape[0] points with zeros, but it slows down a bit
                # transposed_points = pts.transpose()
                out.append(pts)

        # out = list(filter(lambda item: item is not None, out))

        print("out is ", len(out))
        out = np.stack(out, axis=0)
        print(out.shape)
        return out


def load_label_codes():
    label_codes = pd.read_csv(cfg.label_codes, sep="\t").values
    return {label: code for label, code in list(label_codes)}


def load_pts(path, size, category):
    str_path = str(path)
    folder = Path(os.path.dirname(str_path))
    file_name = os.path.basename(str_path)
    data_path = folder / "points" / (file_name+".pts")
    labels_path = folder / "points_label" / (file_name+".seg")

    current_category = (str(folder).split("/"))[-1]

    if current_category.strip() not in cfg.allowed_categories:  # We only allow categories specified in proj. spec.
        return np.empty((0, 3))
    if category != "all" and current_category.strip() != category.strip():
        return np.empty((0, 3))

    data = pd.read_csv(data_path, sep=" ")
    labels = pd.read_csv(labels_path)

    df = pd.concat([data, labels], axis=1)
    to_be_removed = max(df.shape[0] - size, 0)
    drop_indices = np.random.choice(df.index, to_be_removed, replace=False) # TODO: try changing replace=True
    # TODO: normalize point cloud (is it already normalized?)
    df_subset = df.drop(drop_indices)

    return df_subset.values


def drop_points_to(data, max):
    if data.shape[0] < max:
        return data

    return np.random.choice(data, max, 4)
