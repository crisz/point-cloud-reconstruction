import sys
from pathlib import Path

from torch import Tensor
from tqdm import tqdm
import argparse

import numpy as np
import torch

from autoencoder import PointNet_AutoEncoder
from models.GraphCNN import GraphCNN
from models.GraphCNN2 import GraphCNN2
from models.GraphCNN3 import GraphCNN3
from models.GraphCNN_VAE import GraphCNN_VAE
from models.PointNetPlusPlus_AutoEncoder import PointNetPlusPlus_AutoEncoder
from utils.chamfer_loss import PointLoss
from utils.farthest_point_sample import random_occlude_batch
from utils.load_data import load_data, load_novel_categories
from config import config as cfg
from utils.remove_random_part import remove_random_part
from utils.save_model import save_model, load_model

torch.manual_seed(42)


def get_input_tensor(mode="train", path=None):
    if path:
        data = load_data(mode=mode, folder=path)
    else:
        data = load_data(mode=mode)

    data = np.delete(data, 3, axis=2)
    # data = data[:1]
    # data = np.repeat(data, 10, axis=0)

    np.save(str(Path(".") / "original.npy"), data)
    # data[:, 1:] = (0, 0, 0)

    data = torch.from_numpy(data).float()
    return data


def train(radius, args):
    model_name = args['model']
    mode = args['mode']
    multi_resolution = args['multi-resolution']
    use_max = args['use-max']

    if model_name == 'pointnet':
        model = PointNet_AutoEncoder()
    elif model_name == 'pointnetpp':
        model = PointNetPlusPlus_AutoEncoder()
    elif model_name == 'dgcnn':
        model = GraphCNN()
    else:
        model = GraphCNN3()
    model.cuda()
    input_tensor_cpu = get_input_tensor(mode="train", path=args['train-dataset'])
    input_tensor = input_tensor_cpu.cuda()
    val_tensor_cpu = get_input_tensor(mode="val", path=args['eval-dataset'])
    val_tensor = val_tensor_cpu.cuda()

    # novel_categories_sim = load_novel_categories(similar=True)
    # novel_categories_dissim = load_novel_categories(similar=False)

    # val_tensor = torch.from_numpy(novel_categories_sim).float().cuda()
    # print(input_tensor.shape)

    epochs = 40
    criterion = PointLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    i = 0
    loss = None
    epochs_done = 0
    loss_vals = []
    if len(sys.argv) > 2:
        print("Loading model...")
        model_name = sys.argv[2]
        model_state, optimizer_state, epochs_done = load_model(model_name)
        print("Loaded: ", model_state, optimizer_state, epochs_done)
        if model_state is not None:
            print("Loaded {} epochs".format(epochs_done))
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
            epochs = epochs - epochs_done - 1

        print("Loaded model, remaining epochs: ", epochs)
    for i in range(epochs):
        print("Epoch {}/{}".format(i+epochs_done+1, epochs+epochs_done))
        for batch in tqdm(torch.split(input_tensor, 16)):
            optimizer.zero_grad()
            if mode == "reconstruction":
                occluded, remaining = random_occlude_batch(batch, 200)
                _, _, y_pred = model.forward(occluded, add_noise=False, multi_resolution=multi_resolution, use_max=use_max)
                loss = criterion(remaining, y_pred)
            else:
                _, _, y_pred = model.forward(batch, add_noise=False, multi_resolution=multi_resolution, use_max=use_max)
                loss = criterion(batch, y_pred)

            loss.backward()
            loss_vals.append(loss.item())
            with torch.no_grad():
                optimizer.step()
        print("Recon loss is: ", np.array(loss_vals).sum()/len(loss_vals))
        loss_vals = []
        if i%6 == 0 and len(sys.argv) > 2:
            model_name = sys.argv[2]+str(i)
            print(">> Saving the model")
            save_model(model_name, model=model, optimizer=optimizer, epoch=i+epochs_done)
    i += 1
    if len(sys.argv) > 2:
        print(">> Saving the model")
        save_model(sys.argv[2], model=model, optimizer=optimizer, epoch=i+epochs_done)
    print(">> Done! Saving the result on {}".format(str(cfg.y_pred_path)))

    with torch.no_grad():
        result = np.empty((0, 1024, 3))
        for batch in tqdm(torch.split(val_tensor, 16)):
            if mode == "reconstruction":
                occluded, remaining = random_occlude_batch(batch, 200)
                _, _, y_pred = model.forward(occluded, add_noise=False, multi_resolution=False, use_max=False)
                y_pred_npy = y_pred.detach().cpu().numpy()
                original = batch.detach().cpu().numpy()
                complete_cloud = torch.cat([y_pred, occluded], dim=1).detach().cpu().numpy()

                occluded = occluded.detach().cpu().numpy()
                rnd_indices = np.random.choice(complete_cloud.shape[1], size=1024, replace=True)
                complete_cloud = complete_cloud[:, rnd_indices, :]

                rnd_indices = np.random.choice(occluded.shape[1], size=1024, replace=True)
                occluded = occluded[:, rnd_indices, :]
                result = np.concatenate([result, original, occluded, y_pred_npy, complete_cloud], axis=0)


        # result = np.concatenate(result, axis=0)

    print("Result is ", result[:10])
    np.save(cfg.y_pred_path, result)
    print("Saved!")
    # Decomment in case you have open3d installed
    # out = y_pred[0].detach().numpy()
    # show_point_cloud(out)


if __name__ == '__main__':
    print("recon v1.4.43")
    radius = float(sys.argv[1])

    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--mode')
    parser.add_argument('--train-dataset')
    parser.add_argument('--eval-dataset')
    parser.add_argument('--multi-resolution')
    parser.add_argument('--use-max')
    args = parser.parse_args()
    train(radius, args)
