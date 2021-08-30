import sys
from pathlib import Path

from torch import Tensor
from tqdm import tqdm

import numpy as np
import torch

from models.GraphCNN import GraphCNN
from models.GraphCNN2 import GraphCNN2
from models.GraphCNN_VAE import GraphCNN_VAE
from utils.chamfer_loss import PointLoss
from utils.farthest_point_sample import farthest_point_sample, fps_batch, random_occlude_batch
from utils.load_data import load_data, load_novel_categories
from config import config as cfg
from utils.remove_random_part import remove_random_part
from utils.save_model import save_model, load_model

torch.manual_seed(42)


def get_input_tensor(mode="train"):
    data = load_data(mode=mode)
    data = np.delete(data, 3, axis=2)
    # data = data[:1]
    # data = np.repeat(data, 10, axis=0)

    np.save(str(Path(".") / "original.npy"), data)
    # data[:, 1:] = (0, 0, 0)

    data = torch.from_numpy(data).float()
    return data


def train(radius):
    model = GraphCNN2(crop_point_num=radius)
    model.cuda()
    input_tensor_cpu = get_input_tensor(mode="train")
    input_tensor = input_tensor_cpu.cuda()
    val_tensor_cpu = get_input_tensor(mode="val")
    val_tensor = val_tensor_cpu.cuda()

    novel_categories_sim = load_novel_categories(similar=True)
    novel_categories_dissim = load_novel_categories(similar=False)

    # val_tensor = torch.from_numpy(novel_categories_sim).float().cuda()
    #todo: remove this
    val_tensor_cpu = get_input_tensor(mode="train")
    val_tensor = val_tensor_cpu.cuda()
    print(input_tensor.shape)

    epochs = 165
    criterion = PointLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    i = 0
    loss = None
    epochs_done = 0
    if len(sys.argv) > 2:
        print("Loading model...")
        model_name = sys.argv[2]
        model_state, optimizer_state, epochs_done = load_model(model_name)
        if model_state is not None:
            print("Loaded {} epochs".format(epochs_done))
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
            epochs = epochs - epochs_done - 1

        print("Loaded model, remaining epochs: ", epochs)
    for i in range(epochs):
        print("Epoch {}".format(i+1))
        for batch in tqdm(torch.split(input_tensor, 16)):
            optimizer.zero_grad()
            # batch_partially_removed, batch_remaining, radius = remove_random_part(batch, radius)
            # input = [
            #     fps_batch(batch, 1024),
            #     fps_batch(batch, 512),
            #     fps_batch(batch, 256),
            # ]

            occluded, remaining = random_occlude_batch(batch, n_drop=200)

            # Also think about parellelize the encoder on 3 resolutions + torch.max
            res1, res2, res3 = model.forward(occluded, add_noise=False, multi_resolution=False)
            # loss1 = criterion(input[2], res1)
            # loss2 = criterion(input[1], res2)
            loss = criterion(batch, res3)
            # loss = loss1 + loss2 + loss3
            # loss = criterion(batch, res3)
            # loss = recon_loss + 10*auto_loss
            loss.backward()

            with torch.no_grad():
                optimizer.step()
        print("Recon loss is: ", loss)
        if i%2 == 0 and len(sys.argv) > 2:
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
            # batch_partially_removed, batch_remaining, radius = remove_random_part(batch, radius)
            occluded, remaining = random_occlude_batch(batch, n_drop=200)
            res1, res2, res3 = model.forward(occluded, add_noise=False, multi_resolution=False)
            final = torch.cat((occluded, res3), dim=1).detach().cpu().numpy()
            original = batch.detach().cpu().numpy()
            y_pred_npy = res3.detach().cpu().numpy()[:, np.random.choice(res3.shape[1], 1024, replace=True), :]
            occluded = occluded.detach().cpu().numpy()[:, np.random.choice(occluded.shape[1], 1024, replace=True), :]
            final = final[:, np.random.choice(final.shape[1], 1024, replace=True), :]

            result = np.concatenate([result, original, y_pred_npy, occluded, final], axis=0)


        # result = np.concatenate(result, axis=0)

    print("Result is ", result[:10])
    np.save(cfg.y_pred_path, result)
    print("Saved!")
    # Decomment in case you have open3d installed
    # out = y_pred[0].detach().numpy()
    # show_point_cloud(out)


if __name__ == '__main__':
    print("recon v1.4.62")
    radius = int(sys.argv[1])
    train(radius)
