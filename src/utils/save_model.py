import torch
from pathlib import Path


def save_model(name, model, optimizer, epoch):
    checkpoint_path = Path(".") / name
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)


def load_model(name):
    checkpoint_path = Path(".") / name
    cp = torch.load(checkpoint_path)
    return cp['model_state_dict'], cp['optimizer_state_dict'], cp['epoch']