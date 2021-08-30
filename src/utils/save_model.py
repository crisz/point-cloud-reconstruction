import torch
from pathlib import Path
import os


def save_model(name, model, optimizer, epoch):
    checkpoint_path = Path("/content/drive/MyDrive/checkpoints/") / name
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)


def load_model(name):
    checkpoint_path = Path(".") / name
    if checkpoint_path.exists():
        cp = torch.load(checkpoint_path)
        return cp['model_state_dict'], cp['optimizer_state_dict'], cp['epoch']
    else:
        return None, None, 0