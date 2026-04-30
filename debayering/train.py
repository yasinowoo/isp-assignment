"""
Implement and CNN Model for Debayering
"""
import torch
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path

import debayering.data_preparation as dp
from debayering.metrics import MSE, PSNR, SSIM

class DebayeringModel(nn.Module):
    def __init__(self):
        super(DebayeringModel, self).__init__()
        # TOOD: Implement

    def forward(self, x):
        # TOOD: Implement
        return ...

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path: Path):
        # TODO: Implement
        self.data_path = data_path

    def __len__(self):
        # TODO: Implement
        return ...

    def __getitem__(self, idx: int):
        # TODO: Implement
        return mosaicked, rgb 


def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in tqdm(range(num_epochs)):
        for mosaicked, rgb in train_loader:
            optimizer.zero_grad()
            predicted_rgb = model(mosaicked)
            loss = criterion(predicted_rgb, rgb).mean()
            loss.backward()
            optimizer.step()


@torch.no_grad()
def evaluate_model(model, val_loader, criterion):
    total_loss = 0
    for images, labels in val_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
    return total_loss / len(val_loader)
