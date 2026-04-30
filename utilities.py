from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, Subset


def build_model(**kwargs: Any) -> UNet:
    defaults: dict[str, Any] = {
        "spatial_dims": 2,
        "in_channels": 3,
        "out_channels": 1,
        "channels": (16, 32, 64, 128, 256),
        "strides": (2, 2, 2, 2),
        "num_res_units": 2,
    }
    defaults.update(kwargs)
    return UNet(**defaults)


def build_losses(
    *,
    include_background: bool = True,
) -> tuple[DiceLoss, nn.BCEWithLogitsLoss]:
    dice_loss = DiceLoss(sigmoid=True, include_background=include_background)
    bce_loss = nn.BCEWithLogitsLoss()
    return dice_loss, bce_loss


def subset_dataset(dataset: Dataset, max_samples: int) -> Subset:
    n = min(max_samples, len(dataset))
    return Subset(dataset, list(range(n)))


def dataloader_options(*, pin_memory: bool | None = None) -> dict[str, Any]:
    use_cuda = torch.cuda.is_available()
    return {"pin_memory": use_cuda if pin_memory is None else pin_memory}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Adam,
    device: torch.device,
    dice_loss: DiceLoss,
    bce_loss: nn.BCEWithLogitsLoss,
    w_dice: float,
    w_bce: float,
) -> tuple[float, float, float]:
    model.train()
    sum_total = 0.0
    sum_dice = 0.0
    sum_bce = 0.0
    n_samples = 0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        d = dice_loss(logits, masks)
        b = bce_loss(logits, masks)
        loss = w_dice * d + w_bce * b
        loss.backward()
        optimizer.step()
        bs = images.size(0)
        sum_total += loss.item() * bs
        sum_dice += d.item() * bs
        sum_bce += b.item() * bs
        n_samples += bs
    inv = 1.0 / max(n_samples, 1)
    return sum_total * inv, sum_dice * inv, sum_bce * inv


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    w_dice: float,
    w_bce: float,
    device: torch.device | None = None,
    include_background: bool = True,
) -> list[dict[str, float]]:
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    dice_loss, bce_loss = build_losses(include_background=include_background)
    optimizer = Adam(model.parameters(), lr=lr)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        total, dice_term, bce_term = train_one_epoch(
            model,
            train_loader,
            optimizer,
            dev,
            dice_loss,
            bce_loss,
            w_dice,
            w_bce,
        )
        history.append(
            {
                "epoch": float(epoch),
                "loss": total,
                "dice_loss": dice_term,
                "bce_loss": bce_term,
            }
        )
    return history
