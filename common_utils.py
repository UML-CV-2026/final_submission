import torch
from torch.utils.data import DataLoader, Subset
from fundus_dataset import FundusVesselDataset, AugmentPair
from monai.networks.nets import UNet
from safetensors.torch import save_file
from pathlib import Path


def make_unet(channels=(16, 32, 64, 128, 256), device=torch.device("cuda")):
    model = UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=1,
        channels=channels,
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    return model.to(device)

def get_datasets(img_dir, mask_dir, transform=AugmentPair(crop_size=(512, 512))):
    train_dataset = FundusVesselDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        transform=transform,
    )
    val_dataset = FundusVesselDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        transform=None,
    )
    return train_dataset, val_dataset

def hard_dice_score(logits, masks, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    # flatten each image separately: [B, 1, H, W] -> [B, pixels]
    preds = preds.flatten(start_dim=1)
    masks = masks.flatten(start_dim=1)

    intersection = (preds * masks).sum(dim=1)
    denominator = preds.sum(dim=1) + masks.sum(dim=1)

    dice = (2 * intersection + eps) / (denominator + eps)

    return dice.mean()

def validate_full_image(model, loader, loss_fn, threshold=0.5, device=torch.device("cuda")):
    model.eval()
    val_loss = 0.0
    val_dice = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = loss_fn(logits, masks)
            dice = hard_dice_score(logits, masks, threshold=threshold)

            val_loss += loss.item() * images.size(0)
            val_dice += dice.item() * images.size(0)

    val_loss /= len(loader.dataset)
    val_dice /= len(loader.dataset)
    return val_loss, val_dice


def train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs, save_path=None, save_name="baseline.safetensors", device=torch.device("cuda")):
    history = []
    best_val_dice = 0.0
    best_epoch = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_loss_steps = []

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = loss_fn(logits, masks)

            loss.backward()
            optimizer.step()

            step_loss = loss.item()
            train_loss += step_loss * images.size(0)
            train_loss_steps.append(step_loss)

        train_loss /= len(train_loader.dataset)

        # Default validation protocol: full-image metrics on val_loader.
        val_loss, val_dice = validate_full_image(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            threshold=0.5,
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_loss_steps": train_loss_steps,
            "val_loss": val_loss,
            "val_dice": val_dice,
        })

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch + 1

            if save_path is not None:
                save_dir = Path(save_path)
                save_dir.mkdir(parents=True, exist_ok=True)
                st_path = save_dir / save_name
                save_file(model.state_dict(), st_path)

        print(
            f"Epoch {epoch + 1:03d}/{epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Best: {best_val_dice:.4f} @ {best_epoch}"
        )

    return history

def get_dataloaders(train_dataset, val_dataset, batch_size=4, num_workers=4):
    num_samples = len(train_dataset)
    generator = torch.Generator().manual_seed(42)

    # Shuffle the indices
    indices = torch.randperm(num_samples, generator=generator).tolist()

    train_size = int(0.8 * num_samples)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # We will always use a batch size of 1 for validation
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")

    return train_loader, val_loader

# print(f"Train size: {len(train_dataset)}")
# print(f"Val size: {len(val_dataset)}")

# images, masks = next(iter(train_loader))
# val_images, val_masks = next(iter(val_loader))

# print("Train images:", images.shape, images.dtype, images.min().item(), images.max().item())
# print("Train masks: ", masks.shape, masks.dtype, torch.unique(masks))

# print("Val images:", val_images.shape, val_images.dtype, val_images.min().item(), val_images.max().item())
# print("Val masks: ", val_masks.shape, val_masks.dtype, torch.unique(val_masks))