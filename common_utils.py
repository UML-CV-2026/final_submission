import torch
from torch.utils.data import DataLoader, Subset
from fundus_dataset import FundusVesselDataset, AugmentPair

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