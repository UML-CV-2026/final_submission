import os

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import ImageReadMode, decode_image

_IMG_EXTS = {".png", ".jpg", ".jpeg"}


def _list_raster_images(dir_path):
    return sorted(
        f for f in os.listdir(dir_path)
        if os.path.splitext(f)[1].lower() in _IMG_EXTS
    )


class FundusVesselDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = _list_raster_images(img_dir)
        self.masks = _list_raster_images(mask_dir)

        assert len(self.images) == len(self.masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = decode_image(img_path, mode=ImageReadMode.RGB).float() / 255.0
        mask = decode_image(mask_path, mode=ImageReadMode.GRAY).float() / 255.0

        mask = (mask > 0.5).float()

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask


class AugmentPair:
    def __init__(self, crop_size=(256, 256)):
        self.crop_size = crop_size

    def __call__(self, image, mask):
        i, j, h, w = transforms.RandomCrop.get_params(image, self.crop_size)

        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        if torch.rand(1).item() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if torch.rand(1).item() < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        return image, mask

class CenterCropPair:
    def __init__(self, crop_size):
        self.crop_size = crop_size  # (height, width)

    def __call__(self, image, mask):
        image = TF.center_crop(image, self.crop_size)
        mask = TF.center_crop(mask, self.crop_size)
        return image, mask
