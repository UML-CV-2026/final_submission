import torch


class PadToSizePair:
    def __init__(self, target_size=(640, 640)):
        self.target_h, self.target_w = target_size

    def __call__(self, image, mask):
        _, h, w = image.shape
        pad_h = max(self.target_h - h, 0)
        pad_w = max(self.target_w - w, 0)

        # Symmetric padding keeps structures centered and avoids cropping.
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        image = torch.nn.functional.pad(
            image, (left, right, top, bottom), mode="constant", value=0.0
        )
        mask = torch.nn.functional.pad(
            mask, (left, right, top, bottom), mode="constant", value=0.0
        )
        return image, mask
