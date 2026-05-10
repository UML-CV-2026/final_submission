import torch
import numpy as np
from scipy import ndimage

def dice_score(pred, mask):
    return (2 * (pred * mask).sum()) / (pred.sum() + mask.sum() + 1e-8)

def beta0_beta1_2d(binary_mask):
    binary_mask = binary_mask.astype(bool)

    fg_structure = ndimage.generate_binary_structure(2, 1)
    _, beta0 = ndimage.label(binary_mask, structure=fg_structure)

    background = ~binary_mask
    bg_structure = ndimage.generate_binary_structure(2, 2)
    bg_labels, num_bg = ndimage.label(background, structure=bg_structure)

    if num_bg == 0:
        return beta0, 0

    border_labels = set()
    border_labels.update(np.unique(bg_labels[0, :]))
    border_labels.update(np.unique(bg_labels[-1, :]))
    border_labels.update(np.unique(bg_labels[:, 0]))
    border_labels.update(np.unique(bg_labels[:, -1]))

    border_labels.discard(0)

    all_bg_labels = set(range(1, num_bg + 1))
    hole_labels = all_bg_labels - border_labels

    beta1 = len(hole_labels)

    return beta0, beta1

def betti_error_one_image(logits, mask, threshold=0.5, b0_weight=1.0, b1_weight=1.0):
    probs = torch.sigmoid(logits)
    pred = (probs > threshold).float()

    pred_np = pred.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()

    if pred_np.ndim == 3:
        pred_np = pred_np[0]

    if mask_np.ndim == 3:
        mask_np = mask_np[0]

    b0_pred, b1_pred = beta0_beta1_2d(pred_np)
    b0_gt, b1_gt = beta0_beta1_2d(mask_np)

    b0_error = abs(b0_pred - b0_gt)
    b1_error = abs(b1_pred - b1_gt)

    total_error = b0_weight * b0_error + b1_weight * b1_error
    loss = torch.tensor(total_error, device=logits.device, dtype=logits.dtype)

    stats = {
        "betti_loss": float(total_error),
        "b0_error": b0_error,
        "b1_error": b1_error,
        "betti_error": b0_error + b1_error,
        "b0_pred": b0_pred,
        "b1_pred": b1_pred,
        "b0_gt": b0_gt,
        "b1_gt": b1_gt,
    }

    return loss, stats