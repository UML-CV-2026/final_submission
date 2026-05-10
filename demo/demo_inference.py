"""Shared inference and metrics for the Streamlit fundus demo (aligned with evals notebooks)."""

from __future__ import annotations

import io
import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from monai.networks.nets import UNet
from PIL import Image
from safetensors.torch import load_file
from scipy import ndimage
from skimage.morphology import binary_dilation, skeletonize

from cldice import clDice

REPO_ID = "yu-alvin/Computer-Vision-2026"

# Height/width must be multiples of this for MONAI UNet skip connections (2 ** len(strides)).
UNET_SPATIAL_DIVISOR = 16

# (short label for UI, HuggingFace filename under repo root)
MODEL_SPECS: tuple[tuple[str, str], ...] = (
    ("Dice + BCE baseline", "models_baseline_batch_8/baseline.safetensors"),
    ("0.333 Dice + 0.3333 BCE + 0.3333 clDice", "models_clDice/clDice.safetensors"),
    ("0.8 Dice + 0.2 clDice", "models_cldice_no_bce/clDice_no_bce.safetensors"),
    ("0.4 Dice + 0.4 BCE + 0.2 clDice", "models_clDice_new/clDice_new.safetensors"),
    ("0.4 Dice + 0.4 BCE + 0.2 cbDice", "models_cbdice/cbDice.safetensors"),
    ("0.5 Dice + 0.5 BCE + 1e-5 Betti", "models_betti/betti.safetensors"),
)


def make_unet(
    channels: tuple[int, ...] = (32, 64, 128, 256, 512),
) -> UNet:
    return UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=1,
        channels=channels,
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )


def load_weights(model: UNet, path: str, device: torch.device) -> None:
    state = load_file(path)
    model.load_state_dict(state)
    model.to(device)
    model.eval()


def image_bytes_to_tensor(data: bytes) -> torch.Tensor:
    """RGB float tensor [3, H, W] in [0, 1]."""
    pil = Image.open(io.BytesIO(data)).convert("RGB")
    arr = np.asarray(pil, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy(arr)


def pad_chw_for_unet(
    image_chw: torch.Tensor,
    divisor: int = UNET_SPATIAL_DIVISOR,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad bottom/right so H,W are divisible by ``divisor``. Returns (padded, (orig_h, orig_w))."""
    _c, h, w = image_chw.shape
    pad_h = (-int(h)) % divisor
    pad_w = (-int(w)) % divisor
    if pad_h == 0 and pad_w == 0:
        return image_chw, (h, w)
    # (left, right, top, bottom) for the (W, H) dimensions of (C, H, W).
    padded = F.pad(image_chw, (0, pad_w, 0, pad_h), mode="replicate")
    return padded, (h, w)


def crop_bchw_spatial(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Crop NCHW tensor to ``height`` x ``width`` from the top-left."""
    return x[:, :, :height, :width]


def mask_bytes_to_tensor(data: bytes, target_hw: tuple[int, int]) -> torch.Tensor:
    """Binary mask float tensor [1, H, W] in {0, 1}, resized to target (H, W)."""
    pil = Image.open(io.BytesIO(data)).convert("L")
    mask = torch.from_numpy(np.array(pil, dtype=np.float32) / 255.0).unsqueeze(0)
    mask = TF.resize(mask, list(target_hw), interpolation=TF.InterpolationMode.NEAREST)
    return (mask > 0.5).float()


def forward_logits(model: UNet, image_chw: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = image_chw.unsqueeze(0).to(device)
    with torch.no_grad():
        return model(x)


def predict_mask_from_logits(logits: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    probs = torch.sigmoid(logits)
    return (probs > threshold).cpu().numpy()[0, 0].astype(np.float32)


def skeleton(pred_mask: np.ndarray) -> np.ndarray:
    sk = skeletonize(pred_mask > 0)
    return binary_dilation(sk, footprint=np.ones((3, 3), dtype=bool))


def beta0_beta1_2d(binary_mask: np.ndarray) -> tuple[int, int]:
    binary_mask = binary_mask.astype(bool)
    fg_structure = ndimage.generate_binary_structure(2, 1)
    _, beta0 = ndimage.label(binary_mask, structure=fg_structure)

    background = ~binary_mask
    bg_structure = ndimage.generate_binary_structure(2, 2)
    bg_labels, num_bg = ndimage.label(background, structure=bg_structure)

    if num_bg == 0:
        return beta0, 0

    border_labels: set[int] = set()
    border_labels.update(np.unique(bg_labels[0, :]))
    border_labels.update(np.unique(bg_labels[-1, :]))
    border_labels.update(np.unique(bg_labels[:, 0]))
    border_labels.update(np.unique(bg_labels[:, -1]))
    border_labels.discard(0)

    all_bg_labels = set(range(1, num_bg + 1))
    hole_labels = all_bg_labels - border_labels
    beta1 = len(hole_labels)
    return beta0, beta1


def binary_segmentation_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> dict[str, float]:
    y_true = y_true.detach()
    y_pred = y_pred.detach()
    y_true = (y_true > 0.5).float()
    y_pred = (y_pred > threshold).float()
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)

    tp = torch.sum((y_pred == 1) & (y_true == 1)).float()
    fp = torch.sum((y_pred == 1) & (y_true == 0)).float()
    fn = torch.sum((y_pred == 0) & (y_true == 1)).float()

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * precision * recall + eps) / (precision + recall + eps)

    return {
        "dice": dice.item(),
        "iou": iou.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
    }


def betti_error_one_image(
    logits: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.5,
    b0_weight: float = 1.0,
    b1_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
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
        "b0_error": float(b0_error),
        "b1_error": float(b1_error),
        "betti_error": float(b0_error + b1_error),
    }
    return loss, stats


def hard_cldice_score(
    logits: torch.Tensor,
    masks: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    del eps
    probs = torch.sigmoid(logits)
    preds = probs > threshold
    masks_b = masks > 0.5
    preds = preds.squeeze(1)
    masks_b = masks_b.squeeze(1)

    cldice_scores: list[float] = []
    for pred_i, mask_i in zip(preds, masks_b):
        pred_np = pred_i.detach().cpu().numpy().astype(np.bool_)
        mask_np = mask_i.detach().cpu().numpy().astype(np.bool_)
        if not pred_np.any() and not mask_np.any():
            score = 1.0
        else:
            score = float(clDice(pred_np, mask_np))
            if not np.isfinite(score):
                score = 0.0
        cldice_scores.append(score)

    if len(cldice_scores) == 0:
        return torch.tensor(0.0, device=logits.device)
    return torch.tensor(sum(cldice_scores) / len(cldice_scores), device=logits.device)


def compute_all_metrics(
    logits: torch.Tensor,
    mask_1chw: torch.Tensor,
) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    m = binary_segmentation_metrics(mask_1chw, probs, threshold=0.5)
    hcl = hard_cldice_score(logits, mask_1chw, threshold=0.5).item()
    _, betti = betti_error_one_image(logits[0], mask_1chw[0], threshold=0.5)
    m["hard_cldice"] = float(hcl)
    m.update(betti)
    return m


def make_comparison_grid_figure(
    image_chw: torch.Tensor,
    gt_hw: np.ndarray | None,
    predictions: list[tuple[str, np.ndarray]],
    *,
    ncols: int = 4,
    suptitle: str | None = None,
) -> plt.Figure:
    """One figure: original fundus, ground truth (once each), then every model prediction (+ skeleton)."""
    original = image_chw.detach().cpu().numpy().transpose(1, 2, 0)
    n_panels = 2 + len(predictions)
    nrows = math.ceil(n_panels / ncols)
    fig_w = 3.8 * ncols
    fig_h = 3.8 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    idx = 0
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r][c]
            ax.axis("off")
            if idx >= n_panels:
                idx += 1
                continue
            if idx == 0:
                ax.imshow(np.clip(original, 0.0, 1.0))
                ax.set_title("Original", fontsize=10)
            elif idx == 1:
                if gt_hw is None:
                    ax.text(
                        0.5,
                        0.5,
                        "No mask\nuploaded",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=11,
                    )
                    ax.set_facecolor("#eee")
                else:
                    ax.imshow(gt_hw, cmap="gray", vmin=0, vmax=1)
                ax.set_title("Ground truth", fontsize=10)
            else:
                label, pred_mask = predictions[idx - 2]
                sk = skeleton(pred_mask)
                ax.imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
                ax.imshow(
                    np.ma.masked_where(~sk, sk),
                    cmap="autumn",
                    alpha=0.95,
                )
                ax.set_title(label, fontsize=9)
            idx += 1

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=1.02)
    plt.tight_layout()
    return fig
