import os

import streamlit as st
import torch
import torchvision.transforms.functional as TF
import numpy as np

from PIL import Image
from scipy import ndimage
from monai.networks.nets import UNet
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

REPO_ID = "yu-alvin/Computer-Vision-2026"

MODEL_SPECS = (
    ("Dice + BCE baseline", "models_baseline_batch_8/baseline.safetensors"),
    ("0.333 Dice + 0.3333 BCE + 0.3333 clDice", "models_clDice/clDice.safetensors"),
    ("0.8 Dice + 0.2 clDice", "models_cldice_no_bce/clDice_no_bce.safetensors"),
    ("0.4 Dice + 0.4 BCE + 0.2 clDice", "models_clDice_new/clDice_new.safetensors"),
    ("0.4 Dice + 0.4 BCE + 0.2 cbDice", "models_cbdice/cbDice.safetensors"),
    ("0.5 Dice + 0.5 BCE + 1e-5 Betti", "models_betti/betti.safetensors"),
)

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_models():
    repo_path = snapshot_download(repo_id=REPO_ID, allow_patterns=[path for _, path in MODEL_SPECS])
    models = []

    for name, filename in MODEL_SPECS:
        model = UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=1,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(device)

        state = load_file(os.path.join(repo_path, filename))
        model.load_state_dict(state)
        model.eval()
        models.append((name, model))

    return models

def load_image(file):
    image = Image.open(file).convert("RGB")
    return TF.pil_to_tensor(image).float() / 255.0

def load_mask(file):
    mask = Image.open(file).convert("L")
    mask = TF.pil_to_tensor(mask).float() / 255.0
    return (mask > 0.5).float()

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

st.set_page_config(layout="centered")
st.title("Retinal Vessel Segmentation")

models = load_models()

image_file = st.file_uploader("Upload retinal image", type=["png", "jpg", "jpeg"])
mask_file = st.file_uploader("Upload ground-truth mask", type=["png", "jpg", "jpeg"])

if image_file and mask_file:
    image = load_image(image_file)
    mask = load_mask(mask_file)

    x = image.unsqueeze(0).to(device)
    y = mask.unsqueeze(0).to(device)

    st.subheader("Input")

    input_cols = st.columns(2, gap="small")

    with input_cols[0]:
        st.image(image.permute(1, 2, 0).cpu().numpy(), caption="Retinal image", width=300)

    with input_cols[1]:
        st.image(mask[0].cpu().numpy(), caption="Ground-truth mask", clamp=True, width=300)

    st.subheader("Predictions")

    preds = []

    for name, model in models:
        with torch.no_grad():
            logits = model(x)
            pred = (torch.sigmoid(logits) > 0.5).float()

        dice = dice_score(pred, y)
        _, betti_stats = betti_error_one_image(logits[0], y[0])

        preds.append(
            (
                name,
                pred[0, 0].cpu().numpy(),
                dice.item(),
                betti_stats,
            )
        )

    for row_start in range(0, len(preds), 3):
        cols = st.columns(3, gap="medium")

        for col, (name, pred, dice, betti_stats) in zip(cols, preds[row_start:row_start + 3]):
            with col:
                st.image(pred, clamp=True, width=220)
                st.markdown(f"**{name}**")
                st.write(f"Dice: {dice:.4f}")
                st.write(f"Betti Error: {betti_stats['betti_error']}")
                st.write(f"B0 Error: {betti_stats['b0_error']}")
                st.write(f"B1 Error: {betti_stats['b1_error']}")