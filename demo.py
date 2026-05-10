import os

import streamlit as st
import torch
import torchvision.transforms.functional as TF

from PIL import Image
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

def make_unet():
    return UNet(spatial_dims=2, in_channels=3, out_channels=1, channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2), num_res_units=2)


@st.cache_resource
def load_models():
    repo_path = snapshot_download(repo_id=REPO_ID, allow_patterns=[path for _, path in MODEL_SPECS])
    models = []

    for name, filename in MODEL_SPECS:
        model = make_unet().to(device)
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


st.set_page_config(layout="wide")
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

    input_cols = st.columns([0.35, 1, 0.3], gap="small")

    with input_cols[0]: st.image(image.permute(1, 2, 0).cpu().numpy(), caption="Retinal image", width=320)
    with input_cols[1]: st.image(mask[0].cpu().numpy(), caption="Ground-truth mask", clamp=True, width=320)

    st.subheader("Predictions")
    preds = []

    for name, model in models:
        with torch.no_grad():
            pred = (torch.sigmoid(model(x)) > 0.5).float()

        dice = dice_score(pred, y)
        preds.append((name, pred[0, 0].cpu().numpy(), dice.item()))

    cols = st.columns(3)

    for row_start in range(0, len(preds), 3):
        cols = st.columns([1, 1, 1, 2.5], gap="small")

        for col, (name, pred, dice) in zip(cols, preds[row_start:row_start + 3]):
            with col:
                st.image(pred, clamp=True, width=280)
                st.markdown(f"**{name}**\nDice: {dice:.4f}")