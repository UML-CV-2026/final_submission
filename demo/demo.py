import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

import demo_inference as di


@st.cache_resource
def load_models(repo_id, device_name):
    device = torch.device(device_name)
    models = []

    for label, filename in di.MODEL_SPECS:
        weights_path = hf_hub_download(repo_id=repo_id, filename=filename)

        model = di.make_unet()
        di.load_weights(model, weights_path, device)

        models.append((label, model))

    return models


def resize_if_needed(image, max_side):
    height, width = image.shape[1], image.shape[2]

    if max_side <= 0 or max(height, width) <= max_side: return image

    scale = max_side / max(height, width)
    new_height = round(height * scale)
    new_width = round(width * scale)

    resized = F.interpolate(image.unsqueeze(0), size=(new_height, new_width), mode="bilinear", align_corners=False)

    st.info(f"Resized image from {height}x{width} to {new_height}x{new_width} (max side {max_side}).")

    return resized[0]


def load_mask(mask_file, image_shape):
    if mask_file is None: return None
    return di.mask_bytes_to_tensor(mask_file.getvalue(), image_shape)


def pad_for_unet(image):
    height, width = image.shape[1], image.shape[2]
    padded_image, _ = di.pad_chw_for_unet(image)

    if padded_image.shape[1:] != (height, width):
        st.info(
            f"Padded image from {height}×{width} to "
            f"{padded_image.shape[1]}×{padded_image.shape[2]} "
            f"(UNet requires H,W divisible by {di.UNET_SPATIAL_DIVISOR})."
        )

    return padded_image, height, width


def run_model(model, image, height, width, device):
    logits = di.forward_logits(model, image, device)
    logits = di.crop_bchw_spatial(logits, height, width)
    prediction = di.predict_mask_from_logits(logits)

    return logits, prediction


def show_metrics_table(rows):
    columns = [
        "model",
        "dice",
        "iou",
        "precision",
        "recall",
        "f1",
        "hard_cldice",
        "b0_error",
        "b1_error",
        "betti_error",
    ]

    df = pd.DataFrame(rows)[columns]

    score_columns = ["dice", "iou", "precision", "recall", "f1", "hard_cldice"]
    error_columns = ["b0_error", "b1_error", "betti_error"]

    df[score_columns] = df[score_columns].round(4)
    df[error_columns] = df[error_columns].round(2)

    st.subheader("Hard metrics per model")
    st.dataframe(df, use_container_width=True)


def main():
    st.set_page_config(page_title="Fundus vessel segmentation demo", layout="wide")

    st.title("Fundus vessel segmentation multi-model demo")
    st.caption(
        "Runs each of our models on an uploaded fundus image. "
        "You need to upload a ground-truth mask to compute Dice, IoU, clDice, and Betti errors."
    )

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    st.sidebar.write(f"**Device:** `{device_name}`")

    max_side = st.sidebar.number_input(
        "Max side length (pixels, 0 = no resize)",
        min_value=0,
        max_value=4096,
        value=2048,
        step=64,
        help="Downscale large uploads to reduce GPU memory use.",
    )

    image_file = st.file_uploader("Fundus image", type=["png", "jpg", "jpeg", "webp"])
    mask_file = st.file_uploader(
        "Ground-truth mask (optional)",
        type=["png", "jpg", "jpeg", "webp"],
    )

    run_button = st.button(
        "Run all models",
        type="primary",
        disabled=image_file is None,
    )

    if image_file is None or not run_button:
        return

    image = di.image_bytes_to_tensor(image_file.getvalue())
    image = resize_if_needed(image, max_side)

    mask = load_mask(mask_file, image.shape[1:])
    ground_truth = mask[0] if mask is not None else None

    model_input, height, width = pad_for_unet(image)

    try:
        models = load_models(di.REPO_ID, device_name)
    except Exception as error:
        st.error(f"Failed to load models from Hugging Face: {error}")
        return

    rows = []
    predictions: list[tuple[str, np.ndarray]] = []

    for label, model in models:
        logits, prediction = run_model(model, model_input, height, width, device)

        row = {"model": label}

        if mask is not None:
            mask_bchw = mask.unsqueeze(0).to(device)
            metrics = di.compute_all_metrics(logits, mask_bchw)
            row.update(metrics)

        rows.append(row)
        predictions.append((label, prediction))

    gt_array = ground_truth.numpy() if ground_truth is not None else None
    grid_fig = di.make_comparison_grid_figure(
        image,
        gt_array,
        predictions,
        suptitle="Fundus input, ground truth, and model predictions",
    )
    st.pyplot(grid_fig)
    plt.close(grid_fig)

    if mask is not None:
        show_metrics_table(rows)
    else:
        st.warning(
            "Upload a ground-truth mask and run again to see Dice, IoU, clDice, "
            "and Betti error metrics."
        )


if __name__ == "__main__":
    main()