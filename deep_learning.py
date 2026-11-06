
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn


# ==========================================
# 1. CPU CONFIGURATION
# ==========================================

torch.set_num_threads(4)
DEVICE = torch.device("cpu")


# ==========================================
# 2. ZERO-DCE MODEL ARCHITECTURE
# ==========================================

class ZeroDCE(nn.Module):

    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        self.e_conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.e_conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.e_conv3 = nn.Conv2d(32, 32, 3, 1, 1)
        self.e_conv4 = nn.Conv2d(32, 32, 3, 1, 1)
        self.e_conv5 = nn.Conv2d(64, 32, 3, 1, 1)
        self.e_conv6 = nn.Conv2d(64, 32, 3, 1, 1)
        self.e_conv7 = nn.Conv2d(64, 24, 3, 1, 1)

    def forward(self, x):

        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(
            self.e_conv5(torch.cat((x3, x4), dim=1))
        )

        x6 = self.relu(
            self.e_conv6(torch.cat((x2, x5), dim=1))
        )

        x_r = self.e_conv7(
            torch.cat((x1, x6), dim=1)
        )

        curves = torch.tanh(x_r)

        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(
            curves, 3, dim=1
        )

        enhanced = x

        for r in (r1, r2, r3, r4, r5, r6, r7, r8):

            enhanced = enhanced + r * (
                enhanced * enhanced - enhanced
            )

        return curves, enhanced


# ==========================================
# 3. LOAD PRE-TRAINED MODEL
# ==========================================

def load_zero_dce_model(model_path="models/Epoch99.pth"):

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Zero-DCE model checkpoint not found: {model_path}"
        )

    model = ZeroDCE().to(DEVICE)

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE
    )

    # Handle wrapped checkpoints.
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid Zero-DCE checkpoint format.")

    # Remove DataParallel prefix if present.
    cleaned_checkpoint = {}

    for key, value in checkpoint.items():

        new_key = key.replace("module.", "", 1)

        cleaned_checkpoint[new_key] = value

    # Load trained weights.
    model.load_state_dict(
        cleaned_checkpoint,
        strict=True
    )

    model.eval()

    return model


# ==========================================
# 4. COLOUR PRESERVATION
# ==========================================

def preserve_original_colours(
    original,
    enhanced,
    strength=0.35
):
    """
    Preserve original colour information while retaining
    the brightness produced by Zero-DCE.

    strength:
        0.0 = original colours
        1.0 = Zero-DCE colours
    """

    original = np.clip(
        original, 0.0, 1.0
    ).astype(np.float32)

    enhanced = np.clip(
        enhanced, 0.0, 1.0
    ).astype(np.float32)

    strength = float(np.clip(strength, 0.0, 1.0))

    original_lab = cv2.cvtColor(
        original,
        cv2.COLOR_RGB2LAB
    )

    enhanced_lab = cv2.cvtColor(
        enhanced,
        cv2.COLOR_RGB2LAB
    )

    result_lab = enhanced_lab.copy()

    # Keep Zero-DCE enhanced brightness.
    result_lab[:, :, 0] = enhanced_lab[:, :, 0]

    # Blend original and enhanced colour channels.
    result_lab[:, :, 1] = (
        (1.0 - strength) * original_lab[:, :, 1]
        + strength * enhanced_lab[:, :, 1]
    )

    result_lab[:, :, 2] = (
        (1.0 - strength) * original_lab[:, :, 2]
        + strength * enhanced_lab[:, :, 2]
    )

    result = cv2.cvtColor(
        result_lab,
        cv2.COLOR_LAB2RGB
    )

    return np.clip(result, 0.0, 1.0)


# ==========================================
# 5. ZERO-DCE IMAGE ENHANCEMENT
# ==========================================

def enhance_zero_dce(
    image,
    model,
    max_side=256
):

    start_time = time.perf_counter()

    # Validate input.
    image = np.asarray(image)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "Input image must have 3 RGB channels."
        )

    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Input image is empty.")

    if max_side <= 0:
        raise ValueError("max_side must be greater than zero.")

    # Convert input to uint8 if required.
    if image.dtype != np.uint8:

        if np.issubdtype(image.dtype, np.floating):
            if image.max() <= 1.0:
                image = image * 255.0

        image = np.clip(
            image, 0, 255
        ).astype(np.uint8)

    original_height, original_width = image.shape[:2]

    # ======================================
    # Resize image for faster CPU inference
    # ======================================

    scale = min(
        1.0,
        max_side / max(original_height, original_width)
    )

    if scale < 1.0:

        new_width = max(
            1, int(original_width * scale)
        )

        new_height = max(
            1, int(original_height * scale)
        )

        small_image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

    else:
        small_image = image.copy()

    # ======================================
    # Convert image into tensor
    # ======================================

    image_float = (
        small_image.astype(np.float32) / 255.0
    )

    image_tensor = torch.from_numpy(
        np.ascontiguousarray(
            image_float.transpose(2, 0, 1)
        )
    ).unsqueeze(0).to(DEVICE)

    # ======================================
    # Zero-DCE inference
    # ======================================

    model.eval()

    with torch.inference_mode():

        _, enhanced = model(image_tensor)

    # ======================================
    # Convert tensor to NumPy RGB image
    # ======================================

    enhanced = (
        enhanced.squeeze(0)
        .permute(1, 2, 0)
        .clamp(0.0, 1.0)
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    # ======================================
    # Preserve original colours
    # ======================================

    enhanced = preserve_original_colours(
        image_float,
        enhanced,
        strength=0.35
    )

    # ======================================
    # Resize to original dimensions
    # ======================================

    if scale < 1.0:

        enhanced = cv2.resize(
            enhanced,
            (original_width, original_height),
            interpolation=cv2.INTER_CUBIC
        )

    # ======================================
    # Convert to uint8 output
    # ======================================

    result = np.clip(
        enhanced * 255.0,
        0,
        255
    ).round().astype(np.uint8)

    processing_time = (
        time.perf_counter() - start_time
    )

    return result, processing_time
