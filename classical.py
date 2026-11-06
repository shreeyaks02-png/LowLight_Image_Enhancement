
import time
import cv2
import numpy as np


def estimate_illumination(image):
    """Estimate illumination and normalize it for visualization."""

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    illumination = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=25,
        sigmaY=25
    )

    # Normalize illumination map to the full 0-255 range
    illumination = cv2.normalize(
        illumination,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return illumination.astype(np.uint8)


def adaptive_gamma(image):
    """Apply adaptive gamma correction."""

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    mean_brightness = np.mean(gray) / 255.0

    gamma = np.clip(
        1.0 - mean_brightness,
        0.4,
        0.9
    )

    lookup_table = np.array([
        ((i / 255.0) ** gamma) * 255
        for i in range(256)
    ]).astype(np.uint8)

    return cv2.LUT(image, lookup_table)


def apply_retinex(image):
    """Apply Retinex-based illumination enhancement."""

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

    l_float = l_channel.astype(np.float32) + 1.0

    blurred = cv2.GaussianBlur(
        l_float,
        (0, 0),
        sigmaX=30
    )

    retinex = np.log(l_float) - np.log(blurred + 1.0)

    retinex = cv2.normalize(
        retinex,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    enhanced_l = cv2.addWeighted(
        l_channel,
        0.75,
        retinex,
        0.25,
        0
    )

    result = cv2.merge([
        enhanced_l,
        a_channel,
        b_channel
    ])

    return cv2.cvtColor(
        result,
        cv2.COLOR_LAB2RGB
    )


def apply_clahe(image):
    """Improve local contrast using CLAHE."""

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l_channel)

    result = cv2.merge([
        enhanced_l,
        a_channel,
        b_channel
    ])

    return cv2.cvtColor(
        result,
        cv2.COLOR_LAB2RGB
    )


def denoise_image(image):
    """Reduce noise while preserving image details."""

    return cv2.bilateralFilter(
        image,
        d=7,
        sigmaColor=50,
        sigmaSpace=50
    )


def enhance_classical(image):
    """Complete classical low-light image enhancement pipeline."""

    start_time = time.perf_counter()

    image = np.asarray(image, dtype=np.uint8)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be an RGB image.")

    # Step 1: Estimate illumination
    illumination = estimate_illumination(image)

    # Step 2: Adaptive gamma correction
    result = adaptive_gamma(image)

    # Step 3: Retinex enhancement
    result = apply_retinex(result)

    # Step 4: CLAHE contrast enhancement
    result = apply_clahe(result)

    # Step 5: Noise reduction
    result = denoise_image(result)

    processing_time = time.perf_counter() - start_time

    return result, illumination, processing_time
