
import cv2
import numpy as np

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)


def calculate_quality_metrics(image, reference_image=None):
    """
    Calculate image quality metrics.

    Parameters:
        image: Enhanced RGB image (NumPy array).
        reference_image: Optional ground-truth RGB image.

    Returns:
        Dictionary containing quality metrics.
    """

    image = np.asarray(image)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be an RGB image.")

    if image.size == 0:
        raise ValueError("Input image is empty.")

    image = np.clip(image, 0, 255).astype(np.uint8)

    # Convert RGB to grayscale.
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    # 1. Average brightness.
    brightness = float(np.mean(gray))

    # 2. Contrast.
    contrast = float(np.std(gray))

    # 3. Sharpness.
    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

    metrics = {
        "Brightness": round(brightness, 2),
        "Contrast": round(contrast, 2),
        "Sharpness": round(sharpness, 2)
    }

    # Calculate reference-based metrics only
    # when a ground-truth image is supplied.
    if reference_image is not None:

        reference_image = np.asarray(reference_image)

        if (
            reference_image.ndim != 3
            or reference_image.shape[2] != 3
        ):
            raise ValueError(
                "Reference image must be an RGB image."
            )

        reference_image = np.clip(
            reference_image, 0, 255
        ).astype(np.uint8)

        if reference_image.shape != image.shape:
            raise ValueError(
                "Reference and enhanced images "
                "must have the same dimensions."
            )

        psnr = peak_signal_noise_ratio(
            reference_image,
            image,
            data_range=255
        )

        ssim = structural_similarity(
            reference_image,
            image,
            channel_axis=2,
            data_range=255
        )

        metrics["PSNR"] = round(float(psnr), 2)
        metrics["SSIM"] = round(float(ssim), 4)

    return metrics
