# Robust Low-Light Image Restoration

## Project Overview
A low-light image enhancement system that combines Classical Computer Vision and a pretrained Zero-DCE deep-learning model to improve image brightness, contrast and visibility.

## Features
- Upload low-light RGB images.
- Classical CV enhancement using adaptive gamma correction, Retinex, CLAHE and bilateral filtering.
- Deep-learning enhancement using pretrained Zero-DCE.
- Hybrid enhancement using weighted fusion (85% Classical CV + 15% Zero-DCE).
- Brightness, contrast and sharpness evaluation.
- Processing-time comparison.
- Download enhanced images.

## Technologies Used
- Python
- OpenCV
- PyTorch
- Streamlit
- NumPy
- Pillow

## Installation

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd LowLight_Image_Enhancement

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```text
LowLight_Image_Enhancement/
├── app.py
├── requirements.txt
├── models/
│   └── Epoch99.pth
├── src/
│   ├── __init__.py
│   ├── classical.py
│   ├── deep_learning.py
│   └── quality_metrics.py
├── evaluation/
└── outputs/
```

## Evaluation Metrics
- Brightness
- Contrast
- Sharpness (Laplacian variance)
- Processing time

PSNR, SSIM, LPIPS and NIQE are not currently calculated by the implementation.

## Team Details
Add team member names and USNs here.