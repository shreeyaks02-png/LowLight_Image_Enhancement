1RUA24CSE0439,1RUA24CSE0460,1RUA24CSE0424,1RUA24CSE0479

## Comparative Evaluation Results

### 1. Performance Metrics

| Method | Brightness | Contrast | Sharpness | Processing Time |
|---|---:|---:|---:|---:|
| Original Image | 18.72 | 31.83 | 600.21 | N/A |
| Classical CV | 61.22 | 39.60 | 778.13 | 660.9 ms |
| Zero-DCE | 57.38 | 48.29 | 12.84 | 24,619.6 ms |
| Hybrid Fusion | 60.68 | 40.51 | 723.21 | 25,296.2 ms |

### 2. Hybrid Configuration

The hybrid approach combines Classical Computer Vision and the pretrained Zero-DCE model using the following weights:

- Classical CV: 85%
- Zero-DCE: 15%

### 3. Evaluation Metrics Not Yet Measured

- PSNR: Not measured
- SSIM: Not measured
- LPIPS: Not measured
- NIQE: Not measured

### 4. Observation

The classical CV method produced images with a higher increase in brightness and clarity than the Zero-DCE method, which uses Deep Learning. However, the Zero-DCE method has a longer run time. The hybrid fusion method performs both methods.
