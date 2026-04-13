"""
Image Quality Metrics Module
=============================

Comprehensive collection of image quality metrics for evaluating
instruction-driven image restoration.

Reference (full-reference) metrics  — require ground truth:
  - PSNR, SSIM

Usage:
    from src.metrics import compute_psnr, compute_ssim
    p = compute_psnr(clean, restored)
    s = compute_ssim(clean, restored)

    # Convenience function
    psnr, ssim = calculate_psnr_ssim(clean, restored)
"""

import numpy as np
from typing import Dict, Optional
from skimage.metrics import structural_similarity as _ssim
from skimage.metrics import peak_signal_noise_ratio as _psnr


# ══════════════════════════════════════════════════════════════════════
#  Reference metrics  (need ground truth)
# ══════════════════════════════════════════════════════════════════════

def compute_psnr(
    ground_truth: np.ndarray,
    test_image: np.ndarray,
    data_range: float = 1.0,
) -> float:
    """
    Peak Signal-to-Noise Ratio (dB).

    Higher is better. Typical restored-image values: 25–40 dB.
    """
    return float(_psnr(ground_truth, test_image, data_range=data_range))


def compute_ssim(
    ground_truth: np.ndarray,
    test_image: np.ndarray,
    data_range: float = 1.0,
    multichannel: bool = True,
) -> float:
    """
    Structural Similarity Index (SSIM).

    Range [0, 1]. Higher is better.
    """
    kwargs = dict(data_range=data_range)
    if multichannel and test_image.ndim == 3:
        kwargs["channel_axis"] = 2
    return float(_ssim(ground_truth, test_image, **kwargs))


def calculate_psnr_ssim(
    ground_truth: np.ndarray,
    test_image: np.ndarray,
    data_range: float = 1.0,
):
    """
    Convenience wrapper to compute both PSNR and SSIM together.

    This is useful for notebook usage where both metrics are required.
    """
    psnr = compute_psnr(ground_truth, test_image, data_range)
    ssim = compute_ssim(ground_truth, test_image, data_range)
    return psnr, ssim


# ══════════════════════════════════════════════════════════════════════
#  Composite helper (optional, lightweight)
# ══════════════════════════════════════════════════════════════════════

def compute_reference_metrics(
    ground_truth: np.ndarray,
    test_image: np.ndarray,
    data_range: float = 1.0,
) -> Dict[str, float]:
    """
    Compute both PSNR and SSIM and return as a dictionary.
    """
    return {
        "psnr": compute_psnr(ground_truth, test_image, data_range),
        "ssim": compute_ssim(ground_truth, test_image, data_range),
    }