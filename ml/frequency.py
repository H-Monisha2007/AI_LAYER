"""
DeepForensics Frequency-Domain Forensic Signal Generator

Analyzes spatial frequency distributions using:
- 2D Discrete Cosine Transform (2D-DCT)
- 2D Fast Fourier Transform (2D-FFT)
- High-Frequency Spectral Residual Energy Ratio
- Periodic Grid Artifact Detection (Transposed convolution checkerboard patterns)

Note: Frequency evidence is evaluated statistically as auxiliary signal.
"""
import numpy as np
from PIL import Image
from scipy.fftpack import dct
from typing import Dict, Any, Tuple


def compute_spectral_residual_features(img_np: np.ndarray) -> Dict[str, Any]:
    """
    Computes statistical 2D-FFT and 2D-DCT frequency features.
    
    Generative models (GANs, Diffusion models) often exhibit unnatural high-frequency power decay
    or grid-like spectral spikes due to upsampling / transposed convolutions.
    """
    if img_np.ndim == 3:
        gray = np.mean(img_np, axis=2).astype(np.float32)
    else:
        gray = img_np.astype(np.float32)

    h, w = gray.shape

    # 1. 2D Fast Fourier Transform (FFT)
    fft_2d = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft_2d)
    magnitude_spectrum = np.abs(fft_shift)
    log_fft = np.log(magnitude_spectrum + 1e-8)

    # Calculate high-frequency energy ratio
    cy, cx = h // 2, w // 2
    r = min(h, w) // 4
    y, x = np.ogrid[:h, :w]
    low_freq_mask = (y - cy) ** 2 + (x - cx) ** 2 <= r ** 2

    total_energy = np.sum(magnitude_spectrum) + 1e-8
    high_freq_energy = np.sum(magnitude_spectrum[~low_freq_mask])
    high_freq_ratio = float(high_freq_energy / total_energy)

    # 2. 2D Discrete Cosine Transform (DCT)
    dct_rows = dct(gray.astype(np.float64), axis=1, norm='ortho')
    dct_2d = dct(dct_rows, axis=0, norm='ortho')
    log_dct = np.log(np.abs(dct_2d) + 1e-8)

    # High frequency DCT energy ratio (bottom-right quadrant)
    dct_high_freq = np.mean(np.abs(dct_2d[h//2:, w//2:]))
    dct_total = np.mean(np.abs(dct_2d)) + 1e-8
    dct_high_ratio = float(dct_high_freq / dct_total)

    # 3. Detect periodic grid peakiness (spectral spike variance)
    spectral_std = float(np.std(log_fft))
    spectral_max_mean_ratio = float(np.max(log_fft) / (np.mean(log_fft) + 1e-8))

    # Anomaly indicator
    freq_anomaly_score = float(np.clip(
        0.5 + (high_freq_ratio - 0.45) * 1.5 + (spectral_max_mean_ratio - 2.5) * 0.1,
        0.0, 1.0
    ))

    return {
        "high_freq_energy_ratio": round(high_freq_ratio, 4),
        "dct_high_ratio": round(dct_high_ratio, 4),
        "spectral_spike_ratio": round(spectral_max_mean_ratio, 4),
        "spectral_std": round(spectral_std, 4),
        "frequency_anomaly_score": round(freq_anomaly_score, 4),
    }
