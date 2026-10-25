"""Preprocessing sinyal WiFi. Fungsi BVP dipakai untuk eksperimen; fungsi CSI
(amplitude/phase, sanitasi fase) disediakan untuk tahap hulu pipeline Widar3.0."""
import numpy as np
from scipy.signal import butter, filtfilt

def csi_to_amp_phase(csi):
    """Ekstraksi amplitude dan phase dari matriks CSI kompleks."""
    return np.abs(csi), np.angle(csi)

def butter_lowpass(x, cutoff, fs, order=4, axis=-1):
    """Low-pass Butterworth zero-phase."""
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, x, axis=axis)

def sanitize_phase(phase):
    """Phase sanitization: unwrap lalu hilangkan slope & offset linear."""
    ph = np.unwrap(phase, axis=-1)
    n = ph.shape[-1]
    k = np.arange(n)
    slope = (ph[..., -1] - ph[..., 0]) / (n - 1)
    offset = ph.mean(axis=-1)
    return ph - slope[..., None] * k - offset[..., None]

def resample_time(x, T):
    """Interpolasi linear sumbu waktu: (H, W, T0) -> (H, W, T)."""
    H, W, T0 = x.shape
    if T0 == T:
        return x
    old, new = np.linspace(0, 1, T0), np.linspace(0, 1, T)
    out = np.stack([np.interp(new, old, x[i, j]) for i in range(H) for j in range(W)])
    return out.reshape(H, W, T).astype(np.float32)

def preprocess_bvp(x, T=20, lowpass=False, cutoff=0.4):
    """BVP mentah (H,W,T0) -> (T,H,W): interpolasi, low-pass opsional, normalisasi."""
    if x.ndim == 2:
        x = x[..., None]
    x = resample_time(x, T)
    if lowpass:
        x = butter_lowpass(x, cutoff, fs=2.0, order=2, axis=-1).astype(np.float32)
    x = x / (x.max() + 1e-8)
    return np.transpose(x, (2, 0, 1))
