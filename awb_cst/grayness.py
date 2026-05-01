import numpy as np


def _box_blur_wrap(x: np.ndarray, k: int = 7) -> np.ndarray:
    """Uniform filter with circular padding, implemented in pure NumPy."""
    if k % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    p = k // 2
    xp = np.pad(x, ((p, p), (p, p)), mode="wrap")
    integral = np.cumsum(np.cumsum(xp, axis=0), axis=1)
    integral = np.pad(integral, ((1, 0), (1, 0)), mode="constant")
    window_sum = (
        integral[k:, k:]
        - integral[:-k, k:]
        - integral[k:, :-k]
        + integral[:-k, :-k]
    )
    return window_sum / float(k * k)


def _gradient_magnitude(x: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(x)
    return np.sqrt(gx * gx + gy * gy) * 0.5


def _estimate_illuminant_grayness_index(
    image01: np.ndarray,
    percentage_of_gps: float = 0.1,
    delta_threshold: float = 1e-4,
    eps: float = 1e-7,
) -> np.ndarray:
    """Grayness Index illuminant estimation (NumPy-only approximation)."""
    h, w, _ = image01.shape
    num_pixels = h * w
    num_gps = max(1, int(np.floor(percentage_of_gps * num_pixels / 100.0)))

    R = _box_blur_wrap(image01[:, :, 0], k=7)
    G = _box_blur_wrap(image01[:, :, 1], k=7)
    B = _box_blur_wrap(image01[:, :, 2], k=7)

    sat_or_dark = (np.max(image01, axis=-1) >= 0.95) | (np.sum(image01, axis=-1) <= 0.0315)
    zero_mask = (R < 1e-3) | (G < 1e-3) | (B < 1e-3)
    mask = sat_or_dark | zero_mask

    R = np.clip(R, eps, None)
    G = np.clip(G, eps, None)
    B = np.clip(B, eps, None)
    norm1 = R + G + B

    dR = _gradient_magnitude(R)
    dG = _gradient_magnitude(G)
    dB = _gradient_magnitude(B)
    mask |= (dR <= delta_threshold) & (dG <= delta_threshold) & (dB <= delta_threshold)

    logR = np.log(R) - np.log(norm1)
    logB = np.log(B) - np.log(norm1)
    dlogR = _gradient_magnitude(logR)
    dlogB = _gradient_magnitude(logB)

    gi_map = np.sqrt(dlogR * dlogR + dlogB * dlogB)
    gi_map[mask] = gi_map.max()
    gi_map = _box_blur_wrap(gi_map, k=7)

    flat = gi_map.reshape(-1)
    threshold = np.partition(flat, num_gps - 1)[num_gps - 1]
    gp_mask = gi_map <= threshold

    chosen = image01[gp_mask]
    if chosen.size == 0:
        illum = image01.reshape(-1, 3).mean(axis=0)
    else:
        illum = chosen.mean(axis=0)

    illum = np.clip(illum, eps, None)
    return illum / np.linalg.norm(illum)


def correct(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must have shape [H, W, 3].")

    illum = _estimate_illuminant_grayness_index(image)
    illum /= illum[1] 
    return illum.astype(np.float32, copy=False)
