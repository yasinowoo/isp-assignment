import numpy as np

# ==========================================
# Part 1: Color Space Conversions
# ==========================================

def rgb_to_linear_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert an sRGB image to Linear RGB.
    
    Args:
        image: sRGB image with shape (*, H, W, 3).
    
    Returns:
        Linear RGB image with shape (*, H, W, 3).
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Input type is not a np.ndarray. Got {type(image)}")

    lin_rgb = np.where(
        image <= 0.04045,
        image / 12.92,
        np.power((image + 0.055) / 1.055, 2.4)
    )
    return lin_rgb.astype(np.float32, copy=False)


def rgb_to_xyz(image: np.ndarray) -> np.ndarray:
    """
    Convert a Linear RGB image to XYZ.
    
    Args:
        image: Linear RGB image with shape (*, H, W, 3).
    
    Returns:
        XYZ image with shape (*, H, W, 3).
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Input type is not a np.ndarray. Got {type(image)}")

    if image.ndim < 3 or image.shape[-1] != 3:
        raise ValueError(f"Input size must have a shape of (*, H, W, 3). Got {image.shape}")

    # Coefficients for RGB to XYZ conversion
    r = image[..., 0]
    g = image[..., 1]
    b = image[..., 2]

    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    return np.stack([x, y, z], axis=-1).astype(np.float32, copy=False)


def xyz_to_lab(image: np.ndarray) -> np.ndarray:
    r"""Convert an XYZ image to Lab.

    The input XYZ image is assumed to use D65 white point scaling.

    Args:
        image: XYZ image with shape (*, H, W, 3).

    Returns:
        Lab image with shape (*, H, W, 3).
        The L channel values are in the range 0..100. a and b are in the range -128..127.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Input type is not a np.ndarray. Got {type(image)}")

    if image.ndim < 3 or image.shape[-1] != 3:
        raise ValueError(f"Input size must have a shape of (*, H, W, 3). Got {image.shape}")

    # Normalize for D65 white point
    xyz_ref_white = np.array([0.95047, 1.0, 1.08883])
    xyz_normalized = image / xyz_ref_white

    threshold = 0.008856
    power = np.power(np.maximum(xyz_normalized, threshold), 1.0 / 3.0)
    scale = 7.787 * xyz_normalized + 4.0 / 29.0
    xyz_int = np.where(xyz_normalized > threshold, power, scale)

    x = xyz_int[..., 0]
    y = xyz_int[..., 1]
    z = xyz_int[..., 2]

    L = (116.0 * y) - 16.0
    a = 500.0 * (x - y)
    _b = 200.0 * (y - z)

    return np.stack([L, a, _b], axis=-1).astype(np.float32, copy=False)


def rgb_to_lab(image: np.ndarray) -> np.ndarray:
    r"""Convert a RGB image to Lab.

    The input RGB image is assumed to be in the range of [0, 1]. Lab
    color is computed using the D65 illuminant and Observer 2.

    Args:
        image: RGB image to be converted to Lab with shape (*, H, W, 3).

    Returns:
        Lab version of the image with shape (*, H, W, 3).
        The L channel values are in the range 0..100. a and b are in the range -128..127.

    Example:
        >>> import numpy as np
        >>> input_img = np.random.rand(2, 4, 5, 3)
        >>> output_img = rgb_to_lab(input_img)  # Shape: (2, 4, 5, 3)
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Input type is not a np.ndarray. Got {type(image)}")

    if image.ndim < 3 or image.shape[-1] != 3:
        raise ValueError(f"Input size must have a shape of (*, H, W, 3). Got {image.shape}")

    # 1. Convert to Linear RGB
    lin_rgb = rgb_to_linear_rgb(image)

    # 2. Convert to XYZ
    xyz_im = rgb_to_xyz(lin_rgb)

    # 3. Convert XYZ -> Lab
    return xyz_to_lab(xyz_im)


# ==========================================
# Part 2: Delta E Calculation Logic
# ==========================================

def _split_lab_channels(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Helper to split Lab input into L, a, b components.
    Assumes input comes from rgb_to_lab/xyz_to_lab with shape (*, H, W, 3).
    """
    # Keep all calculations in float32
    a = a.astype(np.float32, copy=False)
    return a[..., 0], a[..., 1], a[..., 2]


def _delta_E_CIE2000(
    Lab_1: np.ndarray, Lab_2: np.ndarray, textiles: bool = False
) -> np.ndarray:
    """
    Return the difference Delta E_00 between two given CIE L*a*b* arrays.
    
    Args:
        Lab_1: Lab image 1 with shape (*, H, W, 3)
        Lab_2: Lab image 2 with shape (*, H, W, 3)
        textiles: If True, uses specific parametric factors for textiles.
        
    Returns:
        Delta E difference map with shape (*, H, W)
    """
    L_1, a_1, b_1 = _split_lab_channels(Lab_1)
    L_2, a_2, b_2 = _split_lab_channels(Lab_2)

    k_L = 2 if textiles else 1
    k_C = 1
    k_H = 1

    C_1_ab = np.hypot(a_1, b_1)
    C_2_ab = np.hypot(a_2, b_2)

    C_bar_ab = (C_1_ab + C_2_ab) / 2
    C_bar_ab_7 = C_bar_ab**7

    G = 0.5 * (1 - np.sqrt(C_bar_ab_7 / (C_bar_ab_7 + 25**7)))

    a_p_1 = (1 + G) * a_1
    a_p_2 = (1 + G) * a_2

    C_p_1 = np.hypot(a_p_1, b_1)
    C_p_2 = np.hypot(a_p_2, b_2)

    # Calculate Hue
    h_p_1 = np.where(
        (b_1 == 0) & (a_p_1 == 0),
        0,
        np.degrees(np.arctan2(b_1, a_p_1)) % 360,
    )
    h_p_2 = np.where(
        (b_2 == 0) & (a_p_2 == 0),
        0,
        np.degrees(np.arctan2(b_2, a_p_2)) % 360,
    )

    delta_L_p = L_2 - L_1
    delta_C_p = C_p_2 - C_p_1

    h_p_2_s_1 = h_p_2 - h_p_1
    C_p_1_m_2 = C_p_1 * C_p_2

    # Hue difference handling
    delta_h_p = np.select(
        [
            C_p_1_m_2 == 0,
            np.abs(h_p_2_s_1) <= 180,
            h_p_2_s_1 > 180,
            h_p_2_s_1 < -180,
        ],
        [
            0,
            h_p_2_s_1,
            h_p_2_s_1 - 360,
            h_p_2_s_1 + 360,
        ],
    )

    delta_H_p = 2 * np.sqrt(C_p_1_m_2) * np.sin(np.radians(delta_h_p / 2))

    L_bar_p = (L_1 + L_2) / 2
    C_bar_p = (C_p_1 + C_p_2) / 2

    a_h_p_1_s_2 = np.abs(h_p_1 - h_p_2)
    h_p_1_a_2 = h_p_1 + h_p_2

    h_bar_p = np.select(
        [
            C_p_1_m_2 == 0,
            a_h_p_1_s_2 <= 180,
            (a_h_p_1_s_2 > 180) & (h_p_1_a_2 < 360),
            (a_h_p_1_s_2 > 180) & (h_p_1_a_2 >= 360),
        ],
        [
            h_p_1_a_2,
            h_p_1_a_2 / 2,
            (h_p_1_a_2 + 360) / 2,
            (h_p_1_a_2 - 360) / 2,
        ],
    )

    T = (
        1
        - 0.17 * np.cos(np.radians(h_bar_p - 30))
        + 0.24 * np.cos(np.radians(2 * h_bar_p))
        + 0.32 * np.cos(np.radians(3 * h_bar_p + 6))
        - 0.20 * np.cos(np.radians(4 * h_bar_p - 63))
    )

    delta_theta = 30 * np.exp(-(((h_bar_p - 275) / 25) ** 2))

    C_bar_p_7 = C_bar_p**7
    R_C = 2 * np.sqrt(C_bar_p_7 / (C_bar_p_7 + 25**7))

    L_bar_p_2 = (L_bar_p - 50) ** 2
    S_L = 1 + ((0.015 * L_bar_p_2) / np.sqrt(20 + L_bar_p_2))
    S_C = 1 + 0.045 * C_bar_p
    S_H = 1 + 0.015 * C_bar_p * T

    R_T = -np.sin(np.radians(2 * delta_theta)) * R_C

    d_E = np.sqrt(
        (delta_L_p / (k_L * S_L)) ** 2
        + (delta_C_p / (k_C * S_C)) ** 2
        + (delta_H_p / (k_H * S_H)) ** 2
        + R_T * (delta_C_p / (k_C * S_C)) * (delta_H_p / (k_H * S_H))
    )

    return d_E.astype(np.float32, copy=False)


class DeltaE:
    """
    Computes the DeltaE Metric (CIE 2000) using NumPy.
    It expects inputs to be in XYZ format with shape (*, H, W, 3).
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0.0
        self.total = 0

    def compute(self, preds: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Update state with predictions and targets.
        
        Parameters
        ----------
        preds : np.ndarray
            Predicted sRGB images. Shape: (*, H, W, 3)
        target : np.ndarray
            Ground truth sRGB images. Shape: (*, H, W, 3)
        """
        if preds.shape != target.shape:
            raise ValueError(f"preds and target must have the same shape. Got {preds.shape} and {target.shape}")

        if preds.ndim < 3 or preds.shape[-1] != 3:
            raise ValueError(f"Input size must have a shape of (*, H, W, 3). Got {preds.shape}")

        # 1. Convert XYZ -> Lab
        preds_lab = rgb_to_lab(preds)
        target_lab = rgb_to_lab(target)
        
        # 2. Calculate Delta E Map
        return _delta_E_CIE2000(preds_lab, target_lab)
        