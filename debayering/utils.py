import numpy as np


def as_float_array(x):
    return np.asarray(x, dtype=np.float32)


def tstack(arrays):
    return np.stack(arrays, axis=-1)


def masks_CFA_Bayer(shape, pattern="RGGB"):
    pattern = pattern.upper()
    if pattern not in {"RGGB", "BGGR", "GRBG", "GBRG"}:
        raise ValueError(f"Unsupported Bayer pattern: {pattern}")

    channels = {channel: np.zeros(shape, dtype=bool) for channel in "RGB"}
    for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        channels[channel][y::2, x::2] = True
    return channels["R"], channels["G"], channels["B"]
