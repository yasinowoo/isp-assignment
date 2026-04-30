import numpy as np
from scipy.ndimage import convolve, convolve1d

from .utils import as_float_array, masks_CFA_Bayer, tstack


def _cnv_h(x, y):
    return convolve1d(x, y, mode="mirror")


def _cnv_v(x, y):
    return convolve1d(x, y, mode="mirror", axis=0)


def bayer2rgb(CFA, pattern):
    CFA = as_float_array(CFA)
    R_m, G_m, B_m = masks_CFA_Bayer(CFA.shape, pattern)

    h_0 = as_float_array([0.0, 0.5, 0.0, 0.5, 0.0])
    h_1 = as_float_array([-0.25, 0.0, 0.5, 0.0, -0.25])

    R = CFA * R_m
    G = CFA * G_m
    B = CFA * B_m

    G_H = np.where(G_m == 0, _cnv_h(CFA, h_0) + _cnv_h(CFA, h_1), G)
    G_V = np.where(G_m == 0, _cnv_v(CFA, h_0) + _cnv_v(CFA, h_1), G)

    C_H = np.where(R_m == 1, R - G_H, 0)
    C_H = np.where(B_m == 1, B - G_H, C_H)
    C_V = np.where(R_m == 1, R - G_V, 0)
    C_V = np.where(B_m == 1, B - G_V, C_V)

    D_H = np.abs(C_H - np.pad(C_H, ((0, 0), (0, 2)), mode="reflect")[:, 2:])
    D_V = np.abs(C_V - np.pad(C_V, ((0, 2), (0, 0)), mode="reflect")[2:, :])

    k = as_float_array(
        [
            [0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 3.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0],
        ]
    )

    d_H = convolve(D_H, k, mode="constant")
    d_V = convolve(D_V, np.transpose(k), mode="constant")

    mask = d_V >= d_H
    G = np.where(mask, G_H, G_V)
    M = np.where(mask, 1, 0)

    R_r = np.transpose(np.any(R_m == 1, axis=1)[None]) * np.ones(R.shape)
    B_r = np.transpose(np.any(B_m == 1, axis=1)[None]) * np.ones(B.shape)
    k_b = as_float_array([0.5, 0, 0.5])

    R = np.where(np.logical_and(G_m == 1, R_r == 1), G + _cnv_h(R, k_b) - _cnv_h(G, k_b), R)
    R = np.where(np.logical_and(G_m == 1, B_r == 1), G + _cnv_v(R, k_b) - _cnv_v(G, k_b), R)
    B = np.where(np.logical_and(G_m == 1, B_r == 1), G + _cnv_h(B, k_b) - _cnv_h(G, k_b), B)
    B = np.where(np.logical_and(G_m == 1, R_r == 1), G + _cnv_v(B, k_b) - _cnv_v(G, k_b), B)

    R = np.where(
        np.logical_and(B_r == 1, B_m == 1),
        np.where(M == 1, B + _cnv_h(R, k_b) - _cnv_h(B, k_b), B + _cnv_v(R, k_b) - _cnv_v(B, k_b)),
        R,
    )
    B = np.where(
        np.logical_and(R_r == 1, R_m == 1),
        np.where(M == 1, R + _cnv_h(B, k_b) - _cnv_h(R, k_b), R + _cnv_v(B, k_b) - _cnv_v(R, k_b)),
        B,
    )

    return tstack([R, G, B])
