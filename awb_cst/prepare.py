import numpy as np
from pathlib import Path

this_folder = Path(__file__).parent.resolve()
data_folder = this_folder.parent / 'data'


def read_illuminants() -> dict[str, np.ndarray]:
    return np.load(data_folder / 'illuminants.npy', allow_pickle=True).item()

def hsi_to_raw(hsi: np.ndarray, sensitivity: np.ndarray) -> np.ndarray:
    return hsi @ sensitivity

def augment_whitepoints(hsi: np.ndarray, whitepatch_mask: np.ndarray, 
    new_whitepoint: np.ndarray) -> np.ndarray:
    # TODO: Implement 
    return hsi / spectra_whitepoint * new_whitepoint
