import numpy as np
from pathlib import Path
import einops

project_folder = Path(__file__).parent.parent.resolve()
data_folder = project_folder / 'data'

def load_sensitivity(name):
    return np.load(data_folder / 'sensitivity' / f'{name}.npy')


def load_image(idx: int) -> dict:
    return np.load(data_folder / 'hsi' / f'{idx}.npy', allow_pickle=True).item()


def before_debayering(hsi: np.ndarray, sensitivity: np.ndarray) -> np.ndarray:
    # TODO: Implement 
    result = ...
    return np.clip(result, 0, 1)

def after_debayering(hsi: np.ndarray, sensitivity: np.ndarray) -> np.ndarray:
    # TODO: Implement 
    result = ...
    return np.clip(result, 0, 1)
