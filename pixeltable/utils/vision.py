import numpy as np


def find_boundaries(mask: np.ndarray) -> np.ndarray:
    """Find boundaries using 8-connectivity."""
    assert mask.dtype == bool
    padded = np.pad(mask, 1, mode='constant', constant_values=False)
    eroded = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
        & padded[:-2, :-2]
        & padded[:-2, 2:]
        & padded[2:, :-2]
        & padded[2:, 2:]
    )
    return mask & ~eroded


def dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Binary dilation with 4-connectivity."""
    result = mask.astype(bool)
    for _ in range(iterations):
        padded = np.pad(result, 1, mode='constant', constant_values=False)
        result = padded[1:-1, 1:-1] | padded[:-2, 1:-1] | padded[2:, 1:-1] | padded[1:-1, :-2] | padded[1:-1, 2:]
    return result


def get_contours(mask: np.ndarray, thickness: int = 1) -> np.ndarray:
    """Get contour mask with specified thickness."""
    boundaries = find_boundaries(mask)
    if thickness > 1:
        boundaries = dilate(boundaries, thickness - 1)
    return boundaries
