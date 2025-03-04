from numpy.typing import NDArray
import numpy as np


def mean_distance_from_median(coeffs: NDArray, median_coeffs: NDArray) -> float:
    distances = np.sqrt(np.sum((coeffs - median_coeffs) ** 2, axis=1))
    return np.mean(distances)


def total_distance_from_median(coeffs: NDArray, median_coeffs: NDArray) -> float:
    distances = np.sqrt(np.sum((coeffs - median_coeffs) ** 2, axis=1))
    return np.sum(distances)
