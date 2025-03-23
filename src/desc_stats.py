from numpy.typing import NDArray
import numpy as np


def mean_distance_from_median(coeffs: NDArray[np.float_], median_coeffs: NDArray[np.float_]) -> float:
    distances: NDArray[np.float_] = np.sqrt(np.sum((coeffs - median_coeffs) ** 2, axis=1))
    return np.mean(distances)   # type: ignore


def total_distance_from_median(coeffs: NDArray[np.float_], median_coeffs: NDArray[np.float_]) -> float:
    distances = np.sqrt(np.sum((coeffs - median_coeffs) ** 2, axis=1))
    return np.sum(distances)


def standard_deviation(points: NDArray[np.float_]) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
    centroid = np.mean(points, axis=0)
    squared_deviations = (points - centroid)**2
    variance = np.mean(squared_deviations, axis=0)

    return np.sqrt(variance), centroid
