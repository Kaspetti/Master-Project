from numpy.typing import NDArray
import numpy as np

from fitting import bernstein_polynomial


def mean_distance_from_median(coeffs: NDArray[np.float_], median_coeffs: NDArray[np.float_]) -> float:
    distances: NDArray[np.float_] = np.sqrt(np.sum((coeffs - median_coeffs) ** 2, axis=1))
    return np.mean(distances)   # type: ignore


def total_distance_from_median(coeffs: NDArray[np.float_], median_coeffs: NDArray[np.float_]) -> float:
    distances = np.sqrt(np.sum((coeffs - median_coeffs) ** 2, axis=1))
    return np.sum(distances)


def standard_deviation(points: NDArray[np.float_]) -> tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]]:
    centroid = np.mean(points, axis=0)
    squared_deviations = (points - centroid)**2
    variance = np.mean(squared_deviations, axis=0)

    return np.sqrt(variance), variance, centroid


def spline_sd(degree: int, num_pts: int, variances: NDArray[np.float_]):
    ts = np.linspace(0, 1, num_pts)

    vars_ts = []
    for t in ts:
        var_t = 0
        for j in range(degree+1):
            weight = bernstein_polynomial(t, j, degree)
            weight_sqrd = weight**2
            var_tj = weight_sqrd * variances[j]

            var_t += var_tj

        vars_ts.append(var_t)

    return np.sqrt(vars_ts)


def detect_outlier_splines(coeffs: NDArray[np.float_]):
    outlier_indices = set()
    
    for i in range(coeffs.shape[1]):
        control_points = coeffs[:, i]
        
        q1 = np.percentile(control_points, 25, axis=0)
        q3 = np.percentile(control_points, 75, axis=0)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        for j, point in enumerate(control_points):
            if np.any(point < lower_bound) or np.any(point > upper_bound):
                outlier_indices.add(j)
    
    return list(outlier_indices)
