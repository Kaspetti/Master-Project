import math
from coords import Coord3D
from line_reader import Line, dateline_fix

from numpy.typing import NDArray
from scipy.interpolate import make_splprep
import numpy as np
from kneed import KneeLocator


def fit_lines_spline(lines: list[Line]) -> list[Line]:
    new_lines = [] 

    for line in lines:
        xyz, _ = fit_spline(line)
        new_coords = [Coord3D(xyz[0][i], xyz[1][i], xyz[2][i]).to_lon_lat() for i in range(len(xyz[0]))]

        lons = [coord.lon for coord in new_coords]
        if max(lons) - min(lons) > 180:
            new_coords = dateline_fix(new_coords)
    
        new_lines.append(Line(id=line.id, coords=[coord for coord in new_coords]))

    return new_lines


def fit_spline(line: Line) -> tuple[NDArray, NDArray]:
    coords_3D = [coord.to_3D() for coord in line.coords]
    xs = [coord.x for coord in coords_3D]
    ys = [coord.y for coord in coords_3D]
    zs = [coord.z for coord in coords_3D]

    spl, _ = make_splprep([xs, ys, zs], k=5, s=1) # type: ignore
    new_points = spl(np.linspace(0, 1, 100))

    return new_points, spl.c


def fit_bezier_all(lines: list[Line], get_points: bool = False) -> dict[str, tuple[NDArray, NDArray, float]]:
    max_degree = 0
    for line in lines:
        errs: dict[int, float] = {}
        for i in range(2, 10):
            _, _, err = fit_bezier(line, i, False)
            errs[i] = err

        x = list(errs.keys())
        y = list(errs.values())
        kneedle = KneeLocator(x, y, S=1.0, curve="convex", direction="decreasing")
        elbow = kneedle.elbow
        if elbow == None:
            print("Couldn't find elbow")
            continue

        max_degree = max(max_degree, elbow)

    print(f"Fit bezier splines of degree: {max_degree}")
    return {line.id: fit_bezier(line, max_degree, get_points) for line in lines}


def fit_bezier(line: Line, degree: int, get_points: bool = False) -> tuple[NDArray, NDArray, float]:
    A = get_bezier_matrix(line, degree)
    xs = np.array([coord.to_3D().x for coord in line.coords])
    ys = np.array([coord.to_3D().y for coord in line.coords])
    zs = np.array([coord.to_3D().z for coord in line.coords])

    cx = np.linalg.lstsq(A, xs, rcond=None)[0]
    cy = np.linalg.lstsq(A, ys, rcond=None)[0]
    cz = np.linalg.lstsq(A, zs, rcond=None)[0]

    cs = np.column_stack((cx, cy, cz))

    error = get_error(np.column_stack((xs, ys, zs)), np.dot(A, cs))

    if not get_points:
        return cs, np.array(0), error

    interval = np.linspace(0, 1, 100)
    E = np.zeros((len(interval), degree+1))
    for i, t in enumerate(interval):
        for j in range(degree+1):
            E[i, j] = bernstein_polynomial(t, j, degree)

    curve_points = np.dot(E, cs)
    return cs, curve_points, error


def get_ts(line: Line) -> list[float]:
    total_length = 0
    dists = []
    for i in range(len(line.coords)-1):
        l0 = line.coords[i]
        l1 = line.coords[i+1]
        dist = l0.to_3D().dist(l1.to_3D())

        dists.append(dist)
        total_length += dist

    ts: list[float] = []
    cur_dist = 0

    for dist in dists:
        ts.append(cur_dist / total_length)
        cur_dist += dist

    ts.append(cur_dist / total_length)

    return ts


def bernstein_polynomial(t: float, v: int, n: int) -> float:
    nv = math.factorial(n) / (math.factorial(v) * math.factorial(n - v))
    return nv * math.pow(t, v) * math.pow(1-t, n-v)


def get_bezier_matrix(line: Line, n: int) -> NDArray:
    ts = get_ts(line)

    A = np.zeros(shape=(len(ts), n+1))

    for i in range(len(ts)):
        for j in range(n+1):
            A[i, j] = bernstein_polynomial(ts[i], j, n)

    return A


def get_error(real_values: NDArray, approximation: NDArray) -> float:
    squared_diff = np.sum((real_values - approximation)**2, axis=1)
    return np.sum(squared_diff)
