from coords import Coord3D
from line_reader import Line, dateline_fix

from scipy.interpolate import make_splprep
import numpy as np


def fit_lines_spline(lines: list[Line]) -> list[Line]:
    new_lines = [] 

    for line in lines:
        xyz = fit_spline(line)
        new_coords = [Coord3D(xyz[0][i], xyz[1][i], xyz[2][i]).to_lon_lat() for i in range(len(xyz[0]))]

        lons = [coord.lon for coord in new_coords]
        if max(lons) - min(lons) > 180:
            new_coords = dateline_fix(new_coords)
    
        new_lines.append(Line(id=line.id, coords=[coord for coord in new_coords]))

    return new_lines


def fit_spline(line: Line):
    coords_3D = [coord.to_3D() for coord in line.coords]
    xs = [coord.x for coord in coords_3D]
    ys = [coord.y for coord in coords_3D]
    zs = [coord.z for coord in coords_3D]

    spl, _ = make_splprep([xs, ys, zs], k=6, s=1) # type: ignore
    new_points = spl(np.linspace(0, 1, 100))

    return new_points


def get_ts(line: Line) -> list[float]:
    coords_3D = [coord.to_3D() for coord in line.coords]
    line_length = get_line_length(coords_3D) 

    ts = [0.]
    cur_dist = 0.
    c0 = coords_3D[0]

    for c1 in coords_3D[1:]:
        cur_dist += c0.dist(c1)
        ts.append(cur_dist / line_length)

        c0 = c1

    return ts


def get_line_length(line_coords: list[Coord3D]) -> float:
    dist = 0
    for i, c0 in enumerate(line_coords[:-1]):
        dist += c0.dist(line_coords[i+1]) 

    return dist


def bezier_point(cs: list[Coord3D], t: float) -> Coord3D:
    return cs[0]
