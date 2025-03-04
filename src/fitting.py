from numpy.typing import NDArray
from coords import Coord3D
from line_reader import Line, dateline_fix

from scipy.interpolate import make_splprep
import numpy as np


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

    spl, _ = make_splprep([xs, ys, zs], k=6, s=1) # type: ignore
    new_points = spl(np.linspace(0, 1, 100))

    return new_points, spl.c
