"""
Program for visualizing ensembles of mta lines or
jet lines on a map.
Clusters the lines to create an aggregate visualization
of the lines.
Uses a multiscale approach on the lines in order to improve
performance of the distance checks and reduce effects of
line starting points being shifted.
"""


from dataclasses import dataclass
import argparse
from typing import Literal

from line_reader import Line, get_all_lines
from multiscale import IcoPoint, multiscale
from visualization import plot_map, plot_3D

import matplotlib.pyplot as plt
import cartopy.crs as ccrs  # type: ignore


@dataclass
class Settings:
    show_ico_points: bool
    show_3D_vis: bool
    show_centroids: bool

    sim_start: str
    time_offset: int
    line_type: Literal["jet", "mta"]


@dataclass
class Data:
    lines: list[Line]
    ico_points_ms: dict[int, IcoPoint]
    line_points_ms: dict[str, dict[int, dict[int, tuple[int, float]]]]


def init() -> tuple[Settings, Data]:
    valid_timeoffsets = list(range(0, 73, 3)) + list(range(78, 241, 6))
 
    parser = argparse.ArgumentParser("MTA and Jet lines ensemble vizualizer")
    parser.add_argument("--sphere", action="store_true", help="Show 3D visualization")
    parser.add_argument("--ico", action="store_true", help="Show IcoPoints on map and 3D visualization")
    parser.add_argument("--simstart", type=str, default="2025021100", help="Start of the simulation in the format 'YYYYMMDDHH'")
    parser.add_argument("--timeoffset", type=int, default=0, choices=valid_timeoffsets, help="Time offset from the simstart")
    parser.add_argument("--linetype", type=str, default="jet", choices=["jet", "mta"], help="Type of line (must be 'jet' or 'mta')")
    parser.add_argument("--centroids", action="store_true", help="Show centroids of lines")

    args = parser.parse_args()
    settings = Settings(show_3D_vis=args.sphere,
                        show_ico_points=args.ico,
                        show_centroids=args.centroids,
                        sim_start=args.simstart,
                        time_offset=args.timeoffset,
                        line_type=args.linetype)

    lines = get_all_lines(settings.sim_start, settings.time_offset, settings.line_type)
    ico_points_ms, line_points_ms = multiscale(lines, 4)
    data = Data(lines=lines, ico_points_ms=ico_points_ms, line_points_ms=line_points_ms)

    return settings, data


if __name__ == "__main__":
    settings, data = init()

    fig = plt.figure(figsize=(16, 9))

    if settings.show_3D_vis:
        ax1 = fig.add_subplot(121, projection=ccrs.PlateCarree())
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z') # type: ignore

        plot_3D(data.lines, ax2, data.ico_points_ms, settings.show_ico_points, settings.show_centroids)
    else:
        ax1 = fig.add_subplot(111, projection=ccrs.PlateCarree())

    plot_map(data.lines, ax1, data.ico_points_ms, settings.show_ico_points, settings.show_centroids)

    plt.tight_layout()
    plt.show()
