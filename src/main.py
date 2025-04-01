"""
Program for visualizing ensembles of mta lines or
jet lines on a map.
Clusters the lines to create an aggregate visualization
of the lines.
Uses a multiscale approach on the lines in order to improve
performance of the distance checks and reduce effects of
line starting points being shifted.
"""


import argparse

from download_ens import download
from fitting import fit_lines_spline, fit_spline
from line_reader import get_all_lines
from multiscale import multiscale
from utility import Data, Settings
from visualization import get_legend_elements, plot_map, plot_3D, plot_single_line
import test_script as testing

import matplotlib.pyplot as plt
import cartopy.crs as ccrs  # type: ignore


def init() -> tuple[Settings, Data]:
    valid_timeoffsets = list(range(0, 73, 3)) + list(range(78, 241, 6))
 
    parser = argparse.ArgumentParser("MTA and Jet lines ensemble vizualizer")
    _ = parser.add_argument("--sphere", action="store_true", help="Show 3D visualization")
    _ = parser.add_argument("--ico", action="store_true", help="Show IcoPoints on map and 3D visualization")
    _ = parser.add_argument("--simstart", type=str, default="2025021100", help="Start of the simulation in the format 'YYYYMMDDHH'")
    _ = parser.add_argument("--timeoffset", type=int, default=0, choices=valid_timeoffsets, help="Time offset from the simstart")
    _ = parser.add_argument("--linetype", type=str, default="jet", choices=["jet", "mta", "both"], help="Type of line (must be 'jet', 'mta', or 'both')")
    _ = parser.add_argument("--centroids", action="store_true", help="Show centroids of lines")
    _ = parser.add_argument("--oneline", type=int, default=-1, help="If this is set only one line will be visualized. The number provided is the index of that line")
    _ = parser.add_argument("--bspline", action="store_true", help="Approximate all lines with a BSpline")

    args = parser.parse_args()
    settings = Settings(show_3D_vis=args.sphere,
                        show_ico_points=args.ico,
                        show_centroids=args.centroids,
                        sim_start=args.simstart,
                        time_offset=args.timeoffset,
                        line_type=args.linetype,
                        oneline=args.oneline,
                        bspline=args.bspline)

    if settings.line_type != "both":
        download(settings.sim_start, args.linetype)

        print("Processing lines")
        lines = get_all_lines(settings.sim_start, settings.time_offset, settings.line_type)
        ico_points_ms, line_points_ms = multiscale(lines, 4)

        if settings.bspline:
            lines = fit_lines_spline(lines)

        data = Data(lines=lines, ico_points_ms=ico_points_ms, line_points_ms=line_points_ms)
    else:
        download(settings.sim_start, "jet") 
        download(settings.sim_start, "mta") 

        print("Processing lines 1")
        lines = get_all_lines(settings.sim_start, settings.time_offset, "jet")
        ico_points_ms, line_points_ms = multiscale(lines, 4)

        print("Processing lines 2")
        lines_2 = get_all_lines(settings.sim_start, settings.time_offset, "mta")
        ico_points_ms_2, line_points_ms_2 = multiscale(lines_2, 4)

        if settings.bspline:
            lines = fit_lines_spline(lines)
            lines_2 = fit_lines_spline(lines_2)

        data = Data(lines=lines,
                    ico_points_ms=ico_points_ms,
                    line_points_ms=line_points_ms,
                    lines_2=lines_2,
                    ico_points_ms_2=ico_points_ms_2,
                    line_points_ms_2=line_points_ms_2)


    return settings, data


def main(settings: Settings, data: Data):
    fig = plt.figure(figsize=(16, 9))

    if settings.oneline != -1:
        line = data.lines[settings.oneline]

        ax1 = fig.add_subplot(111, projection="3d")

        fitted_points = fit_spline(line)
        plot_single_line(line, ax1)

        ax1.plot(fitted_points[0], fitted_points[1], fitted_points[2])  # type: ignore

        plt.tight_layout()
        plt.show()
        exit()

    if settings.show_3D_vis:
        ax1 = fig.add_subplot(121, projection=ccrs.PlateCarree())
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z') # type: ignore

        plot_3D(data, settings, ax2)
    else:
        ax1 = fig.add_subplot(111, projection=ccrs.PlateCarree())

    plot_map(data, settings, ax1)

    legend_elements = get_legend_elements(data, settings)
    fig.legend(handles=legend_elements, loc='upper right', 
               bbox_to_anchor=(0.98, 0.98),
               frameon=True, framealpha=0.9,
               fontsize=8, title='Map Elements')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    settings, data = init()
    print("Initialized")

    testing.test_double_clustering_centroids(settings, data)
    # testing.test_clustering(settings, data)
   
    # testing.test_standard_deviation(settings, data)
    # testing.test_confidence_band(settings, data)

    # main(settings, data)
