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

from coords import Coord3D
from line_reader import Line, get_all_lines
from multiscale import IcoPoint, multiscale

import numpy as np
import pandas as pd
import geopandas as gpd # type: ignore
from shapely.geometry import LineString, Point
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import cartopy.crs as ccrs  # type: ignore
import cartopy.feature as cfeature # type: ignore


@dataclass
class Settings:
    show_ico_points: bool
    show_3D_vis: bool


@dataclass
class Data:
    lines: list[Line]
    ico_points_ms: dict[int, IcoPoint]
    line_points_ms: dict[str, dict[int, dict[int, tuple[int, float]]]]


def plot_map(lines: list[Line], ax: Axes, ico_points: dict[int, IcoPoint] | None = None, show_ico: bool = False):
    ids = [line.id for line in lines]
    df = pd.DataFrame(ids, columns=["id"])  
    geometry = [LineString([coord.to_list() for coord in line.coords]) for line in lines]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")  # type: ignore
    gdf.plot(ax=ax, transform=ccrs.PlateCarree(), linewidth=1, color="#0000ff")

    ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="black")   # type: ignore
    ax.add_feature(cfeature.OCEAN, facecolor="lightgrey")     # type: ignore 
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor="darkgrey")    # type: ignore

    if show_ico:
        if not ico_points:
            print("'show_ico' was set to True for 'plot_map' but 'ico_points' was not set")
            return

        geo_points = []
        for point in ico_points.values():
            n = point.coord_3D.to_ndarray() / np.linalg.norm(point.coord_3D.to_ndarray())
            geo_points.append(Coord3D(n[0], n[1], n[2]).to_lon_lat().to_list())

        geometry = [Point(pt) for pt in geo_points]
        gdf = gpd.GeoDataFrame(pd.DataFrame(), geometry=geometry, crs="EPSG:4326")  # type: ignore
        gdf.plot(ax=ax, transform=ccrs.PlateCarree(), color="#ff0000", zorder=100, markersize=1)


def plot_3D(lines: list[Line], ax: Axes, ico_points: dict[int, IcoPoint] | None = None, show_ico: bool = False):
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = 0.99 * np.outer(np.cos(u), np.sin(v))
    y = 0.99 * np.outer(np.sin(u), np.sin(v))
    z = 0.99 * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_wireframe(x, y, z, color='gray', alpha=0.3)    # type: ignore

    for line in lines:
        xs = [coord.to_3D().x for coord in line.coords]
        ys = [coord.to_3D().y for coord in line.coords]
        zs = [coord.to_3D().z for coord in line.coords]
        ax.plot(xs, ys, zs, color="blue")

    if show_ico:
        if not ico_points:
            print("'show_ico' was set to True for 'plot_map' but 'ico_points' was not set")
            return

        xs = [pt.coord_3D.x for pt in ico_points.values()] 
        ys = [pt.coord_3D.y for pt in ico_points.values()] 
        zs = [pt.coord_3D.z for pt in ico_points.values()] 
        ax.scatter(xs, ys, zs, color='red')


def init() -> tuple[Settings, Data]:
    parser = argparse.ArgumentParser("MTA and Jet lines ensemble vizualizer")
    parser.add_argument("--sphere", action="store_true", help="Show 3D visualization")
    parser.add_argument("--ico", action="store_true", help="Show IcoPoints on map and 3D visualization")

    args = parser.parse_args()
    settings = Settings(show_3D_vis=args.sphere, show_ico_points=args.ico)

    lines = get_all_lines("2025021100", 72, "jet")
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

        plot_3D(data.lines, ax2, data.ico_points_ms, settings.show_ico_points)
    else:
        ax1 = fig.add_subplot(111, projection=ccrs.PlateCarree())

    plot_map(data.lines, ax1, data.ico_points_ms, settings.show_ico_points)

    plt.tight_layout()
    plt.show()
