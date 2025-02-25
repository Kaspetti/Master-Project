
from tokenize import group
from typing import Literal

from matplotlib.lines import Line2D
from coords import Coord3D
from line_reader import Line
from multiscale import IcoPoint
from utility import Data, Settings

import numpy as np
import pandas as pd
import geopandas as gpd # type: ignore
from shapely.geometry import LineString, Point
from matplotlib.axes import Axes
import cartopy.crs as ccrs  # type: ignore
import cartopy.feature as cfeature


def plot_map(data: Data, settings: Settings, ax: Axes):
    ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="black")   # type: ignore
    ax.add_feature(cfeature.OCEAN, facecolor="lightgrey")     # type: ignore 
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor="darkgrey")    # type: ignore

    plot_lines_map(data.lines, ax)
    if data.lines_2:
        plot_lines_map(data.lines_2, ax, 1)

    if settings.show_ico_points:
        plot_ico_points_map(data.ico_points_ms, ax)
        if data.ico_points_ms_2:
            plot_ico_points_map(data.ico_points_ms_2, ax, 1)

    if settings.show_centroids:
        plot_centroids_map(data.lines, ax)
        if data.lines_2:
            plot_centroids_map(data.lines_2, ax, 1)

    add_legend_map(data, settings, ax)


def plot_3D(data: Data, settings: Settings, ax: Axes):
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = 0.99 * np.outer(np.cos(u), np.sin(v))
    y = 0.99 * np.outer(np.sin(u), np.sin(v))
    z = 0.99 * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_wireframe(x, y, z, color='gray', alpha=0.3)    # type: ignore

    for line in data.lines:
        xs = [coord.to_3D().x for coord in line.coords]
        ys = [coord.to_3D().y for coord in line.coords]
        zs = [coord.to_3D().z for coord in line.coords]
        ax.plot(xs, ys, zs, color="#053a8d")

    if settings.show_ico_points:
        xs = [pt.coord_3D.x for pt in data.ico_points_ms.values()] 
        ys = [pt.coord_3D.y for pt in data.ico_points_ms.values()] 
        zs = [pt.coord_3D.z for pt in data.ico_points_ms.values()] 
        ax.scatter(xs, ys, zs, color="#0b9dce")

    if settings.show_centroids:
        centroids = [line.get_centroid().to_3D() for line in data.lines]
        xs = [centroid.x for centroid in centroids]
        ys = [centroid.y for centroid in centroids]
        zs = [centroid.z for centroid in centroids]

        ax.scatter(xs, ys, zs, color="#ff872e")


def plot_lines_map(lines: list[Line], ax: Axes, group: Literal[0, 1] = 0):
    ids = [line.id for line in lines]
    df = pd.DataFrame(ids, columns=["id"])  
    geometry = [LineString([coord.to_list() for coord in line.coords]) for line in lines]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")  # type: ignore
    gdf.plot(ax=ax, transform=ccrs.PlateCarree(), linewidth=1, color="#053a8d" if group == 0 else "#098945")


def plot_ico_points_map(ico_points_ms: dict[int, IcoPoint], ax: Axes, group: Literal[0, 1] = 0):
    geo_points = []
    for point in ico_points_ms.values():
        n = point.coord_3D.to_ndarray() / np.linalg.norm(point.coord_3D.to_ndarray())
        geo_points.append(Coord3D(n[0], n[1], n[2]).to_lon_lat().to_list())

    geometry = [Point(pt) for pt in geo_points]
    gdf = gpd.GeoDataFrame(pd.DataFrame(), geometry=geometry, crs="EPSG:4326")  # type: ignore
    gdf.plot(ax=ax, transform=ccrs.PlateCarree(), color="#0b9dce" if group == 0 else "#83bf1c", zorder=100, markersize=1)


def plot_centroids_map(lines: list[Line], ax: Axes, group: Literal[0, 1] = 0):
    geometry = [Point(line.get_centroid().to_list()) for line in lines]
    gdf = gpd.GeoDataFrame(pd.DataFrame(), geometry=geometry, crs="EPSG:4326")  # type: ignore
    gdf.plot(ax=ax, transform=ccrs.PlateCarree(), color="#ff872e" if group == 0 else "#ffbf00", zorder=101, markersize=5)


def add_legend_map(data: Data, settings: Settings, ax: Axes):
    group_1_name = settings.line_type if settings.line_type != "both" else "jet"

    legend_elements = []
    
    legend_elements.append(Line2D([0], [0], color='#053a8d', lw=2, label=f"Lines {group_1_name}"))
    
    if settings.show_ico_points:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='#0b9dce', 
                                     markersize=5, label=f'ICO Points {group_1_name}'))
    
    if settings.show_centroids:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff872e', 
                                     markersize=8, label=f'Centroids {group_1_name}'))
    
    if data.lines_2:
        legend_elements.append(Line2D([0], [0], color='#098945', lw=2, label='Lines mta'))
        
        if settings.show_ico_points and data.ico_points_ms_2:
            legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='#83bf1c', 
                                         markersize=5, label='ICO Points mta'))
        
        if settings.show_centroids:
            legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffbf00', 
                                         markersize=8, label='Centroids mta'))
    
    ax.legend(handles=legend_elements, loc='lower left', frameon=True, framealpha=0.9,
              fontsize=8, title='Map Elements')
