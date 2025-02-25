
from coords import Coord3D
from line_reader import Line
from multiscale import IcoPoint

import numpy as np
import pandas as pd
import geopandas as gpd # type: ignore
from shapely.geometry import LineString, Point
from matplotlib.axes import Axes
import cartopy.crs as ccrs  # type: ignore
import cartopy.feature as cfeature # type: ignore


def plot_map(lines: list[Line], ax: Axes, ico_points: dict[int, IcoPoint] | None = None, show_ico: bool = False, show_centroids: bool = False):
    ids = [line.id for line in lines]
    df = pd.DataFrame(ids, columns=["id"])  
    geometry = [LineString([coord.to_list() for coord in line.coords]) for line in lines]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")  # type: ignore
    gdf.plot(ax=ax, transform=ccrs.PlateCarree(), linewidth=1, color="#0000ff")

    ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="black")   # type: ignore
    ax.add_feature(cfeature.OCEAN, facecolor="lightgrey")     # type: ignore 
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor="darkgrey")    # type: ignore

    if show_ico and ico_points:
        geo_points = []
        for point in ico_points.values():
            n = point.coord_3D.to_ndarray() / np.linalg.norm(point.coord_3D.to_ndarray())
            geo_points.append(Coord3D(n[0], n[1], n[2]).to_lon_lat().to_list())

        geometry = [Point(pt) for pt in geo_points]
        gdf = gpd.GeoDataFrame(pd.DataFrame(), geometry=geometry, crs="EPSG:4326")  # type: ignore
        gdf.plot(ax=ax, transform=ccrs.PlateCarree(), color="#ff0000", zorder=100, markersize=1)
    elif show_ico:
        print("'show_ico' was set to True for 'plot_map' but 'ico_points' was not set")


    if show_centroids:
        geometry = [Point(line.get_centroid().to_list()) for line in lines]
        gdf = gpd.GeoDataFrame(pd.DataFrame(), geometry=geometry, crs="EPSG:4326")  # type: ignore
        gdf.plot(ax=ax, transform=ccrs.PlateCarree(), color="#00ff00", zorder=101, markersize=2)


def plot_3D(lines: list[Line], ax: Axes, ico_points: dict[int, IcoPoint] | None = None, show_ico: bool = False, show_centroids: bool = False):
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

    if show_ico and ico_points:
        xs = [pt.coord_3D.x for pt in ico_points.values()] 
        ys = [pt.coord_3D.y for pt in ico_points.values()] 
        zs = [pt.coord_3D.z for pt in ico_points.values()] 
        ax.scatter(xs, ys, zs, color='red')
    elif show_ico:
        print("'show_ico' was set to True for 'plot_map' but 'ico_points' was not set")

    if show_centroids:
        centroids = [line.get_centroid().to_3D() for line in lines]
        xs = [centroid.x for centroid in centroids]
        ys = [centroid.y for centroid in centroids]
        zs = [centroid.z for centroid in centroids]

        ax.scatter(xs, ys, zs, color="#00ff00")
