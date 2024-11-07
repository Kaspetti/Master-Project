"""
Program for visualizing ensembles of mta lines or
jet lines on a map.
Clusters the lines to create an aggregate visualization
of the lines.
Uses a multiscale approach on the lines in order to improve
performance of the distance checks and reduce effects of
line starting points being shifted.
"""


from coords import Coord3D
from line_reader import get_all_lines
from multiscale import multiscale

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


if __name__ == "__main__":
    lines = get_all_lines("2024101900", 0, "jet")
    ico_points_ms, line_points_ms = multiscale(lines, 0)

    ids = [line.id for line in lines]
    df = pd.DataFrame(ids, columns=["id"])  # type: ignore
    geometry = [LineString([coord.to_list() for coord in line.coords]) for line in lines]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="black") # type: ignore
    ax.add_feature(cfeature.OCEAN, facecolor="lightgrey")   # type: ignore
    ax.add_feature(cfeature.COASTLINE, edgecolor="black")   # type: ignore
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor="darkgrey")    # type: ignore

    gdf.plot(ax=ax, transform=ccrs.PlateCarree(), linewidth=1, color="blue")

    geo_points = []
    for point in ico_points_ms.values():
        n = point.coord_3D.to_ndarray() / np.linalg.norm(point.coord_3D.to_ndarray())
        geo_points.append(Coord3D(n[0], n[1], n[2]).to_lon_lat().to_list())

    geometry = [Point(pt) for pt in geo_points]
    gdf = gpd.GeoDataFrame(pd.DataFrame(), geometry=geometry, crs="EPSG:4326")
    gdf.plot(ax=ax, transform=ccrs.PlateCarree(), color="red", zorder=100, markersize=1)

    plt.show()
