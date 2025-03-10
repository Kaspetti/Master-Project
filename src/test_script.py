from numpy.typing import NDArray
from shapely.geometry import LineString
from coords import Coord3D
from desc_stats import total_distance_from_median
from fitting import fit_bezier, fit_bezier_all, fit_spline
from line_reader import dateline_fix
from utility import Data, Settings, load_networks

import matplotlib.pyplot as plt
import cartopy.crs as ccrs  # type: ignore
import pandas as pd
import geopandas as gpd
import cartopy.feature as cfeature
import numpy as np
from kneed import KneeLocator


def test_bezier_desc_stats(settings: Settings, data: Data):
    networks = load_networks("networks.json")
    dist_threshold = 50
    dist_ratio = 0.05
    
    key = settings.sim_start + str(settings.time_offset) + str(dist_threshold) + str(dist_ratio) + settings.line_type
    node_clusters = networks[key]["node_clusters"]

    largest_cluster = int(max(networks[key]["clusters"], key=lambda k: len(networks[key]["clusters"][k])))
    line_ids = [line_id for line_id, cluster_id in node_clusters.items() if cluster_id == largest_cluster]
    lines = [line for line in data.lines if line.id in line_ids]

    splines = fit_bezier_all(lines)
    coeffs = [cs for cs, _, _ in splines.values()]
    median_coeffs = np.median(coeffs, axis=0)

    distances: dict[str, float] = {}
    for line_id, (cs, _, _) in splines.items():
        distances[line_id] = total_distance_from_median(cs, median_coeffs)

    avg_dist = sum(distances.values()) / len(distances)

    central_line = min(distances, key=lambda k: distances[k])

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="black")   # type: ignore
    ax.add_feature(cfeature.OCEAN, facecolor="lightgrey")     # type: ignore 
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor="darkgrey")    # type: ignore

    for line in lines:
        geometry = [LineString([coord.to_list() for coord in line.coords])]
        gdf = gpd.GeoDataFrame(pd.DataFrame(), geometry=geometry, crs="EPSG:4326")  # type: ignore

        if line.id == central_line:
            gdf.plot(ax=ax, transform=ccrs.PlateCarree(), color="#ff872e", zorder=2, linewidth=4)
        elif distances[line.id] > avg_dist * 2:
            gdf.plot(ax=ax, transform=ccrs.PlateCarree(), color="#ec000b", zorder=1, linewidth=2, linestyle=":")
        else:
            gdf.plot(ax=ax, transform=ccrs.PlateCarree(), color="#053a8d", zorder=0, linewidth=1)

    plt.tight_layout()
    plt.show()


def test_bezier_all(settings: Settings, data: Data):
    splines = fit_bezier_all(data.lines, True)

    fig = plt.figure(figsize=(16, 9))

    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="black")   # type: ignore
    ax.add_feature(cfeature.OCEAN, facecolor="lightgrey")     # type: ignore 
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor="darkgrey")    # type: ignore

    for _, (_, pts, _) in splines.items():
        coords = [Coord3D(pt[0], pt[1], pt[2]).to_lon_lat() for pt in pts]
        max_lon = max(coords, key=lambda c: c.lon).lon
        min_lon = min(coords, key=lambda c: c.lon).lon
        if max_lon - min_lon > 180:
            coords = dateline_fix(coords)

        geometry = [LineString([coord.to_list() for coord in coords])]
        gdf = gpd.GeoDataFrame(pd.DataFrame(), geometry=geometry, crs="EPSG:4326")  # type: ignore
        gdf.plot(ax=ax, transform=ccrs.PlateCarree(), color="#053a8d", zorder=0, linewidth=1)

    plt.tight_layout()
    plt.show()


def test_bezier_error(settings: Settings, data: Data):
    networks = load_networks("networks.json")
    dist_threshold = 50
    dist_ratio = 0.05
    
    key = settings.sim_start + str(settings.time_offset) + str(dist_threshold) + str(dist_ratio) + settings.line_type
    node_clusters = networks[key]["node_clusters"]

    largest_cluster = int(max(networks[key]["clusters"], key=lambda k: len(networks[key]["clusters"][k])))
    line_ids = [line_id for line_id, cluster_id in node_clusters.items() if cluster_id == largest_cluster]
    lines = [line for line in data.lines if line.id in line_ids]

    max_degree = 0
    for line in lines:
        errs: dict[int, float] = {}
        for i in range(2, 10):
            _, _, err = fit_bezier(line, i, True)
            errs[i] = err

        x = list(errs.keys())
        y = list(errs.values())
        kneedle = KneeLocator(x, y, S=1.0, curve="convex", direction="decreasing")
        elbow = kneedle.elbow
        if elbow == None:
            print("Couldn't find elbow")
            continue

        max_degree = max(max_degree, elbow)

    print(f"Decided on degree: {max_degree}")
    splines = [fit_bezier(line, max_degree, True) for line in lines]
    
    fig = plt.figure(figsize=(16, 9))

    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="black")   # type: ignore
    ax.add_feature(cfeature.OCEAN, facecolor="lightgrey")     # type: ignore 
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor="darkgrey")    # type: ignore

    for cs, pts, _  in splines:
        if len(pts) == 0:
            continue

        geometry = [LineString([Coord3D(pt[0], pt[1], pt[2]).to_lon_lat().to_list() for pt in pts])]
        gdf = gpd.GeoDataFrame(pd.DataFrame(), geometry=geometry, crs="EPSG:4326")  # type: ignore
        gdf.plot(ax=ax, transform=ccrs.PlateCarree(), color="#053a8d", zorder=0, linewidth=1)

    for line in lines:
        geometry = [LineString([coord.to_list() for coord in line.coords])]
        gdf = gpd.GeoDataFrame(pd.DataFrame(), geometry=geometry, crs="EPSG:4326")  # type: ignore
        gdf.plot(ax=ax, transform=ccrs.PlateCarree(), color="#ec000b", zorder=0, linewidth=1, linestyle=":")
        
    plt.tight_layout()
    plt.show()


    plt.show()



def test_bezier_single(settings: Settings, data: Data):
    networks = load_networks("networks.json")
    dist_threshold = 50
    dist_ratio = 0.05
    
    key = settings.sim_start + str(settings.time_offset) + str(dist_threshold) + str(dist_ratio) + settings.line_type
    node_clusters = networks[key]["node_clusters"]

    largest_cluster = int(max(networks[key]["clusters"], key=lambda k: len(networks[key]["clusters"][k])))
    line_ids = [line_id for line_id, cluster_id in node_clusters.items() if cluster_id == largest_cluster]
    lines = [line for line in data.lines if line.id in line_ids]

    splines_pts = [fit_bezier(line, 3, True) for line in lines]

    fig = plt.figure(figsize=(16, 9))

    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="black")   # type: ignore
    ax.add_feature(cfeature.OCEAN, facecolor="lightgrey")     # type: ignore 
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor="darkgrey")    # type: ignore

    for cs, pts, _  in splines_pts:
        if len(pts) == 0:
            continue

        geometry = [LineString([Coord3D(pt[0], pt[1], pt[2]).to_lon_lat().to_list() for pt in pts])]
        gdf = gpd.GeoDataFrame(pd.DataFrame(), geometry=geometry, crs="EPSG:4326")  # type: ignore
        gdf.plot(ax=ax, transform=ccrs.PlateCarree(), color="#053a8d", zorder=0, linewidth=1)

    for line in lines:
        geometry = [LineString([coord.to_list() for coord in line.coords])]
        gdf = gpd.GeoDataFrame(pd.DataFrame(), geometry=geometry, crs="EPSG:4326")  # type: ignore
        gdf.plot(ax=ax, transform=ccrs.PlateCarree(), color="#ec000b", zorder=0, linewidth=1, linestyle=":")
        
    plt.tight_layout()
    plt.show()


def test_bspline_all(settings: Settings, data: Data):
    networks = load_networks("networks.json")
    dist_threshold = 50
    dist_ratio = 0.05
    
    key = settings.sim_start + str(settings.time_offset) + str(dist_threshold) + str(dist_ratio) + settings.line_type
    node_clusters = networks[key]["node_clusters"]

    largest_cluster = int(max(networks[key]["clusters"], key=lambda k: len(networks[key]["clusters"][k])))
    line_ids = [line_id for line_id, cluster_id in node_clusters.items() if cluster_id == largest_cluster]
    lines = [line for line in data.lines if line.id in line_ids]

    splines: dict[str, tuple[NDArray, NDArray]] = {}
    for line in lines:
        fitted_points, coeffs = fit_spline(line) 
        splines[line.id] = (fitted_points, coeffs)

    coeffs = np.array([coeffs for (_, coeffs) in splines.values()])
    median_coeffs = np.median(coeffs, axis=0)

    distances: dict[str, float] = {}
    for line_id, (_, coeffs) in splines.items():
        distances[line_id] = total_distance_from_median(coeffs, median_coeffs)

    avg_dist = sum(distances.values()) / len(distances)

    central_line = min(distances, key=lambda k: distances[k])
    fig = plt.figure(figsize=(16, 9))

    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="black")   # type: ignore
    ax.add_feature(cfeature.OCEAN, facecolor="lightgrey")     # type: ignore 
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor="darkgrey")    # type: ignore

    for line in lines:
        geometry = [LineString([coord.to_list() for coord in line.coords])]
        gdf = gpd.GeoDataFrame(pd.DataFrame(), geometry=geometry, crs="EPSG:4326")  # type: ignore

        if line.id == central_line:
            gdf.plot(ax=ax, transform=ccrs.PlateCarree(), color="#ff872e", zorder=2, linewidth=4)
        elif distances[line.id] > avg_dist * 2:
            gdf.plot(ax=ax, transform=ccrs.PlateCarree(), color="#ec000b", zorder=1, linewidth=2, linestyle=":")
        else:
            gdf.plot(ax=ax, transform=ccrs.PlateCarree(), color="#053a8d", zorder=0, linewidth=1)

    plt.tight_layout()
    plt.show()
