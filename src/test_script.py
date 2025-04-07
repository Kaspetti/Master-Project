from coords import Coord3D
from desc_stats import detect_outlier_splines, standard_deviation, total_distance_from_median
from fitting import evaluate_bezier, fit_bezier, fit_bezier_all, fit_spline
from line_reader import dateline_fix
from utility import Data, Settings, load_networks
import cluster

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import cartopy.crs as ccrs  # type: ignore
import pandas as pd
import geopandas as gpd
import cartopy.feature as cfeature
import numpy as np
from kneed import KneeLocator
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from shapely.geometry import LineString
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import colorcet as cc


def test_clustering_confidence_band(settings: Settings, data: Data):
    splines = fit_bezier_all(data.lines)
    coeffs = np.array([cs for cs, _, _ in splines.values()])
    centroids = np.array([np.sum(cs, axis=0) / len(cs) for cs in coeffs])

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="black")   # type: ignore
    ax.add_feature(cfeature.OCEAN, facecolor="lightgrey")     # type: ignore 
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor="darkgrey")    # type: ignore
    
    colors = cc.b_glasbey_bw

    current_ens = 0
    max_current_ens = 0
    min_max_ens = 100
    for line in data.lines:
        ens_line_id = line.id.split("|")
        if int(ens_line_id[0]) == current_ens:
            max_current_ens = max(max_current_ens, int(ens_line_id[1]))
        else:
            min_max_ens = min(min_max_ens, max_current_ens)
            max_current_ens = int(ens_line_id[1])
            current_ens = int(ens_line_id[0])

    min_k = min_max_ens
    max_k = 25

    outer_clusters = cluster.cluster_centroids(centroids, min_k, max_k)
    # selected_cluster = 8
    selected_cluster = 2

    outer_cluster_lines = [line for i, line in enumerate(data.lines) if outer_clusters[i] == selected_cluster]    # type: ignore

    cluster_splines = [splines[line.id] for line in outer_cluster_lines]
    cluster_coeffs = np.array([cs for cs, _, _ in cluster_splines])
    cluster_centroids = np.array([np.sum(cs, axis=0) / len(cs) for cs in cluster_coeffs])

    min_k = 1
    max_k = 10

    inner_clusters = cluster.cluster_centroids(cluster_centroids, min_k, max_k) 
    for c in set(inner_clusters):   # type: ignore
        focus_lines = [line for i, line in enumerate(outer_cluster_lines) if inner_clusters[i] == c]    # type: ignore
        focus_splines = [splines[line.id] for line in focus_lines]
        focus_coeffs = np.array([cs for cs, _, _ in focus_splines])
        
        outlier_indices = detect_outlier_splines(focus_coeffs)
        clean_coeffs = np.array([cs for i, cs in enumerate(focus_coeffs) if i not in outlier_indices])

        coeffs_i = [clean_coeffs[:, i] for i in range(len(clean_coeffs[0]))]
        svcs = [standard_deviation(cs) for cs in coeffs_i]

        centroids = np.array([svc[2] for svc in svcs])
        degree = len(centroids)-1


        for i, line in enumerate(focus_lines):
            geometry = [LineString([coord.to_list() for coord in line.coords])]
            gdf = gpd.GeoDataFrame(pd.DataFrame(), geometry=geometry, crs="EPSG:4326")  # type: ignore

            if i in outlier_indices:
                gdf.plot(ax=ax, transform=ccrs.PlateCarree(), color="black", zorder=0, linewidth=1, linestyle=":")
            else:
                gdf.plot(ax=ax, transform=ccrs.PlateCarree(), color=colors[c]+"55", zorder=1, linewidth=1)


        spline_points = evaluate_bezier(degree, centroids, 100)
        spline_points_geo = [Coord3D(pt[0], pt[1], pt[2]).to_lon_lat().to_list() for pt in spline_points]
        geometry = [LineString(spline_points_geo)]
        gdf = gpd.GeoDataFrame(pd.DataFrame(), geometry=geometry, crs="EPSG:4326")  # type: ignore
        gdf.plot(ax=ax, transform=ccrs.PlateCarree(), color=colors[c], zorder=100, linewidth=4)

        all_curve_points = []
        for i, line in enumerate(focus_lines):
            if i in outlier_indices:
                continue
            cs, _, _ = splines[line.id]
            curve_points = evaluate_bezier(degree, cs, 100)
            all_curve_points.append(curve_points)
        
        all_curve_points = np.array(all_curve_points)
        
        sds_normal = np.zeros(100)
        for i in range(100):
            points_at_i = all_curve_points[:, i, :]
            mean_point = spline_points[i]
            
            distances = np.sqrt(np.sum((points_at_i - mean_point) ** 2, axis=1))
            
            sds_normal[i] = np.std(distances)
        
        confidence_scale = 1.5
        
        normals = np.zeros((100, 3))
        for i in range(1, 99):
            tangent = spline_points[i+1] - spline_points[i-1]
            tangent = tangent / np.linalg.norm(tangent)
            
            radial = spline_points[i] / np.linalg.norm(spline_points[i])
            
            tangent = tangent - np.dot(tangent, radial) * radial
            tangent = tangent / np.linalg.norm(tangent)
            
            normal = np.cross(tangent, radial)
            normal = normal / np.linalg.norm(normal)
            
            normals[i] = normal
        
        normals[0] = normals[1]
        normals[99] = normals[98]
        
        upper_band = np.zeros_like(spline_points)
        lower_band = np.zeros_like(spline_points)
        
        for i in range(100):
            band_width = sds_normal[i] * confidence_scale
            upper_band[i] = spline_points[i] + normals[i] * band_width
            lower_band[i] = spline_points[i] - normals[i] * band_width
        
        upper_band_geo = [Coord3D(pt[0], pt[1], pt[2]).to_lon_lat().to_list() for pt in upper_band]
        lower_band_geo = [Coord3D(pt[0], pt[1], pt[2]).to_lon_lat().to_list() for pt in lower_band]

        upper_x = [point[0] for point in upper_band_geo]
        upper_y = [point[1] for point in upper_band_geo]
        lower_x = [point[0] for point in lower_band_geo]
        lower_y = [point[1] for point in lower_band_geo]
        
        x_fill = np.concatenate([lower_x, upper_x[::-1]])
        y_fill = np.concatenate([lower_y, upper_y[::-1]])
        
        ax.fill(x_fill, y_fill, color=to_rgba(colors[c], 0.2), zorder=1)

        geometry = [LineString(upper_band_geo), LineString(lower_band_geo)]
        gdf = gpd.GeoDataFrame(pd.DataFrame(), geometry=geometry, crs="EPSG:4326")  # type: ignore
        gdf.plot(ax=ax, transform=ccrs.PlateCarree(), color=colors[c], zorder=2, linewidth=2)

    plt.tight_layout()
    plt.show()


def test_double_clustering_centroids(settings: Settings, data: Data):
    splines = fit_bezier_all(data.lines)
    coeffs = np.array([cs for cs, _, _ in splines.values()])
    centroids = np.array([np.sum(cs, axis=0) / len(cs) for cs in coeffs])

    min_k = 5
    max_k = 25
    kneedle_sensitivity = 0.1

    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(121, projection=ccrs.PlateCarree())
    ax1.add_feature(cfeature.LAND, facecolor="white", edgecolor="black")   # type: ignore
    ax1.add_feature(cfeature.OCEAN, facecolor="lightgrey")     # type: ignore 
    ax1.add_feature(cfeature.BORDERS, linestyle=':', edgecolor="darkgrey")    # type: ignore

    labels = cluster.cluster_centroids(centroids, min_k, max_k, kneedle_sensitivity)

    geometry = []
    for line in data.lines:
        geometry.append(LineString([coord.to_list() for coord in line.coords]))
        
    gdf = gpd.GeoDataFrame(pd.DataFrame(), geometry=geometry, crs="EPSG:4326")  # type: ignore
    gdf["clusters"] = labels
    gdf.plot(ax=ax1, transform=ccrs.PlateCarree(), column="clusters", categorical=True)

    # Test double clustering
    ax2 = fig.add_subplot(122, projection=ccrs.PlateCarree())
    ax2.add_feature(cfeature.LAND, facecolor="white", edgecolor="black")   # type: ignore
    ax2.add_feature(cfeature.OCEAN, facecolor="lightgrey")     # type: ignore 
    ax2.add_feature(cfeature.BORDERS, linestyle=':', edgecolor="darkgrey")    # type: ignore

    selected_cluster = 0
    cluster_lines = [line for i, line in enumerate(data.lines) if labels[i] == selected_cluster]    # type: ignore

    cluster_splines = [splines[line.id] for line in cluster_lines]
    cluster_coeffs = np.array([cs for cs, _, _ in cluster_splines])
    cluster_centroids = np.array([np.sum(cs, axis=0) / len(cs) for cs in cluster_coeffs])

    min_k = 1
    max_k = 10

    labels = cluster.cluster_centroids(cluster_centroids, min_k, max_k, kneedle_sensitivity) 

    geometry = []
    for line in cluster_lines:
        geometry.append(LineString([coord.to_list() for coord in line.coords]))
        
    gdf = gpd.GeoDataFrame(pd.DataFrame(), geometry=geometry, crs="EPSG:4326")  # type: ignore
    gdf["clusters"] = labels
    gdf.plot(ax=ax2, transform=ccrs.PlateCarree(), column="clusters", categorical=True)

    plt.tight_layout()
    plt.show()


def test_clustering(settings: Settings, data: Data):
    splines = fit_bezier_all(data.lines)
    coeffs = np.array([cs for cs, _, _ in splines.values()])

    centroids = np.array([np.sum(cs, axis=0) / len(cs) for cs in coeffs])

    min_k = 5
    max_k = 25
    kneedle_sensitivity = 0.1

    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(121, projection=ccrs.PlateCarree())
    ax1.add_feature(cfeature.LAND, facecolor="white", edgecolor="black")   # type: ignore
    ax1.add_feature(cfeature.OCEAN, facecolor="lightgrey")     # type: ignore 
    ax1.add_feature(cfeature.BORDERS, linestyle=':', edgecolor="darkgrey")    # type: ignore

    inertias_centroids: list[float]= [KMeans(n_clusters=k, random_state=0, n_init="auto").fit(centroids).inertia_ for k in range(min_k, max_k)]   # type: ignore
    kneedle_centroids = KneeLocator(range(min_k, max_k), inertias_centroids, S=kneedle_sensitivity, curve="convex", direction="decreasing")

    kmeans_centroids = KMeans(n_clusters=kneedle_centroids.elbow, random_state=0, n_init="auto").fit(centroids) # type: ignore

    geometry = []
    for i, line in enumerate(data.lines):
        geometry.append(LineString([coord.to_list() for coord in line.coords]))
        
    gdf = gpd.GeoDataFrame(pd.DataFrame(), geometry=geometry, crs="EPSG:4326")  # type: ignore
    gdf["clusters"] = kmeans_centroids.labels_
    gdf.plot(ax=ax1, transform=ccrs.PlateCarree(), column="clusters", categorical=True)
    

    ax2 = fig.add_subplot(122, projection=ccrs.PlateCarree())
    ax2.add_feature(cfeature.LAND, facecolor="white", edgecolor="black")   # type: ignore
    ax2.add_feature(cfeature.OCEAN, facecolor="lightgrey")     # type: ignore 
    ax2.add_feature(cfeature.BORDERS, linestyle=':', edgecolor="darkgrey")    # type: ignore

    # 3 components have been shown in testing to contain the most information
    coeffs_concat = coeffs.reshape(coeffs.shape[0], -1)
    pca = PCA(n_components=3).fit(coeffs_concat)
    coeffs_3d = pca.transform(coeffs_concat)
    inertias_pca: list[float]= [KMeans(n_clusters=k, random_state=0, n_init="auto").fit(coeffs_3d).inertia_ for k in range(min_k, max_k)]   # type: ignore
    kneedle_pca = KneeLocator(range(min_k, max_k), inertias_pca, S=kneedle_sensitivity, curve="convex", direction="decreasing")

    kmeans_pca = KMeans(n_clusters=kneedle_pca.elbow, random_state=0, n_init="auto").fit(coeffs_3d) # type: ignore

    geometry = []
    for i, line in enumerate(data.lines):
        geometry.append(LineString([coord.to_list() for coord in line.coords]))
        
    gdf = gpd.GeoDataFrame(pd.DataFrame(), geometry=geometry, crs="EPSG:4326")  # type: ignore
    gdf["clusters"] = kmeans_pca.labels_
    gdf.plot(ax=ax2, transform=ccrs.PlateCarree(), column="clusters", categorical=True)

    plt.tight_layout()
    plt.show()


def test_confidence_band(settings: Settings, data: Data):
    networks = load_networks("networks.json")
    dist_threshold = 50
    dist_ratio = 0.05
    
    key = settings.sim_start + str(settings.time_offset) + str(dist_threshold) + str(dist_ratio) + settings.line_type
    node_clusters = networks[key]["node_clusters"]

    largest_cluster = int(max(networks[key]["clusters"], key=lambda k: len(networks[key]["clusters"][k])))
    line_ids = [line_id for line_id, cluster_id in node_clusters.items() if cluster_id == largest_cluster]
    lines = [line for line in data.lines if line.id in line_ids]

    splines = fit_bezier_all(lines)
    coeffs = np.array([cs for cs, _, _ in splines.values()])

    outlier_indices = detect_outlier_splines(coeffs)
    clean_coeffs = np.array([cs for i, cs in enumerate(coeffs) if i not in outlier_indices])

    coeffs_i = [clean_coeffs[:, i] for i in range(len(clean_coeffs[0]))]
    svcs = [standard_deviation(cs) for cs in coeffs_i]

    centroids = np.array([svc[2] for svc in svcs])
    degree = len(centroids)-1

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="black")   # type: ignore
    ax.add_feature(cfeature.OCEAN, facecolor="lightgrey")     # type: ignore 
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor="darkgrey")    # type: ignore

    for i, line in enumerate(lines):
        geometry = [LineString([coord.to_list() for coord in line.coords])]
        gdf = gpd.GeoDataFrame(pd.DataFrame(), geometry=geometry, crs="EPSG:4326")  # type: ignore

        if i in outlier_indices:
            gdf.plot(ax=ax, transform=ccrs.PlateCarree(), color="red", zorder=0, linewidth=1, linestyle=":")
        # else:
        #     gdf.plot(ax=ax, transform=ccrs.PlateCarree(), color="#0000ff", zorder=0, linewidth=1)


    spline_points = evaluate_bezier(degree, centroids, 100)
    spline_points_geo = [Coord3D(pt[0], pt[1], pt[2]).to_lon_lat().to_list() for pt in spline_points]
    geometry = [LineString(spline_points_geo)]
    gdf = gpd.GeoDataFrame(pd.DataFrame(), geometry=geometry, crs="EPSG:4326")  # type: ignore
    gdf.plot(ax=ax, transform=ccrs.PlateCarree(), color="orange", zorder=100, linewidth=4)

    all_curve_points = []
    for i, line in enumerate(lines):
        if i in outlier_indices:
            continue
        cs, _, _ = splines[line.id]
        curve_points = evaluate_bezier(degree, cs, 100)
        all_curve_points.append(curve_points)
    
    all_curve_points = np.array(all_curve_points)
    
    sds_normal = np.zeros(100)
    for i in range(100):
        points_at_i = all_curve_points[:, i, :]
        mean_point = spline_points[i]
        
        distances = np.sqrt(np.sum((points_at_i - mean_point) ** 2, axis=1))
        
        sds_normal[i] = np.std(distances)
    
    confidence_scale = 1.5
    
    normals = np.zeros((100, 3))
    for i in range(1, 99):
        tangent = spline_points[i+1] - spline_points[i-1]
        tangent = tangent / np.linalg.norm(tangent)
        
        radial = spline_points[i] / np.linalg.norm(spline_points[i])
        
        tangent = tangent - np.dot(tangent, radial) * radial
        tangent = tangent / np.linalg.norm(tangent)
        
        normal = np.cross(tangent, radial)
        normal = normal / np.linalg.norm(normal)
        
        normals[i] = normal
    
    normals[0] = normals[1]
    normals[99] = normals[98]
    
    upper_band = np.zeros_like(spline_points)
    lower_band = np.zeros_like(spline_points)
    
    for i in range(100):
        band_width = sds_normal[i] * confidence_scale
        upper_band[i] = spline_points[i] + normals[i] * band_width
        lower_band[i] = spline_points[i] - normals[i] * band_width
    
    upper_band_geo = [Coord3D(pt[0], pt[1], pt[2]).to_lon_lat().to_list() for pt in upper_band]
    lower_band_geo = [Coord3D(pt[0], pt[1], pt[2]).to_lon_lat().to_list() for pt in lower_band]

    upper_x = [point[0] for point in upper_band_geo]
    upper_y = [point[1] for point in upper_band_geo]
    lower_x = [point[0] for point in lower_band_geo]
    lower_y = [point[1] for point in lower_band_geo]
    
    x_fill = np.concatenate([lower_x, upper_x[::-1]])
    y_fill = np.concatenate([lower_y, upper_y[::-1]])
    
    ax.fill(x_fill, y_fill, color=to_rgba('purple', 0.2), zorder=1)

    geometry = [LineString(upper_band_geo), LineString(lower_band_geo)]
    gdf = gpd.GeoDataFrame(pd.DataFrame(), geometry=geometry, crs="EPSG:4326")  # type: ignore
    gdf.plot(ax=ax, transform=ccrs.PlateCarree(), color="purple", zorder=2, linewidth=2)

    plt.tight_layout()
    plt.show()


def test_standard_deviation(settings: Settings, data: Data):
    networks = load_networks("networks.json")
    dist_threshold = 50
    dist_ratio = 0.05
    
    key = settings.sim_start + str(settings.time_offset) + str(dist_threshold) + str(dist_ratio) + settings.line_type
    node_clusters = networks[key]["node_clusters"]

    largest_cluster = int(max(networks[key]["clusters"], key=lambda k: len(networks[key]["clusters"][k])))
    line_ids = [line_id for line_id, cluster_id in node_clusters.items() if cluster_id == largest_cluster]
    lines = [line for line in data.lines if line.id in line_ids]

    splines = fit_bezier_all(lines)

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(projection="3d")
    colors = ["#053a8d", "#0b9dce", "#098945", "#83bf1c", "#ec000b", "#ff872e", "#7b45b5", "#d883fc", "#a45700", "#ffbf00"]

    for line in lines:
        coords_3d = np.array([coord.to_3D().to_list() for coord in line.coords])
        plt.plot(coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2], c="blue")

    for i in range(len(splines[list(splines.keys())[0]][0])):
        coeffs = np.array([cs[i] for cs, _, _ in splines.values()])
        sd, _, centroid = standard_deviation(coeffs)

        ax.scatter(coeffs[:, 0], coeffs[:, 1], coeffs[:, 2], c=colors[i], alpha=0.5)

        u = np.linspace(0, 2 * np.pi, 10)
        v = np.linspace(0, np.pi, 10)
        x = sd[0] * np.outer(np.cos(u), np.sin(v)) + centroid[0]
        y = sd[1] * np.outer(np.sin(u), np.sin(v)) + centroid[1]
        z = sd[2] * np.outer(np.ones(np.size(u)), np.cos(v)) + centroid[2]

        ax.plot_surface(x, y, z, alpha=0.25, color=colors[i])   # type: ignore



    plt.tight_layout()
    plt.show()



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

    legend_elements = []
    legend_elements.append(Line2D([0], [0], color='#ff872e', lw=4, label="Central Line"))
    legend_elements.append(Line2D([0], [0], color='#ec000b', lw=2, ls=":", label="Outliers"))
    legend_elements.append(Line2D([0], [0], color='#053a8d', lw=1, label="Jet Lines"))

    ax.legend(handles=legend_elements)

    plt.title(label="Jet Lines with center and outliers (calculated using Bezier control points)")
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

    degree = len(splines[list(splines.keys())[0]][0]) - 1
    plt.title(label=f"Bezier Curve representation of Jet Lines (degree: {degree})")
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

    for line in lines:
        errs: dict[int, float] = {}
        for i in range(3, 11):
            _, _, err = fit_bezier(line, i, True)
            errs[i] = err

        x = list(errs.keys())
        y = list(errs.values())
        kneedle = KneeLocator(x, y, S=1.0, curve="convex", direction="decreasing")
        elbow = kneedle.elbow
        if elbow == None:
            print("Couldn't find elbow")
            continue

        plt.plot(x, y)

    plt.title(label="Errors of all Bezier Curves from 3rd to 10th degree")
    plt.tight_layout()
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
        
    legend_elements = []
    legend_elements.append(Line2D([0], [0], color='#053a8d', lw=1, label="Bezier Curves"))
    legend_elements.append(Line2D([0], [0], color='#ec000b', lw=1, ls=":", label="Jet Lines"))

    ax.legend(handles=legend_elements)
    plt.title(label="Jet lines and their 3rd degree Bezier Curves")
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

    splines: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
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

    legend_elements = []
    legend_elements.append(Line2D([0], [0], color='#ff872e', lw=4, label="Central Line"))
    legend_elements.append(Line2D([0], [0], color='#ec000b', lw=2, ls=":", label="Outliers"))
    legend_elements.append(Line2D([0], [0], color='#053a8d', lw=1, label="Jet Lines"))

    ax.legend(handles=legend_elements)

    plt.title(label="Jet Lines with center and outliers (calculated using B-Spline coefficients)")
    plt.tight_layout()
    plt.show()
