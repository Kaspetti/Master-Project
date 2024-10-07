from icosphere import icosphere
import math
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from typing import List, TypedDict


class Line(TypedDict):
    id: str
    coords: List[List[float]]


def to_lon_lat(v: List[float]) -> List[float]:
    '''
    Converts a 3D coordinate into spherical coordinates (lat/lon)

    Parameters
    ----------
    v : List[float] of shape (3,)
        The 3D (x, y, z) coordinates to convert to lat/lon.

    Returns
    -------
    coord : List[float] of shape (2,)
        The lat/lon coordinates of v
    '''
    lat = math.degrees(math.asin(v[2]))
    lon = math.degrees(math.atan2(v[1], v[0]))

    return [lon, lat]


def to_xyz(v: List[float]) -> List[float]:
    '''
    Converts from spherical coordinates into 3D coordinates (x, y ,z) on the
    unit sphere

    Parameters
    ----------
    v : List[float] of shape (2,)
        The spherical coordinate (lon/lat) to convert

    Returns
    -------
    coord : List[float] of shape (3,)
        The coordinate represented as [x, y, z] coordinates on the unit sphere
    '''

    x = math.cos(math.radians(v[1])) * math.cos(math.radians(v[0]))
    y = math.cos(math.radians(v[1])) * math.sin(math.radians(v[0]))
    z = math.sin(math.radians(v[1]))

    return [x, y, z]


def get_all_lines(start: str, time_offset: int) -> List[Line]:
    '''
    Gets all the lines from a NetCDF file given the start date and time offset

    Parameters
    ----------
    start : str
        The start time of the simulation. Format: 'YYYYMMDDHH'.
        Format important to be consistent with file names
    time_offset : int
        The time offset of the lines from the start time.
        Given as hours from start time

    Returns
    -------
    all_lines : List[Line] of shape (n,) with n = line amount
        An array of line objects. Line objects has "id" and an array
        "coords" containing all lat/lon points
    '''
    all_lines = []

    start_time = np.datetime64(f"{start[0:4]}-{start[4:6]}-{start[6:8]}T{start[8:10]}:00:00")

    for i in range(50):
        ds = xr.open_dataset(
                f"./data/{start}/ec.ens_{i:02d}.{start}.sfc.mta.nc"
            )
        date_ds = ds.where(
                    ds.date == start_time + np.timedelta64(time_offset, "h"),
                    drop=True
                )

        grouped_ds = list(date_ds.groupby("line_id"))

        for id, line in grouped_ds:
            min_lon = min(line.longitude.values)
            max_lon = max(line.longitude.values)

            coords = np.column_stack(
                        (line.longitude.values, line.latitude.values)
                    )

            if max_lon - min_lon > 180:
                coords = dateline_fix(coords)

            all_lines.append({"id": f"{i}|{int(id)}", "coords": coords})

    return all_lines


def dateline_fix(coords: List[float]) -> List[float]:
    '''
    Shifts a list of coordinates by 360 in longitude

    Note: This does not check if the coordinates cross the dateline,
    it simply shifts them

    Parameters
    coords : List[float] of shape (n,) with n = amount of points
        The coordinates to shift. An array of lat/lon points

    Returns
    -------
    shifted_coords : List[float] of shape (n,) with n = amount of points
        The original coordinates shifted by 360 in latitude
    '''
    for i in range(len(coords)):
        coord = coords[i]
        if coord[0] < 0:
            coords[i] = [coord[0] + 360, coord[1]]

    return coords


def get_enclosing_triangle(line_point: List[float],
                           ico_points: pd.DataFrame) -> pd.DataFrame:
    '''
    Gets the vertices on the icosphere forming a triangle enclosing the
    line point.

    Finds the three closest points of the ico_points to the line_point.
    Then checks if the line from line_point to (0, 0, 0) crosses the triangle
    formed by the three closest points. If it doesn't intersect it replaces
    the third closest point with the fourth closest and returns the points.

    Solutions gotten from:
        https://stackoverflow.com/questions/42740765/intersection-between-line-and-triangle-in-3d

    Parameters
    ----------
    line_point : List[float] of shape (3,)
        The 3D point to get the enclosing triangle of (x, y, z)
    ico_points : List[List[float]] of shape (n, 3) with n = verts in icosphere
        All the vertices of the ico sphere

    Returns
    -------
    closest : List[List[float]] of shape (3, 3)
        The coordinates of the 3 closest points on the icosphere to the
        line point. These 3 points form the triangle enclosing the point
    '''

    pts = ico_points[["x", "y", "z"]].to_numpy()

    # Get the three closest points to the line point
    dist_sqrd = np.sum((pts - line_point)**2, axis=1)
    sort_indices = np.argsort(dist_sqrd)

    indices = sort_indices[:3]
    tri = pts[indices]

    # Two far away points on the line going from origo to the line point
    q0 = np.multiply(line_point, 10)
    q1 = np.multiply(line_point, -10)

    # Check if the line passes through the plane created by the three
    # points in the triangle
    v0 = signed_volume(q0, tri[0], tri[1], tri[2])
    v1 = signed_volume(q1, tri[0], tri[1], tri[2])

    if v0 * v1 < 0:
        # Check if the line passes through the triangle
        v2 = signed_volume(q0, q1, tri[0], tri[1])
        v3 = signed_volume(q0, q1, tri[1], tri[2])
        v4 = signed_volume(q0, q1, tri[2], tri[0])

        if v2 * v3 > 0 and v2 * v4 > 0:
            return ico_points.iloc[indices]

    # If it doesn't cross the triangle we replace the third closest point
    # with the fourth closest
    indices[2] = sort_indices[3]

    return ico_points.iloc[indices]


def signed_volume(a: List[List[float]], b: List[List[float]],
                  c: List[List[float]], d: List[List[float]]) -> float:
    """
    Returns the signed volume of the tetrahedron formed by the points
    a, b, c, d.
    """

    return (1.0/6.0) * np.dot(np.cross(b-a, c-a), d-a)


def subdivide_triangle(tri_pts: pd.DataFrame) -> pd.DataFrame:
    '''
    Subdivides a triangle once creating three new points, one
    on the midpoint on all three sides of the triangle

    Parameters
    ----------
    ps : List[List[float]] of shape (3, 3)
        The points making up the triangle

    Returns
    -------
    subdivided_points : List[List[float]] of shape (3, 3)
        3 new points which subdivides the original triangle
    '''

    ps = tri_pts[["x", "y", "z"]].to_numpy()
    ids = tri_pts["id"].to_numpy()

    p_1 = normalize_point([(ps[0][0] + ps[1][0]) / 2,
                           (ps[0][1] + ps[1][1]) / 2,
                           (ps[0][2] + ps[1][2]) / 2])

    p_2 = normalize_point([(ps[0][0] + ps[2][0]) / 2,
                           (ps[0][1] + ps[2][1]) / 2,
                           (ps[0][2] + ps[2][2]) / 2])

    p_3 = normalize_point([(ps[1][0] + ps[2][0]) / 2,
                           (ps[1][1] + ps[2][1]) / 2,
                           (ps[1][2] + ps[2][2]) / 2])

    data = [
        {
            "parent1": min(ids[0], ids[1]),
            "parent2": max(ids[0], ids[1]),
            "x": p_1[0],
            "y": p_1[1],
            "z": p_1[2],
        },
        {
            "parent1": min(ids[0], ids[2]),
            "parent2": max(ids[0], ids[2]),
            "x": p_2[0],
            "y": p_2[1],
            "z": p_2[2],
        },
        {
            "parent1": min(ids[1], ids[2]),
            "parent2": max(ids[1], ids[2]),
            "x": p_3[0],
            "y": p_3[1],
            "z": p_3[2],
        },
    ]

    return pd.DataFrame(data)


def normalize_point(p: List[List[float]]) -> List[List[float]]:
    """
    Normalizes the given point making its length equal 1

    Parameters
    ----------
    p : List[List[float]] of shape (n,) with n = dimension of the point
        The point to normalize

    Return
    ------
    normalized_p : List[List[float]] of shape (n,) with
                    n = dimension of the point
        The normalized point
    """

    return p / np.linalg.norm(p)


def generate_plot(simstart: str, time_offset: int, show: bool = False):
    """
    Generates a plot of the lines from a given simulation start and a
    time offset. The plot will be saved as a svg file with this naming
    convention: <simstart>\_<time\_offset>h.svg

    Parameters
    ----------
    simstart : str
        The simulation start. Must be already downloaded.
        Format: YYYYDDMMHH
    time_offset : int
        The time offset from the simulation start. In hours
    show : bool
        If show is set to True then the plot will only be
        shown and not saved. Defaults to False
    """

    lines = get_all_lines(simstart, time_offset)

    ids = [line["id"] for line in lines]
    df = pd.DataFrame(ids, columns=["id"])
    geometry = [LineString(line["coords"]) for line in lines]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="black")
    ax.add_feature(cfeature.OCEAN, facecolor="lightgrey")
    ax.add_feature(cfeature.COASTLINE, edgecolor="black")
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor="darkgrey")

    # gdf.plot(ax=ax, transform=ccrs.PlateCarree(), linewidth=1)

    # START OF DEBUG #

    # Visualize line 514 in red with a bolder line
    colors = ["blue"] * len(gdf)
    linewidth = [1] * len(gdf)

    colors[514] = "red"
    linewidth[514] = 4

    gdf.plot(ax=ax, transform=ccrs.PlateCarree(),
             linewidth=linewidth, colors=colors)

    # Show points of line 514
    geometry = [Point(coord) for coord in lines[514]["coords"]]
    gdf = gpd.GeoDataFrame(pd.DataFrame(), geometry=geometry, crs="EPSG:4326")
    gdf.plot(ax=ax, transform=ccrs.PlateCarree(),
             color="green", markersize=100, zorder=100)

    # Visualize the vertices of the icosphere
    nu = 4
    ico_vertices, faces = icosphere(nu)
    ico_vertices_geo = [to_lon_lat(coord) for coord in ico_vertices]
    geometry = [Point(coord) for coord in ico_vertices_geo]
    gdf.plot(ax=ax, transform=ccrs.PlateCarree(), color="red")

    # The track points represented at multiscale
    track_points_ms = []

    # All ico points after local subdivision of the icosphere
    ico_points_ms = pd.DataFrame(columns=[
        "id",
        "parent1",
        "parent2",
        "msLevel",
        "x",
        "y",
        "z",
        "lon",
        "lat",
    ])

    # Add the original ico points to the ico_points_ms dataframe
    for i, pt in enumerate(ico_vertices):
        ico_points_ms.loc[len(ico_points_ms)] = {
            "id": i,
            "msLevel": 0,
            "x": pt[0],
            "y": pt[1],
            "z": pt[2],
            "lon": ico_vertices_geo[i][0],
            "lat": ico_vertices_geo[i][1],
        }

    subdivs = 5
    for i, coord in enumerate(lines[514]["coords"]):
        print(f"Subdividing {i}")
        geometry = []
        query_points = ico_points_ms.loc[(ico_points_ms["msLevel"] == 0)][["id", "x", "y", "z"]]

        for i in range(subdivs):
            tri = get_enclosing_triangle(to_xyz(coord), query_points)
            sub = subdivide_triangle(tri)

            next_query = pd.DataFrame(tri, columns=["id", "x", "y", "z"]).reset_index(drop=True)
            for index, pt in sub.iterrows():
                # Does the subdivided point already exists?
                existing = ico_points_ms.loc[
                        (ico_points_ms["parent1"] == pt["parent1"]) &
                        (ico_points_ms["parent2"] == pt["parent2"])
                ]
                if not existing.empty:
                    next_query = pd.concat([next_query,
                                            existing[["id", "x", "y", "z"]]],
                                           ignore_index=True)
                    continue

                # Add the subdivided point to ico_points_ms
                next_query.loc[len(next_query)] = {
                    "id": len(ico_points_ms),
                    "x": pt["x"],
                    "y": pt["y"],
                    "z": pt["z"],
                }

                geo_coord = to_lon_lat([pt["x"], pt["y"], pt["z"]])
                ico_points_ms.loc[len(ico_points_ms)] = {
                    "id": len(ico_points_ms),
                    "parent1": pt["parent1"],
                    "parent2": pt["parent2"],
                    "msLevel": i+1,
                    "x": pt["x"],
                    "y": pt["y"],
                    "z": pt["z"],
                    "lon": geo_coord[0],
                    "lat": geo_coord[1],
                }

            query_points = next_query

            if i == subdivs - 1:
                for index, pt in tri.iterrows():
                    original_pt = ico_points_ms.loc[(ico_points_ms["id"] == pt["id"])]
                    geometry.append(Point(original_pt[["lon", "lat"]].to_numpy()))

        geometry.append(Point(coord))

        gdf = gpd.GeoDataFrame(pd.DataFrame(), geometry=geometry, crs="EPSG:4326")
        colors = ["orange"] * len(gdf)
        colors[-1] = "black"
        # colors[-3] = "orange"
        # colors[-2] = "orange"
        # colors[-1] = "orange"

        gdf.plot(ax=ax, transform=ccrs.PlateCarree(), color=colors, zorder=100)

    # Focus the view on line on index 514
    ax.set_extent([-55, -10, 5, 40], crs=ccrs.PlateCarree())

    # END OF DEBUG #

    # ax.set_global()
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    if show:
        plt.show()
    else:
        plt.savefig(f"./images/{simstart}_{time_offset}h.svg", transparent=True, format="svg")
    plt.close()


def generate_all_plots(simstart: str):
    """
    Generates and saves the plots for each time offset for the given simstart

    Parameters
    ----------
    simstart : str
        The simulation start. Must be already downloaded.
        Format: YYYYDDMMHH
    """
    i = 0
    while (i <= 240):
        print(f"Generating {simstart}_{i}h.svg")

        generate_plot(simstart, i)

        if i < 72:
            i += 3
        else:
            i += 6


if __name__ == "__main__":
    generate_plot("2024082300", 0, show=True)

    # generate_all_plots("2024082300")
