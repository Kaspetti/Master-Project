"""
Program for visualizing ensembles of mta lines or
jet lines on a map.
Clusters the lines to create an aggregate visualization
of the lines.
Uses a multiscale approach on the lines in order to improve
performance of the distance checks and reduce effects of
line starting points being shifted.
"""

import math
from typing import List, TypedDict, Tuple, Literal, Set, Dict

from icosphere import icosphere
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.spatial import KDTree


class IcoPoint(TypedDict):
    """A point on an icosphere.

    Attributes:
        id (int): The unique id of the icopoint.
            Usually the index of the point in an array of icopoints.
        parent_1 (int): The unique id of the first parent.
            If the ico point is the result of a subdivision this will be
            the lowest id point of the edge which got subdivided.
        parent_2 (int): The unique id of the second parent.
            If the ico point is the result of a subdivision this will be
            the highest id point of the edge which got subdivided.
        ms_level (int): The multiscale level of the icopoint
            The level of the the initial icopoints on the sphere is 0,
            increasing by 1 for each subdivision.
        coord_3D (List[float]): The 3D coordinate of the point.
            Represented as a list of three floats: [x, y, z].
        coord_geo (List[float]): The longitude and latitude of the point.
            Represented as a list of two floats: [longitude, latitude].
    """

    id: int
    ms_level: int
    parent_1: int
    parent_2: int
    coord_3D: List[float]
    coord_geo: List[float]


class Line(TypedDict):
    """A line.

    A line is a collection of ordered points.

    Attributes:
        id (str): The unique identifier of the line.
            The id is created by combining the ensemble number
            the line is part of and the line's id in that ensemble.
            'ensemble_nr|line_id'
        coords (List[List[float]]): A list of the coordinates for the line.
            Represented as a list containing the longitude and latitude
            points of the line: [[longitude, latiude], ...].
    """

    id: str
    coords: List[List[float]]


class MultiscaleData():
    """Holds the results of the multiscale operation

    Attributes:
        points_at_level (TypedDict[int, Set(int)]): A dictionary of points
            at each multiscale level. Key is the multiscale level, while
            the value is a set of the unique ids of the points at that level.
        subdivided_edges (TypedDict[Tuple[int, int], int]): A dictionary of
            already subdivided edges. Key is the id of the two points making up
            the edge, while the value is the id of the point dividing the edge.
        ico_points_ms (TypedDict[int, IcoPoint]): A dictionary of icopoints.
            The icopoints are the points making up the icosphere. This
            dicitonary functions as a lookup table for the points. The key is
            the id of icopoint, while the value is the IcoPoint instance with
            its attributes.
        line_points_ms (TypedDict[str, TypedDict[int, List[float]]]): A
            dictionary containing the representations of all lines at all
            multiscale levels. Key is the id of the line, while the value is
            a new dictionary where the key is the multiscale level and the
            value is a list of floats representing the line at the given
            multiscale level.
    """

    points_at_level: Dict[int, Set[int]]
    subdivided_edges: Dict[Tuple[int, int], int]
    ico_points_ms: Dict[int, IcoPoint]
    line_points_ms: Dict[str, Dict[int, List[float]]]

    def __init__(self, ico_points: List[IcoPoint], subdivs: int):
        self.points_at_level = {}
        for i in range(subdivs + 1):
            self.points_at_level[i] = set()

        self.subdivided_edges = {}
        self.ico_points_ms = {}
        for i, pt in enumerate(ico_points):
            self.ico_points_ms[i] = {
                "id": i,
                "parent_1": -1,
                "parent_2": -1,
                "ms_level": 0,
                "coord_3D": pt,
                "coord_geo": to_lon_lat(pt),
            }
            self.points_at_level[0].add(i)
        self.line_points_ms = {}


def to_lon_lat(v: List[float]) -> List[float]:
    """Converts a 3D coordinate into longitude and latitude.

    :param v: A list of three floats representing a 3D coordinate:
        [x, y, z].
    :return: A list of two floats representing v in longtitude and latitude:
        [longitude, latitude].
    """
    lat = math.degrees(math.asin(v[2]))
    lon = math.degrees(math.atan2(v[1], v[0]))

    return [lon, lat]


def to_xyz(v: List[float]) -> List[float]:
    """Converts longitude and latitude to a 3D coordinate.

    :param v: A list of two floats representing longitude and latitude:
        [longitude, latitude].
    :return: A list of three floats representing v in 3D coordinates:
        [x, y, z].
    """

    x = math.cos(math.radians(v[1])) * math.cos(math.radians(v[0]))
    y = math.cos(math.radians(v[1])) * math.sin(math.radians(v[0]))
    z = math.sin(math.radians(v[1]))

    return [x, y, z]


def get_all_lines(
    start: str, time_offset: int, line_type: Literal["mta", "jet"]
) -> List[Line]:
    """Reads all lines from a NETCDF4 file and returns them.

    Reads a group of NETCDF4 files of a specific format and returns the lines
    in the files which has the specified time offset.
    The file names must be of the following format:
        ec.ens_{ens_id}.{start}.{sfc|pv2000}.{mta|jetaxis}.nc
    where ens_id is between 00 and 50.
    The files must be in the following path:
        ./data/{mta|jet}/{start}
    The file needs to have the following attributes:
        longitude
        latitude
        date
    where date is the time offset in hours from the start time.
    The function expects 50 files, or 50 ensembles, to be present
    in the folder './date/{mta|jet}/{start}'

    :param start: The start time of the computation.
        Must be of the format: YYYYMMDDTT where TT is one of
        00 or 12.
    :param time_offset: The time offset from the start to get the lines.
        The offset is given in hours from the start time.
    :param line_type: The type of the lines to get.
        Currently supported line types are 'mta' and 'jet'.
    :return: A list of the lines from the 50 ensembles at the time offset.
    """
    all_lines = []

    start_time = np.datetime64(
        f"{start[0:4]}-{start[4:6]}-{start[6:8]}T{start[8:10]}:00:00"
    )

    for i in range(50):
        base_path = f"./data/{line_type}/{start}/"
        file_path = f"ec.ens_{i:02d}.{start}.sfc.mta.nc"

        if line_type == "jet":
            file_path = f"ec.ens_{i:02d}.{start}.pv2000.jetaxis.nc"
        full_path = base_path + file_path

        ds = xr.open_dataset(full_path)
        date_ds = ds.where(
            ds.date == start_time + np.timedelta64(time_offset, "h"), drop=True
        )

        grouped_ds = list(date_ds.groupby("line_id"))

        for id_, line in grouped_ds:
            coords = np.column_stack(
                        (line.longitude.values, line.latitude.values)
                    )

            if max(line.longitude.values) - min(line.longitude.values) > 180:
                coords = dateline_fix(coords)

            all_lines.append({"id": f"{i}|{int(id_)}", "coords": coords})

    return all_lines


def dateline_fix(coords: List[float]) -> List[float]:
    """
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
    """
    for i, coord in enumerate(coords):
        if coord[0] < 0:
            coords[i] = [coord[0] + 360, coord[1]]

    return coords


def get_enclosing_triangle(
    line_point: List[float], ico_points: List[IcoPoint]
) -> List[IcoPoint]:
    """
    Gets the vertices on the icosphere forming a triangle enclosing the
    line point.

    Parameters
    ----------
    line_point : List[float] of shape (3,)
        The 3D point to get the enclosing triangle of (x, y, z)
    ico_points : List[IcoPoint] of shape (n, 3) with n = verts in icosphere
        All the vertices of the ico sphere

    Returns
    -------
    closest : List[IcoPoint] of shape (3,)
        The 3 closest points on the icosphere to the
        line point. These 3 points form the triangle enclosing the point
    """

    ico_points = np.array(ico_points)

    # Get the three closest points to the line point
    ico_points_sorted = np.array(
        sorted(
            ico_points,
            key=lambda pt: np.sqrt(np.sum((pt["coord_3D"] - line_point) ** 2)),
        )
    )

    tri = np.array([point["coord_3D"] for point in ico_points_sorted[:6]])

    if inside_check(line_point, tri):
        return ico_points_sorted[:3]

    tri[2] = ico_points_sorted[3]["coord_3D"]
    if inside_check(line_point, tri):
        return ico_points_sorted[[0, 1, 3]]

    tri[2] = ico_points_sorted[4]["coord_3D"]
    if inside_check(line_point, tri):
        return ico_points_sorted[[0, 1, 4]]

    return ico_points_sorted[[0, 1, 2]]


def inside_check(pt: List[float], tri: List[List[float]]) -> bool:
    """
    Checks if a point lies "inside" a triangle by checking if the ray
    from the point to origo passes through the triangle.

    Solution gotten from:
        https://stackoverflow.com/questions/42740765/intersection-between-line-and-triangle-in-3d

    Parameters
    ----------
    pt : List[float] of shape (3,)
        The point to check if lies inside the triangle
    tri : List[List[float]] of shape (3, 3)
        The points making up the triangle

    Returns
    -------
    True if the line passes through, False otherwise
    """

    # Two far away points on the line going from origo to the line point
    q0 = np.multiply(pt, 10)
    q1 = np.multiply(pt, -10)

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
            return True

    return False


def signed_volume(
    a: List[List[float]],
    b: List[List[float]],
    c: List[List[float]],
    d: List[List[float]],
) -> float:
    """
    Returns the signed volume of the tetrahedron formed by the points
    a, b, c, d.
    """

    return (1.0 / 6.0) * np.dot(np.cross(b - a, c - a), d - a)


def subdivide_triangle(
    tri_pts: List[IcoPoint],
) -> List[Tuple[Tuple[int, int], List[float]]]:
    """
    Subdivides a triangle once creating three new points, one
    on the midpoint on all three sides of the triangle

    Parameters
    ----------
    ps : List[IcoPoint] of shape (3,)
        The points making up the triangle

    Returns
    -------
    subdivided_points : List[Tuple[Tuple[int, int], List[float]]] of shape (3,)
        A list of tuples containing the id of the parents of the new point,
        aswell as the point itself
    """

    ps = [point["coord_3D"] for point in tri_pts]
    ids = [point["id"] for point in tri_pts]

    p_1 = normalize_point(
        [
            (ps[0][0] + ps[1][0]) / 2,
            (ps[0][1] + ps[1][1]) / 2,
            (ps[0][2] + ps[1][2]) / 2,
        ]
    )

    p_2 = normalize_point(
        [
            (ps[0][0] + ps[2][0]) / 2,
            (ps[0][1] + ps[2][1]) / 2,
            (ps[0][2] + ps[2][2]) / 2,
        ]
    )

    p_3 = normalize_point(
        [
            (ps[1][0] + ps[2][0]) / 2,
            (ps[1][1] + ps[2][1]) / 2,
            (ps[1][2] + ps[2][2]) / 2,
        ]
    )

    data = [
        ((min(ids[0], ids[1]), max(ids[0], ids[1])), p_1),
        ((min(ids[0], ids[2]), max(ids[0], ids[2])), p_2),
        ((min(ids[1], ids[2]), max(ids[1], ids[2])), p_3),
    ]

    return data


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


def multiscale(ico_points: List[IcoPoint], lines: List[Line], subdivs: int):
    """
    Performs a multiscale subdivision of the icosahedron, returning the new
    subdivided points aswell as the lines represented at the different
    subdivision levels.

    Subdivision is performed locally around the lines to prevent too many
    points created

    Parameters
    ----------
    ico_points : List[IcoPoint]
        A list of the vertices of the icosahedron before subdivision
    lines : List[Line]
        A list of the lines which the subdivision and multiscale will
        occur to and around
    subdivs : int
        The amount of subdivision to do. The level of the multiscale

    Returns
    -------
    Two data structures. First being the new list of vertices on the
    icosahedron after subdivision. The second being a data structure
    containing the representation of the lines at different scales.
    """

    ms_data = MultiscaleData(ico_points, subdivs)

    points_ms_0 = np.array([ms_data.ico_points_ms[id]
                            for id in ms_data.points_at_level[0]])

    ms_0_kd_tree = KDTree([point["coord_3D"] for point in points_ms_0])

    for line in lines:
        print(f"Processing line: {line['id']}")

        ms_data.line_points_ms[line["id"]] = {}
        # Create line_points_ms for lowest subdivision
        for k in range(subdivs + 1):
            ms_data.line_points_ms[line["id"]][k] = {}

        for i, coord in enumerate(line["coords"]):
            # This does not have to be computed for each coord.. TODO
            coord3D = to_xyz(coord)
            closest = ms_0_kd_tree.query(coord3D, 6)
            query_points = points_ms_0[closest[1]]

            closest = sorted(
                query_points,
                key=lambda pt: np.sqrt(np.sum((pt["coord_3D"] - coord3D) ** 2)),
            )[0]

            dist1 = np.sqrt(np.sum((coord3D - closest["coord_3D"]) ** 2))
            if closest["id"] in ms_data.line_points_ms[line["id"]][0]:
                pt, dist2 = ms_data.line_points_ms[line["id"]][0][closest["id"]]

                if dist1 < dist2:
                    ms_data.line_points_ms[line["id"]][0][closest["id"]] = (coord3D, dist1)
            else:
                ms_data.line_points_ms[line["id"]][0][closest["id"]] = (coord3D, dist1)

            for j in range(subdivs):
                tri = get_enclosing_triangle(coord3D, query_points)
                sub = subdivide_triangle(tri)

                next_query = tri
                for parents, pt in sub:
                    if parents in ms_data.subdivided_edges:
                        next_query = np.append(
                            next_query, ms_data.ico_points_ms[ms_data.subdivided_edges[parents]]
                        )
                        continue

                    id = len(ms_data.ico_points_ms)
                    ico_point = {
                        "id": id,
                        "parent_1": parents[0],
                        "parent_2": parents[1],
                        "ms_level": j + 1,
                        "coord_3D": pt,
                        "coord_geo": to_lon_lat(pt),
                    }

                    next_query = np.append(next_query, ico_point)
                    ms_data.subdivided_edges[parents] = id
                    ms_data.ico_points_ms[id] = ico_point
                    ms_data.points_at_level[j + 1].add(id)

                    # Create line points for this subdivision
                    closest = sorted(
                        query_points,
                        key=lambda pt: np.sqrt(np.sum((pt["coord_3D"] - coord3D) ** 2)),
                    )[0]

                    dist1 = np.sqrt(np.sum((coord3D - closest["coord_3D"]) ** 2))
                    if closest["id"] in ms_data.line_points_ms[line["id"]][j + 1]:
                        pt, dist2 = ms_data.line_points_ms[line["id"]][j + 1][closest["id"]]

                        if dist1 < dist2:
                            ms_data.line_points_ms[line["id"]][j + 1][closest["id"]] = (
                                coord3D,
                                dist1,
                            )
                    else:
                        ms_data.line_points_ms[line["id"]][j + 1][closest["id"]] = (
                            coord3D,
                            dist1,
                        )

                query_points = next_query

    return ms_data.ico_points_ms, ms_data.line_points_ms


def generate_plot(
    simstart: str,
    time_offset: int,
    line_type: Literal["mta", "jet"],
    show: bool = False,
):
    r"""
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
    line_typ : Literal['mta', 'jet']
        The line type to visualize. Can be one of 'mta' or 'jet'
    show : bool
        If show is set to True then the plot will only be
        shown and not saved. Defaults to False
    """

    lines = get_all_lines(simstart, time_offset, line_type)

    ids = [line["id"] for line in lines]
    df = pd.DataFrame(ids, columns=["id"])
    geometry = [LineString(line["coords"]) for line in lines]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    gdf.plot(ax=ax, transform=ccrs.PlateCarree(), linewidth=1, color="blue")

    ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="black")
    ax.add_feature(cfeature.OCEAN, facecolor="lightgrey")
    ax.add_feature(cfeature.COASTLINE, edgecolor="black")
    ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="darkgrey")

    # Visualize the vertices of the icosphere
    nu = 2
    ico_vertices, faces = icosphere(nu)

    ico_points_ms, track_points_ms = multiscale(ico_vertices, lines, 2)

    # Test visualize MS
    ms_level = 0
    geometry = []
    for line in lines:
        points = [
            ico_points_ms[id]["coord_geo"]
            for id in track_points_ms[line["id"]][ms_level].keys()
        ]
        for pt in points:
            geometry.append(Point(pt))

    gdf = gpd.GeoDataFrame(pd.DataFrame(), geometry=geometry, crs="EPSG:4326")
    gdf.plot(
        ax=ax, transform=ccrs.PlateCarree(), markersize=15, color="black", zorder=1000
    )

    geo_points = [point["coord_geo"] for point in ico_points_ms.values()]
    geometry = [Point(pt) for pt in geo_points]
    gdf = gpd.GeoDataFrame(pd.DataFrame(), geometry=geometry, crs="EPSG:4326")
    gdf.plot(ax=ax, transform=ccrs.PlateCarree(), color="red", zorder=100, markersize=1)

    ax.set_global()
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    if show:
        plt.show()
    else:
        plt.savefig(
            f"./images/{simstart}_{time_offset}h.svg", transparent=True, format="svg"
        )
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
    while i <= 240:
        print(f"Generating {simstart}_{i}h.svg")

        generate_plot(simstart, i)

        if i < 72:
            i += 3
        else:
            i += 6


if __name__ == "__main__":
    generate_plot("2024101900", 0, "jet", show=True)
    # cProfile.run('generate_plot("2024082300", 0, show=True)')

    # generate_all_plots("2024082300")
