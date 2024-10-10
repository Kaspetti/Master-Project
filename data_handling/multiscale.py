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
from typing import List, TypedDict, Tuple


class IcoPoint(TypedDict):
    id: int
    msLevel: int
    coord3D: List[float]
    coordGeo: List[float]


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
                           ico_points: List[IcoPoint]) -> List[IcoPoint]:
    '''
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
    '''

    ico_points = np.array(ico_points)

    # Get the three closest points to the line point
    ico_points_sorted = np.array(
            sorted(ico_points,
                   key=lambda pt: np.sqrt(np.sum((pt["coord3D"] - line_point)**2)))
            )

    tri = np.array([point["coord3D"] for point in ico_points_sorted[:6]])

    if inside_check(line_point, tri):
        return ico_points_sorted[:3]

    tri[2] = ico_points_sorted[3]["coord3D"]
    if inside_check(line_point, tri):
        return ico_points_sorted[[0, 1, 3]]

    tri[2] = ico_points_sorted[4]["coord3D"]
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


def signed_volume(a: List[List[float]], b: List[List[float]],
                  c: List[List[float]], d: List[List[float]]) -> float:
    """
    Returns the signed volume of the tetrahedron formed by the points
    a, b, c, d.
    """

    return (1.0/6.0) * np.dot(np.cross(b-a, c-a), d-a)


def subdivide_triangle(tri_pts: List[IcoPoint]) -> List[Tuple[Tuple[int, int],
                                                              List[float]]]:
    '''
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
    '''

    ps = [point["coord3D"] for point in tri_pts]
    ids = [point["id"] for point in tri_pts]

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


def multiscale(ico_points: List[IcoPoint],
               lines: List[Line],
               subdivs: int):
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

    points_at_level = {}
    for i in range(subdivs+1):
        points_at_level[i] = set()

    ico_points_ms = {}
    for i, pt in enumerate(ico_points):
        ico_points_ms[(-1, i)] = {
            "id": i,
            "msLevel": 0,
            "coord3D": pt,
            "coordGeo": to_lon_lat(pt)
        }
        points_at_level[0].add((-1, i))

    track_points_ms = {}
    points_ms_0 = [ico_points_ms[id] for id in points_at_level[0]]

    for line in lines:
        print(f"Processing line: {line['id']}")

        track_points_ms[line["id"]] = {}
        # Create track_points_ms for lowest subdivision
        for k in range(subdivs+1):
            track_points_ms[line["id"]][k] = {}

        for i, coord in enumerate(line["coords"]):
            # This does not have to be computed for each coord.. TODO
            query_points = points_ms_0
            print(query_points)
            coord3D = to_xyz(coord)

            closest = sorted(query_points,
                             key=lambda pt: np.sqrt(np.sum((pt["coord3D"] - coord3D)**2)))[0]

            dist1 = np.sqrt(np.sum((coord3D - closest["coord3D"])**2))
            if closest["id"] in track_points_ms[line["id"]][0]:
                pt, dist2 = track_points_ms[line["id"]][0][closest["id"]]

                if dist1 < dist2:
                    track_points_ms[line["id"]][0][closest["id"]] = (coord3D, dist1)
            else:
                track_points_ms[line["id"]][0][closest["id"]] = (coord3D, dist1)

            for j in range(subdivs):
                tri = get_enclosing_triangle(coord3D, query_points)
                sub = subdivide_triangle(tri)

                next_query = tri
                for (parents, pt) in sub:
                    if parents in ico_points_ms:
                        next_query = np.append(next_query, ico_points_ms[parents])
                        continue

                    ico_point = {
                        "id": len(ico_points_ms),
                        "msLevel": j+1,
                        "coord3D": pt,
                        "coordGeo": to_lon_lat(pt)
                    }

                    next_query = np.append(next_query, ico_point)
                    ico_points_ms[parents] = ico_point

                    # Create track points for this subdivision
                    closest = sorted(query_points,
                                     key=lambda pt: np.sqrt(np.sum((pt["coord3D"] - coord3D)**2)))[0]

                    dist1 = np.sqrt(np.sum((coord3D - closest["coord3D"])**2))
                    if closest["id"] in track_points_ms[line["id"]][j+1]:
                        pt, dist2 = track_points_ms[line["id"]][j+1][closest["id"]]

                        if dist1 < dist2:
                            track_points_ms[line["id"]][j+1][closest["id"]] = (coord3D, dist1)
                    else:
                        track_points_ms[line["id"]][j+1][closest["id"]] = (coord3D, dist1)

                query_points = next_query

    return ico_points_ms, track_points_ms


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

    gdf.plot(ax=ax, transform=ccrs.PlateCarree(), linewidth=1, color="blue")

    ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="black")
    ax.add_feature(cfeature.OCEAN, facecolor="lightgrey")
    ax.add_feature(cfeature.COASTLINE, edgecolor="black")
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor="darkgrey")

    # Visualize the vertices of the icosphere
    nu = 4
    ico_vertices, faces = icosphere(nu)
    geometry = [Point(to_lon_lat(coord)) for coord in ico_vertices]
    gdf = gpd.GeoDataFrame(pd.DataFrame(), geometry=geometry, crs="EPSG:4326")
    gdf.plot(ax=ax, transform=ccrs.PlateCarree(), color="red", markersize=1)

    ico_points_ms, track_points_ms = multiscale(ico_vertices, lines, 5)

    # Test visualize MS
    ms_level = 5
    geometry = []
    for line in lines:
        points = [track_points_ms[line["id"]][ms_level][ico_pt][0] for ico_pt in track_points_ms[line["id"]][ms_level]]
        for pt in points:
            geometry.append(Point(to_lon_lat(pt)))

    gdf = gpd.GeoDataFrame(pd.DataFrame(), geometry=geometry, crs="EPSG:4326")
    gdf.plot(ax=ax, transform=ccrs.PlateCarree(), markersize=5, color="orange", zorder=1000)

    geo_points = [point["coordGeo"] for point in ico_points_ms.values()]
    geometry = [Point(pt) for pt in geo_points]
    gdf = gpd.GeoDataFrame(pd.DataFrame(), geometry=geometry, crs="EPSG:4326")
    gdf.plot(ax=ax, transform=ccrs.PlateCarree(), color="red", zorder=100, markersize=1)

    ax.set_global()
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
