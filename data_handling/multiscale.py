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


def to_lat_lon(v: List[float]) -> List[float]:
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

    return [lat, lon]


def to_xyz(v: List[float]) -> List[float]:
    '''
    Converts from spherical coordinates into 3D coordinates (x, y ,z) on the
    unit sphere

    Parameters
    ----------
    v : List[float] of shape (2,)
        The spherical coordinate (lat/lon) to convert

    Returns
    -------
    coord : List[float] of shape (3,)
        The coordinate represented as [x, y, z] coordinates on the unit sphere
    '''

    x = math.cos(math.radians(v[0])) * math.cos(math.radians(v[1]))
    y = math.cos(math.radians(v[0])) * math.sin(math.radians(v[1]))
    z = math.sin(math.radians(v[0]))

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
                        (line.latitude.values, line.longitude.values)
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
        if coord[1] < 0:
            coords[i] = (coord[0], coord[1] + 360)

    return coords


def get_enclosing_triangle(line_point: List[float],
                           ico_points: List[List[float]]) -> List[List[float]]:
    '''
    Gets the vertices on the icosphere forming a
    triangle enclosing the line point

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
    dist_sqrd = np.sum((ico_points - line_point)**2, axis=1)
    sort_indices = np.argsort(dist_sqrd)

    return ico_points[sort_indices[:3]]


def subdivide_triangle(ps: List[List[float]]) -> List[List[float]]:
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

    p_1 = [(ps[0][0] + ps[1][0]) / 2,
           (ps[0][1] + ps[1][1]) / 2,
           (ps[0][2] + ps[1][2]) / 2]

    p_2 = [(ps[0][0] + ps[2][0]) / 2,
           (ps[0][1] + ps[2][1]) / 2,
           (ps[0][2] + ps[2][2]) / 2]

    p_3 = [(ps[1][0] + ps[2][0]) / 2,
           (ps[1][1] + ps[2][1]) / 2,
           (ps[1][2] + ps[2][2]) / 2]

    return [normalize_point(p_1), normalize_point(p_2), normalize_point(p_3)]


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


def haversine(c1: List[float], c2: List[float]) -> float:
    """
    Calculates the distance between two lat/lon points using the
    haversine formula

    Parameters
    ----------
    c1 : List[float] of shape (2,)
        First coordinate
    c2 : List[float] of shape (2,)
        Second coordinate

    Returns
    -------
    dist : float
        The distance between c1 and c2
    """

    # Don't bother calculating distance if coords are the same
    if c1 == c2:
        return 0.0

    earth_radius = 6371

    lat_1, lon_1 = math.radians(c1[0]), math.radians(c1[1])
    lat_2, lon_2 = math.radians(c2[0]), math.radians(c2[1])

    d_lat = lat_2 - lat_1
    d_lon = lon_2 - lon_1

    dist = 2 * earth_radius * math.asin(
            math.sqrt(
                (1 - math.cos(d_lat)
                 + math.cos(lat_1)
                 * math.cos(lat_2)
                 * (1 - math.cos(d_lon)))
                / 2))

    return dist


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
    geometry = [LineString(np.flip(line["coords"])) for line in lines]
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
    linewidth[514] = 2

    gdf.plot(ax=ax, transform=ccrs.PlateCarree(),
             linewidth=linewidth, colors=colors)

    # Show points of line 514
    geometry = [Point(np.flip(coord)) for coord in lines[514]["coords"]]
    gdf = gpd.GeoDataFrame(pd.DataFrame(), geometry=geometry, crs="EPSG:4326")
    gdf.plot(ax=ax, transform=ccrs.PlateCarree(),
             color="green", markersize=100, zorder=100)

    # Visualize the vertices of the icosphere
    nu = 4
    ico_vertices, faces = icosphere(nu)
    ico_vertices_geo = [to_lat_lon(coord) for coord in ico_vertices]
    geometry = [Point(np.flip(coord)) for coord in ico_vertices_geo]

    gdf = gpd.GeoDataFrame(pd.DataFrame(), geometry=geometry, crs="EPSG:4326")
    gdf.plot(ax=ax, transform=ccrs.PlateCarree(), color="red")

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
