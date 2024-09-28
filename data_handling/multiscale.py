from icosphere import icosphere
import math
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def to_lat_lon(v):
    '''
    Converts a 3D coordinate into spherical coordinates (lat/lon)

    Parameters
    ----------
    v : the (x, y, z) coordinates to convert to lat/lon.
        Given as an array of size 3

    Returns
    -------
    lat_lon : the [lat, lon] coordinates of v
    '''
    lat = math.degrees(math.asin(v[2]))
    lon = math.degrees(math.atan2(v[1], v[0]))

    return [lat, lon]


def to_xyz(v):
    '''
    Converts a [lat, lon] coordinate into 3D coordinates ([x, y ,z]) on the
    unit sphere

    Parameters
    ----------
    v : the [lat, lon] coordinate to convert

    Returns
    -------
    [x, y, z] : the coordinate represented as [x, y, z] coordinates
    on the unit sphere
    '''

    x = math.cos(math.radians(v[0])) * math.cos(math.radians(v[1]))
    y = math.cos(math.radians(v[0])) * math.sin(math.radians(v[1]))
    z = math.sin(math.radians(v[0]))

    return [x, y, z]


def get_all_lines(start, time_offset):
    '''
    Gets all the lines from a NetCDF file given the start date and time offset

    Parameters
    ----------
    start : the start time of the simulation. Format: 'YYYYMMDDHH'.
        Format important to be consistent with file names
    time_offset : the time offset of the lines from the start time.
        Given as hours from start time

    Returns
    -------
    all_lines : an array of line objects. Line objects has "id" and an array
        "coords" containing all [lat, lon] points
    '''
    all_lines = []

    start_time = np.datetime64(f"{start[0:4]}-{start[4:6]}-{start[6:8]}T{start[8:10]}:00:00")

    for i in range(50):
        ds = xr.open_dataset(
                f"{start}/ec.ens_{i:02d}.{start}.sfc.mta.nc"
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


def dateline_fix(coords):
    '''
    Shifts a list of coordinates by 360 in longitude

    Note: This does not check if the coordinates cross the dateline,
    it simply shifts them

    Parameters
    ----------
    coords : the coordinates to shift. An array of [lat, lon]

    Returns
    -------
    shifted_coords : the original coordinates shifted by 360 in latitude
    '''
    for i in range(len(coords)):
        coord = coords[i]
        if coord[1] < 0:
            coords[i] = (coord[0], coord[1] + 360)

    return coords


def get_enclosing_triangle(line_point, ico_points):
    '''
    Gets the indices of the vertices on the icosphere forming a
    triangle enclosing the line point

    Parameters
    ----------
    line_point : the point to get the enclosing triangle of ([x, y, z])
    ico_points : all the vertices of the ico sphere ([[x, y, z]...])

    Returns
    -------
    closest : the indices of the 3 closest points on the icosphere to the
        line point. These 3 points form the triangle enclosing the point
    '''
    dist_sqrd = np.sum((ico_points - line_point)**2, axis=1)
    sort_indices = np.argsort(dist_sqrd)

    return ico_points[sort_indices[:3]]


def subdivide_triangle(ps):
    '''
    Subdivides a triangle once creating three new points, one
    on the midpoint on all three sides of the triangle

    Parameters
    ----------
    ps : the points making up the triangle

    Returns
    -------
    [p_1, p_2, p_3] : 3 new points which subdivides the original triangle
    '''

    p_1 = [(ps[0][0] + ps[1][0]) / 2, (ps[0][1] + ps[1][1]) / 2, (ps[0][2] + ps[1][2]) / 2]
    p_2 = [(ps[0][0] + ps[2][0]) / 2, (ps[0][1] + ps[2][1]) / 2, (ps[0][2] + ps[2][2]) / 2]
    p_3 = [(ps[1][0] + ps[2][0]) / 2, (ps[1][1] + ps[2][1]) / 2, (ps[1][2] + ps[2][2]) / 2]

    return [normalize_point(p_1), normalize_point(p_2), normalize_point(p_3)]


def normalize_point(p):
    return p / np.linalg.norm(p)


def generate_plot(simstart, time_offset, show=False):
    """
    Generates a plot of the lines from a given simulation start and a time offset.
    The plot will be saved as a svg file with this naming convention:
        <simstart>_<time_offset>h.svg

    Parameters
    ----------
    simstart : the simulation start. Must be already downloaded.
        Format: YYYYDDMMHH
    time_offset : the time offset from the simulation start. In hours
    show : if show is set to "True" then the plot will only be
        shown and not saved
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

    gdf.plot(ax=ax, transform=ccrs.PlateCarree())

    ax.set_global()
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    if show:
        plt.show()
    else:
        plt.savefig(f"./images/{simstart}_{time_offset}h.svg", transparent=True, format="svg")
    plt.close()


def generate_all_plots(simstart):
    """
    Generates and saves the plots for each time offset for the given simstart

    Parameters
    ----------
    simstart : the simulation start. Must be already downloaded.
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
    # generate icosphere
    nu = 4
    ico_vertices, faces = icosphere(nu)

    generate_plot("2024082300", 0, show=True)

    # generate_all_plots("2024082300")
