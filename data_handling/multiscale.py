from icosphere import icosphere
import folium
import math
import xarray as xr
import numpy as np


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

            all_lines.append({"id": id, "coords": coords})

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
    line_point : the point to get the enclosing triangle of ([lat, lon])
    ico_points : all the vertices of the ico sphere ([[lat, lon]...])

    Returns
    -------
    closest : the indices of the 3 closest points on the icosphere to the
        line point. These 3 points form the triangle enclosing the point
    '''
    dist_sqrd = np.sum((ico_points - line_point)**2, axis=1)
    sort_indices = np.argsort(dist_sqrd)

    return sort_indices[:3]


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

    p_1 = [(ps[0][0] + ps[1][0]) / 2, (ps[0][1] + ps[1][1]) / 2]
    p_2 = [(ps[0][0] + ps[2][0]) / 2, (ps[0][1] + ps[2][1]) / 2]
    p_3 = [(ps[1][0] + ps[2][0]) / 2, (ps[1][1] + ps[2][1]) / 2]

    return [p_1, p_2, p_3]


if __name__ == "__main__":
    # generate icosphere
    nu = 4
    vertices, faces = icosphere(nu)

    ico_lat_lons = []
    for v in vertices:
        ico_lat_lons.append(to_lat_lon(v))

    # read data
    lines = get_all_lines("2024082300", 0)
    # print(get_enclosing_triangle(lines[0]["coords"][0], ico_lat_lons))

    # show map
    attr = (
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
        'contributors, &copy; <a href="https://cartodb.com/attributions">CartoDB</a>'
    )
    tiles = "https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png"

    lat = 0
    lon = 0
    zoom_start = 2
    m = folium.Map(location=[lat, lon],
                   tiles=tiles,
                   attr=attr,
                   zoom_start=zoom_start)

    # show the lines
    for line in lines:
        folium.PolyLine(
            locations=line["coords"],
            weight=1,
            color="red",
            tooltip=line["id"],
        ).add_to(m)

    # show the vertices of the icosphere on the map
    circle_radius = 2
    for lat_lon in ico_lat_lons:
        folium.CircleMarker(
            location=lat_lon,
            radius=circle_radius,
            color="blue",
            weight=0,
            fill_opacity=1,
            fill=True,
        ).add_to(m)

    # enclosing triangle test
    line_id = 5

    enc_tri = get_enclosing_triangle(lines[line_id]["coords"][0], ico_lat_lons)
    for tri in enc_tri:
        folium.CircleMarker(
            location=ico_lat_lons[tri],
            color="black",
            weight=0,
            fill_opacity=1,
            fill=True,
            radius=5,
        ).add_to(m)

    folium.CircleMarker(
        location=lines[line_id]["coords"][0],
        color="green",
        weight=0,
        fill_opacity=1,
        fill=True,
        radius=5,
    ).add_to(m)

    m.save("index.html")
