from icosphere import icosphere
import folium
import math
import xarray as xr
import numpy as np


def to_lat_lon(v):
    lat = math.degrees(math.asin(v[2]))
    lon = math.degrees(math.atan2(v[1], v[0]))

    return [lat, lon]


def get_all_lines():
    all_lines = []

    start_time = np.datetime64("2024-08-23T00:00:00")
    current_time = 3

    for i in range(50):
        ds = xr.open_dataset(
                f"2024082300/ec.ens_{i:02d}.2024082300.sfc.mta.nc"
            )
        date_ds = ds.where(
                    ds.date == start_time + np.timedelta64(current_time, "h"),
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
    for i in range(len(coords)):
        coord = coords[i]
        if coord[1] < 0:
            coords[i] = (coord[0], coord[1] + 360)

    return coords


if __name__ == "__main__":
    # generate icosphere
    nu = 8
    vertices, faces = icosphere(nu)

    # read data
    lines = get_all_lines()

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
    circle_radius = 5
    for v in vertices:
        lat_lon = to_lat_lon(v)
        folium.CircleMarker(
            location=lat_lon,
            radius=circle_radius,
            color="blue",
            weight=0,
            fill_opacity=1,
            fill=True,
        ).add_to(m)


    m.save("index.html")
