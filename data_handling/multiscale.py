from icosphere import icosphere
import folium
import math


def to_lat_lon(v):
    lat = math.degrees(math.asin(v[2]))
    lon = math.degrees(math.atan2(v[1], v[0]))

    return [lat, lon]


if __name__ == "__main__":
    nu = 12
    vertices, faces = icosphere(nu)

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
