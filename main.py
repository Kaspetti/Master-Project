from data import get_all_lines_1
import folium

attr = (
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
    'contributors, &copy; <a href="https://cartodb.com/attributions">CartoDB</a>'
)
tiles = "https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png"


lat = 0
lon = 0
zoom_start = 2
m = folium.Map(location=[lat, lon], tiles=tiles,
               attr=attr, zoom_start=zoom_start)

lines = get_all_lines_1(0)

for line in lines:
    folium.PolyLine(line["coords"], weight=1, smooth_factor=0).add_to(m)

m.save("index.html")
