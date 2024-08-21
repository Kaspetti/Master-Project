from netCDF4 import Dataset


def get_all_lines(time):
    all_lines = []

    for i in range(50):
        rootgrp = Dataset(
            f"./2024070112/ec.ens_{i:02d}.2024070112.sfc.mta.nc",
            "r"
        )
        rootgrp.set_auto_maskandscale(False)

        start, end = -1, -1
        for i, t in enumerate(rootgrp["date"]):
            if t == time and start == -1:
                start = i
                continue

            if t != time and start != -1:
                end = i
                break

        latitudes = rootgrp["latitude"][start:end]
        longitudes = rootgrp["longitude"][start:end]
        ids = rootgrp["line_id"][start:end]

        line = {"id": 1, "coords": []}
        for id, lat, lon in zip(ids, latitudes, longitudes):
            if line["id"] != id:
                all_lines.append(line)
                line = {"id": int(id), "coords": []}

            line["coords"].append({"latitude": float(lat), "longitude": float(lon)})

        # all_lines.append([{"id": int(id), "coords": {"latitude": float(lat), "longitude": float(lon)}}
        #                   for lat, lon, id in zip(latitudes, longitudes, ids)])

        rootgrp.close()

    return all_lines


def get_all_lines_1(date):

    all_lines = []

    for i in range(50):
        rootgrp = Dataset(
            f"./2024070112/ec.ens_{i:02d}.2024070112.sfc.mta.nc",
            "r"
        )
        rootgrp.set_auto_maskandscale(False)

        start, end = -1, -1
        for i, t in enumerate(rootgrp["date"]):
            if t == date and start == -1:
                start = i
                continue

            if t != date and start != -1:
                end = i
                break

        latitudes = rootgrp["latitude"][start:end]
        longitudes = rootgrp["longitude"][start:end]
        ids = rootgrp["line_id"][start:end]

        line = {"id": 1, "coords": []}
        min_lon = 200
        max_lon = -200
        for id, lat, lon in zip(ids, latitudes, longitudes):
            if line["id"] != id:
                if max_lon - min_lon > 180:
                    line = dateline_fix(line)

                all_lines.append(line)
                min_lon = 200
                max_lon = -200
                line = {"id": int(id), "coords": []}

            line["coords"].append((float(lat), float(lon)))
            min_lon = min(min_lon, lon)
            max_lon = max(max_lon, lon)

        rootgrp.close()

    return all_lines


def dateline_fix(line):
    for i in range(len(line["coords"])):
        coord = line["coords"][i]
        if coord[1] < 0:
            line["coords"][i] = (coord[0], coord[1] + 360)

    return line
