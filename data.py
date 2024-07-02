from netCDF4 import Dataset


def get_coords(line_id, time):
    rootgrp = Dataset("./2024070112/ec.ens_00.2024070112.sfc.mta.nc", "r")

    line_id_indices = rootgrp["line_id"][:] == line_id
    time_indices = rootgrp["date"][:] == time
    matched_indices = line_id_indices & time_indices

    latitudes = rootgrp["latitude"][matched_indices]
    longitudes = rootgrp["longitude"][matched_indices]

    coords = [{"latitude": float(lat), "longitude": float(lon)}
              for lat, lon in zip(latitudes, longitudes)]

    rootgrp.close()

    return coords
