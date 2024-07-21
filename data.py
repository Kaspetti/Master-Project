from netCDF4 import Dataset


def get_coords(ens_id, line_id, time):
    rootgrp = Dataset(
            f"./2024070112/ec.ens_{ens_id:02d}.2024070112.sfc.mta.nc",
            "r"
    )

    line_id_indices = rootgrp["line_id"][:] == line_id
    time_indices = rootgrp["date"][:] == time
    matched_indices = line_id_indices & time_indices

    latitudes = rootgrp["latitude"][matched_indices]
    longitudes = rootgrp["longitude"][matched_indices]

    coords = [{"latitude": float(lat), "longitude": float(lon)}
              for lat, lon in zip(latitudes, longitudes)]

    rootgrp.close()

    return coords


def get_line_amount(ens_id, time):
    rootgrp = Dataset(
            f"./2024070112/ec.ens_{ens_id:02d}.2024070112.sfc.mta.nc",
            "r"
    )

    line_count = len(set(rootgrp["line_id"][:]))
    rootgrp.close()

    return line_count


def get_all_lines(time):
    all_lines = []

    for i in range(50):
        print(i)
        rootgrp = Dataset(
                f"./2024070112/ec.ens_{i:02d}.2024070112.sfc.mta.nc",
                "r"
        )

        time_indices = rootgrp["date"][:] == time

        latitudes = rootgrp["latitude"][time_indices]
        longitudes = rootgrp["longitude"][time_indices]

        all_lines.append([{"latitude": float(lat), "longitude": float(lon)}
                          for lat, lon in zip(latitudes, longitudes)])

        rootgrp.close()

    return all_lines
