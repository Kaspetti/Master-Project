from netCDF4 import Dataset
import xarray as xr
import dask
import numpy as np


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

        # time_indices = rootgrp["date"][:] == time
        #
        # latitudes = rootgrp["latitude"][time_indices]
        # longitudes = rootgrp["longitude"][time_indices]

        all_lines.append([{"latitude": float(lat), "longitude": float(lon)}
                          for lat, lon in zip(latitudes, longitudes)])

        rootgrp.close()

    return all_lines


def get_all_lines_1(time):
    ds = xr.open_dataset(
        "./2024070112/ec.ens_00.2024070112.sfc.mta.nc", chunks={"time": 100}
    )

    # latitudes = ds.latitude.sel(time=time).compute()
    # longitudes = ds.longitude.sel(time=time).compute()


