from netCDF4 import Dataset
import xarray as xr
import dask
import numpy as np


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


def get_all_lines_1(time):
    ds = xr.open_dataset(
        "./2024070112/ec.ens_00.2024070112.sfc.mta.nc", chunks={"time": 100}
    )

    # latitudes = ds.latitude.sel(time=time).compute()
    # longitudes = ds.longitude.sel(time=time).compute()
