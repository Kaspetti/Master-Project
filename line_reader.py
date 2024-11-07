from typing import Literal, List
from dataclasses import dataclass

from coords import CoordGeo

import numpy as np
import xarray as xr


@dataclass
class Line:
    """A line.

    A line is a collection of ordered points.

    Attributes:
        id (str): The unique identifier of the line.
            The id is created by combining the ensemble number
            the line is part of and the line's id in that ensemble.
            'ensemble_nr|line_id'
        coords (List[CoordGeo]): A list of the coordinates of the line.
    """

    id: str
    coords: List[CoordGeo]


def get_all_lines(
    start: str, time_offset: int, line_type: Literal["mta", "jet"]
) -> List[Line]:
    """Reads all lines from a NETCDF4 file and returns them.

    Reads a group of NETCDF4 files of a specific format and returns the lines
    in the files which has the specified time offset.
    The file names must be of the following format:
        ec.ens_{ens_id}.{start}.{sfc|pv2000}.{mta|jetaxis}.nc
    where ens_id is between 00 and 50.
    The files must be in the following path:
        ./data/{mta|jet}/{start}
    The file needs to have the following attributes:
        longitude
        latitude
        date
    where date is the time offset in hours from the start time.
    The function expects 50 files, or 50 ensembles, to be present
    in the folder './date/{mta|jet}/{start}'

    :param start: The start time of the computation.
        Must be of the format: YYYYMMDDTT where TT is one of
        00 or 12.
    :param time_offset: The time offset from the start to get the lines.
        The offset is given in hours from the start time.
    :param line_type: The type of the lines to get.
        Currently supported line types are 'mta' and 'jet'.
    :return: A list of the lines from the 50 ensembles at the time offset.
    """
    all_lines = []

    start_time = np.datetime64(
        f"{start[0:4]}-{start[4:6]}-{start[6:8]}T{start[8:10]}:00:00"
    )

    for i in range(50):
        base_path = f"./data/{line_type}/{start}/"
        file_path = f"ec.ens_{i:02d}.{start}.sfc.mta.nc"

        if line_type == "jet":
            file_path = f"ec.ens_{i:02d}.{start}.pv2000.jetaxis.nc"
        full_path = base_path + file_path

        ds = xr.open_dataset(full_path)
        date_ds = ds.where(
            ds.date == start_time + np.timedelta64(time_offset, "h"), drop=True
        )

        grouped_ds = list(date_ds.groupby("line_id"))

        for id_, line in grouped_ds:
            coords = [
                CoordGeo(lon, lat)
                for lon, lat in zip(line.longitude.values, line.latitude.values)
            ]

            if max(line.longitude.values) - min(line.longitude.values) > 180:
                coords = dateline_fix(coords)

            all_lines.append(Line(id=f"{i}|{int(id_)}", coords=coords))

    return all_lines


def dateline_fix(coords: List[CoordGeo]) -> List[CoordGeo]:
    """Shifts a list of coordinates by 360 degrees longitude.

    :param coords: The list of coordinates to shift.
    :return: The original coordinates shifted by 360 degrees longitude.
    """
    for i, coord in enumerate(coords):
        if coord.lon < 0:
            coords[i] = CoordGeo(coord.lon + 360, coord.lat)

    return coords
