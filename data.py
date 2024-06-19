from netCDF4 import Dataset


def get_coords(line_id, time):
    rootgrp = Dataset("./ec.ensctrl.2024051112.sfc.mta.nc", "r")

    coords = [{"latitude": float(rootgrp["latitude"][i]),
               "longitude": float(rootgrp["longitude"][i])}
              for i in rootgrp["index"]
              if rootgrp["line_id"][i] == line_id
              and rootgrp["date"][i] == time]

    rootgrp.close()

    return coords
