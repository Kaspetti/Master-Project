from typing import Literal
import urllib.request
import os
import sys

def download(simstart: str, line_type: Literal["jet", "mta"]):
    folder_path = f"./data/{line_type}/{simstart}"
    if os.path.exists(folder_path):
        return

    os.mkdir(folder_path)
    for i in range(50):
        f = filename(f"{i:02d}", simstart, line_type)
        print(f"Downloading '{f}'")

        urllib.request.urlretrieve(
            f"https://iveret.gfi.uib.no/{'jetens' if line_type == 'jet' else 'mtaens'}/{f}",
            f"{folder_path}/{f}",
        )


def filename(x, d, t):
    if t == "jet":
        return f"ec.ens_{x}.{d}.pv2000.jetaxis.nc"
    else:
        return f"ec.ens_{x}.{d}.sfc.mta.nc"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Line type not specified. Must be one of 'mta' and 'jet'")
        exit()

    typ = sys.argv[1]
    if typ not in {"mta", "jet"}:
        print(f"Invalid line type '{typ}'. Must be one of 'mta' and 'jet'")
        exit()


    manual_date = None
    if len(sys.argv) >= 3:
        manual_date = sys.argv[2]


    dates = ["2024101900"]

    if manual_date:
        dates = [manual_date]

    for date in dates:
        folder_path = f"./data/{'jet' if typ == 'jet' else 'mta'}/{date}"
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        for i in range(50):
            f = filename(f"{i:02d}", date, typ)
            print(f"Downloading '{f}'")

            urllib.request.urlretrieve(
                f"https://iveret.gfi.uib.no/{'jetens' if typ == 'jet' else 'mtaens'}/{f}",
                f"{folder_path}/{f}",
            )


