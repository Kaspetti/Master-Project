import urllib.request
import os
import sys


if len(sys.argv) < 2:
    print("Line type not specified. Must be one of 'mta' and 'jet'")
    exit()

typ = sys.argv[1]
if typ not in {"mta", "jet"}:
    print(f"Invalid line type '{typ}'. Must be one of 'mta' and 'jet'")
    exit()


def filename(x, d):
    if typ == "jet":
        return f"ec.ens_{x}.{d}.pv2000.jetaxis.nc"
    else:
        return f"ec.ens_{x}.{d}.sfc.mta.nc"


dates = [
        "2024101900"
        ]


for date in dates:
    folder_path = f"./data/{'jet' if typ == 'jet' else 'mta'}/{date}"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    for i in range(50):
        f = filename(f"{i:02d}", date)
        print(f"Downloading '{f}'")

        urllib.request.urlretrieve(
                f"https://iveret.gfi.uib.no/{'jetens' if typ == 'jet' else 'mtaens'}/{f}",
                f"{folder_path}/{f}"
                )
