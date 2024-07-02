import urllib.request
import os


def filename(x, d):
    return f"ec.ens_{x}.{d}.sfc.mta.nc"


date = "2024070112"

if not os.path.exists(f"./{date}"):
    os.mkdir(f"./{date}")

for i in range(50):
    f = filename(f"{i:02d}", date)
    print(f"Downloading '{f}'")

    urllib.request.urlretrieve(
            f"https://iveret.gfi.uib.no/mtaens/{f}",
            f"./{date}/{f}"
    )
