import urllib.request
import os


def filename(x, d):
    return f"ec.ens_{x}.{d}.sfc.mta.nc"


dates = [
        "2024080212",
        "2024080912",
        "2024081612",
        "2024082312",
        "2024083012",
        "2024092312",
        "2024092412",
        "2024092512",
        "2024092612",
        "2024092712",
        "2024092812",
        "2024092912",
        ]

# date = "2024092300"

for date in dates:
    if not os.path.exists(f"./data/{date}"):
        os.mkdir(f"./data/{date}")

    for i in range(50):
        f = filename(f"{i:02d}", date)
        print(f"Downloading '{f}'")

        urllib.request.urlretrieve(
                f"https://iveret.gfi.uib.no/mtaens/{f}",
                f"./data/{date}/{f}"
                )
