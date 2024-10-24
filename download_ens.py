import urllib.request
import os


typ = "jet"


def filename(x, d):
    if typ == "jet":
        return f"ec.ens_{x}.{d}.pv2000.jetaxis.nc"
    else:
        return f"ec.ens_{x}.{d}.sfc.mta.nc"


dates = [
        "2024101900"
        # "2024101900",
        # "2024101800",
        # "2024101700",
        # "2024101600",
        # "2024101500",
        # "2024101400",
        # "2024101300",
        # "2024101912",
        # "2024101812",
        # "2024101712",
        # "2024101612",
        # "2024101512",
        # "2024101412",
        # "2024101312",
        # "2024090100",
        # "2024090800",
        # "2024091500",
        # "2024092200",
        # "2024092900",
        # "2024090112",
        # "2024090812",
        # "2024091512",
        # "2024092212",
        # "2024092912",
        ]

# date = "2024092300"

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
