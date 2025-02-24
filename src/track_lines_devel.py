#!/usr/bin/env python
# -*- encoding: utf-8


import numpy as np
import xarray as xr
import pandas as pd
from typing import List
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import geopandas as gpd
from shapely.geometry import LineString
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import colormaps
import seaborn as sns
from numba import jit


SUBSAMPLE = 10
TRACK_DIST_THRES = 500.0e3
SUPERSAMPLE_DX = 10.0e3
MINLEN_OVERLAP = 1000.0e3
MINCORR_OVERLAP = 0.55


def dateline_fix(coords: List[List[float]]) -> List[List[float]]:
    """Shifts a list of coordinates by 360 degrees longitude.

    Parameters
    ----------
    coords : List[CoordGeo]
        The list of coordinates to shift.

    Returns
    -------
    l : List[CoordGeo]
        The original coordinates shifted by 360 degrees longitude.
    """

    for i, coord in enumerate(coords):
        if coord[0] < 0:
            coords[i] = [coord[0] + 360, coord[1]]

    return coords


@jit
def dist_sphere(lon1, lat1, lon2, lat2, r=6.37e6):
    ''' Shortest distance on a sphere

    Calculate the great circle distance between two points on the surface of a sphere, 
    using spherical trigonometry. By default, the radius of the sphere is assumed to 
    be the Earth radius, R = 6370 km, but that can be changed via the optional 
    parameter r.

    Both the first and second points can be an array of points. If both points are
    actually arrays of points, these arrays must have compatible shapes in the sense of 
    the numpy broadcasting rules.

    Parameters
    ----------
    lon1 : float or np.ndarray
        Longitude(s) of the first point(s) in degrees.
    lat1 : float or np.ndarray
        Latitude(s) of the first point(s) in degrees.
    lon2 : float or np.ndarray
        Longitude(s) of the second point(s) in degrees.
    lat2 : float or np.ndarray
        Latitude(s) of the second point(s) in degrees.
    r : float or np.ndarray
        *Optional*. Radius of the sphere(s). Defaults to the Earth radius.
    
    Returns
    -------
    float or np.ndarray
        Distance(s) between the first and second points
    '''
        
    dlon = np.pi/180 * (lon2 - lon1)
    lat1r = np.pi/180 * lat1
    lat2r = np.pi/180 * lat2
    acos = np.sin(lat1r)*np.sin(lat2r) + np.cos(lat1r)*np.cos(lat2r)*np.cos(dlon)
    dist = r * np.arccos(np.maximum(np.minimum(acos,1.0),-1.0))

    return dist


def add_length_col(df):
    idxs = df.index.to_numpy()
    line_ids = df.line_id.to_numpy()
    lats = df.latitude.to_numpy()
    lons = df.longitude.to_numpy()

    dists = np.zeros((len(df),))

    prev_line = -1
    for idx, cur_line, cur_lat, cur_lon in zip(idxs, line_ids, lats, lons):
        if prev_line == cur_line:
            dist += dist_sphere(cur_lon, cur_lat, prev_lon, prev_lat)
            dists[idx] = dist
        else:
            dist = 0.0

        prev_line = cur_line
        prev_lat, prev_lon = cur_lat, cur_lon

    df["distance_along_line"] = dists

    return


def line_supersample(df):
    x = df.distance_along_line.to_numpy()
    lon = df.longitude.to_numpy()

    # Check whether line crosses the date line
    dlon = lon[1:] - lon[:-1]
    if np.abs(dlon).max() > 180.0:
        # ... and if so: wrap negative longitudes to positive longitudes >= 180E
        lon[lon < 0.0] += 360.0

    lat = df.latitude.to_numpy()
    # ff = df["ff@maxff"].to_numpy()
    # pt = df["pt@maxff"].to_numpy()

    xnew = np.arange(0, x[-1], SUPERSAMPLE_DX)

    return np.stack(
        (
            np.interp(xnew, x, lon),
            np.interp(xnew, x, lat),
            # np.interp(xnew, x, ff),
            # np.interp(xnew, x, pt),
        ),
        axis=-1,
    )


@jit
def find_best_match(i0ref, line0, i1ref, line1):
    best_score = -1.0
    for ioff in range(-25, 27):
        # Step 1: find matching line segment starting from i0ref, i1ref +/- ioff
        i0 = i0ref
        i1 = i1ref + ioff
        if i1 < 0 or i1 >= line1.shape[0]:
            continue
        while i0 > 0 and i1 > 0:
            dist = dist_sphere(
                line0[i0, 0], line0[i0, 1], line1[i1, 0], line1[i1, 1]
            )
            i0 -= 1
            i1 -= 1
            if dist > TRACK_DIST_THRES:
                break

        i0start, i1start = i0 + 1, i1 + 1

        i0 = i0ref
        i1 = i1ref + ioff
        while i0 < line0.shape[0] and i1 < line1.shape[0]:
            dist = dist_sphere(
                line0[i0, 0], line0[i0, 1], line1[i1, 0], line1[i1, 1]
            )
            i0 += 1
            i1 += 1
            if dist > TRACK_DIST_THRES:
                break

        i0stop, i1stop = i0, i1

        overlap = (i0stop - i0start) * SUPERSAMPLE_DX
        if overlap < MINLEN_OVERLAP:
            continue

        slc0 = slice(i0start, i0stop)
        slc1 = slice(i1start, i1stop)

        # Step 2: calculate correlation of lon, lat, ff, pt along the matching segment
        corrs = 0.0
        for n in range(line0.shape[1]):
            corrs += np.corrcoef(line0[slc0, n], line1[slc1, n])[0, 1]

        corrs /= line0.shape[1]

        if corrs < MINCORR_OVERLAP:
            continue

        # Step 3: Choose best match based on combination of length of overlap and correlations
        #
        # Rationale behind the formula by examples:
        #  - A near-perfect correlation of 0.95 contributes a score of 10, as would a 10000 km long line
        #  - A resonable correlation of 0.80 contributes a score of 4.0, as would a 4000km long line
        #  - A minimum correlation of 0.55 contributes a score of 2.0, similar to a line of minimum length 2000 km
        # Thus, in some sense, correlation and length of overlap weigh in about equally in the overall score
        score = 1.0 / (1.05 - corrs) + overlap / 1.0e6

        # print(ioff, i0start, i0stop, i1start, i1stop, overlap, corrs, score)

        if score > best_score:
            best_score = score
            bestslc0 = slc0
            bestslc1 = slc1

    # No overlap found, return empty slice
    if best_score < 0:
        bestslc0 = slice(1, 0)
        bestslc1 = bestslc0

    return line0[bestslc0, :], line1[bestslc1, :]


def track_lines(df0, df1, debug=False):
    df0_firstidx = df0.index[0]
    df1_firstidx = df1.index[0]

    df0s = df0[::SUBSAMPLE]
    lidx0s = df0s.line_id.to_numpy()
    lats0s = df0s.latitude.to_numpy()
    lons0s = df0s.longitude.to_numpy()

    lidx1 = df1.line_id.to_numpy()
    lats1 = df1.latitude.to_numpy()
    lons1 = df1.longitude.to_numpy()

    dists = dist_sphere(
        lons0s[:, np.newaxis],
        lats0s[:, np.newaxis],
        lons1[np.newaxis, :],
        lats1[np.newaxis, :],
    )
    match = np.argmin(dists, axis=1)
    dists = dists.min(axis=1)

    line_match = {lidx: dict() for lidx in lidx0s}
    for pidx0s, pidx1 in enumerate(match):
        if dists[pidx0s] > TRACK_DIST_THRES:
            continue

        line_match_ = line_match[lidx0s[pidx0s]]
        lidx1_ = lidx1[pidx1]

        if lidx1_ in line_match_ and line_match_[lidx1_][0] < dists[pidx0s]:
            continue

        pidx0 = pidx0s * SUBSAMPLE
        line_match_[lidx1_] = (
            dists[pidx0s],
            pidx0 + df0_firstidx,
            pidx1 + df1_firstidx,
        )

    if debug:
        plt.figure(figsize=(10, 10), dpi=96)
        # m = Basemap(projection='npstere', lon_0=-50, resolution='c', boundinglat=25)
        m = Basemap(projection="spstere", lon_0=0, resolution="c", boundinglat=-25)

        for lidx0 in set(lidx0s):
            line = df0[df0.line_id == lidx0]
            m.plot(
                line.longitude.to_numpy(),
                line.latitude.to_numpy(),
                "k",
                linewidth=2,
                zorder=3,
                latlon=True,
            )
        for lidx1 in set(lidx1):
            line = df1[df1.line_id == lidx1]
            m.plot(
                line.longitude.to_numpy(),
                line.latitude.to_numpy(),
                "b",
                linewidth=2,
                zorder=3,
                latlon=True,
            )

    N = 0
    matches = []
    overlaps = []
    l1super_cache = {}
    for lidx0, matching_lines1 in line_match.items():
        if len(matching_lines1) > 0:
            l0super = line_supersample(df0[df0.line_id == lidx0])

        for lidx1, (dist, pidx0, pidx1) in matching_lines1.items():
            if not lidx1 in l1super_cache:
                l1super_cache[lidx1] = line_supersample(df1[df1.line_id == lidx1])

            l1super = l1super_cache[lidx1]

            i0ref = int(df0.loc[pidx0, "distance_along_line"] / SUPERSAMPLE_DX)
            i1ref = int(df1.loc[pidx1, "distance_along_line"] / SUPERSAMPLE_DX)
            l0overlap, l1overlap = find_best_match(i0ref, l0super, i1ref, l1super)

            # Order of points may be reversed, so if no good match is found so far, try reverse order
            if l0overlap.shape[0] == 0:
                l0overlap, l1overlap = find_best_match(
                    i0ref, l0super, l1super.shape[0] - 1 - i1ref, l1super[::-1, :]
                )

            if l0overlap.shape[0] > 0:
                N += l0overlap.shape[0]
                overlaps.append(np.concatenate((l0overlap, l1overlap), axis=1))

                matches.append((lidx0, lidx1))

                if debug:
                    m.plot(
                        l0overlap[:, 0],
                        l0overlap[:, 1],
                        "r",
                        linewidth=4,
                        zorder=2,
                        latlon=True,
                    )
                    m.plot(
                        l1overlap[:, 0],
                        l1overlap[:, 1],
                        "r",
                        linewidth=4,
                        zorder=2,
                        latlon=True,
                    )

    if debug:
        m.drawcoastlines()
        plt.tight_layout()
        plt.savefig(f'track_lines_debug_{df0.iloc[0].date.strftime("%Y%m%d_%H")}.pdf')
        plt.close()

    return matches, overlaps, N


def update_graph(graph, cur, nxt, line_ids, matches):
    # First date, or first date after a discontinuity
    if len(graph["dates"]) == 0 or not graph["dates"][-1] == cur:
        graph["dates"].append(cur)
        prv = None
    else:
        prv = graph["dates"][-2]

    graph["dates"].append(nxt)

    fwd_matches = {}
    bwd_matches = {}
    for id0, id1 in matches:
        if id0 not in fwd_matches:
            fwd_matches[id0] = [
                id1,
            ]
        else:
            fwd_matches[id0].append(id1)

        if id1 not in bwd_matches:
            bwd_matches[id1] = [
                id0,
            ]
        else:
            bwd_matches[id1].append(id0)

    graph["forward"][cur] = fwd_matches
    graph["backward"][nxt] = bwd_matches

    # Backward matches for next time step saved, from here on bwd_matches is for current time step
    bwd_matches = graph["backward"].get(cur, {})

    graph["genesis"][cur] = []
    graph["lysis"][cur] = []
    graph["single"][cur] = []
    for line_id in line_ids:
        if not line_id in fwd_matches:
            if not line_id in bwd_matches:
                graph["single"][cur].append(line_id)
            else:
                graph["lysis"][cur].append(line_id)
        elif not line_id in bwd_matches:
            graph["genesis"][cur].append(line_id)
        # else: has backwards and forwards matches, no need to record.

    return graph


ipath = "/Data/gfi/spengler/csp001/jetaxis_v3"

periods = [
    (yr, mon) for yr in range(1979, 2023) for mon in range(1, 13)
]  # must be sequential for tracking across boundaries
periodstr = "1979-2022"
# periods = [(1979,1), ]
# periodstr = '197901'

debug_plot = False
debug_one = None  # (1979,1,14,0)

if __name__ == "__main__":
    df = xr.open_dataset(
        # f"./data/jet/2024101900/ec.ens_00.2024101900.pv2000.jetaxis.nc"
        f"./data/mta/2024101900/ec.ens_00.2024101900.sfc.mta.nc"
    ).to_dataframe()
    add_length_col(df)

    available_ids = set(range(1, 1000))
    time_offset = 0
    prev_df = pd.DataFrame()
    geometry = []
    ids = []
    while time_offset <= 240:
        print(time_offset)
        date = np.datetime64(f"2024-10-19T00:00") + np.timedelta64(time_offset,'h')
        df0 = df[df.date == date].copy()
        df0.loc[:, "old_id"] = df0.line_id

        if prev_df.empty:
            line_ids = df0.line_id.unique()
            for id in line_ids:
                available_ids.remove(id)
        else:
            matches, _, _ = track_lines(prev_df, df0)
            id_mapping = {prev_id: next_id for next_id, prev_id in matches}

            for id in df0.line_id.unique():
                if id in id_mapping:
                    df0.loc[df0["old_id"] == id, 'line_id'] = id_mapping[id]
                else:
                    df0.loc[df0["old_id"] == id, 'line_id'] = available_ids.pop()

        df.loc[df.date == date, 'old_id'] = df0['old_id']
        df.loc[df.date == date, 'line_id'] = df0['line_id']

        prev_df = df0
        for id in prev_df.line_id.unique():
            coords = prev_df[prev_df['line_id'] == id][['longitude', 'latitude']].to_numpy()
            if max(coords[:, 0]) - min(coords[:, 0]) > 180:
                coords = dateline_fix(coords)

            geometry.append(LineString(coords))
            ids.append(id)

        if time_offset < 72:
            time_offset += 3
        else:
            time_offset += 6

    # # Create contingency table
    # all_ids = sorted(set(
    #     list(df['old_id'].unique()) + 
    #     list(df['line_id'].unique())
    # ))
    #
    # contingency = pd.DataFrame(
    #     0, 
    #     index=all_ids,
    #     columns=all_ids
    # )    
    # grouped = df.groupby(['date', 'old_id', 'line_id']).size()
    # 
    # for (date, old_id, line_id), _ in grouped.items():  # type: ignore
    #     contingency.loc[old_id, line_id] += 1
    #
    # row_sums = contingency.sum(axis=1)
    # col_sums = contingency.sum(axis=0)
    # 
    # filtered_contingency = contingency.loc[
    #     row_sums > 1,  # Filter rows
    #     col_sums > 1   # Filter columns
    # ]
    #
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(filtered_contingency, 
    #             cmap='YlOrRd',
    #             annot=False,
    #             fmt='d', 
    #             cbar_kws={'label': 'Number of Lines'},
    #             square=True)
    # 
    # plt.title('Line ID Transitions Heatmap')
    # plt.xlabel('New Line ID')
    # plt.ylabel('Old Line ID')
    # 
    # # Rotate labels if there are many IDs
    # plt.xticks(rotation=45, ha='right')
    # plt.yticks(rotation=0)
    # 
    # # Adjust layout to prevent label cutoff
    # plt.tight_layout()
    # plt.show()


    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="black") # type: ignore
    ax.add_feature(cfeature.OCEAN, facecolor="lightgrey")   # type: ignore
    ax.add_feature(cfeature.COASTLINE, edgecolor="black")   # type: ignore
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor="darkgrey")    # type: ignore

    df_id = pd.DataFrame(ids, columns=["id"])   # type: ignore
    gdf = gpd.GeoDataFrame(df_id, geometry=geometry, crs="EPSG:4326")   # type: ignore

    cmap = colormaps["tab20"]
    norm = plt.Normalize(gdf['id'].min(), gdf['id'].max())  # type: ignore

    gdf.plot(ax=ax, transform=ccrs.PlateCarree(), linewidth=1, color=cmap(norm(gdf["id"])))
    plt.show()
