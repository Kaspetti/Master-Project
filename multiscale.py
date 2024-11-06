from __future__ import annotations
from typing import Optional, List, Any, Tuple
from dataclasses import dataclass

from coords import Coord3D, CoordGeo, Coord2D
from line_reader import Line

from icosphere import icosphere
from scipy.spatial import KDTree
import numpy as np
from numpy.typing import NDArray
from alive_progress import alive_it
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


@dataclass
class IcoPoint:
    """A point on an icosphere.

    Attributes:
        id (int): The unique id of the icopoint.
            Usually the index of the point in an array of icopoints.
        parent_1 (Optional[IcoPoint]): First point of the edge
            the current point was subdivided from.
        parent_2 (Optional[IcoPoint]): Second point of the edge
            the current point was subdivided from.
        ms_level (int): The multiscale level of the icopoint
            The level of the the initial icopoints on the sphere is 0,
            increasing by 1 for each subdivision.
        coord_3D (Coord3D): The 3D coordinate of the point.
            Represented as a list of three floats: [x, y, z].
        coord_geo (CoordGeo): The longitude and latitude of the point.
            Represented as a list of two floats: [longitude, latitude].
    """

    id: int
    ms_level: int
    coord_3D: Coord3D
    coord_geo: CoordGeo
    parent_1: int
    parent_2: int


def get_enclosing_triangle(line_point: Coord3D,
                           ico_points: NDArray[Any]) -> NDArray[Any]:
    '''
    Gets the vertices on the icosphere forming a triangle enclosing the
    line point.

    Parameters
    ----------
    line_point : List[float] of shape (3,)
        The 3D point to get the enclosing triangle of (x, y, z)
    ico_points : List[IcoPoint] of shape (n, 3) with n = verts in icosphere
        All the vertices of the ico sphere

    Returns
    -------
    closest : List[IcoPoint] of shape (3,)
        The 3 closest points on the icosphere to the
        line point. These 3 points form the triangle enclosing the point
    '''

    # Get the three closest points to the line point
    ico_points_sorted = np.array(
            sorted(ico_points,
                   key=lambda pt: line_point.dist(pt.coord_3D))
            )

    tri = np.array([point.coord_3D.to_list() for point in ico_points_sorted[:6]])

    if inside_check(line_point.to_ndarray(), tri):
        return ico_points_sorted[:3]

    tri[2] = ico_points_sorted[3].coord_3D.to_list()
    if inside_check(line_point.to_ndarray(), tri):
        return ico_points_sorted[[0, 1, 3]]

    tri[2] = ico_points_sorted[4].coord_3D.to_list()
    if inside_check(line_point.to_ndarray(), tri):
        return ico_points_sorted[[0, 1, 4]]

    return ico_points_sorted[[0, 1, 2]]


def inside_check(pt: NDArray[np.float_], tri: NDArray[np.float_]) -> bool:
    """
    Checks if a point lies "inside" a triangle by checking if the ray
    from the point to origo passes through the triangle.

    Solution gotten from:
        https://stackoverflow.com/questions/42740765/intersection-between-line-and-triangle-in-3d

    Parameters
    ----------
    pt : List[float] of shape (3,)
        The point to check if lies inside the triangle
    tri : List[List[float]] of shape (3, 3)
        The points making up the triangle

    Returns
    -------
    True if the line passes through, False otherwise
    """

    # Two far away points on the line going from origo to the line point
    q0 = np.multiply(pt, 10)
    q1 = np.multiply(pt, -10)

    # Check if the line passes through the plane created by the three
    # points in the triangle
    v0 = signed_volume(q0, tri[0], tri[1], tri[2])
    v1 = signed_volume(q1, tri[0], tri[1], tri[2])

    if v0 * v1 < 0:
        # Check if the line passes through the triangle
        v2 = signed_volume(q0, q1, tri[0], tri[1])
        v3 = signed_volume(q0, q1, tri[1], tri[2])
        v4 = signed_volume(q0, q1, tri[2], tri[0])

        if v2 * v3 > 0 and v2 * v4 > 0:
            return True

    return False


def signed_volume(a: NDArray[np.float_], b: NDArray[np.float_],
                  c: NDArray[np.float_], d: NDArray[np.float_]) -> float:
    """
    Returns the signed volume of the tetrahedron formed by the points
    a, b, c, d.
    """

    return (1.0/6.0) * np.dot(np.cross(b-a, c-a), d-a)


def subdivide_triangle(tri_pts: NDArray[Any]) -> List[Tuple[Tuple[int, int],
                                                              List[float]]]:
    '''
    Subdivides a triangle once creating three new points, one
    on the midpoint on all three sides of the triangle

    Parameters
    ----------
    ps : List[IcoPoint] of shape (3,)
        The points making up the triangle

    Returns
    -------
    subdivided_points : List[Tuple[Tuple[int, int], List[float]]] of shape (3,)
        A list of tuples containing the id of the parents of the new point,
        aswell as the point itself
    '''

    ps = [point.coord_3D for point in tri_pts]
    ids = [point.id for point in tri_pts]

    p_1 = [(ps[0].x + ps[1].x) / 2,
           (ps[0].y + ps[1].y) / 2,
           (ps[0].z + ps[1].z) / 2]

    p_2 = [(ps[0].x + ps[2].x) / 2,
           (ps[0].y + ps[2].y) / 2,
           (ps[0].z + ps[2].z) / 2]

    p_3 = [(ps[1].x + ps[2].x) / 2,
           (ps[1].y + ps[2].y) / 2,
           (ps[1].z + ps[2].z) / 2]

    data = [
        ((min(ids[0], ids[1]), max(ids[0], ids[1])), p_1),
        ((min(ids[0], ids[2]), max(ids[0], ids[2])), p_2),
        ((min(ids[1], ids[2]), max(ids[1], ids[2])), p_3),
    ]

    return data # type: ignore


def normalize_point(p: NDArray[np.float_]) -> NDArray[np.float_]:
    """
    Normalizes the given point making its length equal 1

    Parameters
    ----------
    p : List[List[float]] of shape (n,) with n = dimension of the point
        The point to normalize

    Return
    ------
    normalized_p : List[List[float]] of shape (n,) with
                    n = dimension of the point
        The normalized point
    """

    return p / np.linalg.norm(p)


def multiscale(lines: List[Line],
               subdivs: int):
    """
    Performs a multiscale subdivision of the icosahedron, returning the new
    subdivided points aswell as the lines represented at the different
    subdivision levels.

    Subdivision is performed locally around the lines to prevent too many
    points created

    Parameters
    ----------
    ico_points : List[IcoPoint]
        A list of the vertices of the icosahedron before subdivision
    lines : List[Line]
        A list of the lines which the subdivision and multiscale will
        occur to and around
    subdivs : int
        The amount of subdivision to do. The level of the multiscale

    Returns
    -------
    Two data structures. First being the new list of vertices on the
    icosahedron after subdivision. The second being a data structure
    containing the representation of the lines at different scales.
    """

    ico_points, faces = icosphere(2)

    points_at_level = {}
    for i in range(subdivs+1):
        points_at_level[i] = set()

    # Edges which are already subdivided. Used to check if
    # a subdivided point should be added to ico_points_ms or if
    # it is already present
    subdivided_edges = {}

    ico_points_ms = {}
    for i, pt in enumerate(ico_points):
        coord_3D = Coord3D(x=pt[0], y=pt[1], z=pt[2])
        ico_points_ms[i] = IcoPoint(
            id=i,
            parent_1=-1,
            parent_2=-1,
            ms_level=0,
            coord_3D=coord_3D,
            coord_geo=coord_3D.to_lon_lat()
        )
        points_at_level[0].add(i)

    track_points_ms = {}
    points_ms_0 = np.array([ico_points_ms[id] for id in points_at_level[0]])
    ms_0_kd_tree = KDTree([point.coord_3D.to_list() for point in points_ms_0])

    for line in lines:
        print(f"Processing line: {line.id}")

        track_points_ms[line.id] = {}
        # Create track_points_ms for lowest subdivision
        for k in range(subdivs+1):
            track_points_ms[line.id][k] = {}

        for i, coord in enumerate(line.coords):
            # This does not have to be computed for each coord.. TODO
            coord_3D = coord.to_3D()
            closest = ms_0_kd_tree.query(coord_3D.to_list(), 6)
            query_points = points_ms_0[closest[1]]

            closest = sorted(query_points,
                             key=lambda pt: coord_3D.dist(pt.coord_3D))[0]

            dist1 = coord_3D.dist(closest.coord_3D)
            if closest.id in track_points_ms[line.id][0]:
                pt, dist2 = track_points_ms[line.id][0][closest.id]

                if dist1 < dist2:
                    track_points_ms[line.id][0][closest.id] = (coord_3D, dist1)
            else:
                track_points_ms[line.id][0][closest.id] = (coord_3D, dist1)

            for j in range(subdivs):
                tri = get_enclosing_triangle(coord_3D, query_points)
                sub = subdivide_triangle(tri)

                next_query = tri
                for (parents, pt) in sub:
                    if parents in subdivided_edges:
                        next_query = np.append(next_query, ico_points_ms[subdivided_edges[parents]])
                        continue

                    id = len(ico_points_ms)
                    coord_3D = Coord3D(x=pt[0], y=pt[1], z=pt[2])
                    ico_point = IcoPoint(
                        id=id,
                        parent_1=parents[0],
                        parent_2=parents[1],
                        ms_level=j+1,
                        coord_3D=coord_3D,
                        coord_geo=coord_3D.to_lon_lat()
                    )

                    next_query = np.append(next_query, ico_point)   # type: ignore
                    subdivided_edges[parents] = id
                    ico_points_ms[id] = ico_point
                    points_at_level[j+1].add(id)

                    # Create track points for this subdivision
                    closest = sorted(query_points,
                                     key=lambda pt: coord_3D.dist(pt.coord_3D))[0]

                    dist1 = coord_3D.dist(closest.coord_3D)
                    if closest.id in track_points_ms[line.id][j+1]:
                        pt, dist2 = track_points_ms[line.id][j+1][closest.id]

                        if dist1 < dist2:
                            track_points_ms[line.id][j+1][closest.id] = (coord_3D, dist1)
                    else:
                        track_points_ms[line.id][j+1][closest.id] = (coord_3D, dist1)

                query_points = next_query

    return ico_points_ms, track_points_ms
