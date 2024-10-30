from __future__ import annotations
from typing import Optional, List
from dataclasses import dataclass

from coords import Coord3D, CoordGeo
from line_reader import Line

from icosphere import icosphere
from scipy.spatial import KDTree
import numpy as np


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
    parent_1: Optional[IcoPoint] = None
    parent_2: Optional[IcoPoint] = None


def get_enclosing_triangle(pt: Coord3D,
                           tris: List[List[int]],
                           ico_pts: List[IcoPoint]):
    pass


def inside_check(pt: Coord3D, tri: List[Coord3D]) -> bool:
    pass


def multiscale(lines: List[Line], icosphere_nu: int = 2):
    ico_vertices, faces = icosphere(icosphere_nu)   # ms level 0 icopoints
    # The icopoints at each ms level
    ico_points_ms = {
        0: [IcoPoint(
                id=i,
                ms_level=0,
                coord_3D=Coord3D(coord[0], coord[1], coord[2]),
                coord_geo=Coord3D(coord[0], coord[1], coord[2]).to_lon_lat())
            for i, coord in enumerate(ico_vertices)]
    }

    # KDTree of the initial points
    ms_0_kd_tree = KDTree([pt.coord_3D.to_list() for pt in ico_points_ms[0]])
    query_tris = np.array(faces)

    for line in lines:
        for coord in line.coords:
            coord_3D = coord.to_3D()
            closest_pt = np.array(ms_0_kd_tree.query(coord_3D.to_list(), 1)[1])
            triangles = np.where(np.isin(query_tris, closest_pt))[0]

            tri = get_enclosing_triangle(coord_3D, query_tris[triangles], ico_points_ms[0])
            # print(tri)
