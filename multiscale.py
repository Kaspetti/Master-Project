from __future__ import annotations
from typing import Optional, List, Any
from dataclasses import dataclass

from coords import Coord3D, CoordGeo, Coord2D
from line_reader import Line

from icosphere import icosphere
from scipy.spatial import KDTree
import numpy as np
from numpy.typing import NDArray


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


def get_enclosing_triangle(
    pt: Coord3D, tris: NDArray[np.int_], ico_pts: NDArray[Any]
) -> NDArray[np.int_]:
    """Gets which of the given triangles a point lies inside
    
    The point and the triangle points are "projected" to 2D by dropping the largest
    absolute value axis and then use barycentric coordinates to check if the point 
    is inside. If the point is found to be inside a triangle, the indices of the triangle
    points are returned, if not, an empty list is returned.

    :param pt: The point to check if lies inside a triangle.
    :param tris: A list of indices of the points of each triangle.
    :param ico_pts: All ico points of which the indices of the triangles refer to.
    :return: The indices of the points in the triangle pt is inside, or an empty list.
    """

    largest_axis = np.argmax(np.abs(pt.to_list()))
    pt_2D = pt.drop_axis(largest_axis)  # type: ignore

    for tri in tris:
        if inside_check(
            pt_2D, [tri_pt.coord_3D.drop_axis(largest_axis) for tri_pt in ico_pts[tri]]
        ):
            return tri

    return np.empty(0, np.int_)


def inside_check(pt: Coord2D, tri: List[Coord2D]) -> bool:
    a = (
        (tri[1].y - tri[2].y) * (pt.x - tri[2].x)
        + (tri[2].x - tri[1].x) * (pt.y - tri[2].y)
    ) / (
        (tri[1].y - tri[2].y) * (tri[0].x - tri[2].x)
        + (tri[2].x - tri[1].x) * (tri[0].y - tri[2].y)
    )
    b = (
        (tri[2].y - tri[0].y) * (pt.x - tri[2].x)
        + (tri[0].x - tri[2].x) * (pt.y - tri[2].y)
    ) / (
        (tri[1].y - tri[2].y) * (tri[0].x - tri[2].x)
        + (tri[2].x - tri[1].x) * (tri[0].y - tri[2].y)
    )
    c = 1 - a - b

    return 0 <= a <= 1 and 0 <= b <= 1 and 0 <= c <= 1


def subdivide_edge(
        e1: IcoPoint, e2: IcoPoint, id: int, ms_level: int
) -> IcoPoint:

    mid_point = e1.coord_3D.mid_point(e2.coord_3D)

    return IcoPoint(
        id=id,
        ms_level=ms_level,
        coord_3D=mid_point,
        coord_geo=mid_point.to_lon_lat(),
        parent_1=e1 if e1.coord_3D.x >= e2.coord_3D.x else e2,
        parent_2=e1 if e1.coord_3D.x < e2.coord_3D.x else e2
    )


def multiscale(lines: List[Line], icosphere_nu: int = 2):
    ico_vertices, faces = icosphere(icosphere_nu)  # ms level 0 icopoints
    # The icopoints at each ms level
    ico_points_ms = {
        0: np.array(
            [
                IcoPoint(
                    id=i,
                    ms_level=0,
                    coord_3D=Coord3D(coord[0], coord[1], coord[2]),
                    coord_geo=Coord3D(coord[0], coord[1], coord[2]).to_lon_lat(),
                )
                for i, coord in enumerate(ico_vertices)
            ]
        )
    }

    # KDTree of the initial points
    ms_0_kd_tree = KDTree([pt.coord_3D.to_list() for pt in ico_points_ms[0]])
    query_tris = np.array(faces)

    subdivisions = 2
    for line in lines:
        for coord in line.coords:
            coord_3D = coord.to_3D()
            closest_pt = np.array(ms_0_kd_tree.query(coord_3D.to_list(), 1)[1])

            for i in range(subdivisions):
                triangles = np.where(np.isin(query_tris, closest_pt))[0]
                tri_indices = get_enclosing_triangle(
                    coord_3D, query_tris[triangles], ico_points_ms[0]
                )
                if tri_indices.size == 0:
                    print("The fuck?")
                    exit()

                tri_pts = ico_points_ms[0][tri_indices]
                
                sub_point_1 = subdivide_edge(tri_pts[0], tri_pts[1], len(ico_points_ms[0]), i+1) 
                sub_point_2 = subdivide_edge(tri_pts[0], tri_pts[2], len(ico_points_ms[0]), i+1) 
                sub_point_3 = subdivide_edge(tri_pts[1], tri_pts[2], len(ico_points_ms[0]), i+1) 
