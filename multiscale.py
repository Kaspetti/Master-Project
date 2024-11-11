from __future__ import annotations
from typing import List, Dict, Tuple
from dataclasses import dataclass
from itertools import combinations

from coords import Coord3D, CoordGeo
from line_reader import Line

from icosphere import icosphere
from scipy.spatial import KDTree
import numpy as np
from numpy.typing import NDArray
from alive_progress import alive_it

@dataclass
class IcoPoint:
    """A point on an icosphere.

    Attributes
    ----------
    id : int 
    The unique id of the icopoint.
        Usually the index of the point in an array of icopoints.
    parent_1 : int
        Id of the first point of the edge the current point was subdivided from.
    parent_2 : int
        If of the second point of the edge the current point was subdivided from.
    ms_level : int
        The multiscale level of the icopoint
        The level of the the initial icopoints on the sphere is 0,
        increasing by 1 for each subdivision.
    coord_3D : Coord3D
        The 3D coordinate of the point.
    coord_geo : CoordGeo
        The longitude and latitude of the point.
    """

    id: int
    ms_level: int
    coord_3D: Coord3D
    coord_geo: CoordGeo
    parent_1: int = -1
    parent_2: int = -1


def inside_check(pt: NDArray[np.float64], tri: NDArray[np.float64]) -> bool:
    """Checks if a point lies inside a triangle.

    Uses barycentric coordinates to check if a point lies inside
    the triangle formed by three other points. Expects the points to
    be two dimensional.

    Parameters
    ----------
    pt : NDArray[np.float64] of shape (2,)
        The point to check if lies inside the triangle.
    tri : NDArray[np.float64] of shape (3, 2)
        A list of the three points making up the triangle.

    Returns
    -------
    is_inside : bool
        True if the point is inside, False otherwise.
    """

    a = (
        (tri[1][1] - tri[2][1]) * (pt[0] - tri[2][0])
        + (tri[2][0] - tri[1][0]) * (pt[1] - tri[2][1])
    ) / (
        (tri[1][1] - tri[2][1]) * (tri[0][0] - tri[2][0])
        + (tri[2][0] - tri[1][0]) * (tri[0][1] - tri[2][1])
    )
    b = (
        (tri[2][1] - tri[0][1]) * (pt[0] - tri[2][0])
        + (tri[0][0] - tri[2][0]) * (pt[1] - tri[2][1])
    ) / (
        (tri[1][1] - tri[2][1]) * (tri[0][0] - tri[2][0])
        + (tri[2][0] - tri[1][0]) * (tri[0][1] - tri[2][1])
    )
    c = 1 - a - b

    return 0 <= a <= 1 and 0 <= b <= 1 and 0 <= c <= 1


def multiscale(
        lines: List[Line], subdivs: int
) -> Tuple[Dict[int, IcoPoint], Dict[str, Dict[int, Dict[int, Tuple[int, float]]]]]:
    """Performs a multiscale representation on the lines provided.
    
    The multiscale approach starts by representing the world as an icosphere.
    It then subdivides this icosphere around the points on the line in order to
    get a higher resolution only where its needed.

    How many subdivisions is decided by the subdivs parameter and a representation
    of each line on each subdivision level is also returned.

    Parameters
    ----------
    lines : List[Line]
        The lines to perform the multiscale around.
    subdivs : int
        How many subdivisions will be performed.

    Returns
    -------
    ico_points_ms : Dict[int, IcoPoint]
        The dicitonary of all the icopoints after subdivison.
        The key is the same as the icopoint id.
    line_points_ms : Dict[str, Dict[int, Dict[int, Tuple[int, float]]]]
        A dictionary containing the representation of each line at each subdiv level.
        The structure is as follows:
            line_id
                ms_level
                    ico_point_1
                        closest_line_point, dist
                    ico_point_2
                        closest_line_point, dist
        Each ico point under the ms_level dictionary is any ico point which
        is the closest ico point to any point in the line.
    """

    ico_verts, _ = icosphere(2)
    ico_points_ms = {}
    subdivided_edges = {}
    line_points_ms = {}

    for i, pt in enumerate(ico_verts):
        coord_3D = Coord3D(x=pt[0], y=pt[1], z=pt[2])
        ico_pt = IcoPoint(
            id=i,
            ms_level=0,
            coord_3D=coord_3D,
            coord_geo=coord_3D.to_lon_lat()
        )
        ico_points_ms[i] = ico_pt

    ico_points_base_kd = KDTree([pt.coord_3D.to_list() for pt in ico_points_ms.values()])


    outside = 0
    outside_after_flip = 0
    bar = alive_it(lines, title="Performing multiscale")
    for line in bar:
        line_points_ms[line.id] = {}
        for ms_level in range(0, subdivs+1):
            line_points_ms[line.id][ms_level] = {}

        for i, pt in enumerate(line.coords):
            coord_3D = pt.to_3D().to_list()

            closest_dists, closest_idx = ico_points_base_kd.query(coord_3D, 4)
            closest_points = [ico_points_ms[closest_idx[0]],
                              ico_points_ms[closest_idx[1]],
                              ico_points_ms[closest_idx[2]]]

            # Get two of the edges and the normal of the triangle closest_3 make up
            tri_edge_1 = closest_points[1].coord_3D - closest_points[0].coord_3D
            tri_edge_2 = closest_points[2].coord_3D - closest_points[0].coord_3D
            tri_normal = np.cross(tri_edge_1.to_list(), tri_edge_2.to_list())

            # Project the point onto the plane formed by the three points
            n = np.dot(closest_points[0].coord_3D.to_list(), tri_normal) 
            d = np.dot(coord_3D, tri_normal)
            scale = n / d
            projected_pt_list = np.multiply(coord_3D, scale)
            projected_pt = Coord3D(projected_pt_list[0], projected_pt_list[1], projected_pt_list[2])
            
            # Check if point is inside the triangle
            u = tri_edge_1.to_ndarray() / np.linalg.norm(tri_edge_1.to_ndarray())
            v = np.cross(tri_normal, u)
            v = v / np.linalg.norm(v)
            transform = np.vstack([u, v])

            points = np.vstack([[pt.coord_3D.to_ndarray() for pt in closest_points], projected_pt_list])
            points_2D = np.dot(points - points[0], transform.T)
            is_inside = inside_check(points_2D[3], points_2D[:3])
            if not is_inside:
                outside += 1
                flip_point_projected = np.dot(ico_points_ms[closest_idx[3]].coord_3D.to_ndarray() - points[0], transform.T)
                closest_idx[2] = closest_idx[3]
                points_2D[2] = flip_point_projected

                is_inside = inside_check(points_2D[3], points_2D[:3])
                if not is_inside:
                    outside_after_flip += 1

            # Check if the current point is the closest of any other line point to its closest ico point
            if closest_idx[0] not in line_points_ms[line.id][0]:
                line_points_ms[line.id][0][closest_idx[0]] = (i, closest_dists[0])
            elif line_points_ms[line.id][0][closest_idx[0]][1] > closest_dists[0]:
                line_points_ms[line.id][0][closest_idx[0]] = (i, closest_dists[0])

            for ms_level in range(1, subdivs+1):
                # Check for degen triangle ?
                edges = [tuple(sorted(edge)) for edge in combinations(closest_idx[:3], 2)]
                local_points = [ico_points_ms[closest_idx[0]],
                                ico_points_ms[closest_idx[1]],
                                ico_points_ms[closest_idx[2]]]

                for edge in edges:
                    if edge in subdivided_edges:
                        local_points += [ico_points_ms[subdivided_edges[edge]]]
                        continue
                    

                    subdivided_point = ico_points_ms[edge[0]].coord_3D.mid_point(ico_points_ms[edge[1]].coord_3D)
                    id = len(ico_points_ms)
                    subdivided_edges[edge] = id

                    ico_points_ms[id] = IcoPoint(
                        id=id,
                        ms_level=ms_level,
                        parent_1=edge[0],
                        parent_2=edge[1],
                        coord_3D=subdivided_point,
                        coord_geo=subdivided_point.to_lon_lat()
                    )
                    local_points += [ico_points_ms[id]]

                closest_idx = [ico_pt.id for ico_pt in sorted(local_points, 
                                                              key=lambda ico_pt: projected_pt.dist(ico_pt.coord_3D))][:3]

                dist = ico_points_ms[closest_idx[0]].coord_3D.dist(projected_pt)
                if closest_idx[0] not in line_points_ms[line.id][ms_level]:
                    line_points_ms[line.id][ms_level][closest_idx[0]] = (i, dist)
                elif line_points_ms[line.id][ms_level][closest_idx[0]][1] > dist:
                    line_points_ms[line.id][ms_level][closest_idx[0]] = (i, dist)


    print(f"{outside} points outside and needed flipping...")
    print(f"{outside_after_flip} points still outside after flipping...")

    return ico_points_ms, line_points_ms
