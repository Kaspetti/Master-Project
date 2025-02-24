from __future__ import annotations
from typing import List
from dataclasses import dataclass
from itertools import combinations

from coords import Coord3D, CoordGeo
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
    parent_1: int = -1
    parent_2: int = -1


@dataclass
class LinePointMS:
    id: int
    ms_level: int
    closest_ico_id: int
    closest_ico_dist: float
    line_point: int
    line_point_coord_geo: CoordGeo


def inside_check(pt: NDArray[np.float_], tri: NDArray[np.float_]) -> bool:
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


def multiscale(lines: List[Line], subdivs: int):
    ico_verts, _ = icosphere(0)
    ico_points_ms: dict[int, IcoPoint]= {}
    subdivided_edges: dict[tuple[int, int], int] = {}
    line_points_ms: dict[str, dict[int, dict[int, tuple[int, float]]]] = {}

    for i, pt in enumerate(ico_verts):
        coord = Coord3D(x=pt[0], y=pt[1], z=pt[2])
        ico_pt = IcoPoint(
            id=i,
            ms_level=0,
            coord_3D=coord,
            coord_geo=coord.to_lon_lat()
        )
        ico_points_ms[i] = ico_pt

    ico_points_base_kd = KDTree([pt.coord_3D.to_list() for pt in ico_points_ms.values()])


    outside = 0
    outside_after_flip = 0
    for line in lines:
        line_points_ms[line.id] = {}
        for ms_level in range(0, subdivs+1):
            line_points_ms[line.id][ms_level] = {}

        for i, pt in enumerate(line.coords):
            coord_3D = pt.to_3D().to_list()

            closest_dists: list[float]
            closest_idx: list[int]
            closest_dists, closest_idx = ico_points_base_kd.query(coord_3D, 4)  # type: ignore
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


    # print(f"{outside} points outside and needed flipping...")
    # print(f"{outside_after_flip} points still outside after flipping...")

    return ico_points_ms, line_points_ms
