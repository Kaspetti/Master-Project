from __future__ import annotations


import math
from typing import List, Literal

import numpy as np
from numpy.typing import NDArray


class Coord3D:
    x: float
    y: float
    z: float

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __sub__(self, o: Coord3D) -> Coord3D:
        return Coord3D(
            x=self.x - o.x,
            y=self.y - o.y,
            z=self.z - o.z
        )

    def __mul__(self, s: int | float) -> Coord3D:
        return Coord3D(
            x=self.x * s,
            y=self.y * s,
            z=self.z * s
        )

    def __str__(self) -> str:
        return f"Coord3D({self.x}, {self.y}, {self.z})"

    def to_lon_lat(self) -> CoordGeo:
        """Converts a 3D coordinate into longitude and latitude."""
        lon = math.degrees(math.atan2(self.y, self.x))
        lat = math.degrees(math.asin(self.z))

        return CoordGeo(lon, lat)

    def to_list(self) -> List[float]:
        """Returns the coordinate as a list of three floats"""
        return [self.x, self.y, self.z]

    def to_ndarray(self) -> NDArray[np.float_]:
        return np.array(self.to_list())

    def drop_axis(self, axis: Literal[0, 1, 2]) -> Coord2D:
        match axis:
            case 0:
                return Coord2D(self.y, self.z)
            case 1:
                return Coord2D(self.x, self.z)
            case 2:
                return Coord2D(self.x, self.y)

    def mid_point(self, o: Coord3D) -> Coord3D:
        return Coord3D(
            x=(self.x + o.x) / 2,
            y=(self.y + o.y) / 2,
            z=(self.z + o.z) / 2,
        )

    def dist(self, o: Coord3D) -> float:
        return np.sqrt(np.sum((self.to_ndarray() - o.to_ndarray())**2))


class Coord2D:
    x: float
    y: float

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return f"Coord2D({self.x}, {self.y})"

    def to_list(self) -> List[float]:
        """Returns the coordinate as a list of two floats"""
        return [self.x, self.y]


class CoordGeo:
    lon: float
    lat: float

    def __init__(self, lon: float, lat: float):
        self.lon = lon
        self.lat = lat

    def __str__(self) -> str:
        return f"CoordGeo({self.lon}, {self.lat})"

    def to_3D(self) -> Coord3D:
        """Converts longitude and latitude to a 3D coordinate."""

        x = math.cos(math.radians(self.lat)) * math.cos(math.radians(self.lon))
        y = math.cos(math.radians(self.lat)) * math.sin(math.radians(self.lon))
        z = math.sin(math.radians(self.lat))

        return Coord3D(x, y, z)

    def to_list(self) -> List[float]:
        """Returns the coordinate as a list of two floats"""
        return [self.lon, self.lat]
