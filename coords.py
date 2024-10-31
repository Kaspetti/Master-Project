from __future__ import annotations


import math
from typing import List, Literal


class Coord3D:
    x: float
    y: float
    z: float

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def to_lon_lat(self) -> CoordGeo:
        """Converts a 3D coordinate into longitude and latitude."""
        lat = math.degrees(math.asin(self.z))
        lon = math.degrees(math.atan2(self.y, self.x))

        return CoordGeo(lat, lon)

    def to_list(self) -> List[float]:
        """Returns the coordinate as a list of three floats"""
        return [self.x, self.y, self.z]

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


class Coord2D:
    x: float
    y: float

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def to_list(self) -> List[float]:
        """Returns the coordinate as a list of two floats"""
        return [self.x, self.y]


class CoordGeo:
    lon: float
    lat: float

    def __init__(self, lon: float, lat: float):
        self.lon = lon
        self.lat = lat

    def to_3D(self) -> Coord3D:
        """Converts longitude and latitude to a 3D coordinate."""

        x = math.cos(math.radians(self.lat)) * math.cos(math.radians(self.lon))
        y = math.cos(math.radians(self.lat)) * math.sin(math.radians(self.lon))
        z = math.sin(math.radians(self.lat))

        return Coord3D(x, y, z)

    def to_list(self) -> List[float]:
        """Returns the coordinate as a list of two floats"""
        return [self.lon, self.lat]
