from __future__ import annotations

import math
from typing import List


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
