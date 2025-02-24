""" Module for manipulating coords of different kinds.
Contains three classes: Coord3D, Coord2D, and CoordGeo.
"""
from __future__ import annotations


import math
from typing import List, Literal

import numpy as np
from numpy.typing import NDArray


class Coord3D:
    """ A coordinate in 3D space.

    Contains functions for basic arithmetic operations, converting
    to other coordinate types, distance calculations, and converting
    from class to list.

    Attributes
    ----------
    x : float 
        The x element.
    y : float
        The y element.
    z : float
        The z element.
    """

    x: float
    y: float
    z: float

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, o: Coord3D) -> Coord3D:
        """Performs element wise addition on a Coord3D.
        
        Parameters
        ----------
        o : Coord3D
            The coordinate to add to the coordinate.

        Returns
        -------
        a : Coord3D
            o added to the coordinate.
            A new coordinate with the elements of o added to the
            elements of the coordinate.
        """

        return Coord3D(
            x=self.x + o.x,
            y=self.y + o.y,
            z=self.z + o.z
        )

    def __sub__(self, o: Coord3D) -> Coord3D:
        """Performs element wise subtraction on a Coord3D.
        
        Parameters
        ----------
        o : Coord3D
            The coordinate to subtract from the coordinate.

        Returns
        -------
        a : Coord3D
            o subtracted from the coordinate.
            A new coordinate with the elements of o subtracted from the
            elements of the coordinate.
        """

        return Coord3D(
            x=self.x - o.x,
            y=self.y - o.y,
            z=self.z - o.z
        )

    def __mul__(self, s: int | float) -> Coord3D:
        """Performs a scalar multiplication on a Coord3D.

        Parameters
        ----------
        s : int | float
            The scalar to multiply by.

        Returns
        -------
        a : Coord3D
            The coordinate scaled by s.
        """
        return Coord3D(
            x=self.x * s,
            y=self.y * s,
            z=self.z * s
        )

    def __div__(self, s: int | float) -> Coord3D:
        """Performs a scalar division on a Coord3D.

        Parameters
        ----------
        s : int | float
            The scalar to divide by.

        Returns
        -------
        a : Coord3D
            The coordinate divided by s.
        """
        return Coord3D(
            x=self.x / s,
            y=self.y / s,
            z=self.z / s
        )

    def __str__(self) -> str:
        """Creates a string representation of a Coord3D"""

        return f"Coord3D({self.x}, {self.y}, {self.z})"

    def to_lon_lat(self) -> CoordGeo:
        """Converts a 3D coordinate into longitude and latitude."""

        lon = math.degrees(math.atan2(self.y, self.x))
        lat = math.degrees(math.asin(self.z))

        return CoordGeo(lon, lat)

    def to_list(self) -> List[float]:
        """Returns the coordinate as a list of three floats"""

        return [self.x, self.y, self.z]

    def to_ndarray(self) -> NDArray[np.float64]:
        """Returns the coordinate as an NDArray"""

        return np.array(self.to_list())

    def drop_axis(self, axis: Literal[0, 1, 2]) -> Coord2D:
        """Drops an axis from the coordinate making into a Coord2D

        Parameters
        ----------
        axis : 0, 1, or 2
            The axis to drop. 0 = x, 1 = y, z = 2.

        Returns
        -------
        a : Coord2D
            The coordinate with axis dropped.
        """

        match axis:
            case 0:
                return Coord2D(self.y, self.z)
            case 1:
                return Coord2D(self.x, self.z)
            case 2:
                return Coord2D(self.x, self.y)

    def mid_point(self, o: Coord3D) -> Coord3D:
        """Gets the midpoint between the coordinate and another Coord3D

        Parameters
        ----------
        o : Coord3D
            The coordinate to get the mid point between with.

        Returns
        -------
        m : Coord3D
            The mid point between the coordinate and o.
        """
        return Coord3D(
            x=(self.x + o.x) / 2,
            y=(self.y + o.y) / 2,
            z=(self.z + o.z) / 2,
        )

    def dist(self, o: Coord3D) -> float:
        """Gets the distance from the coordinate to another coordinate.
        
        Parameters
        ----------
        o : Coord3D
            The coordinate to get the distance to.

        Returns
        -------
        d : float
            The distance from the coordinate to o.
        """

        return np.sqrt(np.sum((self.to_ndarray() - o.to_ndarray())**2))


class Coord2D:
    """A 2D coordinate

    Attributes
    ----------
    x : float
        The x element.
    y : float
        The y element.
    """

    x: float
    y: float

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __str__(self) -> str:
        """Creates a string representation of the coordinate."""

        return f"Coord2D({self.x}, {self.y})"

    def to_list(self) -> List[float]:
        """Returns the coordinate as a list of two floats."""

        return [self.x, self.y]


class CoordGeo:
    """A longitude and latitude point.

    Attributes
    ----------
    lon : float
        Longitude.
    lat : float
        Latitude.
    """

    lon: float
    lat: float

    def __init__(self, lon: float, lat: float):
        self.lon = lon
        self.lat = lat

    def __str__(self) -> str:
        """Creates a string representation of the coordinate."""

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
