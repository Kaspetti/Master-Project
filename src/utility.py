from dataclasses import dataclass
from typing import Literal

from line_reader import Line
from multiscale import IcoPoint


@dataclass
class Settings:
    show_ico_points: bool
    show_3D_vis: bool
    show_centroids: bool

    sim_start: str
    time_offset: int
    line_type: Literal["jet", "mta"]

    oneline: int


@dataclass
class Data:
    lines: list[Line]
    ico_points_ms: dict[int, IcoPoint]
    line_points_ms: dict[str, dict[int, dict[int, tuple[int, float]]]]

    lines_2: list[Line] | None = None
    ico_points_ms_2: dict[int, IcoPoint] | None = None
    line_points_ms_2: dict[str, dict[int, dict[int, tuple[int, float]]]] | None = None
