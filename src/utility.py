from dataclasses import dataclass
from typing import Literal, TypedDict

from cartopy.crs import json

from line_reader import Line
from multiscale import IcoPoint


@dataclass
class Settings:
    show_ico_points: bool
    show_3D_vis: bool
    show_centroids: bool

    sim_start: str
    time_offset: int
    line_type: Literal["jet", "mta", "both"]

    oneline: int
    bspline: bool


@dataclass
class Data:
    lines: list[Line]
    ico_points_ms: dict[int, IcoPoint]
    line_points_ms: dict[str, dict[int, dict[int, tuple[int, float]]]]

    lines_2: list[Line] | None = None
    ico_points_ms_2: dict[int, IcoPoint] | None = None
    line_points_ms_2: dict[str, dict[int, dict[int, tuple[int, float]]]] | None = None


@dataclass(frozen=True)
class Connection:
    source: str
    target: str
    weight: float


class TypedConnection(TypedDict):
    source: str
    target: str
    weight: float


class Node(TypedDict):
    id: str


class Network(TypedDict):
    nodes: list[Node]
    clusters: dict[int, list[TypedConnection]]
    node_clusters: dict[str, int]



def load_networks(path: str) -> dict[str, Network]:
    with open(path, "r") as f:
        networks: dict[str, Network] = json.load(f)
        return networks
