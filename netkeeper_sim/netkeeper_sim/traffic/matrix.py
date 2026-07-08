from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class TrafficDemand:
    source: str
    destination: str
    demand: float


@dataclass(frozen=True)
class PrefixTrafficDemand:
    source: str
    prefix: str
    demand: float


@dataclass(frozen=True)
class TrafficMatrix:
    demands: tuple[TrafficDemand, ...]

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        valid_nodes: Iterable[str],
        ignore_self_demands: bool = True,
    ) -> "TrafficMatrix":
        node_set = {str(node) for node in valid_nodes}
        demands: list[TrafficDemand] = []
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"source", "destination", "demand"}
            if set(reader.fieldnames or []) < required:
                raise ValueError("CSV must contain source,destination,demand columns")
            for row in reader:
                demands.extend(
                    _build_demand(
                        row["source"],
                        row["destination"],
                        row["demand"],
                        node_set,
                        ignore_self_demands,
                    )
                )
        return cls(tuple(demands))

    @classmethod
    def from_numpy(
        cls,
        matrix: Any,
        nodes: Sequence[str],
        ignore_self_demands: bool = True,
    ) -> "TrafficMatrix":
        node_ids = [str(node) for node in nodes]
        row_count, col_count = _matrix_shape(matrix)
        if row_count != len(node_ids) or col_count != len(node_ids):
            raise ValueError("Matrix shape must match the number of nodes")

        demands: list[TrafficDemand] = []
        node_set = set(node_ids)
        for row_index, source in enumerate(node_ids):
            for col_index, destination in enumerate(node_ids):
                value = _matrix_value(matrix, row_index, col_index)
                demands.extend(
                    _build_demand(
                        source,
                        destination,
                        value,
                        node_set,
                        ignore_self_demands,
                    )
                )
        return cls(tuple(demands))

    @classmethod
    def random(
        cls,
        nodes: Sequence[str],
        seed: int = 42,
        max_demand: float = 10.0,
        density: float = 1.0,
        ignore_self_demands: bool = True,
    ) -> "TrafficMatrix":
        if max_demand < 0:
            raise ValueError("max_demand must be non-negative")
        if not 0 <= density <= 1:
            raise ValueError("density must be between 0 and 1")
        rng = random.Random(seed)
        node_ids = [str(node) for node in nodes]
        demands: list[TrafficDemand] = []
        for source in node_ids:
            for destination in node_ids:
                if ignore_self_demands and source == destination:
                    continue
                if rng.random() > density:
                    continue
                demand = rng.uniform(0.0, max_demand)
                if demand > 0:
                    demands.append(TrafficDemand(source, destination, demand))
        return cls(tuple(demands))

    @property
    def total_demand(self) -> float:
        return sum(demand.demand for demand in self.demands)

    def __iter__(self):
        return iter(self.demands)

    def __len__(self) -> int:
        return len(self.demands)


@dataclass(frozen=True)
class PrefixTrafficMatrix:
    demands: tuple[PrefixTrafficDemand, ...]

    @property
    def total_demand(self) -> float:
        return sum(demand.demand for demand in self.demands)

    def __iter__(self):
        return iter(self.demands)

    def __len__(self) -> int:
        return len(self.demands)


def _build_demand(
    source: Any,
    destination: Any,
    raw_demand: Any,
    valid_nodes: set[str],
    ignore_self_demands: bool,
) -> list[TrafficDemand]:
    source_id = str(source)
    destination_id = str(destination)
    if source_id not in valid_nodes:
        raise ValueError(f"Unknown traffic source: {source_id!r}")
    if destination_id not in valid_nodes:
        raise ValueError(f"Unknown traffic destination: {destination_id!r}")
    demand = float(raw_demand)
    if demand < 0:
        raise ValueError("Traffic demand must be non-negative")
    if source_id == destination_id and ignore_self_demands:
        return []
    if demand == 0:
        return []
    return [TrafficDemand(source_id, destination_id, demand)]


def _matrix_shape(matrix: Any) -> tuple[int, int]:
    shape = getattr(matrix, "shape", None)
    if shape is not None:
        if len(shape) != 2:
            raise ValueError("Traffic matrix must be two-dimensional")
        return int(shape[0]), int(shape[1])
    rows = list(matrix)
    if not rows:
        return 0, 0
    return len(rows), len(rows[0])


def _matrix_value(matrix: Any, row: int, col: int) -> Any:
    try:
        return matrix[row, col]
    except (TypeError, KeyError, IndexError):
        return matrix[row][col]
