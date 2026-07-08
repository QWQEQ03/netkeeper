from __future__ import annotations

from dataclasses import dataclass
from math import inf

from netkeeper_sim.topology.model import Topology


@dataclass(frozen=True)
class LinkLoadMetrics:
    total_load: dict[str, float]
    utilization: dict[str, float]
    maximum_link_utilization: float


def calculate_link_load_metrics(
    topology: Topology,
    link_loads: dict[str, float],
) -> LinkLoadMetrics:
    total_load = {link_id: float(link_loads.get(link_id, 0.0)) for link_id in topology.links}
    utilization: dict[str, float] = {}
    for link_id, link in topology.links.items():
        load = total_load[link_id]
        if link.capacity <= 0:
            utilization[link_id] = inf if load > 0 else 0.0
        else:
            utilization[link_id] = load / link.capacity
    maximum = max(utilization.values(), default=0.0)
    return LinkLoadMetrics(
        total_load=total_load,
        utilization=utilization,
        maximum_link_utilization=maximum,
    )
