"""Traffic matrix and propagation helpers."""

from netkeeper_sim.traffic.matrix import (
    PrefixTrafficDemand,
    PrefixTrafficMatrix,
    TrafficDemand,
    TrafficMatrix,
)
from netkeeper_sim.traffic.propagation import (
    PropagationResult,
    propagate_bgp_traffic,
    propagate_traffic,
)

__all__ = [
    "PrefixTrafficDemand",
    "PrefixTrafficMatrix",
    "PropagationResult",
    "TrafficDemand",
    "TrafficMatrix",
    "propagate_bgp_traffic",
    "propagate_traffic",
]
