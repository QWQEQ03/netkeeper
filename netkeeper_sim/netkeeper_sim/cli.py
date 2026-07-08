from __future__ import annotations

import argparse
from math import inf
from pathlib import Path

from netkeeper_sim.metrics.load import calculate_link_load_metrics
from netkeeper_sim.routing.ecmp import has_ecmp
from netkeeper_sim.routing.ospf import compute_ospf_routes
from netkeeper_sim.topology.loader import load_topology
from netkeeper_sim.traffic.matrix import TrafficMatrix
from netkeeper_sim.traffic.propagation import propagate_traffic


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OSPF and optional traffic propagation.")
    parser.add_argument("--topology", required=True, type=Path)
    parser.add_argument("--traffic-csv", type=Path)
    parser.add_argument("--random-traffic", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-demand", type=float, default=10.0)
    parser.add_argument("--density", type=float, default=0.2)
    args = parser.parse_args()

    topology = load_topology(args.topology)
    forwarding_table = compute_ospf_routes(topology)

    reachable_pairs = 0
    unreachable_pairs = 0
    ecmp_pairs = 0
    for source, destinations in forwarding_table.items():
        for destination, entry in destinations.items():
            if source == destination:
                continue
            if entry.cost == inf or not entry.reachable:
                unreachable_pairs += 1
            else:
                reachable_pairs += 1
                if has_ecmp(forwarding_table, source, destination):
                    ecmp_pairs += 1

    print(f"topology: {args.topology}")
    print(f"nodes: {topology.node_count}")
    print(f"edges: {topology.edge_count}")
    print(f"reachable_node_pairs: {reachable_pairs}")
    print(f"unreachable_node_pairs: {unreachable_pairs}")
    print(f"ecmp_node_pairs: {ecmp_pairs}")

    traffic_matrix = None
    if args.traffic_csv is not None:
        traffic_matrix = TrafficMatrix.from_csv(args.traffic_csv, topology.nodes)
    elif args.random_traffic:
        traffic_matrix = TrafficMatrix.random(
            sorted(topology.nodes),
            seed=args.seed,
            max_demand=args.max_demand,
            density=args.density,
        )

    if traffic_matrix is not None:
        propagation = propagate_traffic(topology, forwarding_table, traffic_matrix)
        metrics = calculate_link_load_metrics(topology, propagation.link_loads)
        print(f"traffic_demands: {len(traffic_matrix)}")
        print(f"total_input_traffic: {propagation.total_input_traffic:.6f}")
        print(f"delivered_traffic: {propagation.delivered_traffic:.6f}")
        print(f"dropped_traffic: {propagation.dropped_traffic:.6f}")
        print(f"unreachable_demands: {len(propagation.unreachable_demands)}")
        print(f"flow_conserved: {propagation.is_flow_conserved()}")
        print(f"maximum_link_utilization: {metrics.maximum_link_utilization:.6f}")


if __name__ == "__main__":
    main()
