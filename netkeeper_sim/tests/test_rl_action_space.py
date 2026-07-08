from __future__ import annotations

from netkeeper_sim.rl import _tensor as T
from netkeeper_sim.rl.action_space import build_action_masks
from netkeeper_sim.routing.bgp import BGPRoute


def test_action_masks_mark_active_links(diamond_topology):
    link_id = sorted(diamond_topology.links)[0]
    link = diamond_topology.links[link_id]
    diamond_topology.fail_link(link.source, link.target)

    masks = build_action_masks(diamond_topology)
    ospf_mask = T.to_numpy(masks.ospf_weight_mask).tolist()
    performance_mask = T.to_numpy(masks.capacity_mask).tolist()

    assert ospf_mask[0] is False
    assert performance_mask[0] is False
    assert all(isinstance(value, bool) for value in ospf_mask)


def test_bgp_masks_follow_existing_route_targets(diamond_topology):
    candidates = {
        "R1": {
            "203.0.113.0/24": [
                BGPRoute("203.0.113.0/24", "R2", 100, (65001,), 20, "R2", "peer"),
                BGPRoute("203.0.113.0/24", "missing", 100, (65002,), 20, "R9", "peer"),
            ]
        }
    }

    masks = build_action_masks(diamond_topology, candidates)

    assert [target.route_index for target in masks.bgp_route_targets] == [0, 1]
    assert T.to_numpy(masks.local_preference_mask).tolist() == [True, False]
    assert T.to_numpy(masks.as_path_length_mask).tolist() == [True, False]
    assert T.to_numpy(masks.med_mask).tolist() == [True, False]
