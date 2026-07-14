"""Single source of public API numeric limits derived from project contracts."""
from netkeeper_sim.rl.config import RLConfig

_rl = RLConfig()
MAX_QUEUE_PACKETS = 1_000_000
RANGES = {
    "ospf_weight": (_rl.ospf_weight.minimum, _rl.ospf_weight.maximum),
    "local_preference": (_rl.local_preference.minimum, _rl.local_preference.maximum),
    "as_path_length": (_rl.as_path_length.minimum, _rl.as_path_length.maximum),
    "med": (_rl.med.minimum, _rl.med.maximum),
    "queue_packets": (0, MAX_QUEUE_PACKETS),
}
