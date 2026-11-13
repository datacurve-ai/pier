"""Opt-in, host-scoped subnet allocation for isolated Docker trials."""

import fcntl
import hashlib
import ipaddress
import json
import os
from pathlib import Path


def allocate_trial_subnets(trial_path: Path, pool: str) -> tuple[str, str]:
    network = ipaddress.ip_network(pool)
    if network.version != 4 or network.prefixlen > 28:
        raise ValueError("Pier Docker subnet pool must be IPv4 /28 or larger")
    # /29 provides a gateway plus five container addresses per isolated network.
    slots = 1 << (29 - network.prefixlen)
    state_path = Path(os.environ.get("PIER_DOCKER_SUBNET_STATE", "/tmp/pier-docker-subnets.json"))
    state_path.parent.mkdir(parents=True, exist_ok=True)
    key = str(trial_path.resolve())
    with state_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        allocations = state.setdefault(str(network), {})
        if key in allocations:
            return tuple(allocations[key])
        used = {value for pair in allocations.values() for value in pair}
        start = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % slots
        pair = []
        for offset in range(slots):
            index = (start + offset) % slots
            subnet = str(ipaddress.ip_network((int(network.network_address) + index * 8, 29)))
            if subnet not in used:
                pair.append(subnet)
            if len(pair) == 2:
                break
        if len(pair) != 2:
            raise RuntimeError(f"Pier Docker subnet pool {network} is exhausted")
        allocations[key] = pair
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state))
        temporary.replace(state_path)
        return tuple(pair)
