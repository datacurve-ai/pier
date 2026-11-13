"""Opt-in, host-scoped subnet allocation for isolated Docker trials."""

import ipaddress
import json
import os
import tempfile
from pathlib import Path

DEFAULT_STATE_PATH = "/tmp/pier-docker-subnets.json"
# A /29 provides a gateway plus five container addresses per isolated network.
SUBNET_PREFIXLEN = 29
SUBNET_SIZE = 1 << (32 - SUBNET_PREFIXLEN)


def allocate_trial_subnets(trial_path: Path, pool: str) -> tuple[str, str]:
    """Return two distinct /29 subnets of `pool` for the trial at `trial_path`.

    Allocations are recorded in the file named by PIER_DOCKER_SUBNET_STATE under
    a lock, so concurrent Pier processes on one host never share a subnet. They
    are never released: delete the state file between campaigns.
    """
    import fcntl  # POSIX only; imported here so that Pier still imports elsewhere

    try:
        network = ipaddress.ip_network(pool)
    except ValueError as error:
        raise ValueError(
            f"Invalid PIER_DOCKER_SUBNET_POOL {pool!r}: {error}"
        ) from error
    if network.version != 4 or network.prefixlen > 28:
        raise ValueError("Pier Docker subnet pool must be IPv4 /28 or larger")

    state_path = Path(os.environ.get("PIER_DOCKER_SUBNET_STATE", DEFAULT_STATE_PATH))
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = state_path.with_name(state_path.name + ".lock")
    key = str(trial_path.resolve())
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            allocations: dict[str, list[str]] = (
                json.loads(state_path.read_text()) if state_path.exists() else {}
            )
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"Corrupt Pier Docker subnet state {state_path}: delete it ({error})"
            ) from error
        if key in allocations and all(
            ipaddress.ip_network(subnet).subnet_of(network)
            for subnet in allocations[key]
        ):
            return tuple(allocations[key])  # type: ignore[return-value]
        used = {subnet for pair in allocations.values() for subnet in pair}
        free = (
            str(subnet)
            for subnet in network.subnets(new_prefix=SUBNET_PREFIXLEN)
            if str(subnet) not in used
        )
        pair = [subnet for _, subnet in zip(range(2), free)]
        if len(pair) != 2:
            raise RuntimeError(
                f"Pier Docker subnet pool {network} is exhausted "
                f"(state {state_path}: delete it between campaigns)"
            )
        allocations[key] = pair
        with tempfile.NamedTemporaryFile(
            "w", dir=state_path.parent, prefix=state_path.name, delete=False
        ) as temporary:
            json.dump(allocations, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary.name, state_path)
        return pair[0], pair[1]
