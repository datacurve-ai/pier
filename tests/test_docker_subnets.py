import ipaddress
import json
from pathlib import Path

import pytest

from pier.environments.agent_setup import write_docker_proxy_compose
from pier.environments.subnets import allocate_trial_subnets
from pier.models.agent.network import NetworkAllowlist

POOL = "198.18.0.0/24"  # 32 /29 slots


@pytest.fixture(autouse=True)
def subnet_state(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PIER_DOCKER_SUBNET_STATE", str(tmp_path / "state" / "subnets.json"))
    monkeypatch.delenv("PIER_DOCKER_SUBNET_POOL", raising=False)


def test_allocation_gives_two_distinct_slash29_inside_pool(tmp_path: Path):
    internal, egress = allocate_trial_subnets(tmp_path / "trial", POOL)

    pool = ipaddress.ip_network(POOL)
    for subnet in (internal, egress):
        net = ipaddress.ip_network(subnet)
        assert net.prefixlen == 29
        assert net.subnet_of(pool)
    assert internal != egress


def test_allocation_is_stable_per_trial(tmp_path: Path):
    first = allocate_trial_subnets(tmp_path / "trial", POOL)

    assert allocate_trial_subnets(tmp_path / "trial", POOL) == first


def test_trials_never_share_a_subnet(tmp_path: Path):
    seen: list[str] = []
    for i in range(16):  # 16 trials x 2 subnets = all 32 slots
        seen.extend(allocate_trial_subnets(tmp_path / f"trial{i}", POOL))

    assert len(set(seen)) == len(seen) == 32


def test_exhausted_pool_raises(tmp_path: Path):
    for i in range(16):
        allocate_trial_subnets(tmp_path / f"trial{i}", POOL)

    with pytest.raises(RuntimeError, match="exhausted"):
        allocate_trial_subnets(tmp_path / "one-too-many", POOL)


def test_state_is_persisted_and_reused(tmp_path: Path, monkeypatch):
    first = allocate_trial_subnets(tmp_path / "trial", POOL)
    state = json.loads(Path(__import__("os").environ["PIER_DOCKER_SUBNET_STATE"]).read_text())

    assert list(state[str(ipaddress.ip_network(POOL))].values()) == [list(first)]


@pytest.mark.parametrize("pool", ["198.18.0.0/29", "fd00::/64"])
def test_pool_too_small_or_ipv6_is_rejected(tmp_path: Path, pool: str):
    with pytest.raises(ValueError):
        allocate_trial_subnets(tmp_path / "trial", pool)


def _write_compose(tmp_path: Path) -> dict:
    path = write_docker_proxy_compose(
        path=tmp_path / "trial" / "docker-compose.json",
        proxy_dir=tmp_path / "trial" / "egress-proxy",
        allowlist=NetworkAllowlist(domains=["example.com"]),
        token="t",
    )
    return json.loads(path.read_text())


def test_compose_has_no_ipam_without_pool(tmp_path: Path):
    compose = _write_compose(tmp_path)

    assert "ipam" not in compose["networks"]["pier-egress-internal"]
    assert "default" not in compose["networks"]


def test_compose_pins_both_networks_with_pool(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PIER_DOCKER_SUBNET_POOL", POOL)

    compose = _write_compose(tmp_path)

    internal = compose["networks"]["pier-egress-internal"]["ipam"]["config"][0]["subnet"]
    egress = compose["networks"]["default"]["ipam"]["config"][0]["subnet"]
    assert (internal, egress) == allocate_trial_subnets(tmp_path / "trial", POOL)
