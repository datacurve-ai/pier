import ipaddress
import json
import multiprocessing
import os
from pathlib import Path

import pytest

from pier.environments.agent_setup import write_docker_proxy_compose
from pier.environments.subnets import allocate_trial_subnets
from pier.models.agent.network import NetworkAllowlist

POOL = "198.18.0.0/24"  # 32 /29 slots


@pytest.fixture(autouse=True)
def subnet_state(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(
        "PIER_DOCKER_SUBNET_STATE", str(tmp_path / "state" / "subnets.json")
    )
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


def test_exhausted_pool_raises_and_names_the_state_file(tmp_path: Path):
    for i in range(16):
        allocate_trial_subnets(tmp_path / f"trial{i}", POOL)

    with pytest.raises(RuntimeError, match="exhausted") as error:
        allocate_trial_subnets(tmp_path / "one-too-many", POOL)
    assert os.environ["PIER_DOCKER_SUBNET_STATE"] in str(error.value)


def test_state_is_persisted_and_reused(tmp_path: Path):
    first = allocate_trial_subnets(tmp_path / "trial", POOL)
    state = json.loads(Path(os.environ["PIER_DOCKER_SUBNET_STATE"]).read_text())

    assert list(state.values()) == [list(first)]


def test_state_path_named_like_its_lock_still_works(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PIER_DOCKER_SUBNET_STATE", str(tmp_path / "subnets.lock"))
    first = allocate_trial_subnets(tmp_path / "a", POOL)
    second = allocate_trial_subnets(tmp_path / "b", POOL)

    assert not set(first) & set(second)
    assert allocate_trial_subnets(tmp_path / "a", POOL) == first


def test_overlapping_pools_never_share_a_subnet(tmp_path: Path):
    a = allocate_trial_subnets(tmp_path / "a", "198.18.0.0/24")
    b = allocate_trial_subnets(tmp_path / "b", "198.18.0.0/16")

    assert not set(a) & set(b)


def test_trial_whose_pair_is_outside_the_current_pool_is_reallocated(tmp_path: Path):
    allocate_trial_subnets(tmp_path / "trial", "198.18.0.0/24")

    pair = allocate_trial_subnets(tmp_path / "trial", "198.19.0.0/24")

    new_pool = ipaddress.ip_network("198.19.0.0/24")
    assert all(ipaddress.ip_network(subnet).subnet_of(new_pool) for subnet in pair)


def test_corrupt_state_names_the_file(tmp_path: Path):
    state = Path(os.environ["PIER_DOCKER_SUBNET_STATE"])
    state.parent.mkdir(parents=True)
    state.write_text("{truncated")

    with pytest.raises(RuntimeError, match=str(state)):
        allocate_trial_subnets(tmp_path / "trial", POOL)


def _allocate_in_process(args: tuple[str, str]) -> list[str]:
    trial, state = args
    os.environ["PIER_DOCKER_SUBNET_STATE"] = state
    return list(allocate_trial_subnets(Path(trial), POOL))


def test_concurrent_processes_get_disjoint_subnets(tmp_path: Path):
    state = os.environ["PIER_DOCKER_SUBNET_STATE"]
    jobs = [(str(tmp_path / f"trial{i}"), state) for i in range(16)]

    with multiprocessing.get_context("spawn").Pool(8) as pool:
        pairs = pool.map(_allocate_in_process, jobs)

    subnets = [subnet for pair in pairs for subnet in pair]
    assert len(set(subnets)) == len(subnets) == 32


@pytest.mark.parametrize(
    "pool", ["198.18.0.0/29", "fd00::/64", "198.18.0.1/24", "not-a-pool"]
)
def test_invalid_pool_is_rejected(tmp_path: Path, pool: str):
    with pytest.raises(ValueError, match="(?i)pool"):
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

    internal = compose["networks"]["pier-egress-internal"]["ipam"]["config"][0][
        "subnet"
    ]
    egress = compose["networks"]["default"]["ipam"]["config"][0]["subnet"]
    assert (internal, egress) == allocate_trial_subnets(tmp_path / "trial", POOL)
