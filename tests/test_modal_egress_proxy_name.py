import re

from pier.environments.modal import ModalEnvironment


def _modal_environment(session_id: str) -> ModalEnvironment:
    env = ModalEnvironment.__new__(ModalEnvironment)
    env.session_id = session_id
    env._sandbox_name_suffix = env._generate_sandbox_name_suffix()
    return env


def test_retry_attempts_use_distinct_egress_proxy_names():
    first_attempt = _modal_environment("task.session")
    retry_attempt = _modal_environment("task.session")

    assert first_attempt._egress_proxy_sandbox_name.startswith("task.session-")
    assert retry_attempt._egress_proxy_sandbox_name.startswith("task.session-")
    assert (
        first_attempt._egress_proxy_sandbox_name
        != retry_attempt._egress_proxy_sandbox_name
    )


def test_attempt_sandbox_names_are_related_and_valid_modal_identifiers():
    env = _modal_environment(f"task.{'a' * 64}")

    sandbox_name = env._sandbox_name
    egress_proxy_name = env._egress_proxy_sandbox_name

    assert egress_proxy_name == f"{sandbox_name}-egress-proxy"
    assert sandbox_name != egress_proxy_name
    assert sandbox_name.startswith("task.")
    assert len(sandbox_name) <= 64
    assert len(egress_proxy_name) <= 64
    assert re.fullmatch(r"[A-Za-z0-9._-]+", sandbox_name)
    assert re.fullmatch(r"[A-Za-z0-9._-]+", egress_proxy_name)


def test_sandbox_name_suffixes_are_nonempty_and_unique_in_tight_loop():
    suffixes = [ModalEnvironment._generate_sandbox_name_suffix() for _ in range(100)]

    assert all(suffixes)
    assert len(suffixes) == len(set(suffixes))
    assert all(re.fullmatch(r"[0-9a-f]{8}", suffix) for suffix in suffixes)
