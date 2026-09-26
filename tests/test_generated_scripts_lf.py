"""Generated scripts must use LF endings, even when Pier runs on Windows.

The Docker backend generates a handful of files on the host and hands them to a
Linux container -- either as a build context (``COPY``) or through
``upload_dir``. Two of them are executed by bash inside that container:

* ``start-squid.sh`` (the egress proxy entrypoint)
* the agent ``Dockerfile``

On Windows, ``Path.write_text`` defaults to text mode with universal newline
translation, rewriting ``\\n`` as ``\\r\\n``. A CRLF shell script dies at
``set -eu`` with ``set: -: invalid option``, so the proxy never binds its port.
Because the proxy is a ``depends_on: service_healthy`` dependency of ``main``,
the whole trial fails before the agent starts -- and the only user-visible
symptom is a confusing ``Failed to download artifact .../model.patch``.

``upload_dir`` already strips CR from uploaded ``*.sh`` files on Windows, but
that path only covers files copied *into a running container*. These proxy and
agent-build files never go through it: they are written to the build context
and consumed by ``docker compose build``, so nothing normalizes them.

These tests emulate Windows newline translation rather than skipping on
non-Windows hosts, so the regression is caught by Linux CI too.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from pier.environments.agent_setup import (
    write_agent_dockerfile,
    write_docker_proxy_compose,
)
from pier.models.agent.install import AgentInstallSpec, InstallStep
from pier.models.agent.network import NetworkAllowlist


@contextmanager
def emulate_windows_text_mode(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Make ``Path.write_text`` behave as it does on Windows.

    Mirrors CPython's behaviour: when ``newline`` is left as ``None`` the text
    layer translates ``\\n`` to ``\\r\\n``; an explicit ``newline`` disables the
    translation. Running this on Linux lets CI reproduce the Windows failure.
    """
    original_write_text = Path.write_text

    def windows_write_text(
        self: Path,
        data: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        if newline is None:
            # Emulate the translation, then disable the real one by passing
            # newline="" -- otherwise a Windows host would translate a second
            # time and emit \r\r\n, making the emulation platform-dependent.
            data = data.replace("\n", "\r\n")
            newline = ""
        return original_write_text(
            self, data, encoding=encoding, errors=errors, newline=newline
        )

    monkeypatch.setattr(Path, "write_text", windows_write_text)
    yield


def _install_spec() -> AgentInstallSpec:
    return AgentInstallSpec(
        agent_name="test-agent",
        version="1",
        steps=[InstallStep(run="echo install", user="root")],
    )


def _assert_lf(path: Path) -> None:
    raw = path.read_bytes()
    assert b"\r" not in raw, (
        f"{path.name} contains CR bytes; it will fail to execute in the Linux "
        f"container. First line: {raw.splitlines()[0]!r}"
    )


def test_egress_proxy_script_uses_lf_on_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CRLF start-squid.sh kills the proxy with `set: -: invalid option`."""
    with emulate_windows_text_mode(monkeypatch):
        write_docker_proxy_compose(
            path=tmp_path / "docker-compose-egress-proxy.json",
            proxy_dir=tmp_path / "egress-proxy",
            allowlist=NetworkAllowlist(domains=["api.example.com"]),
            token="secret",
        )

    _assert_lf(tmp_path / "egress-proxy" / "start-squid.sh")


def test_egress_proxy_dockerfile_uses_lf_on_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with emulate_windows_text_mode(monkeypatch):
        write_docker_proxy_compose(
            path=tmp_path / "docker-compose-egress-proxy.json",
            proxy_dir=tmp_path / "egress-proxy",
            allowlist=NetworkAllowlist(domains=["api.example.com"]),
            token="secret",
        )

    _assert_lf(tmp_path / "egress-proxy" / "Dockerfile")


def test_agent_dockerfile_uses_lf_on_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_dir = tmp_path / "agent-build-context"
    with emulate_windows_text_mode(monkeypatch):
        write_agent_dockerfile(
            build_dir=build_dir,
            source_environment_dir=build_dir,
            prebuilt_image_name="example/task:latest",
            install=_install_spec(),
            user=None,
        )

    _assert_lf(build_dir / "Dockerfile")


def test_emulation_actually_translates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Guard the guard: the emulation must be able to produce CRLF at all.

    Without this, a future refactor that breaks ``emulate_windows_text_mode``
    would silently turn the tests above into no-ops.
    """
    naive = tmp_path / "naive.sh"
    with emulate_windows_text_mode(monkeypatch):
        naive.write_text("set -eu\n")

    assert b"\r\n" in naive.read_bytes()
