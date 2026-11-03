from __future__ import annotations

import json
import os
import secrets
import shlex
from pathlib import Path

from pier.models.agent.install import AgentInstallSpec, InstallStep
from pier.models.agent.network import NetworkAllowlist

AGENT_INSTALL_DIR = ".pier-agent-install"
EGRESS_PROXY_SERVICE = "pier-egress-proxy"
EGRESS_PROXY_PORT = 8080


def _docker_cmd() -> str:
    return os.environ.get("PIER_DOCKER_CMD", "docker")


def _is_podman() -> bool:
    return os.path.basename(_docker_cmd()).startswith("podman")


def docker_run_command(script: str) -> str:
    return "RUN " + json.dumps(["/bin/bash", "-c", script])


def _run_with_step_env(step: InstallStep) -> str:
    if not step.env:
        return step.run
    exports = "".join(
        f"export {key}={shlex.quote(value)}; " for key, value in step.env.items()
    )
    return exports + step.run


def dockerfile_install_commands(
    install: AgentInstallSpec,
    *,
    user: str | int | None,
) -> list[str]:
    commands: list[str] = []
    docker_agent_user = "root" if user is None else str(user)
    for step in install.steps:
        docker_user = "root" if step.user == "root" else docker_agent_user
        commands.extend(
            [
                f"USER {docker_user}",
                docker_run_command(_run_with_step_env(step)),
            ]
        )
    return commands


def write_agent_dockerfile(
    *,
    build_dir: Path,
    source_environment_dir: Path,
    prebuilt_image_name: str | None,
    install: AgentInstallSpec,
    user: str | int | None,
) -> Path:
    build_dir.mkdir(parents=True, exist_ok=True)
    dockerfile_path = build_dir / "Dockerfile"

    if prebuilt_image_name:
        dockerfile = [f"FROM {prebuilt_image_name}"]
    else:
        source = source_environment_dir / "Dockerfile"
        dockerfile = [source.read_text()]

    fingerprint = install.fingerprint()
    dockerfile.extend(
        [
            f"ARG PIER_AGENT_INSTALL_FINGERPRINT={fingerprint}",
            docker_run_command(
                'printf "Pier agent install fingerprint: %s\\n" '
                '"$PIER_AGENT_INSTALL_FINGERPRINT"'
            ),
        ]
    )
    dockerfile.extend(dockerfile_install_commands(install, user=user))
    dockerfile.append("")
    dockerfile_path.write_text("\n".join(dockerfile))
    return dockerfile_path


def proxy_environment(
    token: str, host: str, port: int = EGRESS_PROXY_PORT
) -> dict[str, str]:
    proxy_url = f"http://agent:{token}@{host}:{port}"
    return {
        "HTTP_PROXY": proxy_url,
        "HTTPS_PROXY": proxy_url,
        "http_proxy": proxy_url,
        "https_proxy": proxy_url,
        "NO_PROXY": "localhost,127.0.0.1",
        "no_proxy": "localhost,127.0.0.1",
    }


def new_proxy_token() -> str:
    return secrets.token_urlsafe(24)


def squid_bootstrap_command() -> str:
    return r"""#!/usr/bin/env bash
set -eu

printf '%s' "$ALLOWLIST_DOMAINS" | tr ',' '\n' | sed '/^[[:space:]]*$/d' \
  > /tmp/allowed_domains.txt

htpasswd -bc /tmp/squid.passwd agent "$PROXY_TOKEN"

cat > /tmp/squid.conf <<'EOF'
http_port 0.0.0.0:8080
pid_filename /tmp/squid.pid
coredump_dir /tmp

auth_param basic program /usr/lib/squid/basic_ncsa_auth /tmp/squid.passwd
auth_param basic realm PierPolicyProxy
acl authenticated proxy_auth REQUIRED

acl SSL_ports port 443
acl Safe_ports port 80 443
acl CONNECT method CONNECT
acl allowed_domains dstdomain "/tmp/allowed_domains.txt"

http_access deny !Safe_ports
http_access deny CONNECT !SSL_ports
http_access allow authenticated allowed_domains
http_access deny all

cache deny all
access_log stdio:/tmp/squid_access.log
cache_log /tmp/squid_cache.log
log_mime_hdrs off
shutdown_lifetime 1 seconds
EOF

exec squid -N -f /tmp/squid.conf -d 1
"""


def proxy_policy_env(allowlist: NetworkAllowlist, token: str) -> dict[str, str]:
    return {
        "PROXY_TOKEN": token,
        "ALLOWLIST_DOMAINS": ",".join(allowlist.domains),
    }


def write_docker_proxy_compose(
    *,
    path: Path,
    proxy_dir: Path,
    allowlist: NetworkAllowlist,
    token: str,
) -> Path:
    proxy_dir.mkdir(parents=True, exist_ok=True)
    (proxy_dir / "Dockerfile").write_text(
        "\n".join(
            [
                "FROM ubuntu:24.04",
                "RUN apt-get update && "
                "DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "
                "apache2-utils ca-certificates squid && "
                "rm -rf /var/lib/apt/lists/*",
                "COPY start-squid.sh /usr/local/bin/start-squid.sh",
                "RUN chmod +x /usr/local/bin/start-squid.sh",
                'CMD ["bash", "/usr/local/bin/start-squid.sh"]',
                "",
            ]
        )
    )
    (proxy_dir / "start-squid.sh").write_text(squid_bootstrap_command())
    compose = {
        "services": {
            "main": {
                "networks": ["pier-egress-internal"],
                "depends_on": {
                    EGRESS_PROXY_SERVICE: {
                        "condition": "service_healthy",
                    },
                },
            },
            EGRESS_PROXY_SERVICE: {
                "build": {"context": str(proxy_dir.resolve().absolute())},
                "environment": proxy_policy_env(allowlist, token),
                "healthcheck": {
                    "test": ["CMD-SHELL", "bash -lc '</dev/tcp/127.0.0.1/8080'"],
                    "interval": "1s",
                    "timeout": "1s",
                    "retries": 30,
                },
                "networks": ["pier-egress-internal", "default"],
            },
        },
        "networks": {
            "pier-egress-internal": {
                "internal": True,
            },
        },
    }
    if _is_podman():
        # podman-compose raises "missing networks: default" because the egress
        # proxy service references ["pier-egress-internal", "default"] but only
        # pier-egress-internal is declared top-level. Docker auto-creates the
        # default network, so this is podman-only to keep docker output identical.
        compose["networks"]["default"] = {}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(compose, indent=2))
    return path


def shell_export_env(env: dict[str, str]) -> str:
    return " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
