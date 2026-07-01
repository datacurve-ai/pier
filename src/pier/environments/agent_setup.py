from __future__ import annotations

import json
import secrets
import shlex
from pathlib import Path

from pier.models.agent.install import AgentInstallSpec, InstallStep
from pier.models.agent.network import NetworkAllowlist

AGENT_INSTALL_DIR = ".pier-agent-install"
EGRESS_PROXY_SERVICE = "pier-egress-proxy"
EGRESS_PROXY_PORT = 8080


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

LOG_DIR=/var/log/pier-egress-proxy
ALLOWED_DOMAINS_FILE="$LOG_DIR/allowed_domains.txt"

mkdir -p "$LOG_DIR"
touch "$LOG_DIR/squid_access.log" "$LOG_DIR/squid_cache.log"

printf '%s' "$ALLOWLIST_DOMAINS" | tr ',' '\n' | sed '/^[[:space:]]*$/d' \
  > "$ALLOWED_DOMAINS_FILE"
chown -R proxy:proxy "$LOG_DIR"


htpasswd -bc /tmp/squid.passwd agent "$PROXY_TOKEN"

cat > /tmp/squid.conf <<EOF
http_port 0.0.0.0:8080
pid_filename /tmp/squid.pid
coredump_dir /tmp

auth_param basic program /usr/lib/squid/basic_ncsa_auth /tmp/squid.passwd
auth_param basic realm PierPolicyProxy
acl authenticated proxy_auth REQUIRED

acl SSL_ports port 443
acl Safe_ports port 80 443
acl CONNECT method CONNECT
acl allowed_domains dstdomain "$ALLOWED_DOMAINS_FILE"

http_access deny !Safe_ports
http_access deny CONNECT !SSL_ports
http_access allow authenticated allowed_domains
http_access deny all

# Keep restricted inference egress resilient during long agent runs. The Docker
# embedded resolver and Squid's default negative caching can turn one transient
# DNS/connect miss into a burst of immediate CONNECT 503s for an otherwise
# allowed provider host.
dns_nameservers 1.1.1.1 1.0.0.1 8.8.8.8 8.8.4.4
positive_dns_ttl 5 minutes
negative_dns_ttl 5 seconds
connect_timeout 60 seconds
connect_retries 3
forward_timeout 60 seconds

cache deny all
access_log stdio:$LOG_DIR/squid_access.log
cache_log $LOG_DIR/squid_cache.log
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
    proxy_logs_dir = proxy_dir / "logs"
    proxy_logs_dir.mkdir(parents=True, exist_ok=True)
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
                    "test": [
                        "CMD-SHELL",
                        "test -s /tmp/squid.pid && kill -0 $(cat /tmp/squid.pid)",
                    ],
                    "interval": "5s",
                    "timeout": "2s",
                    "retries": 30,
                },
                "volumes": [
                    (
                        f"{proxy_logs_dir.resolve().absolute()}"
                        ":/var/log/pier-egress-proxy"
                    ),
                ],
                "dns": [
                    "1.1.1.1",
                    "1.0.0.1",
                    "8.8.8.8",
                    "8.8.4.4",
                ],
                "networks": ["pier-egress-internal", "default"],
            },
        },
        "networks": {
            "pier-egress-internal": {
                "internal": True,
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(compose, indent=2))
    return path


def shell_export_env(env: dict[str, str]) -> str:
    return " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
