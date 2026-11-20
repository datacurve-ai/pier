from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse

from pier.models.agent.network import NetworkAllowlist


# The ports the egress proxy allows to every allowlisted domain.
DEFAULT_PORTS = frozenset({80, 443})


def _parse_url(value: Any):
    if not isinstance(value, str):
        return None

    raw = value.strip()
    if not raw or raw.startswith("{env:") or raw.startswith("${"):
        return None

    has_scheme = "://" in raw
    if not has_scheme and "." not in raw and raw != "localhost" and ":" not in raw:
        return None

    parsed = urlparse(raw if has_scheme else f"https://{raw}")
    if not parsed.hostname:
        return None
    return parsed


def hostname_from_url(value: Any) -> str | None:
    """Return the hostname for a URL-like config value."""
    parsed = _parse_url(value)
    return parsed.hostname.lower().rstrip(".") if parsed else None


def endpoint_from_url(value: Any) -> str | None:
    """Return ``host:port`` for a URL-like config value that names a port other than
    80 or 443, which the egress proxy does not allow by domain alone."""
    parsed = _parse_url(value)
    try:
        port = parsed.port if parsed else None
    except ValueError:
        return None
    if port is None or port in DEFAULT_PORTS:
        return None
    host = parsed.hostname.lower().rstrip(".")
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


def allowlist_from_urls(
    values: Iterable[Any],
    *,
    default_domains: Iterable[str] = (),
) -> NetworkAllowlist:
    domains = {domain for domain in default_domains if domain}
    endpoints: set[str] = set()
    for value in values:
        if hostname := hostname_from_url(value):
            domains.add(hostname)
        if endpoint := endpoint_from_url(value):
            endpoints.add(endpoint)
    return NetworkAllowlist(domains=sorted(domains), endpoints=sorted(endpoints))


def collect_url_values(value: Any, *, keys: set[str] | None = None) -> list[str]:
    """Collect URL-like values from nested config dictionaries.

    This intentionally keys off common provider/base-url field names instead of
    trying to interpret every string in an arbitrary config blob.
    """
    url_keys = keys or {
        "api",
        "api_base",
        "api_url",
        "base_url",
        "baseurl",
        "baseURL",
        "openai_base_url",
        "url",
    }

    found: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            if isinstance(key, str) and key in url_keys and isinstance(nested, str):
                found.append(nested)
            elif isinstance(nested, dict | list):
                found.extend(collect_url_values(nested, keys=url_keys))
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict | list):
                found.extend(collect_url_values(item, keys=url_keys))
    return found
