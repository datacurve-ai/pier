from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class NetworkAllowlist(BaseModel):
    domains: list[str] = Field(
        default_factory=list,
        description=(
            "Exact domain or leading-dot suffix allowed by network policy, "
            "for example 'api.anthropic.com' or '.anthropic.com'."
        ),
    )

    endpoints: list[str] = Field(
        default_factory=list,
        description=(
            "An exact host and port allowed by network policy, for example "
            "'192.0.2.10:8000' or 'models.example.com:8080': the endpoint a URL "
            "names on a port other than 80 or 443, which the domains alone do not allow."
        ),
    )

    @field_validator("endpoints")
    @classmethod
    def normalize_endpoints(cls, endpoints: list[str]) -> list[str]:
        normalized: set[str] = set()
        for value in endpoints:
            host, sep, port = value.strip().lower().rpartition(":")
            host = host.rstrip(".")
            if not sep or not host or not port.isdigit() or not 0 < int(port) < 65536:
                raise ValueError("endpoint must be host:port")
            if any(char in host for char in "/*"):
                raise ValueError("endpoint host must be an exact host")
            normalized.add(f"{host}:{int(port)}")
        return sorted(normalized)

    @field_validator("domains")
    @classmethod
    def normalize_domains(cls, domains: list[str]) -> list[str]:
        normalized: set[str] = set()
        for value in domains:
            domain = value.strip().lower().rstrip(".")
            if not domain:
                raise ValueError("domain cannot be empty")
            if any(char in domain for char in "/:*"):
                raise ValueError("domain must be an exact domain or leading-dot suffix")
            normalized.add(domain)
        return sorted(normalized)
