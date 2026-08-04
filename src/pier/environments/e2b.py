from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shlex
import stat
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Literal, TypedDict

from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from pier.environments.agent_setup import _run_with_step_env
from pier.environments.base import BaseEnvironment, ExecResult
from pier.environments.capabilities import (
    EnvironmentCapabilities,
    EnvironmentResourceCapabilities,
)
from pier.models.environment_type import EnvironmentType
from pier.models.task.config import EnvironmentConfig
from pier.models.trial.paths import EnvironmentPaths, TrialPaths
from pier.utils.optional_import import MissingExtraError

try:
    import httpcore
    import httpx
    from connectrpc.errors import ConnectError as RpcConnectError
    from e2b import (
        ALL_TRAFFIC,
        AsyncCommandHandle,
        AsyncSandbox,
        AsyncTemplate,
        BuildException,
        BuildInfo,
        CommandExitException,
        FileType,
        NotFoundException,
        RateLimitException,
        SandboxNetworkOpts,
        Template,
        TemplateBuildStatus,
    )

    _API_RETRYABLE: tuple[type[BaseException], ...] = (
        httpcore.ConnectError,
        httpcore.ConnectTimeout,
        httpcore.PoolTimeout,
        # httpx's HTTP/2 layer raises LocalProtocolError ("Invalid input
        # ConnectionInputs.RECV_HEADERS in state ConnectionState.CLOSED")
        # when a pooled connection the server already closed is reused.
        # API calls guarded by this list are idempotent, so retrying is safe.
        httpcore.LocalProtocolError,
        httpx.ConnectError,
        httpx.LocalProtocolError,
        httpx.ReadError,
        httpx.RemoteProtocolError,
        httpx.TimeoutException,
        RateLimitException,
    )
    # Command dispatch and sandbox creation are not idempotent: a ReadError
    # or timeout can arrive after envd already started the process (or after
    # E2B already provisioned the sandbox — there is no idempotency key, so a
    # retried create makes a second sandbox and leaks the first as billing
    # until its TTL). Only retry errors raised before the request is sent.
    _COMMAND_DISPATCH_RETRYABLE: tuple[type[BaseException], ...] = (
        httpcore.ConnectError,
        httpcore.ConnectTimeout,
        httpcore.PoolTimeout,
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.PoolTimeout,
        RateLimitException,
    )
    _COMMAND_STREAM_RETRYABLE: tuple[type[BaseException], ...] = (RpcConnectError,)
    _HAS_E2B = True
except ImportError:
    _API_RETRYABLE = ()
    _COMMAND_DISPATCH_RETRYABLE = ()
    _COMMAND_STREAM_RETRYABLE = ()
    _HAS_E2B = False


# v5: agent installs default to root and Docker context fingerprints include
# filesystem metadata that affects COPY semantics.
_TEMPLATE_SCHEMA_VERSION = "pier-e2b-v5"
_AGENT_FINGERPRINT_PATH = "/etc/pier/agent-install-fingerprint"
_IMAGE_CONFIG_PATH = "/etc/pier/image-config.json"
# E2B template names (alias included) are capped at 128 characters.
_TEMPLATE_NAME_MAX_LEN = 128
_TEMPLATE_LOCKS: dict[str, asyncio.Lock] = {}
_IMAGE_ENV_CACHE: dict[str, dict[str, str]] = {}
_IMAGE_ENV_LOCKS: dict[str, asyncio.Lock] = {}
# E2B enforces a team-wide cap on concurrent template builds -- 20 is default
_BUILD_CONCURRENCY = int(os.environ.get("PIER_E2B_MAX_CONCURRENT_BUILDS", "20"))
_BUILD_SEMAPHORE: asyncio.Semaphore | None = None


class _WriteEntry(TypedDict):
    path: str
    data: BinaryIO


def _build_semaphore() -> asyncio.Semaphore:
    global _BUILD_SEMAPHORE
    if _BUILD_SEMAPHORE is None:
        _BUILD_SEMAPHORE = asyncio.Semaphore(_BUILD_CONCURRENCY)
    return _BUILD_SEMAPHORE


def _template_lock(name: str) -> asyncio.Lock:
    """Serialize same-template builds inside one Pier process."""
    return _TEMPLATE_LOCKS.setdefault(name, asyncio.Lock())


def _image_env_lock(ref: str) -> asyncio.Lock:
    return _IMAGE_ENV_LOCKS.setdefault(ref, asyncio.Lock())


def _join_continuations(dockerfile_text: str) -> str:
    """Fold backslash line continuations so each instruction is one line."""
    return re.sub(r"\\\s*\n", " ", dockerfile_text)


_DOCKER_ENV_REF = re.compile(
    r"(?<!\\)\$(?:"
    r"\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?:(?P<operator>:-|:\+)(?P<word>[^}]*))?\}"
    r"|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))"
)
_ESCAPED_DOLLAR = "\ue000"


def _interpolate_docker_value(value: str, resolved: dict[str, str]) -> str:
    """Expand the Dockerfile variable forms used by ENV/WORKDIR/USER/FROM."""

    def replace(match: re.Match[str]) -> str:
        name = match.group("braced") or match.group("plain")
        current = resolved.get(name, "")
        operator = match.group("operator")
        word = match.group("word") or ""
        if operator == ":-":
            return current or _interpolate_docker_value(word, resolved)
        if operator == ":+":
            return _interpolate_docker_value(word, resolved) if current else ""
        return current

    return _DOCKER_ENV_REF.sub(replace, value).replace(r"\$", "$")


def _parse_dockerfile(
    dockerfile_text: str,
) -> tuple[str | None, list[str], str | None, str | None]:
    """Return (base image, ENV lines, last WORKDIR, last USER) of a single-stage Dockerfile.

    Multi-stage Dockerfiles are rejected up front in ``_validate_definition``
    because E2B's ``from_dockerfile`` does not support them.
    """
    base: str | None = None
    env_lines: list[str] = []
    workdir: str | None = None
    user: str | None = None
    args: dict[str, str] = {}
    resolved_env: dict[str, str] = {}
    for raw in _join_continuations(dockerfile_text).splitlines():
        line = raw.strip()
        upper = line.upper()
        if upper.startswith("ARG "):
            assignment = line.split(None, 1)[1]
            key, separator, value = assignment.partition("=")
            if separator:
                args[key.strip()] = _interpolate_docker_value(
                    value.strip(), {**args, **resolved_env}
                )
        elif upper.startswith("FROM "):
            tokens = [t for t in line.split()[1:] if not t.startswith("--")]
            if tokens:
                base = _interpolate_docker_value(tokens[0], args) or None
        elif upper.startswith("ENV "):
            assignment = line[4:]
            env_lines.append(assignment)
            resolved_env.update(_parse_env_assignments(assignment, resolved_env))
        elif upper.startswith("WORKDIR "):
            value = _interpolate_docker_value(
                line.split(None, 1)[1].strip("\"'"),
                {**args, **resolved_env},
            )
            if value.startswith("/") or workdir is None:
                workdir = value
            else:
                workdir = str(PurePosixPath(workdir) / value)
        elif upper.startswith("USER "):
            user = _interpolate_docker_value(
                line.split(None, 1)[1].strip("\"'"),
                {**args, **resolved_env},
            )
    return base, env_lines, workdir, user


def _resolve_from_instruction(dockerfile_text: str, base_image: str | None) -> str:
    """Replace an ARG-backed single-stage FROM with its resolved image."""
    if base_image is None:
        raise ValueError("E2B could not resolve the Dockerfile's FROM image.")
    return re.sub(
        r"(?im)^(?P<prefix>[ \t]*FROM[ \t]+(?:--[^ \t]+[ \t]+)*)[^ \t\n]+",
        lambda match: f"{match.group('prefix')}{base_image}",
        dockerfile_text,
        count=1,
    )


def _strip_boot_instructions(dockerfile_text: str) -> str:
    """Drop CMD/ENTRYPOINT so E2B's ``from_dockerfile`` does not turn the
    task boot command into a template start command (run and snapshotted
    during template creation, plus a readiness delay). Pier's Docker backend
    likewise overrides the task command with ``sleep infinity``."""
    # Continuation lines (ending in a backslash) are consumed first, then the
    # final line of the instruction.
    return re.sub(
        r"(?im)^[ \t]*(?:CMD|ENTRYPOINT)\b(?:[^\n]*\\[ \t]*\n)*[^\n]*\n?",
        "",
        dockerfile_text,
    )


def _is_retryable_registry_error(exc: BaseException) -> bool:
    """Registries throttle bursts of anonymous pulls (ECR returns HTTP 429
    under concurrent sweeps); raise_for_status surfaces that as
    HTTPStatusError, which the connection-level _API_RETRYABLE misses."""
    if isinstance(exc, _API_RETRYABLE):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and (
        exc.response.status_code == 429 or exc.response.status_code >= 500
    )


def _normalize_user(user: str | None) -> str | None:
    """Docker's USER may carry a group ("user:group", "uid:gid"); command
    execution wants just the user part. Empty means the image default (root).
    """
    if not user:
        return None
    return user.split(":", 1)[0].strip() or None


def _numeric_uid(user: str | int | None) -> int | None:
    value = str(user) if user is not None else ""
    return int(value) if value.isdigit() else None


def _template_command_as_user(command: str, user: str | int) -> tuple[str, str]:
    """Resolve numeric UIDs inside the template, where envd requires a name."""
    uid = _numeric_uid(user)
    if uid is None:
        return command, str(user)
    wrapped = (
        f'__pier_user="$(getent passwd {uid} | cut -d: -f1)"; '
        f'if [ -z "$__pier_user" ]; then '
        f"echo 'E2B cannot run as UID {uid}: no passwd entry' >&2; exit 1; fi; "
        f'exec su -m "$__pier_user" -s /bin/bash -c {shlex.quote(command)}'
    )
    return wrapped, "root"


def _parse_env_assignments(text: str, resolved: dict[str, str]) -> dict[str, str]:
    """Parse Dockerfile ENV assignments, interpolating against ``resolved``."""
    parsed: dict[str, str] = {}

    try:
        tokens = shlex.split(text.replace(r"\$", _ESCAPED_DOLLAR))
    except ValueError:
        tokens = text.replace(r"\$", _ESCAPED_DOLLAR).split()
    if tokens and "=" not in tokens[0]:
        # Legacy `ENV key value` form: everything after the key is the value.
        key = tokens[0]
        value = " ".join(tokens[1:])
        parsed[key] = _interpolate_docker_value(value, resolved).replace(
            _ESCAPED_DOLLAR, "$"
        )
        return parsed
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        # Docker expands every value against the environment as it existed
        # before this ENV instruction, not against sibling assignments.
        parsed[key] = _interpolate_docker_value(value, resolved).replace(
            _ESCAPED_DOLLAR, "$"
        )
    return parsed


def _parse_image_ref(image_ref: str) -> tuple[str, str, str]:
    """Split an image reference into (registry host, repository, tag/digest)."""
    name, digest = image_ref.split("@", 1) if "@" in image_ref else (image_ref, None)
    first, _, rest = name.partition("/")
    if rest and ("." in first or ":" in first or first == "localhost"):
        host, repo = first, rest
    else:
        host, repo = "registry-1.docker.io", name
        if "/" not in repo:
            repo = f"library/{repo}"
    if digest is not None:
        # A ``name:tag@digest`` ref pulls by digest; drop the tag so the
        # manifest URL stays a bare repository path.
        tagless, separator, _ = repo.rpartition(":")
        return host, tagless if separator else repo, digest
    repo, _, tag = repo.rpartition(":")
    if not repo:
        repo, tag = tag, "latest"
    return host, repo, tag


async def _fetch_image_config(image_ref: str) -> dict:
    """Fetch ENV, WorkingDir, and User from an image's registry config
    (anonymous pull only).

    E2B's envd exposes none of the Docker image config's ENV, WORKDIR, or
    USER to sandbox commands, unlike ``docker exec`` (which starts in the
    image WORKDIR with the image ENV as the image USER). Pier re-applies all
    three per command so tasks behave identically across execution backends.
    """
    host, repo, reference = _parse_image_ref(image_ref)
    accept = ", ".join(
        [
            "application/vnd.docker.distribution.manifest.v2+json",
            "application/vnd.docker.distribution.manifest.list.v2+json",
            "application/vnd.oci.image.manifest.v1+json",
            "application/vnd.oci.image.index.v1+json",
        ]
    )
    async with httpx.AsyncClient(
        follow_redirects=True, timeout=30, max_redirects=5
    ) as client:
        headers = {"Accept": accept}

        async def get(url: str) -> httpx.Response:
            response = await client.get(url, headers=headers)
            if response.status_code == 401:
                challenge = response.headers.get("www-authenticate", "")
                params = dict(
                    re.findall(r'(\w+)="([^"]*)"', challenge.partition(" ")[2])
                )
                realm = params.pop("realm", None)
                if realm:
                    # The realm comes from the registry's response; never
                    # follow it to a plaintext endpoint (SSRF hardening —
                    # cloud metadata services are plain HTTP).
                    if not realm.startswith("https://"):
                        raise RuntimeError(
                            f"Refusing non-HTTPS registry auth realm "
                            f"{realm!r} for image {image_ref!r}."
                        )
                    token_response = await client.get(
                        realm,
                        params={
                            **params,
                            "scope": params.get("scope", f"repository:{repo}:pull"),
                        },
                    )
                    token_response.raise_for_status()
                    token = token_response.json().get("token") or (
                        token_response.json().get("access_token")
                    )
                    headers["Authorization"] = f"Bearer {token}"
                    response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response

        manifest_url = f"https://{host}/v2/{repo}/manifests/{reference}"
        manifest = (await get(manifest_url)).json()
        if "manifests" in manifest:  # multi-arch index: E2B sandboxes are amd64
            digest = next(
                entry["digest"]
                for entry in manifest["manifests"]
                if entry.get("platform", {}).get("architecture") == "amd64"
                and entry.get("platform", {}).get("os") == "linux"
            )
            manifest = (
                await get(f"https://{host}/v2/{repo}/manifests/{digest}")
            ).json()
        config_digest = manifest["config"]["digest"]
        config = (await get(f"https://{host}/v2/{repo}/blobs/{config_digest}")).json()

    env: dict[str, str] = {}
    for entry in config.get("config", {}).get("Env") or []:
        key, _, value = entry.partition("=")
        env[key] = value
    return {
        "env": env,
        "workdir": config.get("config", {}).get("WorkingDir") or None,
        "user": _normalize_user(config.get("config", {}).get("User")),
    }


def _egress_rules(domains: list[str]) -> list[str]:
    """Translate Pier allowlist domains into E2B egress rule syntax.

    Pier's leading-dot suffix domains (``.amazonaws.com``) mean "the domain
    and every subdomain"; E2B expresses subdomain matches as ``*.domain``
    wildcards, so each suffix entry expands to the bare domain plus its
    wildcard.
    """
    rules: list[str] = []
    for domain in sorted(set(domains)):
        if domain.startswith("."):
            base = domain.lstrip(".")
            rules.extend([base, f"*.{base}"])
        else:
            rules.append(domain)
    return rules


class E2BEnvironment(BaseEnvironment):
    """Pier execution environment backed by reusable E2B templates.

    Template identity includes the task image/context, requested resources,
    integration schema, and Pier's agent-install fingerprint, so repeated
    trials of the same task+agent reuse a prepared template instead of
    installing the coding agent inside every billable sandbox. Changing any of
    those inputs mints a new template automatically. Identity hashes the
    image *reference*, not a resolved manifest digest: a mutable tag
    (``:latest``) that moves upstream keeps matching the cached template.
    Pin task images, or pass ``force_build`` to rebuild (which also refetches
    the image config).

    Keyword args (via ``environment.kwargs``):

    - ``template_prefix``: namespace prefix for template names.
    - ``template_mode``: ``"build-if-missing"`` (default) builds the template
      on first use; ``"required"`` fails closed when the template is absent —
      prewarm each unique task/agent pair once, then require cache hits in CI.
    - ``sandbox_timeout_secs``: sandbox TTL (default two hours). On expiry the
      sandbox pauses with memory intact and auto-resumes on the next request,
      so longer trials survive; if Pier dies before cleanup, compute billing
      stops at the TTL and the paused sandbox remains until deleted.

    Network allowlists map to native E2B filtered egress (suffix domains
    become ``*.domain`` wildcards); tasks with ``allow_internet = false`` and
    no allowlist get no outbound network. Isolated tasks also disable E2B's
    public per-port ingress URLs so agent-bound ports are not reachable from
    the internet. E2B domain filtering covers HTTP (80) and TLS (443) only.

    Sandbox commands run with the Docker image config's ENV re-applied, the
    image WORKDIR as the default cwd, and the image USER as the default user:
    envd exposes none of these to commands,
    starts in ``$HOME`` instead of the image WORKDIR, and does not protect
    API-passed env vars from login-shell profile scripts. Pier resolves all
    three from the image registry once at template build time and bakes them
    into the template (so warm trials never depend on registry availability),
    then injects them per command. See ``_resolve_image_config``,
    ``_load_image_config``, and ``_export_prefix``. Registry access is
    anonymous-pull only, and only needed when a template is first built.

    Task Dockerfile CMD/ENTRYPOINT instructions are ignored: E2B would otherwise
    run them while creating the reusable template and snapshot their state.
    Pier's Docker backend similarly overrides the boot command with
    ``sleep infinity``, though it leaves image ENTRYPOINTs in place — tasks
    must not rely on CMD/ENTRYPOINT side effects under E2B.

    ``PIER_E2B_MAX_CONCURRENT_BUILDS`` caps concurrent template builds per
    Pier process (default 20, matching E2B's default team-wide concurrent
    build limit).
    """

    _UPLOAD_BATCH_SIZE = 20
    _DOWNLOAD_CONCURRENCY = 8

    @classmethod
    def preflight(cls) -> None:
        if not os.environ.get("E2B_API_KEY"):
            raise SystemExit(
                "E2B requires E2B_API_KEY to be set. Set it and try again."
            )

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.E2B

    @classmethod
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        # E2B CPU and memory are fixed on the template build. Pier models these as
        # resource requests rather than container-style cgroup ceilings.
        return EnvironmentResourceCapabilities(
            cpu_request=True,
            memory_request=True,
        )

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(
            disable_internet=True,
            filtered_egress=True,
            preinstall_agents=True,
        )

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        *,
        template_prefix: str = "pier",
        template_mode: Literal["build-if-missing", "required"] = "build-if-missing",
        sandbox_timeout_secs: int = 2 * 60 * 60,
        **kwargs,
    ):
        if not _HAS_E2B:
            raise MissingExtraError(package="e2b", extra="e2b")
        if template_mode not in {"build-if-missing", "required"}:
            raise ValueError(
                "E2B template_mode must be 'build-if-missing' or 'required'."
            )

        self._template_prefix = template_prefix
        self._template_mode = template_mode
        self._sandbox_timeout_secs = sandbox_timeout_secs
        self._sandbox: AsyncSandbox | None = None
        self._template_cache_hit = False
        self._template_build_seconds = 0.0
        self._sandbox_create_seconds = 0.0
        self._image_env: dict[str, str] | None = None
        self._image_workdir: str | None = None
        self._image_user: str | None = None
        self._uid_name_cache: dict[int, str] = {}

        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            **kwargs,
        )

        self._template_name = self._build_template_name()

    @property
    def template_name(self) -> str:
        return self._template_name

    @property
    def _environment_definition_path(self) -> Path:
        return self.environment_dir / "Dockerfile"

    def _validate_definition(self) -> None:
        if (self.environment_dir / "docker-compose.yaml").exists():
            raise ValueError("E2B does not support Docker Compose task environments.")
        if self.task_env_config.docker_image:
            return
        if not self._environment_definition_path.exists():
            raise FileNotFoundError(
                f"{self._environment_definition_path} not found. "
                "E2B supports a prebuilt docker_image or a single Dockerfile task."
            )
        dockerfile = _join_continuations(self._environment_definition_path.read_text())
        from_count = sum(
            1
            for line in dockerfile.splitlines()
            if line.strip().upper().startswith("FROM ")
        )
        if from_count > 1:
            # Fail fast: E2B's from_dockerfile rejects multi-stage builds,
            # but only once the template build starts.
            raise ValueError(
                "E2B does not support multi-stage Dockerfile task environments."
            )

    def _template_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(f"{_TEMPLATE_SCHEMA_VERSION}\0".encode())
        # storage_mb is deliberately absent: E2B disk size is fixed by plan
        # tier, with no knob on template build or sandbox create.
        digest.update(
            (
                f"cpu={self._effective_cpus or 2}\0"
                f"memory={self._effective_memory_mb or 1024}\0"
                f"user={self._resolve_user(None)}\0"
            ).encode()
        )

        if self.task_env_config.docker_image:
            digest.update(f"image={self.task_env_config.docker_image}\0".encode())
        else:
            for path in sorted(self.environment_dir.rglob("*")):
                relative = path.relative_to(self.environment_dir).as_posix()
                digest.update(f"path={relative}\0".encode())
                metadata = path.lstat()
                digest.update(f"mode={stat.S_IMODE(metadata.st_mode):o}\0".encode())
                if stat.S_ISLNK(metadata.st_mode):
                    digest.update(f"symlink={os.readlink(path)}\0".encode())
                elif stat.S_ISREG(metadata.st_mode):
                    digest.update(b"file\0")
                    digest.update(path.read_bytes())
                elif stat.S_ISDIR(metadata.st_mode):
                    digest.update(b"directory\0")
                else:
                    digest.update(f"type={stat.S_IFMT(metadata.st_mode):o}\0".encode())
                digest.update(b"\0")

        if self.agent_install_spec is not None:
            digest.update(f"agent={self.agent_install_spec.fingerprint()}\0".encode())
        else:
            digest.update(b"agent=none\0")
        return digest.hexdigest()[:20]

    def _build_template_name(self) -> str:
        role = "agent" if self.agent_install_spec is not None else "verifier"
        slug = re.sub(r"[^a-z0-9_-]+", "-", self.environment_name.lower()).strip("-")
        prefix = re.sub(r"[^a-z0-9_-]+", "-", self._template_prefix.lower()).strip("-")
        prefix = (prefix or "pier")[:32]
        suffix = f"-{role}-{self._template_fingerprint()}"
        slug_budget = _TEMPLATE_NAME_MAX_LEN - len(prefix) - len(suffix) - 1
        slug = (slug or "task")[:slug_budget]
        return f"{prefix}-{slug}{suffix}"

    async def _template_definition(self, *, refresh_image_config: bool = False):
        image_env, image_workdir, image_user = await self._resolve_image_config(
            refresh=refresh_image_config
        )

        if self.task_env_config.docker_image:
            definition = Template().from_image(self.task_env_config.docker_image)
        else:
            # CMD/ENTRYPOINT are stripped: E2B's from_dockerfile would turn
            # them into a template start command, running the task boot
            # command during template creation (see _strip_boot_instructions).
            dockerfile_text = _strip_boot_instructions(
                self._environment_definition_path.read_text()
            )
            base_image, _, _, _ = _parse_dockerfile(dockerfile_text)
            # This conversion targets the Deep SWE Dockerfile shape, not general
            # Dockerfile semantic parity. Runtime config is corrected below;
            # variable-dependent build-time WORKDIR/USER/ENV would need a fuller
            # conversion before those constructs are added to task Dockerfiles.
            definition = Template(
                file_context_path=str(self.environment_dir)
            ).from_dockerfile(_resolve_from_instruction(dockerfile_text, base_image))

        # Bake the docker-exec-equivalent ENV/WORKDIR into the template:
        # install steps below then run with the image env (docker build
        # layering), attach sessions inherit them as sandbox defaults, and
        # exec() reads the JSON copy at runtime with no registry round-trip.
        if image_env:
            definition = definition.set_envs(image_env)
        if image_workdir:
            definition = definition.set_workdir(image_workdir)
        config_json = json.dumps(
            {"env": image_env, "workdir": image_workdir, "user": image_user}
        )
        definition = definition.run_cmd(
            f"mkdir -p /etc/pier && printf '%s' {shlex.quote(config_json)} "
            f"> {_IMAGE_CONFIG_PATH}",
            user="root",
        )

        install = self.agent_install_spec
        if install is None:
            return definition

        # Match Docker, Modal, and Daytona: an explicit task agent user wins;
        # otherwise agent installation and verification run as root.
        agent_user = self._resolve_user(None) or "root"
        for step in install.steps:
            user = "root" if step.user == "root" else agent_user
            command, template_user = _template_command_as_user(
                _run_with_step_env(step), user
            )
            definition = definition.run_cmd(command, user=template_user)

        if install.verification_command:
            command, template_user = _template_command_as_user(
                install.verification_command, agent_user
            )
            definition = definition.run_cmd(
                command,
                user=template_user,
            )

        marker = shlex.quote(install.fingerprint())
        definition = definition.run_cmd(
            f"mkdir -p /etc/pier && printf '%s\\n' {marker} "
            f"> {_AGENT_FINGERPRINT_PATH}",
            user="root",
        )
        return definition

    def _on_build_log(self, entry) -> None:
        self.logger.debug("E2B template %s: %s", self._template_name, entry)

    async def _template_exists(self) -> bool:
        # E2B creates the bare alias before the build is runnable. Requiring the
        # default tag avoids treating an in-progress build as a warm cache hit.
        return await AsyncTemplate.exists(f"{self._template_name}:default")

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(multiplier=1, max=10),
        retry=retry_if_exception_type(_API_RETRYABLE),
        reraise=True,
    )
    async def _get_build_status(self, build_info: BuildInfo, logs_offset: int):
        return await AsyncTemplate.get_build_status(
            build_info,
            logs_offset=logs_offset,
        )

    async def _build_template(self, *, force_build: bool) -> None:
        build_info = await AsyncTemplate.build_in_background(
            # A force build must refetch the image config too, or a rebuild
            # triggered to pick up a moved mutable tag would bake stale
            # ENV/WORKDIR/USER from this process's cache.
            await self._template_definition(refresh_image_config=force_build),
            self._template_name,
            cpu_count=self._effective_cpus or 2,
            memory_mb=self._effective_memory_mb or 1024,
            skip_cache=force_build,
            on_build_logs=self._on_build_log,
        )
        logs_offset = 0
        while True:
            status = await self._get_build_status(build_info, logs_offset)
            logs_offset += len(status.log_entries)
            for entry in status.log_entries:
                self._on_build_log(entry)

            if status.status == TemplateBuildStatus.READY:
                return
            if status.status == TemplateBuildStatus.ERROR:
                message = (
                    status.reason.message
                    if status.reason is not None
                    else "E2B template build failed"
                )
                raise BuildException(message)
            await asyncio.sleep(1)

    async def _ensure_template(self, *, force_build: bool) -> None:
        started = time.monotonic()
        async with _template_lock(self._template_name):
            exists = await self._template_exists()
            self._template_cache_hit = exists and not force_build
            if self._template_cache_hit:
                self._template_build_seconds = time.monotonic() - started
                return

            if self._template_mode == "required" and not force_build:
                raise RuntimeError(
                    f"Required E2B template {self._template_name!r} does not exist. "
                    "Run once with template_mode='build-if-missing' or force_build=true."
                )

            async with _build_semaphore():
                await self._build_template(force_build=force_build)
            self._template_build_seconds = time.monotonic() - started

    def _sandbox_network_options(self) -> SandboxNetworkOpts:
        # E2B exposes a public per-port ingress URL for every sandbox.
        # Public traffic always requires the sandbox's access token, independent
        # of the task's outbound internet policy.
        options: SandboxNetworkOpts = {"allow_public_traffic": False}
        if not self.task_env_config.allow_internet and self.network_allowlist.domains:
            options["allow_out"] = _egress_rules(self.network_allowlist.domains)
            options["deny_out"] = [ALL_TRAFFIC]
        return options

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_random_exponential(multiplier=1, max=10),
        # Sandbox creation is not idempotent (no idempotency key in the E2B
        # API): an error after the request is sent may mean the sandbox was
        # provisioned, and a retry would create a second one and leak the
        # first. Retry only errors raised before the request goes out.
        retry=retry_if_exception_type(_COMMAND_DISPATCH_RETRYABLE),
        reraise=True,
    )
    async def _create_sandbox(self) -> None:
        create_task = asyncio.ensure_future(
            AsyncSandbox.create(
                template=self._template_name,
                metadata={
                    "environment_name": self.environment_name,
                    "session_id": self.session_id,
                    "pier_template_schema": _TEMPLATE_SCHEMA_VERSION,
                },
                envs=self._persistent_env or None,
                timeout=self._sandbox_timeout_secs,
                # TTL expiry pauses (memory intact) instead of killing, and any
                # envd activity auto-resumes. A trial that outlives the TTL
                # self-heals: the command stream drops on pause and the
                # PID-reconnect in exec() resumes the sandbox. If Pier dies,
                # compute billing still stops at the TTL.
                lifecycle={"on_timeout": "pause", "auto_resume": True},
                allow_internet_access=(
                    self.task_env_config.allow_internet
                    or bool(self.network_allowlist.domains)
                ),
                network=self._sandbox_network_options(),
            )
        )
        try:
            self._sandbox = await asyncio.shield(create_task)
        except asyncio.CancelledError:
            # A trial start timeout can cancel us mid-create. Capture the
            # sandbox if the API call still completes so start()'s cleanup
            # can kill it instead of leaking a billing sandbox until its TTL.
            try:
                self._sandbox = await asyncio.wait_for(create_task, timeout=30)
            except BaseException:
                create_task.cancel()
            raise

    def _runtime_image_ref(
        self,
    ) -> tuple[str | None, list[str], str | None, str | None]:
        """The image whose config governs runtime commands, plus any ENV
        lines, WORKDIR, and USER the task Dockerfile layers on top of it."""
        if self.task_env_config.docker_image:
            return self.task_env_config.docker_image, [], None, None
        return _parse_dockerfile(self._environment_definition_path.read_text())

    @retry(
        stop=stop_after_attempt(6),
        wait=wait_random_exponential(multiplier=1, max=30),
        retry=retry_if_exception(_is_retryable_registry_error),
        reraise=True,
    )
    async def _fetch_image_config(self, image_ref: str) -> dict:
        return await _fetch_image_config(image_ref)

    async def _resolve_image_config(
        self, *, refresh: bool = False
    ) -> tuple[dict[str, str], str | None, str | None]:
        """Resolve the Docker image config ENV, WORKDIR, and USER that
        ``docker exec`` would provide. envd applies none of them to sandbox
        commands, so exec() merges the ENV back in (at the lowest
        precedence), defaults the cwd to the image WORKDIR, and runs as the
        image USER for backend equivalence. Called at template build time
        only; the result is baked into the template."""
        image_ref, env_lines, dockerfile_workdir, dockerfile_user = (
            self._runtime_image_ref()
        )
        if image_ref is None:
            return {}, dockerfile_workdir, _normalize_user(dockerfile_user)
        async with _image_env_lock(image_ref):
            if refresh or image_ref not in _IMAGE_ENV_CACHE:
                try:
                    _IMAGE_ENV_CACHE[image_ref] = await self._fetch_image_config(
                        image_ref
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to read the image config for {image_ref!r}. "
                        "Pier needs it to replicate the image ENV "
                        "(PYTHONPATH, PATH, ...), WORKDIR, and USER inside "
                        "E2B sandbox commands."
                    ) from exc
        cached = _IMAGE_ENV_CACHE[image_ref]
        env = dict(cached["env"])
        for line in env_lines:
            env.update(_parse_env_assignments(line, env))
        user = _normalize_user(dockerfile_user) or cached.get("user")
        workdir = dockerfile_workdir or cached["workdir"]
        if workdir and not workdir.startswith("/"):
            workdir = str(PurePosixPath(cached["workdir"] or "/") / workdir)
        return env, workdir, user

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(multiplier=1, max=10),
        retry=retry_if_exception_type(_API_RETRYABLE),
        reraise=True,
    )
    async def _read_image_config(self) -> str:
        if self._sandbox is None:
            raise RuntimeError("Sandbox not found. Start the environment first.")
        return await self._sandbox.files.read(
            _IMAGE_CONFIG_PATH, format="text", user="root"
        )

    async def _load_image_config(self) -> None:
        """Load the ENV/WORKDIR/USER the template build baked into the
        sandbox."""
        try:
            raw = await self._read_image_config()
        except NotFoundException as exc:
            raise RuntimeError(
                f"E2B template {self._template_name!r} does not contain "
                f"{_IMAGE_CONFIG_PATH}; rebuild it with force_build=true."
            ) from exc
        config = json.loads(raw)
        self._image_env = dict(config.get("env") or {})
        self._image_workdir = config.get("workdir")
        self._image_user = _normalize_user(config.get("user"))
        if _numeric_uid(self._image_user) is not None:
            self._image_user = await self._resolve_e2b_user(self._image_user)

    def _merge_env(self, env: dict[str, str] | None) -> dict[str, str] | None:
        merged = dict(self._image_env or {})
        explicit = super()._merge_env(env)
        if explicit:
            merged.update(explicit)
        return merged or None

    async def _resolve_e2b_user(self, user: str | int | None) -> str:
        resolved = self._resolve_user(user)
        if resolved is None:
            resolved = self._image_user or "root"
        uid = _numeric_uid(resolved)
        if uid is None:
            return str(resolved)
        if uid in self._uid_name_cache:
            return self._uid_name_cache[uid]
        if self._sandbox is None:
            raise RuntimeError("Sandbox not found. Start the environment first.")
        try:
            result = await self._sandbox.commands.run(
                f"getent passwd {uid} | cut -d: -f1",
                cwd="/",
                user="root",
                timeout=15,
            )
        except CommandExitException as exc:
            raise RuntimeError(
                f"E2B cannot run as UID {uid}: the image has no passwd entry."
            ) from exc
        name = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        if not name:
            raise RuntimeError(
                f"E2B cannot run as UID {uid}: the image has no passwd entry."
            )
        self._uid_name_cache[uid] = name
        return name

    async def _verify_preinstalled_agent(self) -> None:
        install = self.agent_install_spec
        if install is None:
            return
        expected = shlex.quote(install.fingerprint())
        result = await self.exec(
            f'test "$(cat {_AGENT_FINGERPRINT_PATH} 2>/dev/null)" = {expected}',
            timeout_sec=15,
            user="root",
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"E2B template {self._template_name!r} does not contain the "
                f"expected Pier agent fingerprint {install.fingerprint()}."
            )

    async def start(self, force_build: bool) -> None:
        total_started = time.monotonic()
        await self._ensure_template(force_build=force_build)

        create_started = time.monotonic()
        try:
            await self._create_sandbox()
            self._sandbox_create_seconds = time.monotonic() - create_started
            await self._load_image_config()
            await self._verify_preinstalled_agent()
            await self.exec(
                f"mkdir -p {EnvironmentPaths.agent_dir} "
                f"{EnvironmentPaths.verifier_dir} {EnvironmentPaths.artifacts_dir} "
                f"&& chmod 777 {EnvironmentPaths.agent_dir} "
                f"{EnvironmentPaths.verifier_dir} {EnvironmentPaths.artifacts_dir}",
                user="root",
            )
        except BaseException:
            # BaseException so a trial start timeout (CancelledError) also
            # kills the sandbox instead of leaking it until its TTL. Shielded
            # so a second cancellation cannot interrupt the kill.
            await asyncio.shield(self.stop(delete=True))
            raise

        self.logger.info(
            "E2B startup metrics template=%s cache_hit=%s template_sec=%.3f "
            "sandbox_sec=%.3f total_sec=%.3f",
            self._template_name,
            self._template_cache_hit,
            self._template_build_seconds,
            self._sandbox_create_seconds,
            time.monotonic() - total_started,
        )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(multiplier=1, max=10),
        retry=retry_if_exception_type(_API_RETRYABLE),
        reraise=True,
    )
    async def _kill_sandbox(self) -> None:
        if self._sandbox is None:
            return
        try:
            await self._sandbox.kill()
        except NotFoundException:
            # Already gone; nothing left to bill.
            pass

    async def stop(self, delete: bool):
        if not delete:
            self.logger.info(
                "E2B trial sandboxes are ephemeral; stop always terminates them."
            )
        if self._sandbox is None:
            return
        try:
            await self._kill_sandbox()
        except Exception as exc:
            sandbox_id = getattr(self._sandbox, "sandbox_id", "unknown")
            self.logger.warning(
                "Error stopping E2B sandbox %s after retries; run "
                "'e2b sandbox kill %s' to stop billing: %s",
                sandbox_id,
                sandbox_id,
                exc,
            )
        finally:
            self._sandbox = None

    async def attach(self) -> None:
        if self._sandbox is None:
            raise RuntimeError("Sandbox not found. Start the environment first.")
        os.execvp("e2b", ["e2b", "sandbox", "connect", self._sandbox.sandbox_id])

    # File transfers run as root to match the Docker-cp semantics of Pier's
    # other backends: targets like /tests, /solution, and the task workdir are
    # typically root-owned, and E2B's default file-operation user is "user".

    async def upload_file(self, source_path: Path | str, target_path: str):
        if self._sandbox is None:
            raise RuntimeError("Sandbox not found. Start the environment first.")
        with Path(source_path).open("rb") as source:
            await self._sandbox.files.write(target_path, source, user="root")

    async def upload_dir(self, source_dir: Path | str, target_dir: str):
        if self._sandbox is None:
            raise RuntimeError("Sandbox not found. Start the environment first.")
        source = Path(source_dir)
        files: list[_WriteEntry] = []
        handles = []

        async def flush() -> None:
            if not files:
                return
            try:
                await self._sandbox.files.write_files(files, user="root")
            finally:
                for handle in handles:
                    handle.close()
                files.clear()
                handles.clear()

        try:
            for path in source.rglob("*"):
                if path.is_file():
                    handle = path.open("rb")
                    handles.append(handle)
                    files.append(
                        _WriteEntry(
                            path=str(
                                PurePosixPath(target_dir)
                                / path.relative_to(source).as_posix()
                            ),
                            data=handle,
                        )
                    )
                    if len(files) == self._UPLOAD_BATCH_SIZE:
                        await flush()
            await flush()
        finally:
            for handle in handles:
                handle.close()

    async def download_file(self, source_path: str, target_path: Path | str):
        if self._sandbox is None:
            raise RuntimeError("Sandbox not found. Start the environment first.")
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        stream = await self._sandbox.files.read(
            source_path, format="stream", user="root"
        )
        async with stream:
            with target.open("wb") as destination:
                async for chunk in stream:
                    destination.write(chunk)

    async def download_dir(self, source_dir: str, target_dir: Path | str):
        if self._sandbox is None:
            raise RuntimeError("Sandbox not found. Start the environment first.")
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        # envd's list API needs a finite depth (0 is clamped to 1 server-side,
        # so there is no unlimited option); 100 levels of nesting far exceeds
        # any real task artifact tree.
        entries = await self._sandbox.files.list(source_dir, depth=100, user="root")
        file_entries: list[tuple[str, Path]] = []
        for entry in entries:
            relative = Path(entry.path).relative_to(Path(source_dir))
            local = target / relative
            if entry.type == FileType.DIR:
                local.mkdir(parents=True, exist_ok=True)
            elif entry.type == FileType.FILE:
                file_entries.append((entry.path, local))

        semaphore = asyncio.Semaphore(self._DOWNLOAD_CONCURRENCY)

        async def fetch(remote: str, local: Path) -> None:
            async with semaphore:
                await self.download_file(remote, local)

        await asyncio.gather(*(fetch(remote, local) for remote, local in file_entries))

    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        return await self._path_type(path) == FileType.DIR

    async def is_file(self, path: str, user: str | int | None = None) -> bool:
        return await self._path_type(path) == FileType.FILE

    async def _path_type(self, path: str) -> FileType | None:
        if self._sandbox is None:
            raise RuntimeError("Sandbox not found. Start the environment first.")
        try:
            return (await self._sandbox.files.get_info(path, user="root")).type
        except NotFoundException:
            # Only a confirmed missing path maps to False; transient API
            # errors propagate so callers do not mistake them for absence.
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, max=10),
        retry=retry_if_exception_type(_COMMAND_DISPATCH_RETRYABLE),
        reraise=True,
    )
    async def _dispatch_command(
        self,
        command: str,
        *,
        cwd: str | None,
        env: dict[str, str] | None,
        timeout_sec: int | None,
        user: str,
    ) -> AsyncCommandHandle:
        if self._sandbox is None:
            raise RuntimeError("Sandbox not found. Start the environment first.")
        return await self._sandbox.commands.run(
            cmd=command,
            background=True,
            cwd=cwd,
            envs=env,
            timeout=timeout_sec or 0,
            user=user,
        )

    @staticmethod
    def _export_prefix(env: dict[str, str] | None) -> str:
        """Shell exports for values that login profiles are known to replace.

        envd starts commands through a login shell, so profile scripts run
        after the ``envs`` API values are applied and can clobber them —
        Debian's /etc/profile unconditionally resets PATH, which silently
        dropped image PATH entries like /opt/venv/bin. Exporting inside the
        command runs after the profiles and wins deterministically.
        """
        if not env:
            return ""
        return "".join(
            f"export {key}={shlex.quote(value)}; "
            for key, value in env.items()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
        )

    @staticmethod
    def _capture_command(command: str, capture_dir: str) -> str:
        stdout_path = shlex.quote(f"{capture_dir}/stdout")
        stderr_path = shlex.quote(f"{capture_dir}/stderr")
        status_path = shlex.quote(f"{capture_dir}/status")
        quoted_dir = shlex.quote(capture_dir)
        return (
            f"mkdir -p {quoted_dir} || exit $?; "
            f"( {command}\n) >{stdout_path} 2>{stderr_path}; "
            f"__pier_status=$?; printf '%s\\n' \"$__pier_status\" >{status_path}; "
            'exit "$__pier_status"'
        )

    async def _read_command_capture(self, capture_dir: str) -> ExecResult | None:
        if self._sandbox is None:
            raise RuntimeError("Sandbox stopped while command was running.")
        try:
            status = await self._sandbox.files.read(
                f"{capture_dir}/status", format="text", user="root"
            )
        except NotFoundException:
            return None
        stdout, stderr = await asyncio.gather(
            self._sandbox.files.read(
                f"{capture_dir}/stdout", format="text", user="root"
            ),
            self._sandbox.files.read(
                f"{capture_dir}/stderr", format="text", user="root"
            ),
        )
        return ExecResult(
            stdout=stdout,
            stderr=stderr,
            return_code=int(status.strip()),
        )

    async def _cleanup_command_capture(self, capture_dir: str) -> None:
        if self._sandbox is None:
            return
        try:
            await self._sandbox.files.remove(capture_dir, user="root")
        except Exception as exc:
            self.logger.debug(
                "Failed to remove E2B command capture %s: %s", capture_dir, exc
            )

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        resolved_user = await self._resolve_e2b_user(user)
        merged_env = self._merge_env(env)
        post_profile_env = (
            {"PATH": merged_env["PATH"]}
            if merged_env is not None and "PATH" in merged_env
            else None
        )
        capture_dir = f"/tmp/.pier-exec-{uuid.uuid4().hex}"
        handle = await self._dispatch_command(
            self._capture_command(
                self._export_prefix(post_profile_env) + command,
                capture_dir,
            ),
            # docker exec starts in the image WORKDIR, or "/" when the image
            # has none; envd starts in $HOME, which may not even exist for
            # non-root image users (e.g. nginx's /nonexistent).
            cwd=cwd or self.task_env_config.workdir or self._image_workdir or "/",
            env=merged_env,
            timeout_sec=timeout_sec,
            user=resolved_user,
        )
        # The stream drops on transient network errors and when the sandbox TTL
        # pauses it mid-command. Output and status are captured in the sandbox,
        # so reconnecting cannot silently omit bytes emitted between streams.
        max_attempts = 4
        fallback_result: ExecResult | None = None
        for attempt in range(max_attempts):
            try:
                await handle.wait()
                break
            except CommandExitException as exc:
                fallback_result = ExecResult(
                    stdout=exc.stdout,
                    stderr=exc.stderr,
                    return_code=exc.exit_code,
                )
                break
            except _COMMAND_STREAM_RETRYABLE:
                captured = await self._read_command_capture(capture_dir)
                if captured is not None:
                    await self._cleanup_command_capture(capture_dir)
                    return captured
                if attempt == max_attempts - 1:
                    raise
                self.logger.warning(
                    "E2B command stream disconnected for pid=%s; reconnecting",
                    handle.pid,
                )
                await asyncio.sleep(2**attempt)
                if self._sandbox is None:
                    raise RuntimeError("Sandbox stopped while command was running.")
                try:
                    handle = await self._sandbox.commands.connect(
                        handle.pid,
                        timeout=timeout_sec or 0,
                    )
                except NotFoundException:
                    captured = await self._read_command_capture(capture_dir)
                    if captured is None:
                        raise
                    await self._cleanup_command_capture(capture_dir)
                    return captured
        captured = await self._read_command_capture(capture_dir)
        if captured is None:
            if fallback_result is not None:
                await self._cleanup_command_capture(capture_dir)
                return fallback_result
            raise RuntimeError(
                f"E2B command pid={handle.pid} exited without a durable status."
            )
        await self._cleanup_command_capture(capture_dir)
        return captured
