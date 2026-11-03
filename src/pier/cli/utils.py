import asyncio
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from collections.abc import Iterator
from typing import Any, Coroutine, TypeVar

from pier.utils.logger import logger

T = TypeVar("T")


@contextmanager
def prevent_system_sleep() -> Iterator[None]:
    """Keep the host awake while a CLI job is running.

    macOS ships ``caffeinate`` as part of the operating system. Running it with
    ``-w`` also makes the assertion self-cleaning if Pier exits unexpectedly.
    Other platforms currently remain a no-op.
    """
    process: subprocess.Popen[bytes] | None = None
    if sys.platform == "darwin":
        try:
            process = subprocess.Popen(
                ["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            logger.getChild(__name__).warning(
                "Could not prevent host sleep while the Pier job is running: %s",
                exc,
            )

    try:
        yield
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine with proper Windows subprocess support.

    On Windows, the default SelectorEventLoop doesn't support subprocesses.
    We explicitly pass ProactorEventLoop as the loop_factory to ensure it's used.

    Requires Python 3.12+ (which is the minimum version for this project).
    """
    if sys.platform == "win32":
        return asyncio.run(coro, loop_factory=asyncio.ProactorEventLoop)
    return asyncio.run(coro)


def parse_kwargs(kwargs_list: list[str] | None) -> dict[str, Any]:
    """Parse key=value strings into a dictionary.

    Values are parsed as JSON or Python literals if valid, otherwise treated as strings.
    This allows non-string parameters like numbers, booleans, lists, and dictionaries.

    Examples:
        key=value -> {"key": "value"}
        key=123 -> {"key": 123}
        key=true -> {"key": True}
        key=True -> {"key": True}
        key=[1,2,3] -> {"key": [1, 2, 3]}
        key={"a":1} -> {"key": {"a": 1}}
    """
    if not kwargs_list:
        return {}

    result = {}
    for kwarg in kwargs_list:
        if "=" not in kwarg:
            raise ValueError(f"Invalid kwarg format: {kwarg}. Expected key=value")
        key, value = kwarg.split("=", 1)
        key = key.strip()
        value = value.strip()

        # Try to parse as JSON first
        try:
            result[key] = json.loads(value)
        except json.JSONDecodeError:
            # Handle Python-style literals that JSON doesn't recognize
            if value == "True":
                result[key] = True
            elif value == "False":
                result[key] = False
            elif value == "None":
                result[key] = None
            else:
                # If JSON parsing fails and not a Python literal, treat as string
                result[key] = value

    return result


def parse_env_vars(env_list: list[str] | None) -> dict[str, str]:
    """Parse KEY=VALUE strings into a dictionary of strings.

    Unlike parse_kwargs, values are always treated as literal strings
    without any JSON or Python literal parsing.

    Examples:
        KEY=value -> {"KEY": "value"}
        KEY=123 -> {"KEY": "123"}
        KEY=true -> {"KEY": "true"}
        KEY={"a":1} -> {"KEY": "{\"a\":1}"}
    """
    if not env_list:
        return {}

    result = {}
    for item in env_list:
        if "=" not in item:
            raise ValueError(f"Invalid env var format: {item}. Expected KEY=VALUE")
        key, value = item.split("=", 1)
        result[key.strip()] = value.strip()

    return result
