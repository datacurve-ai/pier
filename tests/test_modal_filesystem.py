"""Unit tests for Modal sandbox filesystem path checks."""

import functools
from types import SimpleNamespace

import pytest

pytest.importorskip("modal")

from modal.exception import (
    SandboxFilesystemNotADirectoryError,
    SandboxFilesystemNotFoundError,
)

from pier.environments.modal import _ModalStrategy


def run_async(fn):
    """Drive an async test with asyncio.run (pier has no pytest-asyncio)."""
    import asyncio

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


class _ListFiles:
    def __init__(self, outcomes: dict[str, object]) -> None:
        self.outcomes = outcomes
        self.calls: list[str] = []

    async def aio(self, path: str):
        self.calls.append(path)
        outcome = self.outcomes[path]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _strategy_with_list_files(outcomes: dict[str, object]) -> tuple[_ModalStrategy, _ListFiles]:
    list_files = _ListFiles(outcomes)

    class _Filesystem:
        pass

    filesystem = _Filesystem()
    filesystem.list_files = list_files

    class _Sandbox:
        pass

    sandbox = _Sandbox()
    sandbox.filesystem = filesystem
    strategy = _ModalStrategy(SimpleNamespace(_sandbox=sandbox))
    return strategy, list_files


@run_async
async def test_is_dir_and_is_file_use_filesystem_list_files() -> None:
    strategy, list_files = _strategy_with_list_files(
        {
            "/dir": [],
            "/file": SandboxFilesystemNotADirectoryError("not a directory"),
            "/missing": SandboxFilesystemNotFoundError("not found"),
        }
    )

    assert await strategy.is_dir("/dir") is True
    assert await strategy.is_file("/dir") is False
    assert await strategy.is_dir("/file") is False
    assert await strategy.is_file("/file") is True
    assert await strategy.is_dir("/missing") is False
    assert await strategy.is_file("/missing") is False
    assert list_files.calls == [
        "/dir",
        "/dir",
        "/file",
        "/file",
        "/missing",
        "/missing",
    ]
    assert not hasattr(strategy._env._sandbox, "ls")


@run_async
async def test_is_dir_and_is_file_also_map_builtin_os_errors() -> None:
    strategy, _list_files = _strategy_with_list_files(
        {
            "/file": NotADirectoryError("not a directory"),
            "/missing": FileNotFoundError("not found"),
        }
    )

    assert await strategy.is_dir("/file") is False
    assert await strategy.is_file("/file") is True
    assert await strategy.is_dir("/missing") is False
    assert await strategy.is_file("/missing") is False
