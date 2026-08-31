"""Unit tests for Modal sandbox filesystem path checks."""

import functools
from types import SimpleNamespace

import pytest

pytest.importorskip("modal")

from modal.exception import (
    SandboxFilesystemNotADirectoryError,
    SandboxFilesystemNotFoundError,
)

from pier.environments.modal import ModalEnvironment, _ModalStrategy


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


def _strategy_with_list_files(
    outcomes: dict[str, object],
) -> tuple[_ModalStrategy, _ListFiles]:
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


@run_async
async def test_sdk_exec_stream_forwards_control_queue_to_process_stdin() -> None:
    import asyncio

    closed = asyncio.Event()
    writes: list[bytes] = []

    class _EmptyAfterDrain:
        def __aiter__(self):
            return self

        async def __anext__(self):
            await closed.wait()
            raise StopAsyncIteration

    class _Stdin:
        def __init__(self) -> None:
            self.drain = SimpleNamespace(aio=self._drain)

        def write(self, value: bytes) -> None:
            writes.append(value)

        def write_eof(self) -> None:
            writes.append(b"<eof>")
            closed.set()

        async def _drain(self) -> None:
            return None

    process = SimpleNamespace(
        stdin=_Stdin(),
        stdout=_EmptyAfterDrain(),
        stderr=_EmptyAfterDrain(),
        wait=SimpleNamespace(aio=lambda: asyncio.sleep(0, result=0)),
    )

    async def exec_aio(*_args, **_kwargs):
        return process

    env = object.__new__(ModalEnvironment)
    env._persistent_env = {}
    env._sandbox = SimpleNamespace(exec=SimpleNamespace(aio=exec_aio))
    queue = asyncio.Queue()
    message = '{"type":"request.pause","paused":false}\n'
    queue.put_nowait(message)
    queue.put_nowait(None)

    result = await env._sdk_exec_stream(
        "command",
        on_output=lambda _stream, _chunk: asyncio.sleep(0),
        input_queue=queue,
    )

    assert result.return_code == 0
    assert writes == [message.encode(), b"<eof>"]
