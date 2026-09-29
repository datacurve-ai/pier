import subprocess

from pier.cli.utils import prevent_system_sleep


class FakeProcess:
    def __init__(self, *, running: bool = True):
        self.running = running
        self.terminated = False
        self.killed = False
        self.wait_timeouts: list[float | None] = []

    def poll(self):
        return None if self.running else 0

    def terminate(self):
        self.terminated = True
        self.running = False

    def kill(self):
        self.killed = True
        self.running = False

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        return 0


def test_prevent_system_sleep_uses_scoped_macos_assertion(monkeypatch):
    process = FakeProcess()
    observed = {}

    def fake_popen(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return process

    monkeypatch.setattr("pier.cli.utils.sys.platform", "darwin")
    monkeypatch.setattr("pier.cli.utils.os.getpid", lambda: 1234)
    monkeypatch.setattr("pier.cli.utils.subprocess.Popen", fake_popen)

    with prevent_system_sleep():
        assert not process.terminated

    assert observed["command"] == [
        "/usr/bin/caffeinate",
        "-i",
        "-w",
        "1234",
    ]
    assert observed["kwargs"] == {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    assert process.terminated
    assert process.wait_timeouts == [5]


def test_prevent_system_sleep_is_noop_off_macos(monkeypatch):
    def unexpected_popen(*args, **kwargs):
        raise AssertionError("Popen should not be called")

    monkeypatch.setattr("pier.cli.utils.sys.platform", "linux")
    monkeypatch.setattr("pier.cli.utils.subprocess.Popen", unexpected_popen)

    with prevent_system_sleep():
        pass


def test_prevent_system_sleep_fails_open_when_caffeinate_cannot_start(
    monkeypatch, caplog
):
    def failing_popen(*args, **kwargs):
        raise OSError("not available")

    monkeypatch.setattr("pier.cli.utils.sys.platform", "darwin")
    monkeypatch.setattr("pier.cli.utils.subprocess.Popen", failing_popen)

    with prevent_system_sleep():
        pass

    assert "Could not prevent host sleep" in caplog.text
