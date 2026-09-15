"""Opt-in checks against real E2B sandboxes; requires E2B_API_KEY.

Run from the repository with:
    uv run python scripts/verify_e2b.py --image ubuntu:24.04

Creates a reusable template and a temporary sandbox, then terminates the sandbox.
Use --template-mode required on a second run to also verify template reuse.
The image must provide Bash, tar, and standard POSIX utilities.
"""

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from e2b import AsyncSandbox, NotFoundException

from pier.environments.e2b import E2BEnvironment
from pier.models.task.config import EnvironmentConfig
from pier.models.trial.paths import TrialPaths


async def wait_for_file(env: E2BEnvironment, path: str) -> None:
    async with asyncio.timeout(10):
        while not await env.is_file(path):
            await asyncio.sleep(0.1)


async def check_resume(env: E2BEnvironment, *, ttl: bool) -> None:
    name = "ttl" if ttl else "pause"
    root = f"/tmp/pier-check-{name}"
    reconnected = asyncio.Event()
    connect_command = env._connect_command

    async def observe_reconnect(pid: int):
        handle = await connect_command(pid)
        reconnected.set()
        return handle

    env._connect_command = observe_reconnect
    command = asyncio.create_task(
        env.exec(
            f"echo invocation >> {root}-count; printf before; "
            f"touch {root}-ready; "
            f"while [ ! -f {root}-release ]; do sleep 0.1; done; printf after",
            timeout_sec=30,
        )
    )
    try:
        await wait_for_file(env, f"{root}-ready")
        assert env._sandbox is not None
        if ttl:
            await env._sandbox.set_timeout(1)
        else:
            await env._sandbox.pause()
        async with asyncio.timeout(20):
            await reconnected.wait()
        await env._sandbox.set_timeout(120)
        release = await env.exec(f"touch {root}-release", timeout_sec=10)
        assert release.return_code == 0, release
        result = await command
        assert result.return_code == 0 and result.stdout == "beforeafter", result
        count = await env.exec(f"cat {root}-count", timeout_sec=10)
        assert count.stdout == "invocation\n", "Command was dispatched twice"
    finally:
        env._connect_command = connect_command
        if not command.done():
            command.cancel()
            try:
                await command
            except asyncio.CancelledError:
                pass


async def check_termination(env: E2BEnvironment) -> None:
    try:
        await env.exec("sleep 2; touch /tmp/pier-check-timeout-late", timeout_sec=1)
    except TimeoutError:
        pass
    else:
        raise AssertionError("Command deadline was not enforced")

    command = asyncio.create_task(
        env.exec(
            "touch /tmp/pier-check-cancel-ready; "
            "sleep 2 | cat; touch /tmp/pier-check-cancel-late"
        )
    )
    try:
        await wait_for_file(env, "/tmp/pier-check-cancel-ready")
    finally:
        command.cancel()
        try:
            await command
        except asyncio.CancelledError:
            pass
    await asyncio.sleep(2.5)
    assert not await env.is_file("/tmp/pier-check-timeout-late")
    assert not await env.is_file("/tmp/pier-check-cancel-late")


async def verify(args: argparse.Namespace) -> None:
    E2BEnvironment.preflight()
    with tempfile.TemporaryDirectory(prefix="pier-e2b-check-") as directory:
        root = Path(directory)
        paths = TrialPaths(trial_dir=root / "trial")
        paths.mkdir()
        env = E2BEnvironment(
            environment_dir=root,
            environment_name="pier/e2b-validation",
            session_id=f"e2b-check-{root.name}",
            trial_paths=paths,
            task_env_config=EnvironmentConfig(
                docker_image=args.image, cpus=2, memory_mb=1024, allow_internet=False
            ),
            template_prefix=args.template_prefix,
            template_mode=args.template_mode,
            sandbox_timeout_secs=120,
        )
        checks = []
        try:
            await env.start(force_build=False)
            result = await env.exec(
                "printf out; printf err >&2; exit 7", timeout_sec=10
            )
            assert (result.stdout, result.stderr, result.return_code) == (
                "out",
                "err",
                7,
            )
            checks.append("stdout/stderr/exit status")

            source = root / "upload"
            source.mkdir()
            (source / "empty").mkdir()
            (source / "helper.sh").write_text("#!/bin/sh\nprintf helper-ok\n")
            (source / "helper.sh").chmod(0o755)
            (source / "link").symlink_to("helper.sh")
            await env.upload_dir(source, "/tmp/pier-check-upload")
            result = await env.exec(
                "cd /tmp/pier-check-upload && test -d empty && test -L link && ./link",
                timeout_sec=10,
            )
            assert result.return_code == 0 and result.stdout == "helper-ok", result
            await env.download_file(
                "/tmp/pier-check-upload/helper.sh", root / "download"
            )
            assert (root / "download").read_bytes() == (
                source / "helper.sh"
            ).read_bytes()
            checks.append("file transfer and upload metadata")

            await check_resume(env, ttl=False)
            checks.append("explicit pause/resume without redispatch")
            await check_resume(env, ttl=True)
            checks.append("TTL pause/resume without redispatch")
            await check_termination(env)
            checks.append("timeout/cancellation terminate descendants")
        finally:
            sandbox_id = env._sandbox.sandbox_id if env._sandbox else None
            await env.stop(delete=True)
            if sandbox_id:
                try:
                    await AsyncSandbox.get_info(sandbox_id)
                except NotFoundException:
                    pass
                else:
                    raise AssertionError(f"Sandbox was not deleted: {sandbox_id}")

        checks.append("sandbox deletion")

        print(
            json.dumps(
                {"template": env.template_name, "checks_passed": checks}, indent=2
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", required=True, help="Published Linux amd64 task image"
    )
    parser.add_argument("--template-prefix", default="pier-check")
    parser.add_argument(
        "--template-mode",
        choices=["build-if-missing", "required"],
        default="build-if-missing",
    )
    asyncio.run(verify(parser.parse_args()))
