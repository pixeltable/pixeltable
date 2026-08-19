"""pxt CLI test fixtures.

The daemon runs in a separate process, so it must inherit the per-worker
PIXELTABLE_* env vars set by the session-scoped init_env fixture. We spawn
our own daemon on a worker-specific port to avoid colliding with the user's
real daemon on 22089.
"""

import json
import os
import pathlib
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import pytest


def _pick_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@dataclass
class PxtResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def json(self) -> Any:
        return json.loads(self.stdout)


@pytest.fixture(scope='session')
def pxt_daemon(init_env: None, tmp_path_factory: pytest.TempPathFactory) -> Iterator[int]:
    port = _pick_port()
    env = {**os.environ, 'PXT_PORT': str(port)}
    log_path = tmp_path_factory.mktemp('pxt-daemon') / 'daemon.log'
    prior_port = os.environ.get('PXT_PORT')
    with open(log_path, 'w', encoding='utf-8') as log:
        proc = subprocess.Popen(
            [sys.executable, '-m', 'pixeltable_cli.server.daemon'],
            env=env,
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
        )
    try:
        from pixeltable_cli.client.utils import is_running

        os.environ['PXT_PORT'] = str(port)
        # Allow for a cold pixeltable import in the daemon subprocess, which on a loaded CI runner can run
        # well past a warm import; matches the client's own startup health timeout.
        startup_timeout = 45
        deadline = time.time() + startup_timeout
        while time.time() < deadline:
            if is_running():
                break
            if proc.poll() is not None:
                tail = log_path.read_text(errors='replace')[-500:]
                raise RuntimeError(f'daemon exited early: {tail}')
            time.sleep(0.1)
        else:
            tail = log_path.read_text(errors='replace')[-500:]
            raise RuntimeError(f'daemon did not come up within {startup_timeout}s; log tail:\n{tail}')

        # The client reports the interpreter behind the pxt script, which need not be spelled the way
        # sys.executable is (python vs python3 in the same environment). ensure_running() restarts a daemon
        # whose identity differs from the caller's and the replacement inherits the caller's environment, so
        # provoke that restart here, with the environment this fixture started the daemon with.
        subprocess.run(['pxt', 'ls', '/'], env=env, capture_output=True, check=False, timeout=60)
        assert is_running()
        yield port
    finally:
        # a test may have restarted the daemon, in which case the process answering on the port is not the
        # one started here; take that one down too, so the session leaves nothing behind
        subprocess.run(
            ['pxt', 'daemon', 'stop', '-f'],
            env={**os.environ, 'PXT_PORT': str(port)},
            capture_output=True,
            check=False,
            timeout=60,
        )
        if prior_port is None:
            os.environ.pop('PXT_PORT', None)
        else:
            os.environ['PXT_PORT'] = prior_port
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


PxtRunner = Callable[..., PxtResult]

_RUN_TIMEOUT_SECS = 300


def _as_text(stream: bytes | str | None) -> str:
    """Normalize captured output: TimeoutExpired carries bytes even when the run was text=True."""
    if stream is None:
        return ''
    return stream if isinstance(stream, str) else stream.decode(errors='replace')


@pytest.fixture
def apps() -> Callable[[str], str]:
    """Resolves the name of a file in the shared app corpus to its path."""
    directory = pathlib.Path(__file__).parent / 'apps'

    def _path(name: str) -> str:
        path = directory / name
        assert path.is_file(), f'no such app file: {path}'
        return str(path)

    return _path


@dataclass
class BackgroundPxt:
    """A `pxt` command still running, for a verb that serves until it is interrupted."""

    proc: subprocess.Popen
    port: int

    @property
    def endpoint(self) -> str:
        return f'http://127.0.0.1:{self.port}'

    def wait_until_serving(self, timeout: float = 60.0) -> None:
        """Block until the command answers on its port, or fail with whatever it printed instead."""
        import httpx

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise AssertionError(f'pxt exited with {self.proc.returncode} before serving')
            try:
                if httpx.get(f'{self.endpoint}/openapi.json', timeout=1.0).status_code == 200:
                    return
            except httpx.HTTPError:
                time.sleep(0.2)
        raise AssertionError(f'nothing was serving on {self.endpoint} within {timeout:.0f}s')


@pytest.fixture
def cli_bg(pxt_daemon: int, make_catalog_path: Callable[[str], str]) -> Iterator[Callable[..., BackgroundPxt]]:
    """Runs a `pxt` command in the background, for one that serves rather than returning."""
    running: list[BackgroundPxt] = []

    def _run(*args: str, port: int | None = None) -> BackgroundPxt:
        bound = _pick_port() if port is None else port
        env = {**os.environ, 'PXT_PORT': str(pxt_daemon), 'BROWSER': 'true'}
        proc = subprocess.Popen(
            ['pxt', *args, '--port', str(bound)],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        handle = BackgroundPxt(proc, bound)
        running.append(handle)
        return handle

    yield _run

    for handle in running:
        handle.proc.terminate()
        try:
            handle.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            handle.proc.kill()


@pytest.fixture
def cli(pxt_daemon: int, make_catalog_path: Callable[[str], str]) -> PxtRunner:
    # make_catalog_path resets the catalog (like uses_db) and pulls in the local/proxy axis, so a test
    # using cli() auto-forks over both backends unless it is marked @pytest.mark.local. The CLI daemon and
    # this test process share PIXELTABLE_HOME, so both resolve a pxt:// path to the same local proxy daemon.
    def _run(
        *args: str,
        check: bool = True,
        cwd: str | os.PathLike[str] | None = None,
        env_overrides: dict[str, str | None] | None = None,
    ) -> PxtResult:
        # BROWSER=true prevents an actual browser tab open on `pxt dashboard` when tests are run on a dev machine.
        env = {**os.environ, 'PXT_PORT': str(pxt_daemon), 'BROWSER': 'true'}
        for name, value in (env_overrides or {}).items():
            if value is None:
                env.pop(name, None)
            else:
                env[name] = value
        try:
            r = subprocess.run(
                ['pxt', *args],
                capture_output=True,
                text=True,
                env=env,
                check=False,
                stdin=subprocess.DEVNULL,
                cwd=cwd,
                timeout=_RUN_TIMEOUT_SECS,
            )
        except subprocess.TimeoutExpired as exc:
            # subprocess.run has already killed the client; report whatever it managed to emit
            raise AssertionError(
                f'{" ".join(("pxt", *args))} did not finish within {_RUN_TIMEOUT_SECS}s\n'
                f'--- stdout ---\n{_as_text(exc.stdout)}\n'
                f'--- stderr ---\n{_as_text(exc.stderr)}'
            ) from exc
        if check and r.returncode != 0:
            raise AssertionError(f'pxt {args} failed (rc={r.returncode}): {r.stderr}')
        return PxtResult(r.returncode, r.stdout, r.stderr)

    return _run
