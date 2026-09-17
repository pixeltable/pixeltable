"""pxt CLI test fixtures.

The daemon runs in a separate process, so it must inherit the per-worker
PIXELTABLE_* env vars set by the session-scoped init_env fixture. We spawn
our own daemon on a worker-specific port to avoid colliding with the user's
real daemon on 22089.
"""

import contextlib
import json
import os
import pathlib
import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import pytest

from pixeltable.config import Config
from pixeltable_cli.client.utils import is_running

from ..utils import CLOUD_DB_ROOT_URIS, DatabaseRoot, cloud_env_configured

_REPO_ROOT = pathlib.Path(__file__).parents[2]
_CORPUS_DIR = pathlib.Path(__file__).parent

# the pin installs these two, so only they need a commit; the corpus is packaged from the working tree,
# so editing an app or this file needs none
_PINNED_PATHS = ('pixeltable', 'pixeltable_cli')


def _requirements_in() -> list[str]:
    """The corpus project's dependencies apart from pixeltable, one per line, comments dropped."""
    lines = (_CORPUS_DIR / 'requirements.in').read_text(encoding='utf-8').splitlines()
    return [line for line in lines if line.strip() != '' and not line.startswith('#')]


# both the corpus project and the per-test ones install these, so their databases share one image
PROJECT_EXTRAS = tuple(_requirements_in())

# the exit statuses `pxt db diff` and `pxt db update` document
EXIT_IN_AGREEMENT = 0
EXIT_ERROR = 1
EXIT_CHANGES_PENDING = 2

# a publish that rebuilds the image waits on CodeBuild, far longer than the default cli timeout allows
APPLY_TIMEOUT = 2400.0


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
        if self.stdout.strip() == '':
            # a command that failed wrote its reason to stderr, and parsing '' would discard it
            raise AssertionError(f'pxt produced no output (rc={self.returncode}): {self.stderr.strip()}')
        return json.loads(self.stdout)


@pytest.fixture(scope='session', autouse=True)
def session_project(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """The project the session's daemon serves.

    The daemon serves one project, and a client standing in another restarts it, so every CLI test works
    inside this one: its application files go in a directory of their own under this root.

    autouse and session-scoped: the local proxy daemon is handed the recorded project root when it starts,
    and any test in this package may be the one that starts it.
    """
    root = tmp_path_factory.mktemp('pxt_project')
    (root / 'pixeltable.toml').write_text('', encoding='utf-8')
    Config.init(reinit=True, project_root=root)
    return root


@pytest.fixture(autouse=True)
def serve_the_session_project(session_project: pathlib.Path) -> None:
    """Point Config at the session's project before each test.

    Every process this package starts is handed the project Config holds at the time, and a test elsewhere in
    the session leaves its own value behind.
    """
    if Config.get().project_root != session_project:
        Config.init(reinit=True, project_root=session_project)


@pytest.fixture
def served_project(session_project: pathlib.Path) -> pathlib.Path:
    return session_project


@pytest.fixture
def project_dir(session_project: pathlib.Path, request: pytest.FixtureRequest) -> pathlib.Path:
    """A directory of the session's project, named after the test that writes its files there.

    A module path is relative to the project root, so files in a directory named after the test are
    imported under names no other test uses.
    """
    name = re.sub(r'\W', '_', request.node.name)
    directory = session_project / name
    directory.mkdir(exist_ok=True)
    return directory


@pytest.fixture(scope='session')
def pxt_daemon(
    init_env: None, tmp_path_factory: pytest.TempPathFactory, session_project: pathlib.Path
) -> Iterator[int]:
    port = _pick_port()
    env = {**os.environ, 'PXT_PORT': str(port)}
    log_path = tmp_path_factory.mktemp('pxt-daemon') / 'daemon.log'
    prior_port = os.environ.get('PXT_PORT')
    with open(log_path, 'w', encoding='utf-8') as log:
        proc = subprocess.Popen(
            # the project this daemon serves, named the way a client names it
            [sys.executable, '-m', 'pixeltable_cli.server.daemon', '--project-root', str(session_project)],
            env=env,
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
        )
    try:
        os.environ['PXT_PORT'] = str(port)
        # Allow for a cold pixeltable import in the daemon subprocess, which on a loaded CI runner can run
        # well past a warm import; matches the client's own startup health timeout.
        startup_timeout = 45
        print(f'Waiting for the test daemon on port {port}, serving {session_project}', flush=True)
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
        # provoke that restart here, with the environment this fixture started the daemon with, and from the
        # project this daemon serves: the replacement takes its project from the caller's working directory.
        subprocess.run(['pxt', 'ls', '/'], env=env, cwd=session_project, capture_output=True, check=False, timeout=60)
        assert is_running()
        print(f'Test daemon is up on port {port}; log at {log_path}', flush=True)
        yield port
    finally:
        # a test may have restarted the daemon, in which case the process answering on the port is not the
        # one started here; take that one down too, so the session leaves nothing behind
        subprocess.run(
            ['pxt', 'daemon', 'stop', '-f'],
            env={**os.environ, 'PXT_PORT': str(port)},
            cwd=session_project,
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

# `pxt db update` against a hosted database builds its image, which runs CodeBuild
BUILD_TIMEOUT = 1800.0

_RUN_TIMEOUT_SECS = 300

_WHEEL_SUBDIR = 'wheels'


def db_diff(cli: PxtRunner, project: pathlib.Path, db_uri: str) -> dict[str, Any]:
    """What `pxt db diff` reports, its exit status under 'returncode'."""
    r = cli('db', 'diff', db_uri, '--json', cwd=project, check=False)
    assert r.returncode in (EXIT_IN_AGREEMENT, EXIT_CHANGES_PENDING), r.stderr
    return {**r.json, 'returncode': r.returncode}


def assert_in_agreement(cli: PxtRunner, project: pathlib.Path, db_uri: str) -> None:
    plan = db_diff(cli, project, db_uri)
    assert plan['in_agreement'], plan['ops']
    assert plan['returncode'] == EXIT_IN_AGREEMENT
    assert plan['ops'] == []


def db_update(cli: PxtRunner, project: pathlib.Path, db_uri: str, *flags: str) -> dict[str, Any]:
    """What `pxt db update` applied, its exit status under 'returncode'."""
    r = cli('db', 'update', db_uri, '-f', '--json', *flags, cwd=project, check=False, timeout=APPLY_TIMEOUT)
    assert r.returncode in (EXIT_IN_AGREEMENT, EXIT_CHANGES_PENDING), r.stderr
    return {**r.json, 'returncode': r.returncode}


def _as_text(stream: bytes | str | None) -> str:
    """Normalize captured output: TimeoutExpired carries bytes even when the run was text=True."""
    if stream is None:
        return ''
    return stream if isinstance(stream, str) else stream.decode(errors='replace')


def copy_app_corpus(session_project: pathlib.Path) -> pathlib.Path:
    """Put the shared app corpus in the session's project, and return where it landed."""
    directory = session_project / 'apps'
    if not directory.exists():
        shutil.copytree(pathlib.Path(__file__).parent / 'apps', directory, ignore=shutil.ignore_patterns('__pycache__'))
    return directory


@pytest.fixture(scope='session')
def pixeltable_wheel(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """A wheel built from this working tree, for a project to install in place of the released pixeltable."""
    repo_root = pathlib.Path(__file__).parents[2]
    out_dir = tmp_path_factory.mktemp('pxt_wheel')
    print(f'Building a Pixeltable wheel from {repo_root}', flush=True)
    r = subprocess.run(
        ['uv', 'build', '--wheel', '-o', str(out_dir), str(repo_root)],
        text=True,
        check=False,
        timeout=_RUN_TIMEOUT_SECS,
    )
    assert r.returncode == 0, f'building a pixeltable wheel from {repo_root} failed:\n{r.stderr}'
    wheels = list(out_dir.glob('*.whl'))
    assert len(wheels) == 1, f'expected one wheel in {out_dir}, found {wheels}'
    return wheels[0]


def read_logs_until(
    cli: PxtRunner, *args: str, contains: str, timeout: float = 60.0, cwd: pathlib.Path | None = None
) -> list[dict[str, Any]]:
    """Run `pxt <args> --json` until a record's line contains the text, and return the records.

    A line reaches the hosted log a few seconds after the pod writes it. Fail on timeout.
    """
    deadline = time.monotonic() + timeout
    while True:
        records: list[dict[str, Any]] = cli(*args, '--json', cwd=cwd).json
        if any(contains in r['line'] for r in records):
            return records
        assert time.monotonic() < deadline, (contains, records[-5:])
        time.sleep(2.0)


@contextlib.contextmanager
def disposable_db(cli: PxtRunner, uri: str, cwd: pathlib.Path) -> Iterator[str]:
    """Delete the database at uri once the caller is done with it, whether or not one was ever created."""
    try:
        yield uri
    finally:
        cli('db', 'delete', uri, cwd=cwd, check=False)


def write_requirements(project: pathlib.Path, wheel: pathlib.Path, *extra: str) -> None:
    """Write project's requirements.txt, installing pixeltable from wheel rather than from PyPI.

    The wheel is copied into the project so that the archive carries it, and named by a path relative to
    the project root, which is where the image build runs pip.
    """
    wheel_dir = project / _WHEEL_SUBDIR
    wheel_dir.mkdir(exist_ok=True)
    shutil.copy(wheel, wheel_dir / wheel.name)
    (project / 'requirements.txt').write_text(
        '\n'.join([f'./{_WHEEL_SUBDIR}/{wheel.name}', *extra]) + '\n', encoding='utf-8'
    )


@pytest.fixture(scope='session')
def cloud_service_db(
    cloud_service_db_uri: str, session_cli: PxtRunner, session_project: pathlib.Path, pixeltable_wheel: pathlib.Path
) -> Iterator[str]:
    """Create the database the 'cloud-service' root names, serving this session's project, and return it.

    test_service.py deploys that project's application files as services and edits them as it goes, and a
    pod reaches an edit only through the database's archive, which `pxt db update` replaces. So the root
    gets a database of its own rather than the corpus database the rest of the package reads.

    Session-scoped, since creating a database provisions storage and runs CodeBuild.
    """
    copy_app_corpus(session_project)
    write_requirements(session_project, pixeltable_wheel, *PROJECT_EXTRAS)
    with disposable_db(session_cli, cloud_service_db_uri, session_project) as uri:
        (session_project / 'pixeltable.toml').write_text(
            f'[[pixeltable.database]]\nname = {json.dumps(uri)}\n', encoding='utf-8'
        )
        # the daemon read the project config when it started
        session_cli('daemon', 'restart', cwd=session_project)
        session_cli('db', 'update', uri, '-f', cwd=session_project, timeout=BUILD_TIMEOUT)
        yield uri


def _git(*args: str) -> str:
    r = subprocess.run(['git', '-C', str(_REPO_ROOT), *args], capture_output=True, text=True, check=True)
    return r.stdout.strip()


def _pixeltable_repo(sha: str) -> str:
    """The https url of a remote that has sha."""
    # '->' skips the symbolic origin/HEAD, which is a second name for a branch already listed
    branches = [line.strip() for line in _git('branch', '-r', '--contains', sha).splitlines() if '->' not in line]
    remotes = list(dict.fromkeys(branch.split('/', maxsplit=1)[0] for branch in branches))
    assert len(remotes) > 0, (
        f'commit {sha[:8]} is on no remote branch, and the image build fetches it from GitHub; '
        'run `git push origin HEAD` and try again'
    )
    url = _git('remote', 'get-url', 'origin' if 'origin' in remotes else remotes[0])
    return re.sub(r'^git@([^:]+):', r'https://\1/', url).removesuffix('.git')


@pytest.fixture(scope='session')
def corpus_pixeltable_pin() -> str | None:
    """Write the corpus project's requirements.txt, pinning pixeltable to this checkout's commit.

    A hosted pod speaks the management protocol to the control plane, so it has to run the pixeltable under
    test rather than the last release. The image build runs in CodeBuild, which reaches GitHub but not this
    machine, so the pin is a commit on a remote rather than a path here.

    Returns None where no control plane is configured, since only an image build reads this file.
    """
    if not cloud_env_configured():
        return None
    # an untracked file sits outside the corpus project and is absent from the archive, so it is not drift
    modified = _git('status', '--porcelain', '--untracked-files=no', '--', *_PINNED_PATHS)
    assert modified == '', f'a pod installs the commit, not this working tree; commit or stash first:\n{modified}'
    sha = _git('rev-parse', 'HEAD')
    pin = f'pixeltable @ git+{_pixeltable_repo(sha)}@{sha}'
    (_CORPUS_DIR / 'requirements.txt').write_text('\n'.join([pin, *_requirements_in()]) + '\n', encoding='utf-8')
    return pin


@pytest.fixture(scope='session', autouse=True)
def _serve_corpus_db(session_cli: PxtRunner, corpus_pixeltable_pin: str | None) -> None:
    """Publish this app corpus and this commit's pixeltable to the CLI database.

    The tests resolve the corpus's udfs, which reach a pod only in the database's project archive, and a
    pod runs the pixeltable that corpus_pixeltable_pin wrote into requirements.txt. Publishing both is part
    of the run.
    """
    if not cloud_env_configured():
        return
    uri = CLOUD_DB_ROOT_URIS['cloud-cli']
    pending = _corpus_db_ops(session_cli, uri)
    if len(pending) == 0:
        return
    print(f'Publishing {_CORPUS_DIR} to {uri}: {"; ".join(op["description"] for op in pending)}', flush=True)
    db_update(session_cli, _CORPUS_DIR, uri)
    remaining = _corpus_db_ops(session_cli, uri)
    assert len(remaining) == 0, f'{uri} still differs from {_CORPUS_DIR} after an update: {remaining}'


def _corpus_db_ops(cli: PxtRunner, uri: str) -> list[dict[str, Any]]:
    """The operations that would reconcile uri with the corpus, limited to the archive and the image."""
    return [op for op in db_diff(cli, _CORPUS_DIR, uri)['ops'] if op['target'] in ('archive', 'image')]


@pytest.fixture
def apps(session_project: pathlib.Path) -> Callable[[str], str]:
    """Returns a Callable that resolves the name of an app file in the shared app corpus to its path.

    The corpus needs to be copied into the session's project in order for cli commands to work.
    """
    directory = copy_app_corpus(session_project)

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
def cli_bg(
    pxt_daemon: int, db_root: DatabaseRoot, session_project: pathlib.Path
) -> Iterator[Callable[..., BackgroundPxt]]:
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
            # in the session's project, so that client and daemon agree on which project this is
            cwd=session_project,
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


@pytest.fixture(scope='session')
def session_cli(pxt_daemon: int, session_project: pathlib.Path) -> PxtRunner:
    """Run the CLI against the session's daemon and project, for work a session does once."""

    def _run(
        *args: str,
        check: bool = True,
        cwd: str | os.PathLike[str] | None = None,
        env_overrides: dict[str, str | None] | None = None,
        timeout: float = _RUN_TIMEOUT_SECS,
    ) -> PxtResult:
        # in the session's project, so that client and daemon agree on which project this is
        cwd = session_project if cwd is None else cwd
        # BROWSER=true prevents an actual browser tab open on `pxt dashboard` when tests are run on a dev machine.
        env = {**os.environ, 'PXT_PORT': str(pxt_daemon), 'BROWSER': 'true'}
        for name, value in (env_overrides or {}).items():
            if value is None:
                env.pop(name, None)
            else:
                env[name] = value
        print(f'Running: pxt {" ".join(args)}', flush=True)
        started = time.monotonic()
        try:
            r = subprocess.run(
                ['pxt', *args],
                capture_output=True,
                text=True,
                env=env,
                check=False,
                stdin=subprocess.DEVNULL,
                cwd=cwd,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            # subprocess.run has already killed the client; report whatever it managed to emit
            raise AssertionError(
                f'{" ".join(("pxt", *args))} did not finish within {timeout}s\n'
                f'--- stdout ---\n{_as_text(exc.stdout)}\n'
                f'--- stderr ---\n{_as_text(exc.stderr)}'
            ) from exc
        print(f'Finished in {time.monotonic() - started:.1f}s (rc={r.returncode}): pxt {" ".join(args)}', flush=True)
        if check and r.returncode != 0:
            raise AssertionError(f'pxt {args} failed (rc={r.returncode}): {r.stderr}')
        print(r.stdout)
        return PxtResult(r.returncode, r.stdout, r.stderr)

    return _run


@pytest.fixture
def cli(db_root: DatabaseRoot, session_cli: PxtRunner) -> PxtRunner:
    # db_root resets the catalog (like uses_db) and parameterizes over the database roots, so a test
    # using cli() auto-forks over all backends unless it is marked @pytest.mark.db_roots. The CLI daemon and
    # this test process share PIXELTABLE_HOME, so both resolve a pxt:// path to the same local proxy daemon.
    return session_cli
