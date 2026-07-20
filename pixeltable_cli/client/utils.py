"""Client-only support: daemon orchestration (probe /api/health, spawn/kill/restart the daemon, read the
pidfile, tail the log on failed startup), the stdlib HTTP client (get/post), and CLI path helpers. Stdlib-only
so importing this on every `pxt` invocation stays cheap."""

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from pixeltable_cli.utils import (
    _IDENTITY_KEYS,
    _resolve_pixeltable_home,
    get_port,
    identity,
    pidfile_path,
    validate_path_shape,
)

_IS_WINDOWS = os.name == 'nt'


def session_key() -> str:
    """Stable per-terminal session id: the invoking shell's pid plus its start time.

    The start time distinguishes a recycled pid, so a stale working directory can't bleed into an unrelated
    shell that the OS later assigns the same pid.

    On POSIX the `pxt` console script is exec'd in place, so os.getppid() is the shell itself. On Windows the
    console-script entry point is an .exe trampoline that launches the interpreter as a child and stays its
    parent, so os.getppid() is that trampoline: a distinct pid for every `pxt` invocation, which would give
    each command its own session and defeat the working directory. Step up to the trampoline's own parent to
    recover the shell that all of a terminal's commands share.
    """
    if _IS_WINDOWS:
        launcher_pid = os.getppid()
        shell_pid = _win_parent_pid(launcher_pid)
        # fall back to the trampoline if its parent can't be resolved (session won't persist, but is stable
        # within a single command rather than crashing)
        pid = shell_pid if shell_pid is not None else launcher_pid
        return f'{pid}:{_win_process_creation_time(pid)}'
    ppid = os.getppid()
    return f'{ppid}:{_parent_start_time(ppid)}'


def _win_parent_pid(pid: int) -> int | None:
    """The parent pid of pid on Windows, via a Toolhelp process snapshot; None if it can't be resolved."""
    # ctypes.wintypes exists only on Windows, so import it inside this platform-guarded path.
    import ctypes
    from ctypes import wintypes

    class ProcessEntry(ctypes.Structure):
        _fields_ = [
            ('dwSize', wintypes.DWORD),
            ('cntUsage', wintypes.DWORD),
            ('th32ProcessID', wintypes.DWORD),
            ('th32DefaultHeapID', ctypes.POINTER(ctypes.c_ulong)),  # ULONG_PTR: pointer-sized, only its width matters
            ('th32ModuleID', wintypes.DWORD),
            ('cntThreads', wintypes.DWORD),
            ('th32ParentProcessID', wintypes.DWORD),
            ('pcPriClassBase', ctypes.c_long),
            ('dwFlags', wintypes.DWORD),
            ('szExeFile', ctypes.c_char * 260),
        ]

    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)  # type: ignore[attr-defined]  # Windows-only
    # Set restype explicitly: a HANDLE is pointer-sized and would be truncated under the default int restype
    # on 64-bit Windows.
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.Process32First.restype = wintypes.BOOL
    kernel32.Process32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel32.Process32Next.restype = wintypes.BOOL
    kernel32.Process32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    snap_all_processes = 0x00000002  # TH32CS_SNAPPROCESS
    invalid_handle = wintypes.HANDLE(-1).value  # INVALID_HANDLE_VALUE
    snapshot = kernel32.CreateToolhelp32Snapshot(snap_all_processes, 0)
    if snapshot == invalid_handle:
        return None
    try:
        entry = ProcessEntry()
        entry.dwSize = ctypes.sizeof(ProcessEntry)
        if not kernel32.Process32First(snapshot, ctypes.byref(entry)):
            return None
        while True:
            if entry.th32ProcessID == pid:
                return int(entry.th32ParentProcessID)
            if not kernel32.Process32Next(snapshot, ctypes.byref(entry)):
                return None
    finally:
        kernel32.CloseHandle(snapshot)


def _win_process_creation_time(pid: int) -> str:
    """The creation time of pid on Windows as an opaque string; empty if it can't be read."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)  # type: ignore[attr-defined]  # Windows-only
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    query_limited_info = 0x1000  # PROCESS_QUERY_LIMITED_INFORMATION: minimal rights to read the creation time
    handle = kernel32.OpenProcess(query_limited_info, False, pid)
    if not handle:
        return ''
    try:
        creation = wintypes.FILETIME()
        unused = wintypes.FILETIME()  # exit/kernel/user times are required out-params but not needed here
        ok = kernel32.GetProcessTimes(
            handle, ctypes.byref(creation), ctypes.byref(unused), ctypes.byref(unused), ctypes.byref(unused)
        )
        if not ok:
            return ''
        return str((creation.dwHighDateTime << 32) | creation.dwLowDateTime)
    finally:
        kernel32.CloseHandle(handle)


def _parent_start_time(pid: int) -> str:
    """Best-effort process start time for pid, as an opaque string; empty if it can't be read."""
    if not _IS_WINDOWS:
        try:
            # On Linux, field 22 of /proc/<pid>/stat is the start time. The comm field (2) can contain
            # spaces and parens, so parse the fields after the last ')'.
            with open(f'/proc/{pid}/stat', encoding='utf-8') as f:
                stat = f.read()
            return stat[stat.rfind(')') + 1 :].split()[19]
        except (OSError, IndexError):
            pass
    try:
        out = subprocess.run(
            ['ps', '-o', 'lstart=', '-p', str(pid)], capture_output=True, text=True, timeout=2.0, check=False
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ''


def base_url() -> str:
    return f'http://127.0.0.1:{get_port()}'


def health_url() -> str:
    return f'{base_url()}/api/health'


def _daemon_log_path() -> str:
    """Per-port log path, matching the per-port pidfile."""
    return os.path.join(_resolve_pixeltable_home(), 'logs', f'pxt-daemon-{get_port()}.log')


def read_pidfile() -> int | None:
    try:
        with open(pidfile_path(), encoding='utf-8') as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def fetch_health(timeout: float = 0.3) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(health_url(), timeout=timeout) as r:
            body = json.loads(r.read())
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError):
        return None
    # Verify this is actually our daemon and not some other service on the same port that
    # happens to return a JSON object with an ok=true field. Require both the pxt service
    # marker and the full set of identity fields the mismatch / kill logic relies on.
    if not isinstance(body, dict) or body.get('service') != 'pxt' or not body.get('ok'):
        return None
    required = ('pid', 'started_at', *_IDENTITY_KEYS)
    if not all(k in body for k in required):
        return None
    return body


def is_running(timeout: float = 0.3) -> bool:
    return fetch_health(timeout) is not None


def _identity_diff(client: dict[str, Any], daemon: dict[str, Any]) -> list[str]:
    """Return the list of identity keys whose values differ."""
    return [k for k in _IDENTITY_KEYS if client.get(k) != daemon.get(k)]


def spawn_detached() -> None:
    log_path = _daemon_log_path()
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        # POSIX: setsid() detaches from the controlling terminal; Windows: a new process
        # group + DETACHED_PROCESS gives the same "survive the parent shell" property.
        popen_kwargs: dict[str, Any] = {'stdin': subprocess.DEVNULL}
        if _IS_WINDOWS:
            popen_kwargs['creationflags'] = (
                subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            )
        else:
            popen_kwargs['start_new_session'] = True

        # `python -m` puts the daemon's working directory at the front of sys.path. If pxt is
        # invoked from a directory that contains a pixeltable/ or pixeltable_cli/ folder, that
        # folder shadows the installed package and the daemon imports the wrong code. Anchor the
        # daemon's cwd to the pixeltable home (which holds no importable packages) and set
        # PYTHONSAFEPATH so the working directory is not prepended to sys.path at all (3.11+).
        env = {**os.environ, 'PYTHONSAFEPATH': '1'}
        with open(log_path, 'a', encoding='utf-8') as log:
            subprocess.Popen(
                [sys.executable, '-m', 'pixeltable_cli.server.daemon'],
                stdout=log,
                stderr=log,
                cwd=_resolve_pixeltable_home(),
                env=env,
                **popen_kwargs,
            )
    except OSError as e:
        reason = e.strerror or e.__class__.__name__
        raise RuntimeError(f'pxt daemon log unavailable ({log_path}): {reason}') from None


_TAIL_BYTES = 64 * 1024  # plenty of headroom for n_lines while bounding memory on huge logs


def _tail_daemon_log(n_lines: int = 10) -> str:
    try:
        with open(_daemon_log_path(), 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - _TAIL_BYTES))
            data = f.read()
    except OSError:
        return ''
    lines = data.decode('utf-8', errors='replace').splitlines()
    return '\n'.join(lines[-n_lines:]).rstrip()


def _await_health(timeout: float) -> bool:
    """Poll /api/health until it responds or the timeout elapses. Returns whether it came up."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_running():
            return True
        time.sleep(0.1)
    return False


# A freshly-spawned daemon doesn't serve /api/health until it finishes importing pixeltable, which on a
# cold, loaded machine (no warm filesystem cache, e.g. the first spawn on a CI runner) can take far longer
# than a warm import. This bounds how long the client waits for that first response; it returns as soon as
# /api/health answers, so a generous value only adds headroom for the cold case and never slows the warm one.
_STARTUP_HEALTH_TIMEOUT_SECS = 45.0


def wait_for_health(timeout: float = _STARTUP_HEALTH_TIMEOUT_SECS) -> None:
    if _await_health(timeout):
        return
    tail = _tail_daemon_log()
    msg = f'pxt daemon did not come up within {timeout}s'
    if tail != '':
        msg += f'\n--- daemon log tail ---\n{tail}'
    raise RuntimeError(msg)


# A daemon that is starting up has already written its pidfile (the bind precedes the write) but
# does not serve /api/health until it finishes importing pixeltable. Before treating a live-but-silent daemon as hung,
# give it this long to come up so we don't kill one that a concurrent pxt invocation just spawned.
_STARTUP_GRACE_PERIOD_SECS = 10.0


# The module the daemon is launched as (see spawn_detached()). We match the `-m <module>` launch
# form rather than a bare substring so a recycled PID whose argv merely mentions the module name
# (eg `python -c '...pixeltable_cli.server.daemon...'`) isn't mistaken for our daemon and killed.
_DAEMON_MODULE = 'pixeltable_cli.server.daemon'
_DAEMON_CMDLINE_RE = re.compile(rf'-m\s+{re.escape(_DAEMON_MODULE)}(\s|$)')


def _pid_cmdline(pid: int) -> str | None:
    """The command line of pid as a single string, or None if it can't be read."""
    # Linux exposes argv directly and cheaply via /proc; no subprocess needed.
    try:
        with open(f'/proc/{pid}/cmdline', 'rb') as f:
            raw = f.read()
        if raw != b'':
            return raw.replace(b'\x00', b' ').decode('utf-8', errors='replace')
    except OSError:
        pass
    if _IS_WINDOWS:
        # No cheap stdlib argv source; the caller falls back to refusing to kill.
        return None
    # macOS/BSD (and Linux without a readable /proc entry): ask ps for the full command line.
    try:
        out = subprocess.run(
            ['ps', '-p', str(pid), '-o', 'args='], capture_output=True, text=True, timeout=2.0, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = out.stdout.strip()
    return line if line != '' else None


def _pid_is_our_daemon(pid: int) -> bool:
    """Best-effort check that pid is one of our daemon processes rather than an unrelated process."""
    cmdline = _pid_cmdline(pid)
    return cmdline is not None and _DAEMON_CMDLINE_RE.search(cmdline) is not None


def _pid_alive(pid: int) -> bool:
    try:
        # signal 0 is the 'are you there?' probe (doesn't kill, just raises if the PID is gone)
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # PID exists but is owned by another user; treat as alive (we can't kill it anyway).
        return True
    except (OSError, SystemError):
        # SystemError surfaces on Windows when os.kill(pid, 0) hits an internal CPython
        # result-handling edge case for an unknown PID; treat the same as OSError ("gone").
        return False
    return True


def kill_and_wait(pid: int, timeout: float = 5.0) -> None:
    # Wait on the PID itself (not /health) so a hung-but-alive daemon that still holds the
    # listen socket is detected and SIGKILLed; otherwise the next spawn would fail with
    # 'address already in use' because we returned early on the health probe.
    # Windows has no SIGKILL; os.kill(pid, SIGTERM) calls TerminateProcess, which is
    # already a hard kill, so the fallback below is a no-op there.
    sigkill = getattr(signal, 'SIGKILL', signal.SIGTERM)
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _pid_alive(pid):
            return
        time.sleep(0.1)
    try:
        os.kill(pid, sigkill)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def ensure_running() -> str:
    health = fetch_health()
    if health is not None:
        client_identity = identity()
        diff = _identity_diff(client_identity, health)
        if len(diff) > 0:
            # Identity mismatch: the daemon was launched against a different install or env snapshot than the
            # client now sees (eg, after pip install -U pixeltable). Restart it ourselves rather than making
            # the user do it: a non-None health response means fetch_health() already verified the responder is
            # our daemon.
            reported_pid = health.get('pid')
            kill_and_wait(reported_pid)
            spawn_detached()
            wait_for_health()
            # Cross-verify: the new responder must have a fresh PID and an identity that fully matches the client.
            # Anything else means the restart did not actually swap in a daemon belonging to this install/env.
            new_health = fetch_health()
            if new_health is None:
                reason = 'new daemon did not respond to /api/health'
            elif new_health.get('pid') == reported_pid:
                reason = f'new daemon kept the killed PID {reported_pid}'
            else:
                new_diff = _identity_diff(client_identity, new_health)
                if len(new_diff) > 0:
                    reason = f'new daemon still differs in: {", ".join(new_diff)}'
                else:
                    reason = ''
            if reason != '':
                raise RuntimeError(
                    f'pxt daemon restart did not produce a matching responder on port {get_port()}: {reason}'
                )
    else:
        # Nothing is answering /api/health. Either no daemon is up, or one we started bound the
        # port and then wedged before serving health. The pidfile records the PID, but PIDs get
        # recycled and the file is only bookkeeping, so a live PID is not proof of ownership: only
        # reclaim the port (kill the holder) once we've confirmed the live process is actually our
        # daemon. An unconfirmed PID is treated as a stale pidfile and we just spawn.
        stale_pid = read_pidfile()
        if stale_pid is not None and _pid_alive(stale_pid) and _pid_is_our_daemon(stale_pid):
            # It may just be slow to start, so give it a grace window before concluding it is
            # hung; a daemon that comes up in the meantime is used as-is.
            if _await_health(_STARTUP_GRACE_PERIOD_SECS):
                return base_url()
            kill_and_wait(stale_pid)
        spawn_detached()
        wait_for_health()
    return base_url()


def _request(method: str, path: str, body: dict[str, Any] | None = None, params: dict[str, Any] | None = None) -> Any:
    try:
        base = ensure_running()
    except RuntimeError as e:
        print(f'pxt: {e}', file=sys.stderr)
        sys.exit(1)

    url = f'{base}{path}'
    if params is not None:
        # Drop unset values so the daemon sees its default; coerce bool to '1'/'0' to
        # match the server's query_bool parser.
        filtered = {k: ('1' if v is True else '0' if v is False else v) for k, v in params.items() if v is not None}
        if len(filtered) > 0:
            # doseq=True expands list values into repeated params (?pk=a&pk=b).
            url += '?' + urllib.parse.urlencode(filtered, doseq=True)

    headers: dict[str, str] = {'X-Pxt-Session': session_key()}
    data: bytes | None = None
    if body is not None:
        data = json.dumps(body).encode()
        headers['Content-Type'] = 'application/json'
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    # No timeout: localhost call, legitimate operations have no defensible upper bound.
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read() or b'null')
    except urllib.error.HTTPError as e:
        try:
            parsed = json.loads(e.read() or b'null')
            body = parsed if isinstance(parsed, dict) else {}
        except ValueError:
            body = {}
        detail = body.get('detail') or e.reason
        print(f'pxt: {e.code} {detail}', file=sys.stderr)
        server_tb = body.get('traceback')
        if server_tb is not None:
            # an unexpected daemon failure carries its traceback; show it so the user can report the error
            print(f'\n--- daemon traceback ---\n{server_tb}', file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f'pxt: cannot reach daemon at {url}: {e.reason}', file=sys.stderr)
        sys.exit(1)


def get_request(path: str, params: dict[str, Any] | None = None) -> Any:
    return _request('GET', path, params=params)


def post_request(path: str, body: dict[str, Any]) -> Any:
    return _request('POST', path, body=body)


def validate_path_arg(path: str) -> str:
    """Validate a pxt path's shape and return it unchanged. Paths travel as query params or body fields,
    which the transport URL-encodes, so no encoding happens here. A bad shape ('.' separator, trailing '/',
    '//') exits 2 with a clear message before any network round-trip."""
    err = validate_path_shape(path)
    if err is not None:
        print(f'pxt: {err}', file=sys.stderr)
        sys.exit(2)
    return path


def display_path(path: str) -> str:
    """Render a catalog path for human-readable output in the CLI's absolute form: a local path gets a
    leading '/' (matching how an absolute path is typed), a pxt:// URI is shown as-is, and the root is '/'."""
    if path.startswith('pxt://') or path.startswith('/'):
        return path
    return '/' + path
