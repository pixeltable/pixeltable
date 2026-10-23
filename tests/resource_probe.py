"""Per-test resource measurements, enabled by PXTTEST_RESOURCE_PROBE=<output .jsonl path>.

Each process that runs tests samples its own memory, its non-Postgres child processes, the machine's available
memory and swap, and the free space on the basetemp filesystem. The peaks for each test travel to the reporting
process (the xdist controller, or the only process without xdist) in the teardown report's user_properties. The
reporting process appends one JSON record per test to the output file and prints a `[probe]` line, so the job log
keeps the data even if the runner dies before the artifact is uploaded.
"""

import json
import os
import platform
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Generator

import psutil
import pytest

_GB = 1024**3
_SAMPLE_INTERVAL_S = 0.25


def _existing_dir(path: Path) -> Path:
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def _reset_os_peak() -> bool:
    """On Linux, reset VmHWM so that _os_peak_rss() measures a single test exactly. Returns whether it did."""
    try:
        Path('/proc/self/clear_refs').write_text('5', encoding='ascii')
        return True
    except OSError:
        return False


def _os_peak_rss(proc: psutil.Process) -> int:
    """The OS's record of this process's peak RSS; since _reset_os_peak() on Linux, and lifetime elsewhere."""
    if platform.system() == 'Linux':
        for line in Path('/proc/self/status').read_text(encoding='ascii').splitlines():
            if line.startswith('VmHWM:'):
                return int(line.split()[1]) * 1024
    if platform.system() == 'Windows':
        return proc.memory_info().peak_wset  # type: ignore[attr-defined]
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS


class _Sampler(threading.Thread):
    def __init__(self, disk_path: Path) -> None:
        super().__init__(name='resource-probe', daemon=True)
        self.disk_path = disk_path
        self.proc = psutil.Process()
        self.lock = threading.Lock()
        self.window: dict[str, float] = {}
        self.reset()

    def _measure(self) -> dict[str, float]:
        rss_self = self.proc.memory_info().rss
        rss_children = 0
        for child in self.proc.children(recursive=True):
            try:
                # Postgres backends count shared buffers in each RSS, which would overstate the test's own use
                if not child.name().lower().startswith('postgres'):
                    rss_children += child.memory_info().rss
            except psutil.Error:
                pass
        vm = psutil.virtual_memory()
        return {
            'rss': rss_self,
            'rss_tree': rss_self + rss_children,
            'avail': vm.available,
            'swap': psutil.swap_memory().used,
            'disk_free': psutil.disk_usage(str(_existing_dir(self.disk_path))).free,
        }

    def reset(self) -> None:
        self.os_peak_exact = _reset_os_peak()
        self.os_peak_start = _os_peak_rss(self.proc)
        m = self._measure()
        with self.lock:
            self.window = {
                'start': time.monotonic(),
                'rss_start': m['rss'],
                'rss_peak': m['rss'],
                'rss_tree_peak': m['rss_tree'],
                'avail_min': m['avail'],
                'swap_max': m['swap'],
                'disk_free_start': m['disk_free'],
                'disk_free_min': m['disk_free'],
            }

    def _update(self, m: dict[str, float]) -> None:
        with self.lock:
            w = self.window
            w['rss_peak'] = max(w['rss_peak'], m['rss'])
            w['rss_tree_peak'] = max(w['rss_tree_peak'], m['rss_tree'])
            w['avail_min'] = min(w['avail_min'], m['avail'])
            w['swap_max'] = max(w['swap_max'], m['swap'])
            w['disk_free_min'] = min(w['disk_free_min'], m['disk_free'])

    def snapshot(self) -> dict[str, Any]:
        self._update(self._measure())
        with self.lock:
            snap: dict[str, Any] = dict(self.window)
        snap['wall_s'] = time.monotonic() - snap.pop('start')
        # A lifetime peak that rose during this test was reached during this test
        os_peak = _os_peak_rss(self.proc)
        if self.os_peak_exact or os_peak > self.os_peak_start:
            snap['rss_peak'] = max(snap['rss_peak'], os_peak)
        return snap

    def run(self) -> None:
        while True:
            time.sleep(_SAMPLE_INTERVAL_S)
            try:
                self._update(self._measure())
            except psutil.Error:
                pass


class ResourceProbe:
    def __init__(self, config: pytest.Config, out_path: str) -> None:
        workerinput = getattr(config, 'workerinput', None)
        is_worker = workerinput is not None
        is_controller = not is_worker and bool(config.getoption('numprocesses', None))
        self.config = config
        self.worker_id = workerinput['workerid'] if workerinput is not None else 'main'
        self.out_path = None if is_worker else Path(out_path)
        self.sampler: _Sampler | None = None
        if not is_controller:
            basetemp = config.option.basetemp or tempfile.gettempdir()
            self.sampler = _Sampler(Path(basetemp))
            self.sampler.start()
        self.outcomes: dict[str, str] = {}
        self.durations: dict[str, float] = {}

    # Test-running side

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_protocol(self, item: pytest.Item) -> Generator[None, object, object]:
        if self.sampler is not None:
            self.sampler.reset()
        return (yield)  # noqa: B901

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_makereport(
        self, item: pytest.Item, call: pytest.CallInfo[None]
    ) -> Generator[None, pytest.TestReport, pytest.TestReport]:
        rep = yield
        if self.sampler is not None and call.when == 'teardown':
            snap = self.sampler.snapshot()
            snap['worker'] = self.worker_id
            rep.user_properties.append(('resource_probe', snap))
        return rep  # noqa: B901

    # Reporting side

    def pytest_sessionstart(self, session: pytest.Session) -> None:
        if self.out_path is None:
            return
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            'header': True,
            'os': os.environ.get('PXTTEST_CI_OS', platform.platform()),
            'cpus_logical': psutil.cpu_count(),
            'cpus_physical': psutil.cpu_count(logical=False),
            'mem_total': psutil.virtual_memory().total,
            'swap_total': psutil.swap_memory().total,
            'disk_free': psutil.disk_usage(str(_existing_dir(Path(self.config.option.basetemp or '.')))).free,
        }
        self._emit(header, f'[probe] host {json.dumps(header)}')

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if self.out_path is None:
            return
        nodeid = report.nodeid
        if report.when not in ('setup', 'call', 'teardown'):
            # xdist reports a crashed worker (eg, killed for running out of memory) with when='???'
            crash = {'nodeid': nodeid, 'outcome': 'crashed', 'detail': str(report.longrepr)[-500:]}
            self._emit(crash, f'[probe] CRASHED {nodeid}')
            return
        self.durations[nodeid] = self.durations.get(nodeid, 0.0) + report.duration
        if report.failed:
            self.outcomes[nodeid] = 'failed'
        elif report.skipped and nodeid not in self.outcomes:
            self.outcomes[nodeid] = 'skipped'
        if report.when != 'teardown':
            return
        snap: dict[str, Any] | None = dict(report.user_properties).get('resource_probe')  # type: ignore[assignment]
        if snap is None:
            return
        outcome = self.outcomes.pop(nodeid, 'passed')
        record: dict[str, Any] = {
            'nodeid': nodeid,
            'outcome': outcome,
            'duration_s': self.durations.pop(nodeid, 0.0),
            **snap,
        }

        def gb(key: str) -> str:
            return f'{snap[key] / _GB:.1f}G'

        line = (
            f'[probe] {snap["worker"]} {outcome.upper():7} {record["duration_s"]:7.1f}s '
            f'rss={gb("rss_peak")} (+{(snap["rss_peak"] - snap["rss_start"]) / _GB:.1f}G) tree={gb("rss_tree_peak")} '
            f'avail_min={gb("avail_min")} swap_max={gb("swap_max")} '
            f'disk_free_min={gb("disk_free_min")} (-{(snap["disk_free_start"] - snap["disk_free_min"]) / _GB:.1f}G) '
            f'{nodeid}'
        )
        self._emit(record, line)

    def _emit(self, record: dict[str, Any], line: str) -> None:
        assert self.out_path is not None
        with self.out_path.open('a', encoding='utf8') as fp:
            fp.write(json.dumps(record) + '\n')
        tr = self.config.pluginmanager.get_plugin('terminalreporter')
        if tr is not None:
            tr.write_line(line)
