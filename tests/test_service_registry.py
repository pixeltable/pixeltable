import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import pixeltable as pxt
from pixeltable.config import Config
from pixeltable.service import service_registry as registry

from .utils import pxt_raises


def _reaped_pid() -> int:
    """The pid of a process that has exited and been reaped, and is therefore no longer alive."""
    proc = subprocess.Popen([sys.executable, '-c', 'pass'])
    proc.wait()
    return proc.pid


def _deployment(name: str, base_path: str, pid: int, spec: Any = None) -> registry.ServiceDeployment:
    if spec is None:
        spec = {'name': name, 'prefix': '', 'routes': []}
    return registry.ServiceDeployment(
        service_name=name,
        base_path=base_path,
        endpoint='http://127.0.0.1:8000',
        pid=pid,
        created_at=1.0,
        app_file='/tmp/app.py',
        spec=spec,
    )


class TestServiceRegistry:
    def test_layout(self, init_env: None) -> None:
        """The tree lives under the Pixeltable home, mirroring the directory a service's models bind to."""
        home = Config.get().home
        assert registry.services_dir() == home / 'services'
        assert registry.target_dir() == home / 'services'
        assert registry.target_dir('d/c') == home / 'services' / 'd' / 'c'
        # a path is a path, however it is written
        assert registry.target_dir('d.c') == registry.target_dir('d/c')

    def test_deployments(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A deployment is recorded per service, and is only a deployment while its process runs."""
        monkeypatch.setattr(registry, 'services_dir', lambda: tmp_path / 'services')
        spec: Any = {'name': 'ingest', 'prefix': '/v1', 'routes': [{'method': 'POST', 'path': '/notes'}]}

        registry.save(_deployment('ingest', 'd/c', os.getpid(), spec=spec))
        deployed = registry.get('ingest', 'd/c')
        assert deployed is not None
        assert deployed['service_name'] == 'ingest'
        assert deployed['base_path'] == 'd/c'
        assert deployed['endpoint'] == 'http://127.0.0.1:8000'
        assert deployed['app_file'] == '/tmp/app.py'
        # the definition survives the round trip, so a diff can compare against what is deployed
        assert deployed['spec'] == spec
        # one file per service, and nothing left over from writing it
        assert [p.name for p in (tmp_path / 'services' / 'd' / 'c').iterdir()] == ['ingest.json']

        # the same name at another target is a different deployment; each target is read on its own
        registry.save(_deployment('ingest', 'd', os.getpid()))
        assert [d['base_path'] for d in registry.list_at('d')] == ['d']
        assert [d['base_path'] for d in registry.list_at('d/c')] == ['d/c']
        assert {(d['service_name'], d['base_path']) for d in registry.list_all()} == {
            ('ingest', 'd'),
            ('ingest', 'd/c'),
        }
        assert registry.get('ingest', 'nowhere') is None
        assert registry.list_at('nowhere') == []

        # a service whose process is gone is not deployed, and needs no cleanup to stop being reported
        registry.save(_deployment('crashed', 'd/c', _reaped_pid()))
        assert registry.get('crashed', 'd/c') is None
        assert [d['service_name'] for d in registry.list_at('d/c')] == ['ingest']
        # ... and the next update of that service takes the record over
        registry.save(_deployment('crashed', 'd/c', os.getpid()))
        assert registry.get('crashed', 'd/c') is not None

        # a file that is not a record is not a deployment
        (tmp_path / 'services' / 'd' / 'c' / 'junk.json').write_text('{not json')
        (tmp_path / 'services' / 'd' / 'c' / 'nopid.json').write_text('{"service_name": "nopid"}')
        assert sorted(d['service_name'] for d in registry.list_at('d/c')) == ['crashed', 'ingest']

        registry.remove('ingest', 'd/c')
        assert registry.get('ingest', 'd/c') is None
        assert [d['service_name'] for d in registry.list_at('d/c')] == ['crashed']
        # forgetting what is already forgotten is not an error
        registry.remove('ingest', 'd/c')

    def test_name_validation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A service name that could name something other than a record in the tree is refused."""
        monkeypatch.setattr(registry, 'services_dir', lambda: tmp_path / 'services')
        for name in ('../../escape', 'has/slash', 'has space', ''):
            with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match='not a valid service name'):
                registry.get(name)
            with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match='not a valid service name'):
                registry.save(_deployment(name, '', os.getpid()))
