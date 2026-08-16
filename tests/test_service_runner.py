import pathlib
from textwrap import dedent

import httpx
import pytest

import pixeltable as pxt
from pixeltable.serving import service_runner
from pixeltable.serving.service_registry import ServiceDeployment
from pixeltable.utils.process import pid_alive

from .utils import pxt_raises, skip_test_if_not_installed

pytestmark = pytest.mark.local('a local service runs against the in-process catalog')

_APP_SRC = """
import pixeltable as pxt
from pixeltable.serving import FastAPIRouter

TableModel = pxt.model_base()


class Notes(TableModel, name='notes'):
    note_id = pxt.Column(type=pxt.Int, primary_key=True)
    val: pxt.Int
    incr = val + 1


ingest = FastAPIRouter(name='ingest')
ingest.add_insert_route(Notes, path='/notes', inputs=['note_id', 'val'], outputs=['incr'])
"""


class TestServiceRunner:
    def _write_app(self, tmp_path: pathlib.Path) -> str:
        app_file = tmp_path / 'app.py'
        app_file.write_text(dedent(_APP_SRC))
        return str(app_file)

    def test_start_serves_and_stop_removes(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """A started service serves its routes against the target it is bound to, until it is stopped."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')
        app_file = self._write_app(tmp_path)
        pxt.create_dir('svc')
        t = pxt.create_table('svc.notes', {'note_id': pxt.Int, 'val': pxt.Int}, primary_key='note_id')
        t.add_computed_column(incr=t.val + 1)

        deployment = service_runner.start(app_file, 'ingest', 'svc')
        try:
            assert deployment.service_name == 'ingest'
            assert deployment.base_path == 'svc'
            assert pid_alive(deployment.pid)
            assert service_runner.health_ok(deployment.endpoint)
            # the record is what a reader of the registry sees, and it names the file it was served from
            recorded = ServiceDeployment.read('ingest', 'svc')
            assert recorded is not None
            assert recorded.endpoint == deployment.endpoint
            assert recorded.app_file == str(pathlib.Path(app_file).resolve())
            assert [route['path'] for route in recorded.spec['routes']] == ['/notes']

            resp = httpx.post(f'{deployment.endpoint}/notes', json={'note_id': 1, 'val': 10}, timeout=30.0)
            assert resp.status_code == 200, resp.text
            assert resp.json() == {'incr': 11}
            # the route wrote to the table the service is bound to
            assert t.where(t.note_id == 1).count() == 1

            # starting again returns the running service rather than a second process
            assert service_runner.start(app_file, 'ingest', 'svc').pid == deployment.pid
        finally:
            service_runner.stop(deployment)

        assert not pid_alive(deployment.pid)
        assert ServiceDeployment.read('ingest', 'svc') is None

    def test_start_rejects_unknown_service(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """A name the file does not declare fails in the calling process, before anything is spawned."""
        skip_test_if_not_installed('fastapi')
        app_file = self._write_app(tmp_path)
        with pxt_raises(pxt.ErrorCode.SERVICE_NOT_FOUND, match='declares no service named'):
            service_runner.start(app_file, 'nosuch', '')
        assert ServiceDeployment.list(recursive=True) == []
