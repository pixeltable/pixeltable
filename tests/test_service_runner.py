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
        app_file.write_text(dedent(_APP_SRC), encoding='utf-8')
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

    def test_update_prune_stop_list(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """update starts what is declared and restarts what changed; prune and stop take deployments down."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')
        from pixeltable_cli.server import bridge
        from pixeltable_cli.utils import PxtPath

        app_file = self._write_app(tmp_path)
        target = PxtPath('svc')
        pxt.create_dir('svc')
        t = pxt.create_table('svc.notes', {'note_id': pxt.Int, 'val': pxt.Int}, primary_key='note_id')
        t.add_computed_column(incr=t.val + 1)

        try:
            plan = bridge.service_update(app_file, target)
            assert [(d['name'], d['status']) for d in plan['services']] == [('ingest', 'applied')]
            first = ServiceDeployment.read('ingest', 'svc')
            assert first is not None and service_runner.health_ok(first.endpoint)

            # a service already serving its declaration is left running, not restarted
            plan = bridge.service_update(app_file, target)
            assert [d['status'] for d in plan['services']] == ['skipped']
            unchanged = ServiceDeployment.read('ingest', 'svc')
            assert unchanged is not None and unchanged.pid == first.pid

            # adding a route changes the declaration, so the service is replaced by one serving it
            app_path = pathlib.Path(app_file)
            app_path.write_text(
                app_path.read_text(encoding='utf-8') + "ingest.add_compute_route(Notes, path='/compute')\n",
                encoding='utf-8',
            )
            plan = bridge.service_update(app_file, target)
            assert [d['status'] for d in plan['services']] == ['applied']
            second = ServiceDeployment.read('ingest', 'svc')
            assert second is not None and second.pid != first.pid
            assert not pid_alive(first.pid)
            assert [route['path'] for route in second.spec['routes']] == ['/notes', '/compute']

            # list reports it, with the file it was served from
            listed = bridge.service_list(target)
            assert [(d['name'], d['base_path'], d['app_file']) for d in listed] == [
                ('ingest', 'svc', str(pathlib.Path(app_file).resolve()))
            ]

            # a deployment the file no longer declares is stopped and forgotten
            app_path.write_text(dedent(_APP_SRC).replace("name='ingest'", "name='other'"), encoding='utf-8')
            plan = bridge.service_prune(app_file, target)
            assert [(op['name'], op['status']) for op in plan['ops']] == [('ingest', 'applied')]
            assert not pid_alive(second.pid)
            assert bridge.service_list(target) == []
        finally:
            for deployment in ServiceDeployment.list('svc'):
                service_runner.stop(deployment)

    def test_stop_is_idempotent(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Stopping a service that is not deployed is reported, not an error."""
        skip_test_if_not_installed('fastapi')
        from pixeltable_cli.server import bridge
        from pixeltable_cli.utils import PxtPath

        ops = bridge.service_stop(['nosuch'], PxtPath(''))
        assert [(op['name'], op['status']) for op in ops] == [('nosuch', 'skipped')]

    def test_example_app_is_servable(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """The file 'pxt service example' writes declares both the tables and the services, as it says."""
        skip_test_if_not_installed('fastapi')
        skip_test_if_not_installed('uvicorn')
        from pixeltable_cli.client.commands.service import _EXAMPLE_APP
        from pixeltable_cli.server import bridge
        from pixeltable_cli.utils import PxtPath

        app_file = tmp_path / 'app.py'
        app_file.write_text(_EXAMPLE_APP, encoding='utf-8')
        target = PxtPath('example')

        # the same file drives both verbs: the models create the tables, the routers serve over them
        bridge.schema_update(str(app_file), target)
        assert pxt.get_table('example.docs') is not None

        try:
            plan = bridge.service_update(str(app_file), target)
            assert [(d['name'], d['status']) for d in plan['services']] == [('ingest', 'applied')]
            deployment = ServiceDeployment.read('ingest', 'example')
            assert deployment is not None

            resp = httpx.post(
                f'{deployment.endpoint}/docs', json={'doc_id': 1, 'title': 'a title', 'body': None}, timeout=30.0
            )
            assert resp.status_code == 200, resp.text
            assert resp.json() == {'title_upper': 'A TITLE'}
        finally:
            for d in ServiceDeployment.list('example'):
                service_runner.stop(d)
