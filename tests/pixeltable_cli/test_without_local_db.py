import json
import os
import pathlib
import subprocess
import sys
from textwrap import dedent
from typing import Any

import pytest

import pixeltable as pxt

from ..utils import DatabaseRoot
from .conftest import PxtRunner

pytestmark = pytest.mark.db_roots(
    'proxy', 'cloud-cli', reason='the commands under test reach a hosted catalog, and must not need the local one'
)

_APP = """
import datetime

import pixeltable as pxt

TableModel = pxt.model_base()


class Events(TableModel, name='events'):
    at: pxt.Timestamp
    recent = at > datetime.datetime(2024, 1, 1)
"""


@pytest.fixture(params=['embedded', 'external'])
def broken_db(request: pytest.FixtureRequest, tmp_path: pathlib.Path) -> dict[str, str | None]:
    """Environment overrides under which a new process cannot start or reach its own database."""
    if request.param == 'embedded':
        pgdata = tmp_path / 'pgdata'
        pgdata.write_text('', encoding='utf-8')  # initdb refuses a data directory that is a regular file
        return {'PIXELTABLE_PGDATA': str(pgdata), 'PIXELTABLE_DB_CONNECT_STR': None}
    return {'PIXELTABLE_DB_CONNECT_STR': 'postgresql://pixeltable@127.0.0.1:1/pixeltable'}  # nothing listens on 1


def _run_sdk(code: str, env_overrides: dict[str, str | None], cwd: pathlib.Path) -> Any:
    """Run code in a new Python process and return the JSON value of the last line it prints."""
    env = {**os.environ}
    for name, value in env_overrides.items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    r = subprocess.run(
        [sys.executable, '-c', 'import json\n\nimport pixeltable as pxt\n' + dedent(code)],
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


class TestWithoutLocalDb:
    def test_schema_update(
        self,
        cli: PxtRunner,
        db_root: DatabaseRoot,
        project_dir: pathlib.Path,
        daemon_port: int,
        broken_db: dict[str, str | None],
    ) -> None:
        app = project_dir / 'app.py'
        app.write_text(_APP, encoding='utf-8')
        target = db_root.make_catalog_path('events_app')
        # a daemon of its own, started with the broken database
        r = cli('schema', 'update', str(app), target, env_overrides={**broken_db, 'PXT_PORT': str(daemon_port)})
        assert 'created' in r.stdout, r.stdout
        assert pxt.get_table(f'{target}/events').columns() == ['at', 'recent']

    def test_insert_and_collect(
        self, db_root: DatabaseRoot, session_project: pathlib.Path, broken_db: dict[str, str | None]
    ) -> None:
        path = db_root.make_catalog_path('docs')
        docs = pxt.create_table(path, {'title': pxt.String})
        docs.add_computed_column(title_upper=docs.title.upper())
        code = f"""
            docs = pxt.get_table({path!r})
            docs.insert(title='Hello')
            print(json.dumps(docs.select(docs.title_upper).collect()['title_upper']))
        """
        assert _run_sdk(code, broken_db, session_project) == ['HELLO']

    def test_ls(self, db_root: DatabaseRoot, session_project: pathlib.Path, broken_db: dict[str, str | None]) -> None:
        pxt.create_dir(db_root.make_catalog_path('ls_dir'))
        code = f"""
            print(json.dumps(list(pxt.ls({db_root.prefix!r})['Name'])))
        """
        assert 'ls_dir' in _run_sdk(code, broken_db, session_project)
