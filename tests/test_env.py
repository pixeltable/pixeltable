import asyncio
import logging
import os
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

import pixeltable as pxt
from pixeltable import env, exceptions as excs
from pixeltable.config import Config
from pixeltable.env import Env
from pixeltable.runtime import get_runtime, reset_runtime
from pixeltable.utils.filecache import FileCache

from .utils import pxt_raises, skip_test_if_not_local

pytestmark = pytest.mark.db_roots('local', reason='exercises process-global Env/Config and runtime reset')


def _reset_env(reinit: bool, db_name: str | None) -> None:
    """Reset the environment for testing. db_name=None keeps whichever database is configured."""
    reset_runtime()
    if db_name is not None:
        os.environ['PIXELTABLE_DB'] = db_name
    Config.init(reinit=True)
    Env._init_env(reinit_db=reinit)
    FileCache.init()


@pytest.fixture(autouse=True)
def restore_env() -> Iterator[None]:
    """Put the process back on its configured database once the test is done.

    PIXELTABLE_DB is how the suite gives each worker its own database, so the value it had on entry is
    what has to come back; dropping it would move every later test to the default database.
    """
    configured_db = os.environ.get('PIXELTABLE_DB')
    yield
    if configured_db is None:
        os.environ.pop('PIXELTABLE_DB', None)
    else:
        os.environ['PIXELTABLE_DB'] = configured_db
    _reset_env(reinit=False, db_name=None)


class TestEnvReset:
    """Run tests for env reset"""

    def test_basic(self, uses_db: None) -> None:
        """Test basic env clear functionality."""
        skip_test_if_not_local('Env reset tests require a local (file-backed) Pixeltable database')
        # Set environment
        dbname = 'test_basic_db'
        _reset_env(reinit=True, db_name=dbname)

        env1 = Env.get()
        assert env1 is not None
        assert env1._db_name == dbname

        # Create a simple table
        t = pxt.create_table('test_table', {'col1': pxt.String | None})
        t.insert([{'col1': 'test_data'}])
        assert t.count() == 1

        # Verify we can create a new env instance with same db
        _reset_env(reinit=False, db_name=dbname)
        env2 = Env.get()
        assert env2 is not None
        assert env2 != env1
        assert env2._db_name == dbname
        t = pxt.get_table('test_table')
        assert t is not None
        assert t.count() == 1

    def test_switch_environments(self, uses_db: None) -> None:
        """Test switching between two environments."""
        skip_test_if_not_local('Env reset tests require a local (file-backed) Pixeltable database')
        dbname1 = 'test_switch_environments_db1'
        dbname2 = 'test_switch_environments_db2'

        # Environment 1
        _reset_env(reinit=True, db_name=dbname1)

        t1 = pxt.create_table('table1', {'name': pxt.String | None})
        t1.insert([{'name': 'env1_data'}])

        # Switch to Environment 2
        _reset_env(reinit=True, db_name=dbname2)

        env2 = Env.get()
        assert env2._db_name == dbname2

        # Create different table in env2
        t2 = pxt.create_table('table2', {'value': pxt.Int | None})
        t2.insert([{'value': 42}])

        # Verify table1 doesn't exist in env2
        with pxt_raises(excs.ErrorCode.PATH_NOT_FOUND):
            pxt.get_table('table1')

        # Switch back to Environment 1
        _reset_env(reinit=False, db_name=dbname1)

        env1_again = Env.get()
        assert env1_again._db_name == dbname1

        # Verify table1 still exists in env1
        t1_again = pxt.get_table('table1')
        assert t1_again.count() == 1

        # Verify table2 doesn't exist in env1
        with pxt_raises(excs.ErrorCode.PATH_NOT_FOUND):
            pxt.get_table('table2')

    def test_metadata_persistence(self, uses_db: None) -> None:
        """Test that metadata persists across environment switches."""
        skip_test_if_not_local('Env reset tests require a local (file-backed) Pixeltable database')
        metadata_db = 'test_metadata_persistence_db'
        # Environment 1 setup
        _reset_env(reinit=True, db_name=metadata_db)

        # Create directory structure
        pxt.create_dir('analytics')
        pxt.create_dir('analytics/reports')

        # Create tables with different features
        t1 = pxt.create_table(
            'users', {'user_id': pxt.Int | None, 'username': pxt.String | None, 'active': pxt.Bool | None}
        )

        t2 = pxt.create_table('analytics/reports/sales', {'sale_id': pxt.Int | None, 'amount': pxt.Float | None})

        # Add computed column
        t2.add_computed_column(amount_doubled=t2.amount * 2)

        # Create view
        v1 = pxt.create_view('analytics/high_sales', t2.where(t2.amount > 100.0))

        # Insert data
        t1.insert(
            [{'user_id': 1, 'username': 'alice', 'active': True}, {'user_id': 2, 'username': 'bob', 'active': False}]
        )

        t2.insert([{'sale_id': 1, 'amount': 150.0}, {'sale_id': 2, 'amount': 50.0}])

        # Record metadata
        dirs_before = set(pxt.list_dirs())
        tables_before = set(pxt.list_tables())
        user_count = t1.count()
        high_sales_count = v1.count()

        # Reinitialize same environment
        _reset_env(reinit=False, db_name=metadata_db)

        # Verify metadata
        assert set(pxt.list_dirs()) == dirs_before
        assert set(pxt.list_tables()) == tables_before

        # Verify data
        t1_new = pxt.get_table('users')
        assert t1_new.count() == user_count

        v1_new = pxt.get_table('analytics/high_sales')
        assert v1_new.count() == high_sales_count

        # Verify computed column still works
        t2_new = pxt.get_table('analytics/reports/sales')
        result = t2_new.where(t2_new.sale_id == 1).select(t2_new.amount_doubled).collect()
        assert result[0]['amount_doubled'] == 300.0


class TestHostedMediaDefault:
    def test_home_bucket_default(self, uses_db: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """On a hosted db's pod, media that has no configured destination goes to the db's home bucket."""
        monkeypatch.delenv('PIXELTABLE_INPUT_MEDIA_DEST', raising=False)
        monkeypatch.delenv('PIXELTABLE_OUTPUT_MEDIA_DEST', raising=False)

        # no hosted-db identity: the defaults stay as configured (here: unset, ie. the local media dir)
        monkeypatch.delenv('PXTCLOUD_ORG', raising=False)
        monkeypatch.delenv('PXTCLOUD_DB', raising=False)
        _reset_env(reinit=False, db_name=None)
        assert Env.get().hosted_db() is None
        assert Env.get().default_input_media_dest is None
        assert Env.get().default_output_media_dest is None

        # identity present: both defaults are the home bucket
        monkeypatch.setenv('PXTCLOUD_ORG', 'org1')
        monkeypatch.setenv('PXTCLOUD_DB', 'db1')
        _reset_env(reinit=False, db_name=None)
        assert Env.get().hosted_db() == ('org1', 'db1')
        assert Env.get().default_input_media_dest == 'pxt://org1:db1/buckets/home'
        assert Env.get().default_output_media_dest == 'pxt://org1:db1/buckets/home'

        # a user-configured default wins over the home bucket, per setting
        monkeypatch.setenv('PIXELTABLE_OUTPUT_MEDIA_DEST', 's3://user-bucket/prefix')
        _reset_env(reinit=False, db_name=None)
        assert Env.get().default_input_media_dest == 'pxt://org1:db1/buckets/home'
        assert Env.get().default_output_media_dest == 's3://user-bucket/prefix'
        monkeypatch.delenv('PIXELTABLE_OUTPUT_MEDIA_DEST')

        # so does the database's own entry in the project config
        project_root = Config.get().project_root
        project = tmp_path / 'project'
        project.mkdir()
        (project / 'pixeltable.toml').write_text(
            "[[pixeltable.database]]\nname = 'pxt://org1:db1'\ndb_input_media_dest = 's3://entry-bucket/input/'\n"
        )
        reset_runtime()
        Config.init(reinit=True, project_root=project)
        Env._init_env(reinit_db=False)
        assert Env.get().default_input_media_dest == 's3://entry-bucket/input/'
        assert Env.get().default_output_media_dest == 'pxt://org1:db1/buckets/home'
        Config.init(reinit=True, project_root=project_root)
        _reset_env(reinit=False, db_name=None)

    def test_pxtfs_media_dest_warns(self, uses_db: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """A media destination configured in the pxtfs:// spelling warns, at the line that set up Pixeltable, and
        still applies. The pxt:// spelling, the home-bucket default and a daemon do not warn."""
        monkeypatch.delenv('PIXELTABLE_INPUT_MEDIA_DEST', raising=False)
        monkeypatch.setenv('PXTCLOUD_ORG', 'org1')
        monkeypatch.setenv('PXTCLOUD_DB', 'db1')
        monkeypatch.setenv('PIXELTABLE_OUTPUT_MEDIA_DEST', 'pxtfs://org1:db1/home/out')
        with pytest.warns(excs.PixeltableDeprecationWarning) as record:
            _reset_env(reinit=False, db_name=None)
        # the record also holds unrelated warnings raised in the block, such as a ResourceWarning when GC collects a
        # socket an earlier test on this worker left open
        deprecations = [w for w in record if issubclass(w.category, excs.PixeltableDeprecationWarning)]
        assert [str(w.message) for w in deprecations] == [
            "output_media_dest 'pxtfs://org1:db1/home/out' uses the deprecated pxtfs:// spelling; write "
            "'pxt://org1:db1/buckets/home/out' instead. Values already stored as pxtfs:// keep reading."
        ]
        assert deprecations[0].filename == __file__
        assert Env.get().default_output_media_dest == 'pxtfs://org1:db1/home/out'
        assert Env.get().default_input_media_dest == 'pxt://org1:db1/buckets/home'

        # the suite turns a Pixeltable warning into an error, so each of these would raise if it warned
        monkeypatch.setenv('PIXELTABLE_PROXY_DAEMON', '1')
        _reset_env(reinit=False, db_name=None)
        monkeypatch.delenv('PIXELTABLE_PROXY_DAEMON')
        monkeypatch.setenv('PIXELTABLE_OUTPUT_MEDIA_DEST', 'pxt://org1:db1/buckets/home/out')
        _reset_env(reinit=False, db_name=None)
        assert Env.get().default_output_media_dest == 'pxt://org1:db1/buckets/home/out'

    def test_local_media_dest_in_project(self, uses_db: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A local media destination inside the project directory is not allowed."""
        media_dir = tmp_path / 'media'
        for dest in (media_dir.as_uri(), str(media_dir)):
            monkeypatch.setenv('PIXELTABLE_OUTPUT_MEDIA_DEST', dest)
            reset_runtime()
            Config.init(reinit=True, project_root=tmp_path)
            with pxt_raises(pxt.ErrorCode.INVALID_CONFIGURATION, match='output_media_dest .* is inside the project'):
                Env._init_env(reinit_db=False)


class TestProxyEndpoint:
    @pytest.mark.parametrize(
        ('cloud_host', 'expected'),
        [
            (None, ('acme-main.pxt.run', 443)),
            ('dev.pxt.run', ('acme-main.dev.pxt.run', 443)),
            ('pxt.run:9000', ('acme-main.pxt.run', 9000)),
            ('localhost:9443', ('acme-main.localhost', 9443)),
        ],
    )
    def test_proxy_endpoint(
        self, cloud_host: str | None, expected: tuple[str, int], init_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if cloud_host is None:
            monkeypatch.delenv('PIXELTABLE_CLOUD_HOST', raising=False)
        else:
            monkeypatch.setenv('PIXELTABLE_CLOUD_HOST', cloud_host)
        assert Env.get().proxy_endpoint('acme', 'main') == expected

    @pytest.mark.parametrize(
        ('cloud_host', 'error'),
        [('dev.pxt.run:https', "port 'https' is not a valid integer"), (':9000', 'missing host')],
    )
    def test_proxy_endpoint_rejects(
        self, cloud_host: str, error: str, init_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('PIXELTABLE_CLOUD_HOST', cloud_host)
        with pxt_raises(excs.ErrorCode.INVALID_CONFIGURATION, match=error):
            Env.get().proxy_endpoint('acme', 'main')


class TestOptionalPackages:
    def test_require_package_missing_parent(self, init_env: None) -> None:
        """A missing parent package is an absent optional dependency, not a raw import error.

        find_spec raises ModuleNotFoundError when the parent is not installed. Registration already
        records that as absent; the recheck in require_package has to do the same.
        """
        env = Env.get()
        name = 'missing_parent.child'
        env._Env__register_package(name)  # type: ignore[attr-defined]
        try:
            with pxt_raises(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                match=(
                    r'This feature requires the `missing_parent\.child` package'
                    r'.*pip install -U missing_parent\.child'
                ),
            ):
                env.require_package(name)
        finally:
            del env._Env__optional_packages[name]  # type: ignore[attr-defined]


class _TestClient:
    """Stands in for an async SDK client, recording how it was closed."""

    closes: int
    loop_was_open: bool | None  # whether the loop was still usable when close() ran

    def __init__(self) -> None:
        self.closes = 0
        self.loop_was_open = None

    async def call(self) -> None:
        """Stands in for an API request."""
        await asyncio.sleep(0)

    async def close(self) -> None:
        self.closes += 1
        self.loop_was_open = not asyncio.get_running_loop().is_closed()


# every client the factory below hands out; the ones created on another thread are reachable nowhere else
_test_clients: list[_TestClient] = []


@env.register_client('test_sdk', credential_param=None)
def _() -> _TestClient:
    client = _TestClient()
    _test_clients.append(client)
    return client


@pxt.udf
async def uses_client(n: int) -> int:
    """Fetches the registered client, the way the udfs of an API integration do."""
    await get_runtime().get_client('test_sdk').call()
    return n + 1


@pxt.udf
async def client_embedding(text: str) -> pxt.Array[(4,), np.float32]:
    """An embedding udf backed by an API client, as the hosted embedding services are."""
    await get_runtime().get_client('test_sdk').call()
    return np.full(4, float(len(text)), dtype=np.float32)


class _CleanupProbe:
    """A stand-in client whose close() is distinct from a registered cleanup callback."""

    closes: int
    cleanups: int
    loop_id: int | None

    def __init__(self) -> None:
        self.closes = 0
        self.cleanups = 0
        self.loop_id = None

    def close(self) -> None:
        self.closes += 1


class TestApiClients:
    """The clients of an API integration are closed on the event loop they were created on."""

    def test_cleanup_callback_while_referenced(self, uses_db: None, caplog: pytest.LogCaptureFixture) -> None:
        async_client = _CleanupProbe()
        sync_client = _CleanupProbe()

        async def async_cleanup(client: _CleanupProbe) -> None:
            await asyncio.sleep(0)
            client.cleanups += 1
            client.loop_id = id(asyncio.get_running_loop())

        def sync_cleanup(client: _CleanupProbe) -> None:
            client.cleanups += 1

        @env.register_client('cleanup_callback_sdk', credential_param=None, cleanup=async_cleanup)
        def _async_factory() -> _CleanupProbe:
            return async_client

        @env.register_client('sync_cleanup_sdk', credential_param=None, cleanup=sync_cleanup)
        def _sync_factory() -> _CleanupProbe:
            return sync_client

        runtime = get_runtime()
        loop_id = id(runtime.event_loop)
        assert runtime.get_client('cleanup_callback_sdk') is async_client
        assert runtime.get_client('sync_cleanup_sdk') is sync_client
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger='pixeltable.runtime'):
            reset_runtime()

        assert async_client.cleanups == 1
        assert async_client.closes == 0
        assert async_client.loop_id == loop_id
        assert sync_client.cleanups == 1
        assert sync_client.closes == 0
        assert not any(rec.message.startswith('Error closing client') for rec in caplog.records)

    def test_reregister_keeps_cleanup_of_existing_client(self, uses_db: None) -> None:
        """A client keeps the cleanup registered when it was created.

        Re-registering the same name replaces the factory for clients created later. It does not
        change the callback that closes a client already cached on the runtime.
        """
        original = _CleanupProbe()
        replacement = _CleanupProbe()
        closed_by: list[tuple[_CleanupProbe, str]] = []

        def cleanup_a(client: _CleanupProbe) -> None:
            closed_by.append((client, 'A'))

        def cleanup_b(client: _CleanupProbe) -> None:
            closed_by.append((client, 'B'))

        @env.register_client('demo', credential_param=None, cleanup=cleanup_a)
        def _factory_a() -> _CleanupProbe:
            return original

        runtime = get_runtime()
        assert runtime.get_client('demo') is original

        @env.register_client('demo', credential_param=None, cleanup=cleanup_b)
        def _factory_b() -> _CleanupProbe:
            return replacement

        assert runtime.get_client('demo') is original
        reset_runtime()
        assert closed_by == [(original, 'A')]
        assert original.closes == 0
        assert replacement.closes == 0

        runtime = get_runtime()
        assert runtime.get_client('demo') is replacement
        reset_runtime()
        assert closed_by == [(original, 'A'), (replacement, 'B')]

    def test_closed_on_reset(self, uses_db: None) -> None:
        t = pxt.create_table('clients', {'n': pxt.Int, 's': pxt.String})
        t.add_computed_column(m=uses_client(t.n))
        t.add_embedding_index('s', embedding=client_embedding)
        n_clients = len(_test_clients)

        t.insert([{'n': 1, 's': 'chartreuse'}])  # the plan runs on the calling thread
        # the embedding of the search string is computed on a thread the runtime keeps for coroutines
        assert t.order_by(t.s.similarity(string='chartreuse'), asc=False).limit(1).collect()['n'] == [1]
        clients = _test_clients[n_clients:]
        assert len(clients) == 2, 'expected a client per thread'

        reset_runtime()

        assert all((c.closes, c.loop_was_open) == (1, True) for c in clients)

    def test_closed_with_running_loop(self, uses_db: None) -> None:
        t = pxt.create_table('clients', {'n': pxt.Int})
        t.add_computed_column(m=uses_client(t.n))
        n_clients = len(_test_clients)

        async def insert_from_a_running_loop() -> None:  # noqa: RUF029
            # a live loop on the calling thread, as in a notebook cell, makes execution fan out to a thread
            # and event loop of its own
            t.insert([{'n': 1}])

        asyncio.run(insert_from_a_running_loop())

        assert t.collect()['m'] == [2]
        clients = _test_clients[n_clients:]
        assert len(clients) == 1
        assert (clients[0].closes, clients[0].loop_was_open) == (1, True)
