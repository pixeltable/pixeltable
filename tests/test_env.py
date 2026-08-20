from typing import Iterator

import pytest

import pixeltable as pxt
from pixeltable import exceptions as excs
from pixeltable.config import Config
from pixeltable.env import Env
from pixeltable.runtime import reset_runtime
from pixeltable.utils.filecache import FileCache

from .utils import pxt_raises, skip_test_if_not_local

pytestmark = pytest.mark.local('exercises process-global Env/Config and runtime reset')


def _reset_env(reinit: bool, db_name: str | None) -> None:
    """Reset the environment for testing. db_name=None restores the default test database."""
    reset_runtime()
    # Reload configs
    config_overrides = {} if db_name is None else {'pixeltable.db': db_name}
    Config.init(config_overrides=config_overrides, reinit=True)
    Env._init_env(reinit_db=reinit)
    FileCache.init()


@pytest.fixture(autouse=True)
def restore_env() -> Iterator[None]:
    """Put the process back on its configured database once the test is done."""
    yield
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


class TestApiKey:
    def test_require_api_key(self, init_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('PIXELTABLE_API_KEY', 'sk-test')
        assert Env.get().require_api_key() == 'sk-test'
        assert Env.get().require_api_key('create a database') == 'sk-test'

        monkeypatch.delenv('PIXELTABLE_API_KEY', raising=False)
        monkeypatch.setattr(Config, 'get_string_value', lambda self, key, section='pixeltable': None)
        with pxt_raises(excs.ErrorCode.MISSING_CREDENTIALS, match='A Pixeltable API key is required\\. Set it with'):
            Env.get().require_api_key()
        with pxt_raises(
            excs.ErrorCode.MISSING_CREDENTIALS, match='API key is required to create a database\\. Set it with'
        ):
            Env.get().require_api_key('create a database')


class TestProxyEndpoint:
    @pytest.mark.parametrize(
        ('cloud_host', 'expected'),
        [
            (None, ('acme-main.pxt.run', 9000)),
            ('dev.pxt.run', ('acme-main.dev.pxt.run', 9000)),
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
        with pxt_raises(excs.ErrorCode.GENERIC_USER_ERROR, match=error):
            Env.get().proxy_endpoint('acme', 'main')
