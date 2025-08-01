import os
import shutil
import tempfile
from typing import Callable, Dict, Generator, Set, TypedDict

import pytest

import pixeltable as pxt
from pixeltable import exceptions as excs
from pixeltable.catalog import Catalog
from pixeltable.config import Config
from pixeltable.env import Env


class MetadataState(TypedDict, total=False):
    """Typed dictionary for tracking metadata state."""

    dirs: Set[str]
    tables: Set[str]
    counts: Dict[str, int]
    data_checks: Dict[str, Callable[[], None]]


@pytest.fixture(scope='function')
def test_setup() -> Generator[Dict[str, str], None, None]:
    """Create test environment directories."""
    test_dir = tempfile.mkdtemp(prefix='pxt_test_env')
    env1_home = os.path.join(test_dir, 'env1')
    env2_home = os.path.join(test_dir, 'env2')

    os.makedirs(env1_home)
    os.makedirs(env2_home)

    yield {'env1_home': env1_home, 'env2_home': env2_home, 'test_dir': test_dir}

    # Cleanup
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


def _reset_env() -> None:
    """Reset the environment for testing."""
    Catalog.clear()
    # Reload configs
    Config.init(config_overrides={}, reinit=True)
    Env._init_env()

@pytest.mark.skip("This test may be affecting other test setup, disabling for now")
class TestEnvReset:

    def test_basic(self, test_setup: Dict[str, str]) -> None:
        """Test basic env clear functionality."""
        # Set environment 1
        os.environ['PIXELTABLE_HOME'] = test_setup['env1_home']
        os.environ['PIXELTABLE_DB'] = 'test_db1'
        _reset_env()

        env1 = Env.get()
        assert env1 is not None
        assert env1._db_name == 'test_db1'

        # Create a simple table
        t = pxt.create_table('test_table', {'col1': pxt.String})
        t.insert([{'col1': 'test_data'}])
        assert t.count() == 1

        # Verify we can create a new instance with same db
        _reset_env()
        env2 = Env.get()
        assert env2 is not None
        assert env2 != env1
        assert env2._db_name == 'test_db1'
        t = pxt.get_table('test_table')
        assert t is not None
        assert t.count() == 1

    def test_switch_environments(self, test_setup: Dict[str, str]) -> None:
        """Test switching between two environments."""
        # Environment 1
        os.environ['PIXELTABLE_HOME'] = test_setup['env1_home']
        os.environ['PIXELTABLE_DB'] = 'db1'
        _reset_env()

        t1 = pxt.create_table('table1', {'name': pxt.String})
        t1.insert([{'name': 'env1_data'}])

        # Switch to Environment 2
        os.environ['PIXELTABLE_HOME'] = test_setup['env2_home']
        os.environ['PIXELTABLE_DB'] = 'db2'
        _reset_env()

        env2 = Env.get()
        assert env2._db_name == 'db2'

        # Create different table in env2
        t2 = pxt.create_table('table2', {'value': pxt.Int})
        t2.insert([{'value': 42}])

        # Verify table1 doesn't exist in env2
        with pytest.raises(excs.Error):
            pxt.get_table('table1')

        # Switch back to Environment 1
        os.environ['PIXELTABLE_HOME'] = test_setup['env1_home']
        os.environ['PIXELTABLE_DB'] = 'db1'
        _reset_env()

        env1_again = Env.get()
        assert env1_again._db_name == 'db1'

        # Verify table1 still exists in env1
        t1_again = pxt.get_table('table1')
        assert t1_again.count() == 1

        # Verify table2 doesn't exist in env1
        with pytest.raises(excs.Error):
            pxt.get_table('table2')

    def test_metadata_persistence(self, test_setup: Dict[str, str]) -> None:
        """Test that metadata persists across environment switches."""
        # Environment 1 setup
        os.environ['PIXELTABLE_HOME'] = test_setup['env1_home']
        os.environ['PIXELTABLE_DB'] = 'metadata_db'
        _reset_env()

        # Create directory structure
        pxt.create_dir('analytics')
        pxt.create_dir('analytics.reports')

        # Create tables with different features
        t1 = pxt.create_table('users', {'user_id': pxt.Int, 'username': pxt.String, 'active': pxt.Bool})

        t2 = pxt.create_table('analytics.reports.sales', {'sale_id': pxt.Int, 'amount': pxt.Float})

        # Add computed column
        t2.add_computed_column(amount_doubled=t2.amount * 2)

        # Create view
        v1 = pxt.create_view('analytics.high_sales', t2.where(t2.amount > 100.0))

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
        _reset_env()

        # Verify metadata
        assert set(pxt.list_dirs()) == dirs_before
        assert set(pxt.list_tables()) == tables_before

        # Verify data
        t1_new = pxt.get_table('users')
        assert t1_new.count() == user_count

        v1_new = pxt.get_table('analytics.high_sales')
        assert v1_new.count() == high_sales_count

        # Verify computed column still works
        t2_new = pxt.get_table('analytics.reports.sales')
        result = t2_new.where(t2_new.sale_id == 1).select(t2_new.amount_doubled).collect()
        assert result[0]['amount_doubled'] == 300.0
