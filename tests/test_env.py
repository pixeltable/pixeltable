"""
Simplified end-to-end test for Env.clear() with database switching.
Compatible with pytest and PyCharm debugger.
"""

import os
import tempfile
import shutil
import pytest

import pixeltable as pxt
from pixeltable.env import Env
from pixeltable.catalog import Catalog


@pytest.fixture(scope="function")
def test_setup():
    """Create test environment directories."""
    test_dir = tempfile.mkdtemp(prefix='pxt_test_env')
    env1_home = os.path.join(test_dir, 'env1')
    env2_home = os.path.join(test_dir, 'env2')

    os.makedirs(env1_home)
    os.makedirs(env2_home)

    yield {
        'env1_home': env1_home,
        'env2_home': env2_home,
        'test_dir': test_dir
    }

    # Cleanup
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


class TestEnvReset:
    """Simplified tests for Env.Simple() functionality."""

    def test_basic(self, test_setup):
        """Test basic env clear functionality."""
        # Set environment 1
        os.environ['PIXELTABLE_HOME'] = test_setup['env1_home']
        os.environ['PIXELTABLE_DB'] = 'test_db1'

        # Initialize environment
        pxt.init()
        env1 = Env.get()
        assert env1 is not None
        assert env1._db_name == 'test_db1'

        # Create a simple table
        t = pxt.create_table('test_table', {'col1': pxt.String})
        t.insert([{'col1': 'test_data'}])
        assert t.count() == 1

        Catalog.clear()
        Env.clear()
        assert Env._instance is None

        # Verify we can create a new instance
        env2 = Env.get()
        assert env2 is not None
        assert env2 != env1

    def test_switch_environments(self, test_setup):
        """Test switching between two environments."""
        # Environment 1
        os.environ['PIXELTABLE_HOME'] = test_setup['env1_home']
        os.environ['PIXELTABLE_DB'] = 'db1'

        pxt.init()
        t1 = pxt.create_table('table1', {'name': pxt.String})
        t1.insert([{'name': 'env1_data'}])

        Catalog.clear()
        Env.clear()

        # Switch to Environment 2
        os.environ['PIXELTABLE_HOME'] = test_setup['env2_home']
        os.environ['PIXELTABLE_DB'] = 'db2'

        pxt.init()
        env2 = Env.get()
        assert env2._db_name == 'db2'

        # Create different table in env2
        t2 = pxt.create_table('table2', {'value': pxt.Int})
        t2.insert([{'value': 42}])

        # Verify table1 doesn't exist in env2
        with pytest.raises(Exception):
            pxt.get_table('table1')

        # Switch back to Environment 1
        Catalog.clear()
        Env.clear()
        os.environ['PIXELTABLE_HOME'] = test_setup['env1_home']
        os.environ['PIXELTABLE_DB'] = 'db1'

        pxt.init()
        env1_again = Env.get()
        assert env1_again._db_name == 'db1'

        # Verify table1 still exists in env1
        t1_again = pxt.get_table('table1')
        assert t1_again.count() == 1

        # Verify table2 doesn't exist in env1
        with pytest.raises(Exception):
            pxt.get_table('table2')

    def test_metadata_persistence(self, test_setup):
        """Test that metadata persists across environment switches."""
        # Environment 1 setup
        os.environ['PIXELTABLE_HOME'] = test_setup['env1_home']
        os.environ['PIXELTABLE_DB'] = 'metadata_db'

        pxt.init()

        # Create directory structure
        pxt.create_dir('analytics')
        pxt.create_dir('analytics.reports')

        # Create tables with different features
        t1 = pxt.create_table('users', {
            'user_id': pxt.Int,
            'username': pxt.String,
            'active': pxt.Bool
        })

        t2 = pxt.create_table('analytics.reports.sales', {
            'sale_id': pxt.Int,
            'amount': pxt.Float,
        })


        # Add computed column
        t2.add_computed_column(amount_doubled=t2.amount * 2)

        # Create view
        v1 = pxt.create_view('analytics.high_sales', t2.where(t2.amount > 100.0))

        # Insert data
        t1.insert([
            {'user_id': 1, 'username': 'alice', 'active': True},
            {'user_id': 2, 'username': 'bob', 'active': False}
        ])

        t2.insert([
            {'sale_id': 1, 'amount': 150.0},
            {'sale_id': 2, 'amount': 50.0}
        ])


        # Record metadata
        dirs_before = set(pxt.list_dirs())
        tables_before = set(pxt.list_tables())
        user_count = t1.count()
        high_sales_count = v1.count()

        Catalog.clear()
        Env.clear()

        # Reinitialize same environment
        pxt.init()

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

    def test_cleanup_on_clear(self, test_setup):
        """Test that resources are properly cleaned up on clear."""
        os.environ['PIXELTABLE_HOME'] = test_setup['env1_home']
        os.environ['PIXELTABLE_DB'] = 'cleanup_test'

        env = Env.get()

        # Verify resources are initialized
        assert env._sa_engine is not None
        assert env._httpd is not None
        assert env._initialized is True

        # Create some data
        t = pxt.create_table('test', {'data': pxt.String})
        t.insert([{'data': 'test'}])

        # Get references to check cleanup
        db_url = env._db_url
        media_dir = env._media_dir

        Env.clear()

        # Verify singleton is cleared
        assert Env._instance is None

        # Verify directories still exist (they should not be deleted)
        assert media_dir.exists()

    def test_incremental_metadata_with_switching(self, test_setup):
        """Test incremental metadata creation while switching between databases."""

        # Helper to switch environment and validate previous state
        def switch_to_env(env_home: str, db_name: str, expected_state: dict = None):
            """Switch to environment and validate its previous state if provided."""
            Catalog.clear()
            Env.clear()
            os.environ['PIXELTABLE_HOME'] = env_home
            os.environ['PIXELTABLE_DB'] = db_name
            pxt.init()

            # Validate previous state after switching
            if expected_state:
                validate_metadata(expected_state)

            return Env.get()

        # Helper to validate metadata state
        def validate_metadata(expected_state: dict):
            """Validate current metadata matches expected state."""
            actual_dirs = set(pxt.list_dirs())
            actual_tables = set(pxt.list_tables())

            expected_dirs = set(expected_state['dirs'])
            expected_tables = set(expected_state['tables'])

            assert actual_dirs == expected_dirs, f"Dirs mismatch: expected {expected_dirs}, got {actual_dirs}"
            assert actual_tables == expected_tables, f"Tables mismatch: expected {expected_tables}, got {actual_tables}"

            # Validate each table's data
            for table_name, expected_count in expected_state['counts'].items():
                if table_name in actual_tables:
                    t = pxt.get_table(table_name)
                    actual_count = t.count()
                    assert actual_count == expected_count, f"{table_name}: expected {expected_count} rows, got {actual_count}"

            # Validate specific data integrity checks if provided
            if 'data_checks' in expected_state:
                for check_name, check_fn in expected_state['data_checks'].items():
                    check_fn()

        # Track metadata state for each environment
        env1_state = {
            'dirs': set(),
            'tables': set(),
            'counts': {},
            'data_checks': {}
        }

        env2_state = {
            'dirs': set(),
            'tables': set(),
            'counts': {},
            'data_checks': {}
        }

        # Step 1: Initialize env1 and create first directory
        env1 = switch_to_env(test_setup['env1_home'], 'incremental_db1')
        assert env1._db_name == 'incremental_db1'

        pxt.create_dir('data')
        env1_state['dirs'].add('data')
        validate_metadata(env1_state)

        # Step 2: Switch to env2 and create different directory
        env2 = switch_to_env(test_setup['env2_home'], 'incremental_db2', env2_state)
        assert env2._db_name == 'incremental_db2'

        pxt.create_dir('models')
        env2_state['dirs'].add('models')
        validate_metadata(env2_state)

        # Step 3: Back to env1 - validate previous state, then create table
        switch_to_env(test_setup['env1_home'], 'incremental_db1', env1_state)

        t1 = pxt.create_table('data.events', {
            'event_id': pxt.Int,
            'event_type': pxt.String,
            'timestamp': pxt.Timestamp
        })
        env1_state['tables'].add('data.events')
        env1_state['counts']['data.events'] = 0
        validate_metadata(env1_state)

        # Step 4: To env2 - validate previous state, then create table
        switch_to_env(test_setup['env2_home'], 'incremental_db2', env2_state)

        t2 = pxt.create_table('models.predictions', {
            'pred_id': pxt.Int,
            'model_name': pxt.String,
            'score': pxt.Float
        })
        env2_state['tables'].add('models.predictions')
        env2_state['counts']['models.predictions'] = 0
        validate_metadata(env2_state)

        # Step 5: Back to env1 - validate, insert data and create view
        switch_to_env(test_setup['env1_home'], 'incremental_db1', env1_state)

        # Insert data
        t1 = pxt.get_table('data.events')
        t1.insert([
            {'event_id': 1, 'event_type': 'login', 'timestamp': '2024-01-01 10:00:00'},
            {'event_id': 2, 'event_type': 'purchase', 'timestamp': '2024-01-01 11:00:00'},
            {'event_id': 3, 'event_type': 'logout', 'timestamp': '2024-01-01 12:00:00'}
        ])
        env1_state['counts']['data.events'] = 3

        # Create view
        v1 = pxt.create_view('data.login_events', t1.where(t1.event_type == 'login'))
        env1_state['tables'].add('data.login_events')
        env1_state['counts']['data.login_events'] = 1

        # Add data integrity check
        def check_env1_events():
            t = pxt.get_table('data.events')
            events = t.select(t.event_type).order_by(t.event_id).collect()
            assert [e['event_type'] for e in events] == ['login', 'purchase', 'logout']

        env1_state['data_checks']['event_types'] = check_env1_events
        validate_metadata(env1_state)

        # Step 6: To env2 - validate, insert data and create computed column
        switch_to_env(test_setup['env2_home'], 'incremental_db2', env2_state)

        t2 = pxt.get_table('models.predictions')
        t2.insert([
            {'pred_id': 1, 'model_name': 'model_v1', 'score': 0.85},
            {'pred_id': 2, 'model_name': 'model_v2', 'score': 0.92}
        ])
        env2_state['counts']['models.predictions'] = 2

        # Add computed column
        t2.add_computed_column(is_high_score=t2.score > 0.9)

        # Add data check for computed column
        def check_env2_scores():
            t = pxt.get_table('models.predictions')
            high_scores = t.where(t.is_high_score == True).count()
            assert high_scores == 1, f"Expected 1 high score, got {high_scores}"

        env2_state['data_checks']['high_scores'] = check_env2_scores
        validate_metadata(env2_state)

        # Step 7: Back to env1 - validate all previous data, create nested directory and table
        switch_to_env(test_setup['env1_home'], 'incremental_db1', env1_state)

        pxt.create_dir('data.processed')
        env1_state['dirs'].add('data.processed')

        t3 = pxt.create_table('data.processed.summary', {
            'summary_id': pxt.Int,
            'event_count': pxt.Int,
            'date': pxt.String
        })
        env1_state['tables'].add('data.processed.summary')
        env1_state['counts']['data.processed.summary'] = 0
        validate_metadata(env1_state)

        # Step 8: To env2 - validate, create view and snapshot
        switch_to_env(test_setup['env2_home'], 'incremental_db2', env2_state)

        t2 = pxt.get_table('models.predictions')
        v2 = pxt.create_view('models.high_scores', t2.where(t2.is_high_score == True))
        env2_state['tables'].add('models.high_scores')
        env2_state['counts']['models.high_scores'] = 1

        s1 = pxt.create_snapshot('models.predictions_snapshot', t2)
        env2_state['tables'].add('models.predictions_snapshot')
        env2_state['counts']['models.predictions_snapshot'] = 2
        validate_metadata(env2_state)

        # Step 9: Back to env1 - validate, then delete a view
        switch_to_env(test_setup['env1_home'], 'incremental_db1', env1_state)

        # Verify view exists before deletion
        assert 'data.login_events' in pxt.list_tables()
        assert pxt.get_table('data.login_events').count() == 1

        pxt.drop_table('data.login_events')
        env1_state['tables'].remove('data.login_events')
        del env1_state['counts']['data.login_events']
        validate_metadata(env1_state)

        # Step 10: To env2 - validate, add more data and create another table
        switch_to_env(test_setup['env2_home'], 'incremental_db2', env2_state)

        t2 = pxt.get_table('models.predictions')
        t2.insert([
            {'pred_id': 3, 'model_name': 'model_v3', 'score': 0.78},
            {'pred_id': 4, 'model_name': 'model_v4', 'score': 0.95}
        ])
        env2_state['counts']['models.predictions'] = 4
        env2_state['counts']['models.high_scores'] = 2  # Now 2 high scores

        # Snapshot count remains 2 (frozen at creation time)

        # Update data check for new high score count
        def check_env2_scores_updated():
            t = pxt.get_table('models.predictions')
            high_scores = t.where(t.is_high_score == True).count()
            assert high_scores == 2, f"Expected 2 high scores, got {high_scores}"

        env2_state['data_checks']['high_scores'] = check_env2_scores_updated

        # Create metrics table
        t4 = pxt.create_table('models.metrics', {
            'metric_id': pxt.Int,
            'metric_name': pxt.String,
            'value': pxt.Float
        })
        env2_state['tables'].add('models.metrics')
        env2_state['counts']['models.metrics'] = 0
        validate_metadata(env2_state)

        # Step 11: Back to env1 - validate, create complex view with joins
        switch_to_env(test_setup['env1_home'], 'incremental_db1', env1_state)

        t1 = pxt.get_table('data.events')
        t3 = pxt.get_table('data.processed.summary')

        # Insert summary data
        t3.insert([
            {'summary_id': 1, 'event_count': 3, 'date': '2024-01-01'}
        ])
        env1_state['counts']['data.processed.summary'] = 1

        # Create view with computed column
        v3 = pxt.create_view('data.event_analysis', t1)
        v3.add_computed_column(is_login=v3.event_type == 'login')
        env1_state['tables'].add('data.event_analysis')
        env1_state['counts']['data.event_analysis'] = 3

        # Add check for computed column
        def check_login_events():
            v = pxt.get_table('data.event_analysis')
            login_count = v.where(v.is_login == True).count()
            assert login_count == 1, f"Expected 1 login event, got {login_count}"
            # Also check the computed column values
            results = v.select(v.event_type, v.is_login).order_by(v.event_id).collect()
            expected = [
                {'event_type': 'login', 'is_login': True},
                {'event_type': 'purchase', 'is_login': False},
                {'event_type': 'logout', 'is_login': False}
            ]
            for i, (result, exp) in enumerate(zip(results, expected)):
                assert result['event_type'] == exp['event_type'], f"Row {i}: event_type mismatch"
                assert result['is_login'] == exp['is_login'], f"Row {i}: is_login mismatch"

        # Update login event check to handle new rows
        def check_login_events_updated():
            v = pxt.get_table('data.event_analysis')
            # Check only original login event
            login_count = v.where((v.is_login == True) & (v.event_id <= 3)).count()
            assert login_count == 1, f"Expected 1 original login event, got {login_count}"
            # Check total count matches
            assert v.count() == env1_state['counts']['data.event_analysis']

        env1_state['data_checks']['login_events'] = check_login_events_updated
        validate_metadata(env1_state)

        # Step 12: Delete table with dependencies in env2
        switch_to_env(test_setup['env2_home'], 'incremental_db2', env2_state)

        # This should fail due to dependencies
        with pytest.raises(Exception):
            pxt.drop_table('models.predictions')

        # Verify nothing was deleted
        validate_metadata(env2_state)

        # Force drop to remove table and all dependents
        pxt.drop_table('models.predictions', force=True)

        # Update state - predictions and its dependents are gone
        for table in ['models.predictions', 'models.high_scores', 'models.predictions_snapshot']:
            env2_state['tables'].remove(table)
            del env2_state['counts'][table]

        # Remove data check that depends on deleted table
        del env2_state['data_checks']['high_scores']

        validate_metadata(env2_state)

        # Step 13: Multiple rapid switches to verify stability
        for i in range(3):
            # Switch to env1
            switch_to_env(test_setup['env1_home'], 'incremental_db1', env1_state)

            # Add a row to verify we can still modify
            t1 = pxt.get_table('data.events')
            t1.insert([{
                'event_id': 100 + i,
                'event_type': f'test_{i}',
                'timestamp': f'2024-01-02 {10 + i}:00:00'
            }])
            env1_state['counts']['data.events'] += 1
            env1_state['counts']['data.event_analysis'] += 1

            # Update the data check to handle current state
            expected_total = 3 + i + 1  # 3 original + events added so far

            def check_env1_events_current(expected=expected_total):
                t = pxt.get_table('data.events')
                # Check original events are still there in correct order
                original_events = t.where(t.event_id <= 3).select(t.event_type).order_by(t.event_id).collect()
                assert [e['event_type'] for e in original_events] == ['login', 'purchase', 'logout']
                # Check total count
                assert t.count() == expected, f"Expected {expected} events, got {t.count()}"

            env1_state['data_checks']['event_types'] = check_env1_events_current

            # Validate current state
            validate_metadata(env1_state)

            # Switch to env2
            switch_to_env(test_setup['env2_home'], 'incremental_db2', env2_state)

            # Verify only metrics table exists
            assert set(pxt.list_tables()) == {'models.metrics'}

        # Step 14: Final comprehensive validation
        switch_to_env(test_setup['env1_home'], 'incremental_db1', env1_state)

        # Update event check for new data
        def check_env1_all_events():
            t = pxt.get_table('data.events')
            assert t.count() == 6  # 3 original + 3 from rapid switches
            # Check original events are still there
            original_events = t.where(t.event_id <= 3).select(t.event_type).order_by(t.event_id).collect()
            assert [e['event_type'] for e in original_events] == ['login', 'purchase', 'logout']

        env1_state['data_checks']['event_types'] = check_env1_all_events
        validate_metadata(env1_state)

        switch_to_env(test_setup['env2_home'], 'incremental_db2', env2_state)

        # Create same-named table that was deleted to verify isolation
        t5 = pxt.create_table('models.predictions', {
            'id': pxt.Int,
            'value': pxt.String
        })
        env2_state['tables'].add('models.predictions')
        env2_state['counts']['models.predictions'] = 0

        # One more switch to verify both environments retain their final state
        switch_to_env(test_setup['env1_home'], 'incremental_db1', env1_state)
        switch_to_env(test_setup['env2_home'], 'incremental_db2', env2_state)