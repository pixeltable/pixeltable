"""
Tests for the @remote decorator functionality.
"""

from typing import Any, Callable

import pytest

from pixeltable.env import Env
from pixeltable.remote_decorator import is_remote_path
from pixeltable.share.remote import RemoteClient


class MockRemoteClient(RemoteClient):
    """Test client that counts calls instead of making real HTTP requests."""

    def __init__(self, base_url: str = 'http://localhost:8000', timeout: int = 30):
        super().__init__(base_url, timeout)
        self.call_counts: dict[str, int] = {}
        self.last_call_args: dict[str, dict[str, Any]] = {}

    def make_remote_call(self, func: Callable, **kwargs: Any) -> Any:
        func_name = func.__name__
        self.call_counts[func_name] = self.call_counts.get(func_name, 0) + 1
        self.last_call_args[func_name] = kwargs
        return f'mock_result_for_{func_name}'

    def get_call_count(self, func_name: str) -> int:
        return self.call_counts.get(func_name, 0)

    def get_last_call_args(self, func_name: str) -> dict[str, Any]:
        return self.last_call_args.get(func_name, {})


class TestRemoteDecorator:
    """Test cases for @remote decorator validation and routing."""

    def test_is_remote_path(self) -> None:
        """Test the is_remote_path utility function."""
        # Test remote paths
        assert is_remote_path('pxt://local/A/B/C')
        assert is_remote_path('pxt://server/D/E/F')
        assert is_remote_path('pxt://remote/X/Y/Z')

        # Test local paths
        assert not is_remote_path('a.b.c')
        assert not is_remote_path('local/path')
        assert not is_remote_path('simple_path')
        assert not is_remote_path('')

        # Test non-string inputs (these should raise TypeError or return False)
        # Note: is_remote_path expects string input, so we test the behavior
        try:
            result = is_remote_path(123)  # type: ignore[arg-type]
            assert result is False
        except TypeError:
            pass  # Expected behavior for non-string input

    def test_create_table(self, reset_db: None) -> None:
        """Test create_table with local and remote paths."""
        test_client = MockRemoteClient()
        env = Env.get()
        original_client = env._remote_client
        env._remote_client = test_client

        try:
            import pixeltable as pxt

            # Test local path
            result = pxt.create_table('test_table', {'col1': pxt.Int})
            assert test_client.get_call_count('create_table') == 0
            assert str(result).startswith("table 'test_table'")

            # Test remote path
            result = pxt.create_table('pxt://local/A/B/C', {'col1': pxt.Int})
            assert test_client.get_call_count('create_table') == 1
            assert result == 'mock_result_for_create_table'

            args = test_client.get_last_call_args('create_table')
            assert args['path'] == 'pxt://local/A/B/C'
            assert args['schema'] == {'col1': pxt.Int}
        finally:
            env._remote_client = original_client

    def test_create_view(self, reset_db: None) -> None:
        """Test create_view with local and remote paths."""
        test_client = MockRemoteClient()
        env = Env.get()
        original_client = env._remote_client
        env._remote_client = test_client

        try:
            import pixeltable as pxt

            # Create a table first
            table = pxt.create_table('test_table', {'col1': pxt.Int})

            # Test local path
            result = pxt.create_view('test_view', table)
            assert test_client.get_call_count('create_view') == 0
            assert str(result).startswith("view 'test_view'")

            # Test remote path
            result = pxt.create_view('pxt://local/A/B/C', table)
            assert test_client.get_call_count('create_view') == 1
            assert result == 'mock_result_for_create_view'

            args = test_client.get_last_call_args('create_view')
            assert args['path'] == 'pxt://local/A/B/C'
        finally:
            env._remote_client = original_client

    def test_create_snapshot(self, reset_db: None) -> None:
        """Test create_snapshot with local and remote paths."""
        test_client = MockRemoteClient()
        env = Env.get()
        original_client = env._remote_client
        env._remote_client = test_client

        try:
            import pixeltable as pxt

            # Create a table first
            table = pxt.create_table('test_table', {'col1': pxt.Int})

            # Test local path
            result = pxt.create_snapshot('test_snapshot', table)
            assert test_client.get_call_count('create_snapshot') == 0
            assert str(result).startswith("snapshot 'test_snapshot'")

            # Test remote path
            result = pxt.create_snapshot('pxt://local/A/B/C', table)
            assert test_client.get_call_count('create_snapshot') == 1
            assert result == 'mock_result_for_create_snapshot'

            args = test_client.get_last_call_args('create_snapshot')
            assert args['path_str'] == 'pxt://local/A/B/C'
        finally:
            env._remote_client = original_client

    def test_get_table(self, reset_db: None) -> None:
        """Test get_table with local and remote paths."""
        test_client = MockRemoteClient()
        env = Env.get()
        original_client = env._remote_client
        env._remote_client = test_client

        try:
            import pixeltable as pxt

            # Create a table first
            pxt.create_table('test_table', {'col1': pxt.Int})

            # Test local path
            result = pxt.get_table('test_table')
            assert test_client.get_call_count('get_table') == 0
            assert str(result).startswith("table 'test_table'")

            # Test remote path
            result = pxt.get_table('pxt://local/A/B/C')
            assert test_client.get_call_count('get_table') == 1
            assert result == 'mock_result_for_get_table'

            args = test_client.get_last_call_args('get_table')
            assert args['path'] == 'pxt://local/A/B/C'
        finally:
            env._remote_client = original_client

    def test_move(self, reset_db: None) -> None:
        """Test move with local, remote, and mixed paths."""
        test_client = MockRemoteClient()
        env = Env.get()
        original_client = env._remote_client
        env._remote_client = test_client

        try:
            import pixeltable as pxt

            # Create a table first
            pxt.create_table('test_table', {'col1': pxt.Int})

            # Test local paths
            pxt.move('test_table', 'test_table_moved')
            assert test_client.get_call_count('move') == 0
            moved_table = pxt.get_table('test_table_moved')
            assert str(moved_table).startswith("table 'test_table_moved'")

            # Test remote paths
            result = pxt.move('pxt://local/A/B/C', 'pxt://local/D/E/F')
            assert test_client.get_call_count('move') == 1
            assert result == 'mock_result_for_move'

            args = test_client.get_last_call_args('move')
            assert args['path'] == 'pxt://local/A/B/C'
            assert args['new_path'] == 'pxt://local/D/E/F'

            # Test mixed paths - should raise error
            with pytest.raises(ValueError, match='Mixed remote and local paths not allowed'):
                pxt.move('local_path', 'pxt://remote/path')

            with pytest.raises(ValueError, match='Mixed remote and local paths not allowed'):
                pxt.move('pxt://remote/path', 'local_path')
        finally:
            env._remote_client = original_client

    def test_drop_table(self, reset_db: None) -> None:
        """Test drop_table with local and remote paths."""
        test_client = MockRemoteClient()
        env = Env.get()
        original_client = env._remote_client
        env._remote_client = test_client

        try:
            import pixeltable as pxt

            # Create a table first
            pxt.create_table('test_table', {'col1': pxt.Int})

            # Test local path
            pxt.drop_table('test_table')
            assert test_client.get_call_count('drop_table') == 0
            # Verify table was dropped
            with pytest.raises((ValueError, pxt.exceptions.Error)):
                pxt.get_table('test_table')

            # Test remote path
            result = pxt.drop_table('pxt://local/A/B/C')
            assert test_client.get_call_count('drop_table') == 1
            assert result == 'mock_result_for_drop_table'

            args = test_client.get_last_call_args('drop_table')
            # The decorator converts string paths to RemoteTable objects
            from pixeltable.share.remote import RemoteTable

            assert isinstance(args['table'], RemoteTable)
            assert args['table'].path == 'pxt://local/A/B/C'
        finally:
            env._remote_client = original_client

    def test_get_dir_contents(self, reset_db: None) -> None:
        """Test get_dir_contents with local and remote paths."""
        test_client = MockRemoteClient()
        env = Env.get()
        original_client = env._remote_client
        env._remote_client = test_client

        try:
            import pixeltable as pxt

            # Test local path
            result = pxt.get_dir_contents()
            assert test_client.get_call_count('get_dir_contents') == 0
            assert hasattr(result, 'tables')
            assert hasattr(result, 'dirs')

            # Test remote path
            result = pxt.get_dir_contents('pxt://local/A/B/C')
            assert test_client.get_call_count('get_dir_contents') == 1
            assert result == 'mock_result_for_get_dir_contents'

            args = test_client.get_last_call_args('get_dir_contents')
            assert args['dir_path'] == 'pxt://local/A/B/C'
        finally:
            env._remote_client = original_client

    def test_list_tables(self, reset_db: None) -> None:
        """Test list_tables with local and remote paths."""
        test_client = MockRemoteClient()
        env = Env.get()
        original_client = env._remote_client
        env._remote_client = test_client

        try:
            import pixeltable as pxt

            # Create a table first
            pxt.create_table('test_table', {'col1': pxt.Int})

            # Test local path
            result = pxt.list_tables()
            assert test_client.get_call_count('list_tables') == 0
            assert isinstance(result, list)
            assert 'test_table' in result

            # Test remote path
            result = pxt.list_tables('pxt://local/A/B/C')
            assert test_client.get_call_count('list_tables') == 1
            assert result == 'mock_result_for_list_tables'

            args = test_client.get_last_call_args('list_tables')
            assert args['dir_path'] == 'pxt://local/A/B/C'
        finally:
            env._remote_client = original_client

    def test_create_dir(self, reset_db: None) -> None:
        """Test create_dir with local and remote paths."""
        test_client = MockRemoteClient()
        env = Env.get()
        original_client = env._remote_client
        env._remote_client = test_client

        try:
            import pixeltable as pxt

            # Test local path
            result = pxt.create_dir('test_dir')
            assert test_client.get_call_count('create_dir') == 0
            assert result is not None  # Dir objects don't have nice string representation

            # Test remote path
            result = pxt.create_dir('pxt://local/A/B/C')
            assert test_client.get_call_count('create_dir') == 1
            assert result == 'mock_result_for_create_dir'

            args = test_client.get_last_call_args('create_dir')
            assert args['path'] == 'pxt://local/A/B/C'
        finally:
            env._remote_client = original_client

    def test_drop_dir(self, reset_db: None) -> None:
        """Test drop_dir with local and remote paths."""
        test_client = MockRemoteClient()
        env = Env.get()
        original_client = env._remote_client
        env._remote_client = test_client

        try:
            import pixeltable as pxt

            # Create a directory first
            pxt.create_dir('test_dir')

            # Test local path
            pxt.drop_dir('test_dir')
            assert test_client.get_call_count('drop_dir') == 0

            # Test remote path
            result = pxt.drop_dir('pxt://local/A/B/C')
            assert test_client.get_call_count('drop_dir') == 1
            assert result == 'mock_result_for_drop_dir'

            args = test_client.get_last_call_args('drop_dir')
            assert args['path'] == 'pxt://local/A/B/C'
        finally:
            env._remote_client = original_client

    def test_ls(self, reset_db: None) -> None:
        """Test ls with local and remote paths."""
        test_client = MockRemoteClient()
        env = Env.get()
        original_client = env._remote_client
        env._remote_client = test_client

        try:
            import pixeltable as pxt

            # Test local path
            result = pxt.ls()
            assert test_client.get_call_count('ls') == 0
            assert hasattr(result, 'columns')

            # Test remote path
            result = pxt.ls('pxt://local/A/B/C')
            assert test_client.get_call_count('ls') == 1
            assert result == 'mock_result_for_ls'

            args = test_client.get_last_call_args('ls')
            assert args['path'] == 'pxt://local/A/B/C'
        finally:
            env._remote_client = original_client

    def test_list_dirs(self, reset_db: None) -> None:
        """Test list_dirs with local and remote paths."""
        test_client = MockRemoteClient()
        env = Env.get()
        original_client = env._remote_client
        env._remote_client = test_client

        try:
            import pixeltable as pxt

            # Test local path
            result = pxt.list_dirs()
            assert test_client.get_call_count('list_dirs') == 0
            assert isinstance(result, list)

            # Test remote path
            result = pxt.list_dirs('pxt://local/A/B/C')
            assert test_client.get_call_count('list_dirs') == 1
            assert result == 'mock_result_for_list_dirs'

            args = test_client.get_last_call_args('list_dirs')
            assert args['path'] == 'pxt://local/A/B/C'
        finally:
            env._remote_client = original_client
