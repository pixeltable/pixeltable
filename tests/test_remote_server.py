"""
Remote server integration tests.

These tests start the remote server locally (as a local server) and test the complete
remote functionality workflow including path conversion, remote calls, and
result conversion.
"""

import threading
import time
from contextlib import suppress
from typing import Generator

import pytest
import requests
import uvicorn

import pixeltable as pxt
from pixeltable.share.remote import RemoteClient, RemoteDir, RemoteTable


class TestRemoteServer:
    """Remote server integration tests - runs remote server locally for testing."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_server(self) -> Generator[None, None, None]:
        """Setup local server for integration tests - class level fixture."""
        # Use a fixed port for testing
        self.port = 8001
        self.server_url = f'http://localhost:{self.port}'

        # Start server programmatically
        self.server_thread, self.server = self._start_server()

        # Wait for server to be ready
        self._wait_for_server()

        # Create client with the correct URL after server is ready
        self.client = RemoteClient(base_url=self.server_url)

        # Set the client in the environment so the @remote decorator uses it
        from pixeltable.env import Env

        env = Env.get()
        env._remote_client = self.client

        yield

        # Cleanup: Shutdown server
        self._shutdown_server()

    def _start_server(self) -> tuple[threading.Thread, uvicorn.Server]:
        """Start the remote server locally in a separate thread."""
        from pixeltable.share.remote import app

        # Create server instance for proper shutdown
        config = uvicorn.Config(app, host='127.0.0.1', port=self.port, log_level='error')
        server = uvicorn.Server(config)

        # Start server in a separate thread
        server_thread = threading.Thread(target=server.run, daemon=False)
        server_thread.start()

        return server_thread, server

    def _wait_for_server(self, timeout: int = 10) -> None:
        """Wait for the local server to be ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            with suppress(requests.RequestException):
                response = requests.get(f'{self.server_url}/health', timeout=1)
                if response.status_code == 200:
                    return
            time.sleep(0.1)
        pytest.fail('Local server failed to start within timeout')

    def _shutdown_server(self) -> None:
        """Shutdown the local server gracefully."""
        # Reset the environment's remote client
        from pixeltable.env import Env

        env = Env.get()
        env._remote_client = None

        if hasattr(self, 'server') and self.server:
            with suppress(Exception):
                self.server.should_exit = True
                # Give the server a moment to shutdown
                time.sleep(0.5)

        if hasattr(self, 'server_thread') and self.server_thread:
            with suppress(Exception):
                # Wait for thread to finish (with timeout)
                self.server_thread.join(timeout=2)

    def test_complete_remote_workflow(self, reset_db: None) -> None:
        """Test complete remote workflow: create all objects and test operations."""

        # === SETUP: Create all test objects ===

        # Create base directory
        base_dir = pxt.create_dir('pxt://local/INTEGRATION_TEST')
        assert isinstance(base_dir, RemoteDir)
        assert base_dir.path == 'pxt://local/integration_test'

        # Verify directory was actually created on server
        # We can check by listing directories and seeing if integration_test appears
        local_dirs = pxt.list_dirs()
        assert 'integration_test' in local_dirs

        # Create subdirectories
        subdir1 = pxt.create_dir('pxt://local/INTEGRATION_TEST/SUBDIR1')
        assert isinstance(subdir1, RemoteDir)
        assert subdir1.path == 'pxt://local/integration_test/subdir1'

        # Verify subdirectory was actually created on server
        local_subdirs = pxt.list_dirs('integration_test')
        assert 'integration_test.subdir1' in local_subdirs

        subdir2 = pxt.create_dir('pxt://local/INTEGRATION_TEST/SUBDIR2')
        assert isinstance(subdir2, RemoteDir)
        assert subdir2.path == 'pxt://local/integration_test/subdir2'

        # Verify subdirectory was actually created on server
        local_subdirs_after = pxt.list_dirs('integration_test')
        assert 'integration_test.subdir2' in local_subdirs_after

        # Create tables with different schemas
        table1 = pxt.create_table(
            'pxt://local/INTEGRATION_TEST/TABLE1', schema={'id': pxt.Int, 'name': pxt.String, 'value': pxt.Float}
        )
        assert isinstance(table1, RemoteTable)
        assert table1.path == 'pxt://local/integration_test/table1'

        # Verify table was actually created on server
        local_table1 = pxt.get_table('integration_test.table1')
        assert local_table1 is not None
        assert local_table1._name == 'table1'

        table2 = pxt.create_table('pxt://local/INTEGRATION_TEST/TABLE2', {'id': pxt.Int, 'data': pxt.String})
        assert isinstance(table2, RemoteTable)
        assert table2.path == 'pxt://local/integration_test/table2'

        # Verify table was actually created on server
        local_table2 = pxt.get_table('integration_test.table2')
        assert local_table2 is not None
        assert local_table2._name == 'table2'

        # === TEST: Parameterized types in table creation ===

        # Create table with parameterized types
        json_schema = {'type': 'object', 'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}}}
        table3 = pxt.create_table(
            'pxt://local/INTEGRATION_TEST/TABLE3',
            {
                'id': pxt.Int,
                'json_data': pxt.Json[json_schema],  # type: ignore[misc]
                'array_data': pxt.Array[(3, 4), pxt.Float],  # type: ignore[misc]
                'image_data': pxt.Image,
                'parameterized_image': pxt.Image[(300, 300), 'RGB'],  # type: ignore[misc]
            },
        )
        assert isinstance(table3, RemoteTable)
        assert table3.path == 'pxt://local/integration_test/table3'

        # Verify table with parameterized types was actually created on server
        local_table3 = pxt.get_table('integration_test.table3')
        assert local_table3 is not None
        assert local_table3._name == 'table3'

        # Create a view
        view1 = pxt.create_view(
            'pxt://local/INTEGRATION_TEST/VIEW1',
            'pxt://local/INTEGRATION_TEST/TABLE1',  # type: ignore[arg-type]
        )
        assert isinstance(view1, RemoteTable)
        assert view1.path == 'pxt://local/integration_test/view1'

        # Verify view was actually created on server
        local_view1 = pxt.get_table('integration_test.view1')
        assert local_view1 is not None
        assert local_view1._name == 'view1'

        # === TEST: Parameterized types in view creation with additional schema ===

        # Create view with additional columns containing parameterized types (must be nullable for views)
        # Create ColumnType objects directly instead of using __metadata__
        from pixeltable.type_system import ArrayType, ImageType, IntType, JsonType

        json_type = JsonType(
            json_schema={'type': 'object', 'properties': {'status': {'type': 'string'}}}, nullable=True
        )
        array_type = ArrayType(shape=(2, 2), dtype=IntType(), nullable=True)
        image_type = ImageType(width=100, height=100, mode='L', nullable=True)

        view2 = pxt.create_view(
            'pxt://local/INTEGRATION_TEST/VIEW2',
            'pxt://local/INTEGRATION_TEST/TABLE1',  # type: ignore[arg-type]
            additional_columns={
                'extra_json': json_type,
                'extra_array': array_type,
                'extra_image': image_type,  # Grayscale image
            },
        )
        assert isinstance(view2, RemoteTable)
        assert view2.path == 'pxt://local/integration_test/view2'

        # Verify view with additional schema was actually created on server
        local_view2 = pxt.get_table('integration_test.view2')
        assert local_view2 is not None
        assert local_view2._name == 'view2'

        # Create a snapshot
        snapshot1 = pxt.create_snapshot(
            'pxt://local/INTEGRATION_TEST/SNAPSHOT1',
            'pxt://local/INTEGRATION_TEST/TABLE1',  # type: ignore[arg-type]
        )
        assert isinstance(snapshot1, RemoteTable)
        assert snapshot1.path == 'pxt://local/integration_test/snapshot1'

        # Verify snapshot was actually created on server
        local_snapshot1 = pxt.get_table('integration_test.snapshot1')
        assert local_snapshot1 is not None
        assert local_snapshot1._name == 'snapshot1'

        # === TEST: Parameterized types in snapshot creation with additional schema ===

        # Create snapshot with additional columns containing parameterized types (must be nullable for snapshots)
        # Create ColumnType objects directly instead of using __metadata__
        from pixeltable.type_system import ArrayType, FloatType, ImageType, JsonType

        json_type2 = JsonType(
            json_schema={'type': 'object', 'properties': {'version': {'type': 'string'}}}, nullable=True
        )
        array_type2 = ArrayType(shape=(5, 5), dtype=FloatType(), nullable=True)
        image_type2 = ImageType(width=50, height=50, mode='RGB', nullable=True)

        snapshot2 = pxt.create_snapshot(
            'pxt://local/INTEGRATION_TEST/SNAPSHOT2',
            'pxt://local/INTEGRATION_TEST/TABLE1',  # type: ignore[arg-type]
            additional_columns={'snapshot_metadata': json_type2, 'data_matrix': array_type2, 'thumbnail': image_type2},
        )
        assert isinstance(snapshot2, RemoteTable)
        assert snapshot2.path == 'pxt://local/integration_test/snapshot2'

        # Verify snapshot with additional schema was actually created on server
        local_snapshot2 = pxt.get_table('integration_test.snapshot2')
        assert local_snapshot2 is not None
        assert local_snapshot2._name == 'snapshot2'

        # === TEST: List operations ===

        # List directories
        dirs = pxt.list_dirs('pxt://local/INTEGRATION_TEST')
        assert isinstance(dirs, list)
        assert 'pxt://local/integration_test/subdir1' in dirs
        assert 'pxt://local/integration_test/subdir2' in dirs

        # List tables
        tables = pxt.list_tables('pxt://local/INTEGRATION_TEST')
        assert isinstance(tables, list)
        assert 'pxt://local/integration_test/table1' in tables
        assert 'pxt://local/integration_test/table2' in tables
        assert 'pxt://local/integration_test/view1' in tables
        assert 'pxt://local/integration_test/snapshot1' in tables

        # Test get_dir_contents
        contents = pxt.get_dir_contents('pxt://local/INTEGRATION_TEST')
        assert isinstance(contents, dict)
        assert 'dirs' in contents
        assert 'tables' in contents
        assert len(contents['dirs']) >= 2  # At least our two subdirs
        assert len(contents['tables']) >= 4  # Our tables, view, and snapshot

        # Test ls operation
        ls_result = pxt.ls('pxt://local/INTEGRATION_TEST')
        assert hasattr(ls_result, 'shape')  # Should return a pandas DataFrame

        # === TEST: Get operations ===

        # Get table
        retrieved_table = pxt.get_table('pxt://local/INTEGRATION_TEST/TABLE1')
        assert isinstance(retrieved_table, RemoteTable)
        assert retrieved_table.path == 'pxt://local/integration_test/table1'

        # === TEST: Move operation ===

        # Move a table to a new location
        move_result = pxt.move('pxt://local/INTEGRATION_TEST/TABLE1', 'pxt://local/INTEGRATION_TEST/MOVED_TABLE')
        assert move_result is None  # Move returns None on success

        # Verify the table was moved (remote check)
        moved_table = pxt.get_table('pxt://local/INTEGRATION_TEST/MOVED_TABLE')
        assert isinstance(moved_table, RemoteTable)
        assert moved_table.path == 'pxt://local/integration_test/moved_table'

        # Verify the table was actually moved on server (local check)
        local_moved_table = pxt.get_table('integration_test.moved_table')
        assert local_moved_table is not None
        assert local_moved_table._name == 'moved_table'

        # Verify original location no longer exists
        with pytest.raises((ValueError, pxt.exceptions.Error)):
            pxt.get_table('integration_test.table1')

        # === TEST: Drop operations ===

        # Drop a table
        drop_result = pxt.drop_table('pxt://local/INTEGRATION_TEST/TABLE2')
        assert drop_result is None  # Drop returns None on success

        # Verify the table was dropped (remote check)
        tables_after_drop = pxt.list_tables('pxt://local/INTEGRATION_TEST')
        assert 'pxt://local/integration_test/table2' not in tables_after_drop

        # Verify the table was actually dropped on server (local check)
        with pytest.raises((ValueError, pxt.exceptions.Error)):
            pxt.get_table('integration_test.table2')

        # Drop a directory
        drop_dir_result = pxt.drop_dir('pxt://local/INTEGRATION_TEST/SUBDIR1')
        assert drop_dir_result is None  # Drop returns None on success

        # Verify the directory was dropped (remote check)
        dirs_after_drop = pxt.list_dirs('pxt://local/INTEGRATION_TEST')
        assert 'pxt://local/integration_test/subdir1' not in dirs_after_drop

        # Verify the directory was actually dropped on server (local check)
        local_dirs_after_drop = pxt.list_dirs('integration_test')
        assert 'integration_test.subdir1' not in local_dirs_after_drop

        # === CLEANUP ===

        # Drop remaining resources
        remaining_tables = [
            'pxt://local/INTEGRATION_TEST/MOVED_TABLE',
            'pxt://local/INTEGRATION_TEST/TABLE3',  # Table with parameterized types
            'pxt://local/INTEGRATION_TEST/VIEW1',
            'pxt://local/INTEGRATION_TEST/VIEW2',  # View with additional schema
            'pxt://local/INTEGRATION_TEST/SNAPSHOT1',
            'pxt://local/INTEGRATION_TEST/SNAPSHOT2',  # Snapshot with additional schema
        ]

        for table_path in remaining_tables:
            with suppress(Exception):
                pxt.drop_table(table_path)

        # Drop remaining directories
        with suppress(Exception):
            pxt.drop_dir('pxt://local/INTEGRATION_TEST/SUBDIR2')

        # Drop base directory
        with suppress(Exception):
            pxt.drop_dir('pxt://local/INTEGRATION_TEST')


class TestRemotePathOperations:
    """Test remote path operations and conversions."""

    def test_remote_path_detection(self) -> None:
        """Test that remote paths are properly detected."""
        from pixeltable.share.remote import is_remote_path

        assert is_remote_path('pxt://local/A/B/C')
        assert is_remote_path('pxt://server/D/E/F')
        assert not is_remote_path('a.b.c')
        assert not is_remote_path('local/path')

    def test_path_conversion(self) -> None:
        """Test path conversion between local and remote formats."""
        from pixeltable.share.remote import convert_local_path_to_remote, convert_remote_path_to_local

        # Test remote to local conversion
        assert convert_remote_path_to_local('pxt://local/A/B/C') == 'a.b.c'
        assert convert_remote_path_to_local('pxt://server/DIR/SUBDIR') == 'dir.subdir'
        assert convert_remote_path_to_local('pxt://local/') == ''

        # Test local to remote conversion
        assert convert_local_path_to_remote('a.b.c') == 'pxt://local/a/b/c'
        assert convert_local_path_to_remote('dir.subdir') == 'pxt://local/dir/subdir'
        assert convert_local_path_to_remote('') == 'pxt://local/'

    def test_remote_table_and_dir_objects(self) -> None:
        """Test RemoteTable and RemoteDir object behavior."""
        table = RemoteTable(path='pxt://local/A/B/C')
        assert table.path == 'pxt://local/A/B/C'
        assert str(table) == "RemoteTable(path='pxt://local/A/B/C')"
        assert table == RemoteTable(path='pxt://local/A/B/C')
        assert table != RemoteTable(path='pxt://local/D/E/F')

        dir_obj = RemoteDir(path='pxt://local/A/B/C')
        assert dir_obj.path == 'pxt://local/A/B/C'
        assert str(dir_obj) == "RemoteDir(path='pxt://local/A/B/C')"
        assert dir_obj == RemoteDir(path='pxt://local/A/B/C')
        assert dir_obj != RemoteDir(path='pxt://local/D/E/F')
