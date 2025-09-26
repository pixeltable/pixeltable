"""
Unit tests for remote_utils.py
"""

from pixeltable.share.remote.utils import ModelCache, convert_local_path_to_remote, convert_remote_path_to_local


class TestPathConversion:
    """Test cases for path conversion functions."""

    def test_convert_remote_path_to_local(self) -> None:
        """Test converting remote path to local path."""
        assert convert_remote_path_to_local('pxt://local/A/B/C') == 'a.b.c'
        assert convert_remote_path_to_local('pxt://server/DIR/SUBDIR') == 'dir.subdir'
        # Empty path represents root directory
        assert convert_remote_path_to_local('pxt://local/') == ''
        assert convert_remote_path_to_local('local.path') == 'local.path'

    def test_convert_remote_path_to_local_invalid_scheme(self) -> None:
        """Test converting remote path with invalid scheme."""
        # Non-pxt schemes are returned as-is
        assert convert_remote_path_to_local('http://local/A/B/C') == 'http://local/A/B/C'

    def test_convert_local_path_to_remote(self) -> None:
        """Test converting local path to remote path."""
        assert convert_local_path_to_remote('a.b.c') == 'pxt://local/a/b/c'
        assert convert_local_path_to_remote('dir.subdir') == 'pxt://local/dir/subdir'
        assert convert_local_path_to_remote('') == 'pxt://local/'

    def test_convert_local_path_to_remote_already_remote(self) -> None:
        """Test converting already remote path."""
        remote_path = 'pxt://local/A/B/C'
        assert convert_local_path_to_remote(remote_path) == remote_path


class TestModelCache:
    """Test cases for ModelCache functionality."""

    def test_model_cache_initialization(self) -> None:
        """Test that ModelCache loads all @remote functions."""
        cache = ModelCache()

        # Check that cache has loaded functions
        assert len(cache._cache) > 0

        # Check specific functions are loaded
        expected_functions = ['create_table', 'get_table', 'move', 'list_tables', 'create_dir']
        for func_name in expected_functions:
            assert func_name in cache._cache

    def test_get_model(self) -> None:
        """Test getting Pydantic model for a function."""
        cache = ModelCache()

        # Test with create_table function
        func = cache.get_function('create_table')
        model = cache.get_model(func)
        assert model is not None

        # Test that model has expected attributes
        assert hasattr(model, 'model_fields')

    def test_get_function(self) -> None:
        """Test getting original function from cache."""
        cache = ModelCache()

        # Test getting a function
        func = cache.get_function('create_table')
        assert func is not None
        assert callable(func)

        # Test that function has expected signature
        import inspect

        sig = inspect.signature(func)
        assert 'path' in sig.parameters
        assert 'schema' in sig.parameters

    def test_get_union_return_type(self) -> None:
        """Test getting union type of all return types."""
        cache = ModelCache()

        union_type = cache.get_union_return_type()
        assert union_type is not None

        # Since we simplified to use Any, just check it's Any
        from typing import Any

        assert union_type == Any
