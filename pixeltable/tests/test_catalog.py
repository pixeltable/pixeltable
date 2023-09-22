from pixeltable.catalog import is_valid_identifier, is_valid_path

class TestCatalog:
    """Tests for miscellanous catalog functions."""
    def test_valid_identifier(self) -> None:
        valid_ids = ['a', 'a1', 'a_1', 'a_']
        invalid_ids = ['', '_', '__', '_a', '1a', 'a.b', '.a', 'a-b']
        for valid_id in valid_ids:
            assert is_valid_identifier(valid_id), valid_ids

        for invalid_id in invalid_ids:
            assert not is_valid_identifier(invalid_id), invalid_ids

    def test_valid_path(self) -> None:
        assert is_valid_path('', empty_is_valid=True)
        assert not is_valid_path('', empty_is_valid=False)

        valid_paths = ['a', 'a_.b_', 'a.b.c', 'a.b.c.d']
        invalid_paths = ['.', '..', 'a.', '.a', 'a..b']
        
        for valid_path in valid_paths:
            assert is_valid_path(valid_path, empty_is_valid=False), valid_path
            assert is_valid_path(valid_path, empty_is_valid=True), valid_path

        for invalid_path in invalid_paths:
            assert not is_valid_path(invalid_path, empty_is_valid=False), invalid_path
            assert not is_valid_path(invalid_path, empty_is_valid=True), invalid_path