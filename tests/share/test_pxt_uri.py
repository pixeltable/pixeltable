"""Tests for PxtUri parsing and construction."""

from uuid import UUID

import pytest

from pixeltable.share.protocol import PxtUri


class TestPxtUri:
    """Test PxtUri parsing and construction."""

    @pytest.mark.parametrize(
        'uri_str,expected_org,expected_db,expected_path,expected_id,expected_version',
        [
            ('pxt://org_name/table_name', 'org_name', None, 'table_name', None, None),
            ('pxt://org_name:db_name/table_name', 'org_name', 'db_name', 'table_name', None, None),
            ('pxt://org_name:db_name/table_name:5', 'org_name', 'db_name', 'table_name', None, 5),
        ],
    )
    def test_parse_path_uris(
        self,
        uri_str: str,
        expected_org: str,
        expected_db: str | None,
        expected_path: str,
        expected_id: None,
        expected_version: int | None,
    ) -> None:
        """Test parsing URIs with paths."""
        uri = PxtUri(uri_str)
        assert uri.org == expected_org
        assert uri.db == expected_db
        assert uri.path == expected_path
        assert uri.id == expected_id
        assert uri.version == expected_version

    @pytest.mark.parametrize(
        'uri_str,expected_db,expected_version',
        [
            ('pxt://org_name/550e8400-e29b-41d4-a716-446655440000', None, None),
            ('pxt://org_name:db_name/550e8400-e29b-41d4-a716-446655440000', 'db_name', None),
            ('pxt://org_name:db_name/550e8400-e29b-41d4-a716-446655440000:10', 'db_name', 10),
        ],
    )
    def test_parse_uuid_uris(self, uri_str: str, expected_db: str | None, expected_version: int | None) -> None:
        """Test parsing URIs with UUID."""
        test_uuid = '550e8400-e29b-41d4-a716-446655440000'
        uri = PxtUri(uri_str)
        assert uri.org == 'org_name'
        assert uri.db == expected_db
        assert uri.path is None
        assert uri.id == UUID(test_uuid)
        assert uri.version == expected_version

    def test_parse_root_path(self) -> None:
        """Test parsing root path."""
        uri = PxtUri('pxt://org_name:db_name/')
        assert uri.org == 'org_name'
        assert uri.db == 'db_name'
        assert uri.path == ''
        assert uri.id is None
        assert uri.version is None

    def test_parse_nested_path(self) -> None:
        """Test parsing nested paths."""
        uri = PxtUri('pxt://org_name/dir/subdir/table')
        assert uri.org == 'org_name'
        assert uri.path == 'dir/subdir/table'
        assert uri.id is None

    def test_parse_path_with_invalid_version(self) -> None:
        """Test parsing paths with colons that should not be treated as version."""
        with pytest.raises(ValueError, match='Invalid table version'):
            PxtUri('pxt://org_name/table:')
        with pytest.raises(ValueError, match='Invalid table version'):
            PxtUri('pxt://org_name/table:version')

    def test_parse_from_dict_input(self) -> None:
        """Test parsing from dict input."""
        uri = PxtUri({'uri': 'pxt://org_name/table'})
        assert uri.path == 'table'

    def test_parse_invalid_scheme(self) -> None:
        """Test parsing with invalid scheme."""
        with pytest.raises(ValueError, match='Expected a pxt:// URI or a Pixeltable URL'):
            PxtUri('http://org_name/table')

    def test_parse_missing_org_slug(self) -> None:
        """Test parsing with missing org."""
        with pytest.raises(ValueError, match='URI must have an organization'):
            PxtUri('pxt:///table')

    def test_parse_missing_path(self) -> None:
        """Test parsing with missing path - empty path is allowed, so this should pass."""
        # pxt://org_name actually parses with empty path, which is valid for root
        uri = PxtUri('pxt://org_name')
        assert uri.path == ''
        assert uri.org == 'org_name'

    def test_parse_invalid_data_type(self) -> None:
        """Test parsing with invalid data type."""
        with pytest.raises(ValueError, match='Invalid data type'):
            PxtUri(123)  # type: ignore

    def test_parse_dict_missing_uri_key(self) -> None:
        """Test parsing dict without uri key."""
        with pytest.raises(ValueError, match='URI must be provided in dict'):
            PxtUri({'invalid': 'key'})

    @pytest.mark.parametrize(
        'kwargs,error_match',
        [
            ({}, 'Either path or id must be provided'),
            ({'path': 'table', 'id': UUID('550e8400-e29b-41d4-a716-446655440000')}, 'Cannot specify both path and id'),
        ],
    )
    def test_from_components_validation(self, kwargs: dict, error_match: str) -> None:
        """Test from_components validation."""
        with pytest.raises(ValueError, match=error_match):
            PxtUri.from_components('org_name', **kwargs)

    def test_str_returns_original_uri(self) -> None:
        """Test that __str__ returns the original URI."""
        uri_str = 'pxt://org_name/table:5'
        uri = PxtUri(uri_str)
        assert str(uri) == uri_str

    def test_version(self) -> None:
        """Test that negative version numbers in URI raise ValueError."""
        with pytest.raises(ValueError, match='Version must be a non-negative integer'):
            PxtUri('pxt://org_name/table:-1')
        uri = PxtUri('pxt://org_name/table:0')
        assert uri.version == 0
        uri = PxtUri('pxt://org_name/table:42')
        assert uri.version == 42

    def test_version_in_from_components(self) -> None:
        """Test that negative version numbers in from_components raise ValueError."""
        with pytest.raises(ValueError, match='Version must be a non-negative integer'):
            PxtUri.from_components('org_name', path='table', version=-12)
        uri = PxtUri.from_components('org_name', path='table', version=0)
        assert uri.version == 0
        uri = PxtUri.from_components('org_name', path='table', version=10)
        assert uri.version == 10

    @pytest.mark.parametrize(
        'url,expected_uri,expected_org,expected_db,expected_path',
        [
            (
                'https://pixeltable.com/t/pixeltable:fiftyone/coco_mini_2017',
                'pxt://pixeltable:fiftyone/coco_mini_2017',
                'pixeltable',
                'fiftyone',
                'coco_mini_2017',
            ),
            (
                'https://www.pixeltable.com/t/pixeltable:fiftyone/coco_mini_2017',
                'pxt://pixeltable:fiftyone/coco_mini_2017',
                'pixeltable',
                'fiftyone',
                'coco_mini_2017',
            ),
            ('http://pixeltable.com/t/myorg/mytable', 'pxt://myorg/mytable', 'myorg', None, 'mytable'),
            ('http://www.pixeltable.com/t/myorg/mytable', 'pxt://myorg/mytable', 'myorg', None, 'mytable'),
            (
                'https://pixeltable.com/t/org:db/dir/subdir/table',
                'pxt://org:db/dir/subdir/table',
                'org',
                'db',
                'dir/subdir/table',
            ),
            ('https://pixeltable.com/t/org:db/table:5', 'pxt://org:db/table:5', 'org', 'db', 'table'),
        ],
    )
    def test_normalize_https_url(
        self, url: str, expected_uri: str, expected_org: str, expected_db: str | None, expected_path: str
    ) -> None:
        uri = PxtUri(url)
        assert str(uri) == expected_uri
        assert uri.org == expected_org
        assert uri.db == expected_db
        assert uri.path == expected_path

    def test_normalize_preserves_version(self) -> None:
        uri = PxtUri('https://pixeltable.com/t/org:db/table:5')
        assert uri.version == 5
        assert str(uri) == 'pxt://org:db/table:5'

    def test_normalize_uuid_url(self) -> None:
        test_uuid = '550e8400-e29b-41d4-a716-446655440000'
        uri = PxtUri(f'https://pixeltable.com/t/org:db/{test_uuid}')
        assert uri.id == UUID(test_uuid)
        assert uri.path is None
        assert str(uri) == f'pxt://org:db/{test_uuid}'

    def test_normalized_uri_stored_not_original_url(self) -> None:
        url = 'https://www.pixeltable.com/t/org:db/table'
        uri = PxtUri(url)
        assert str(uri).startswith('pxt://')
        assert 'pixeltable.com' not in str(uri)

    def test_dict_input_with_https_url(self) -> None:
        uri = PxtUri({'uri': 'https://pixeltable.com/t/org:db/table'})
        assert str(uri) == 'pxt://org:db/table'
        assert uri.org == 'org'

    def test_reject_invalid_urls(self) -> None:
        with pytest.raises(ValueError, match=r'Invalid URI .* Expected a pxt:// URI or a Pixeltable URL'):
            PxtUri('https://noop.com/t/org/table')  # wrong domain
        with pytest.raises(ValueError, match=r'Invalid URI .* Expected a pxt:// URI or a Pixeltable URL'):
            PxtUri('https://notpixeltable.com/t/org/table')  # wrong domain but with pixeltable.com in it
        with pytest.raises(ValueError, match=r'Invalid URI .* Expected a pxt:// URI or a Pixeltable URL'):
            PxtUri('https://pixeltable.com/org/table')  # missing /t/ prefix
        with pytest.raises(ValueError, match='URI must have an organization'):
            PxtUri('https://pixeltable.com/t/')  # missing org
        with pytest.raises(ValueError, match=r'Invalid URI .* Expected a pxt:// URI or a Pixeltable URL'):
            PxtUri('ftp://pixeltable.com/t/org/table')  # unsupported scheme
        with pytest.raises(ValueError, match=r'Invalid URI .* Expected a pxt:// URI or a Pixeltable URL'):
            PxtUri('http://t/org/table')  # missing domain
