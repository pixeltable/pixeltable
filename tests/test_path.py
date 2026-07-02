import pytest

import pixeltable.exceptions as excs
from pixeltable.catalog import Path, is_valid_identifier
from tests.utils import pxt_raises


class TestPath:
    """Unit tests for identifier and Path parsing, construction, navigation, and comparison."""

    def test_valid_identifier(self) -> None:
        valid_ids = ['a', 'a1', 'a_1', 'a_']
        invalid_ids = ['', '_', '__', '_a', '1a', 'a.b', '.a', 'a-b']
        for valid_id in valid_ids:
            assert is_valid_identifier(valid_id), valid_ids

        for invalid_id in invalid_ids:
            assert not is_valid_identifier(invalid_id), invalid_ids

    def test_valid_path(self) -> None:
        """Test path validation using Path.parse()."""
        # Test empty path
        Path.parse('', allow_empty_path=True)  # Should succeed
        with pxt_raises(excs.ErrorCode.INVALID_PATH):
            Path.parse('', allow_empty_path=False)  # Should fail

        valid_paths = ['a', 'a_.b_', 'a.b.c', 'a.b.c.d']
        invalid_paths = ['.', '..', 'a.', '.a', 'a..b']

        for valid_path in valid_paths:
            # Should succeed with both empty_is_valid settings
            Path.parse(valid_path, allow_empty_path=False)
            Path.parse(valid_path, allow_empty_path=True)

        for invalid_path in invalid_paths:
            # Should fail regardless of empty_is_valid setting
            with pxt_raises(excs.ErrorCode.INVALID_PATH):
                Path.parse(invalid_path, allow_empty_path=False)
            with pxt_raises(excs.ErrorCode.INVALID_PATH):
                Path.parse(invalid_path, allow_empty_path=True)

    def test_path_parse(self) -> None:
        """Test Path.parse() with '/' delimiter and backward compatibility with '.'."""
        # Test valid paths with SLASH
        valid_slash_paths = ['a', 'a_', 'a/b', 'a/b/c', 'a_/b_']
        for valid_path in valid_slash_paths:
            parsed = Path.parse(valid_path)
            assert parsed.components == tuple(valid_path.split('/'))
            assert str(parsed) == valid_path

        # Test backward compatibility with DOT delimiter
        valid_dot_paths = ['a', 'a_', 'a.b', 'a.b.c', 'a_.b_']
        for valid_path in valid_dot_paths:
            parsed = Path.parse(valid_path)
            assert parsed.components == tuple(valid_path.split('.'))
            # String representation always uses SLASH
            assert str(parsed) == valid_path.replace('.', '/')

        # Test empty path
        empty_parsed = Path.parse('', allow_empty_path=True)
        assert empty_parsed.components == ()
        assert str(empty_parsed) == ''

        # Test versioned paths with SLASH
        versioned = Path.parse('a/b/c:5', allow_versioned_path=True)
        assert versioned.components == ('a', 'b', 'c')
        assert versioned.version == 5
        assert str(versioned) == 'a/b/c:5'

        # Test versioned paths with DOT (backward compatibility)
        versioned_dot = Path.parse('a.b.c:5', allow_versioned_path=True)
        assert versioned_dot.components == ('a', 'b', 'c')
        assert versioned_dot.version == 5
        assert str(versioned_dot) == 'a/b/c:5'

    def test_local_catalog_uri(self) -> None:
        # A plain path lives in the local catalog (empty uri, no org/db).
        local = Path.parse('a.b')
        assert local.org is None
        assert local.db is None
        assert local.uri == ''
        assert local.catalog_uri == Path()

    def test_hosted_path_parse(self) -> None:
        """Path.parse() understands pxt:// URIs and Pixeltable web URLs."""
        hosted = Path.parse('pxt://variata:main/dir/tbl')
        assert hosted.org == 'variata'
        assert hosted.db == 'main'
        assert hosted.components == ('dir', 'tbl')
        assert hosted.uri == 'pxt://variata:main'
        assert hosted.catalog_uri == Path(org='variata', db='main')
        assert str(hosted) == 'pxt://variata:main/dir/tbl'

        # Versioned hosted path.
        versioned = Path.parse('pxt://local:testdb/dir/tbl:7', allow_versioned_path=True)
        assert (versioned.org, versioned.db, versioned.components, versioned.version) == (
            'local',
            'testdb',
            ('dir', 'tbl'),
            7,
        )

        # Org without a db.
        no_db = Path.parse('pxt://variata/tbl')
        assert no_db.org == 'variata'
        assert no_db.db is None
        assert no_db.uri == 'pxt://variata'

        # A Pixeltable web URL normalizes to the same parse as its pxt:// form.
        assert Path.parse('https://pixeltable.com/t/variata:main/dir/tbl') == Path.parse('pxt://variata:main/dir/tbl')

        # str() round-trips for both local and hosted paths.
        assert all(
            Path.parse(str(p), allow_versioned_path=True) == p for p in (Path.parse('a/b'), hosted, versioned, no_db)
        )

    def test_hosted_path_errors(self) -> None:
        # pxt:// with no org.
        for bad in ('pxt://', 'pxt:///tbl'):
            with pxt_raises(excs.ErrorCode.INVALID_PATH):
                Path.parse(bad)
        # Negative version.
        with pxt_raises(excs.ErrorCode.INVALID_PATH):
            Path.parse('pxt://variata:main/tbl:-1', allow_versioned_path=True)
        # Bad identifier component in a hosted path.
        with pxt_raises(excs.ErrorCode.INVALID_PATH):
            Path.parse('pxt://variata:main/a..b')
        # Org slug parses out of the netloc but isn't a valid identifier.
        with pxt_raises(excs.ErrorCode.INVALID_PATH):
            Path.parse('pxt://bad org/tbl')
        # An extra colon lands in the db slug, which then fails identifier validation.
        with pxt_raises(excs.ErrorCode.INVALID_PATH):
            Path.parse('pxt://variata:main:extra/tbl')

    def test_path_construction_invariants(self) -> None:
        # Invariants enforced at construction, so they hold for from_components() (and direct
        # construction), not only for parse().
        # A db requires an org.
        with pxt_raises(excs.ErrorCode.INVALID_PATH):
            Path.from_components(('tbl',), db='main')
        # Org and db must be valid slugs.
        with pxt_raises(excs.ErrorCode.INVALID_PATH):
            Path.from_components(('tbl',), org='bad org')
        with pxt_raises(excs.ErrorCode.INVALID_PATH):
            Path.from_components(('tbl',), org='variata', db='bad:db')
        # Version must be non-negative.
        with pxt_raises(excs.ErrorCode.INVALID_PATH):
            Path.from_components(('tbl',), version=-1)
        # Components must be valid, non-empty identifiers; the empty tuple is the root.
        with pxt_raises(excs.ErrorCode.INVALID_PATH):
            Path.from_components(('a', 'bad name'))
        with pxt_raises(excs.ErrorCode.INVALID_PATH):
            Path.from_components(('a', ''))
        with pxt_raises(excs.ErrorCode.INVALID_PATH):
            Path.from_components(('',))  # a single empty component is not the root
        assert Path.from_components(('a', 'b')).components == ('a', 'b')
        assert Path.from_components(()).is_root  # the empty tuple is the root
        # Hyphenated org/db slugs are accepted.
        hosted = Path.parse('pxt://my-org:my-db/tbl')
        assert (hosted.org, hosted.db) == ('my-org', 'my-db')
        assert Path.from_components(('tbl',), org='my-org', db='my-db').uri == 'pxt://my-org:my-db'

    def test_hosted_path_navigation(self) -> None:
        # Navigation preserves the catalog (org/db) and drops the version.
        path = Path.parse('pxt://variata:main/a/b/c:3', allow_versioned_path=True)
        assert path.parent == Path.from_components(('a', 'b'), org='variata', db='main')
        assert path.append('d') == Path.from_components(('a', 'b', 'c', 'd'), org='variata', db='main')
        assert path.ancestors() == [
            Path.from_components((), org='variata', db='main'),
            Path.from_components(('a',), org='variata', db='main'),
            Path.from_components(('a', 'b'), org='variata', db='main'),
        ]
        # Same-named local and hosted paths are distinct.
        assert Path.parse('a/b') != Path.parse('pxt://variata:main/a/b')
        # is_ancestor is false across catalogs
        assert not Path.parse('a').is_ancestor(Path.parse('pxt://variata:main/a/b'))

    @pytest.mark.parametrize('path_str', ['a.b.c', 'a/b/c'])
    def test_path_ancestors(self, path_str: str) -> None:
        # Test with both dot and slash paths (both result in '/' representation)
        # multiple ancestors in path
        path = Path.parse(path_str)
        expected_ancestors = [Path.from_components(()), Path.from_components(('a',)), Path.from_components(('a', 'b'))]
        assert path.ancestors() == expected_ancestors

        # single element in path
        path = Path.parse('a')
        assert path.ancestors() == [Path.from_components(())]

        # root
        path = Path.parse('', allow_empty_path=True)
        assert path.ancestors() == []

    def test_path_delimiter_str_hash_compare(self) -> None:
        """Test that paths with different input delimiters but same components compare equal and hash the same."""
        # Parse with DOT delimiter (backward compatibility)
        dotted_path = Path.parse('a.b.c')

        # Parse with SLASH delimiter
        unix_path = Path.parse('a/b/c')

        assert dotted_path.components == unix_path.components == ('a', 'b', 'c')

        # String representation always uses SLASH
        assert str(dotted_path) == 'a/b/c'
        assert str(unix_path) == 'a/b/c'

        # both paths should be equal
        assert dotted_path == unix_path

        # both paths should have the same hash
        assert hash(dotted_path) == hash(unix_path)

        # Test with versioned paths
        dotted_versioned = Path.parse('a.b.c:5', allow_versioned_path=True)
        unix_versioned = Path.parse('a/b/c:5', allow_versioned_path=True)

        assert dotted_versioned.components == unix_versioned.components == ('a', 'b', 'c')
        assert dotted_versioned.version == unix_versioned.version == 5
        assert str(dotted_versioned) == 'a/b/c:5'
        assert str(unix_versioned) == 'a/b/c:5'
        assert dotted_versioned == unix_versioned
        assert hash(dotted_versioned) == hash(unix_versioned)

        # Test parent always uses SLASH
        dotted_parent = dotted_path.parent
        unix_parent = unix_path.parent
        assert str(dotted_parent) == 'a/b'
        assert str(unix_parent) == 'a/b'
        assert dotted_parent == unix_parent

        # Test append always uses SLASH
        dotted_appended = dotted_path.append('d')
        unix_appended = unix_path.append('d')
        assert str(dotted_appended) == 'a/b/c/d'
        assert str(unix_appended) == 'a/b/c/d'
        assert dotted_appended.components == unix_appended.components == ('a', 'b', 'c', 'd')
