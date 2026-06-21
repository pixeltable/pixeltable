from textwrap import dedent
from typing import Any, Callable

import psycopg
import pydantic
import pytest
import sqlalchemy.exc as sql_exc

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable.catalog import Dir, InsertableTableProxy, Path, ViewProxy, is_valid_identifier
from pixeltable.runtime import get_runtime
from pixeltable.utils.fault_injection import FaultLocation
from tests.coordinator import MultiThreadedScenario
from tests.fault_injection import BlockFault, ExceptionFault
from tests.utils import pxt_raises


class TestCatalog:
    """Tests for miscellanous catalog functions."""

    @pytest.fixture(autouse=True)
    def _skip_local_proxy_without_serve_deps(self, request: pytest.FixtureRequest) -> None:
        # the standalone local-proxy tests spawn the daemon, which needs fastapi/uvicorn (the serve extra)
        if request.node.name.startswith('test_local_proxy'):
            pytest.importorskip('fastapi')
            pytest.importorskip('uvicorn')

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

    def test_local_proxy(self, init_env: None) -> None:
        # End-to-end against a real local-proxy daemon over HTTP: pxt.create_table('pxt://local:<db>/...')
        # routes through CatalogProxy -> HTTP -> the daemon's own catalog, and returns a metadata-backed
        # proxy handle. A server-side error is re-raised as the identical exception type on the client.
        from pixeltable.service import proxy_daemon

        def column_signature(tbl: pxt.Table) -> dict:
            # schema-intrinsic column properties, comparable across two distinct tables
            return {
                name: (col['type_'], col['is_primary_key'], col['media_validation'])
                for name, col in tbl.get_metadata()['columns'].items()
            }

        db = 'proxytest'
        proxy_daemon.start(db)
        try:
            t = pxt.create_table(f'pxt://local:{db}/proxy_foo', {'a': pxt.Int, 's': pxt.String})
            assert isinstance(t, InsertableTableProxy)
            assert t.get_metadata()['name'] == 'proxy_foo'
            assert set(t.get_metadata()['columns']) == {'a', 's'}

            # A full schema -- scalars, every media type, and ColumnSpecs (primary key + media validation) --
            # created over the proxy must yield the same column metadata as creating it locally.
            schema: dict[str, Any] = {
                'id': pxt.Required[pxt.Int],
                's': pxt.String,
                'img': pxt.Image,
                'vid': pxt.Video,
                'aud': pxt.Audio,
                'doc': pxt.Document,
                'img_validated': {'type': pxt.Image, 'media_validation': 'on_read'},
            }
            local = pxt.create_table('proxy_local_ref', schema, primary_key='id')
            remote = pxt.create_table(f'pxt://local:{db}/proxy_full', schema, primary_key='id')
            assert isinstance(remote, InsertableTableProxy)
            assert column_signature(remote) == column_signature(local)
            assert remote.get_metadata()['primary_key'] == local.get_metadata()['primary_key'] == ['id']

            # get_table returns a metadata-equivalent handle; the same columns as the created table.
            fetched = pxt.get_table(f'pxt://local:{db}/proxy_full')
            assert isinstance(fetched, InsertableTableProxy)
            assert column_signature(fetched) == column_signature(remote)
            assert pxt.get_table(f'pxt://local:{db}/missing', if_not_exists='ignore') is None
            with pytest.raises(excs.NotFoundError):
                pxt.get_table(f'pxt://local:{db}/missing')

            # drop_table removes it; a subsequent lookup yields None (or raises with the default).
            pxt.drop_table(f'pxt://local:{db}/proxy_full')
            assert pxt.get_table(f'pxt://local:{db}/proxy_full', if_not_exists='ignore') is None
            pxt.drop_table(f'pxt://local:{db}/proxy_full', if_not_exists='ignore')  # no-op, no error
            with pytest.raises(excs.NotFoundError):
                pxt.drop_table(f'pxt://local:{db}/proxy_full')

            with pytest.raises(excs.NotFoundError):
                pxt.create_table(f'pxt://local:{db}/no_such_dir/foo', {'a': pxt.Int})
        finally:
            proxy_daemon.delete(db)

    def test_table_path_key(self, init_env: None) -> None:
        from uuid import uuid4

        from pixeltable.catalog import TablePathKey, TableVersionPath
        from pixeltable.catalog.table_version import TableVersionKey

        # recursive {tbl_version, base} as_dict/from_dict round-trips, including a base element
        key = TablePathKey((TableVersionKey(uuid4(), None), TableVersionKey(uuid4(), 3)))
        assert TablePathKey.from_dict(key.as_dict()) == key
        assert key.as_dict()['base'] is not None

        # against a real path, key() reproduces the legacy nested as_dict() byte-for-byte (no migration)
        t = pxt.create_table('tpk', {'a': pxt.Int})
        tvp = t._tbl_path
        assert isinstance(tvp, TableVersionPath)
        assert tvp.key().as_dict() == tvp.as_dict()
        # a live table's effective key has version None; its snapshot key has the concrete version
        assert tvp.key().leaf.effective_version is None
        assert tvp.snapshot_key().leaf.effective_version == tvp.version()

    def test_local_proxy_cross_thread(self, init_env: None) -> None:
        # A thread that didn't construct the proxy has an empty thread-local md path; _tbl_md_path must
        # lazily fetch it via the catalog so schema introspection works from any thread.
        import threading

        from pixeltable.service import proxy_daemon

        db = 'proxy_xthread'
        proxy_daemon.start(db)
        try:
            t = pxt.create_table(f'pxt://local:{db}/xt', {'a': pxt.Int, 's': pxt.String})
            captured: dict[str, Any] = {}

            def worker() -> None:
                captured['columns'] = set(t.columns())
                captured['name'] = t.get_metadata()['name']

            th = threading.Thread(target=worker)
            th.start()
            th.join()
            assert captured['columns'] == {'a', 's'}
            assert captured['name'] == 'xt'
        finally:
            proxy_daemon.delete(db)

    def test_local_proxy_query(self, init_env: None) -> None:
        # A query built against a hosted table routes collect()/count() to the daemon, runs there, and
        # returns the (scalar) result set synchronously.
        from pixeltable.service import proxy_daemon

        db = 'proxy_query'
        proxy_daemon.start(db)
        try:
            t = pxt.create_table(f'pxt://local:{db}/q', {'a': pxt.Int, 's': pxt.String})
            t.insert([{'a': 1, 's': 'x'}, {'a': 2, 's': 'y'}, {'a': 3, 's': 'z'}])

            assert t.count() == 3
            res = t.where(t.a >= 2).order_by(t.a).select(t.a, t.s).collect()
            assert res['a'] == [2, 3]
            assert res['s'] == ['y', 'z']
            assert t.where(t.a >= 2).count() == 2
            assert len(t.limit(1).collect()) == 1
        finally:
            proxy_daemon.delete(db)

    def test_local_proxy_mutations(self, init_env: None) -> None:
        # update/batch_update/delete/revert/list_views/get_versions dispatch to the daemon; delete is rejected
        # on a view (mirroring View.delete).
        from pixeltable.service import proxy_daemon

        db = 'proxy_mut'
        proxy_daemon.start(db)
        try:
            t = pxt.create_table(f'pxt://local:{db}/m', {'a': pxt.Required[pxt.Int], 's': pxt.String}, primary_key='a')
            t.insert([{'a': 1, 's': 'x'}, {'a': 2, 's': 'y'}, {'a': 3, 's': 'z'}])
            assert t.count() == 3

            t.update({'s': 'U'}, where=t.a == 1)
            assert t.where(t.a == 1).select(t.s).collect()['s'] == ['U']

            t.batch_update([{'a': 2, 's': 'B'}])
            assert t.where(t.a == 2).select(t.s).collect()['s'] == ['B']

            t.delete(where=t.a == 3)
            assert t.count() == 2

            assert len(t.get_versions()) >= 1

            t.revert()  # undo the delete
            assert t.count() == 3

            v = pxt.create_view(f'pxt://local:{db}/m_view', t)
            assert isinstance(v, ViewProxy)
            assert any('m_view' in path for path in t.list_views())
            with pxt_raises(excs.ErrorCode.UNSUPPORTED_OPERATION):
                v.delete()  # delete is unsupported on a view
        finally:
            proxy_daemon.delete(db)

    def test_function_serialization(self) -> None:
        # Functions (e.g. embedding UDFs passed to add_embedding_index) cross the wire via proxy_protocol.
        import pixeltable.func as func
        from pixeltable.functions import string as pxt_str
        from pixeltable.service import proxy_protocol

        f = pxt_str.contains
        assert isinstance(f, func.Function)
        round_tripped = proxy_protocol.deserialize(proxy_protocol.serialize(f))
        assert round_tripped.self_path == f.self_path

    def test_local_proxy_schema_changes(self, init_env: None) -> None:
        # add_columns / add_column / add_computed_column / rename_column / drop_column / recompute_columns
        # dispatch to the daemon as gated mutations; the client's md refreshes after each so the next op sees
        # the new schema.
        from pixeltable.service import proxy_daemon

        db = 'proxy_ddl'
        proxy_daemon.start(db)
        try:
            t = pxt.create_table(f'pxt://local:{db}/d', {'a': pxt.Int, 's': pxt.String})
            t.insert([{'a': 1, 's': 'x'}, {'a': 2, 's': 'yy'}])

            t.add_columns({'b': pxt.Float, 'c': pxt.String})
            assert {'a', 's', 'b', 'c'} <= set(t.columns())

            t.add_column(e=pxt.Int)
            assert 'e' in t.columns()

            t.add_computed_column(a1=t.a + 1)
            assert t.where(t.a == 2).select(t.a1).collect()['a1'] == [3]

            t.rename_column('c', 'c2')
            cols = set(t.columns())
            assert 'c2' in cols and 'c' not in cols

            t.drop_column('b')
            assert 'b' not in t.columns()

            t.recompute_columns('a1')
            assert t.where(t.a == 1).select(t.a1).collect()['a1'] == [2]
        finally:
            proxy_daemon.delete(db)

    def test_local_proxy_view(self, init_env: None) -> None:
        # A view created over the proxy (a Table base plus a computed column referencing the base) must
        # yield the same column metadata as creating the equivalent view locally.
        from pixeltable.service import proxy_daemon

        def column_signature(tbl: pxt.Table) -> dict:
            return {
                name: (col['type_'], col['is_primary_key'], col['media_validation'], col['is_computed'])
                for name, col in tbl.get_metadata()['columns'].items()
            }

        db = 'proxyviewtest'
        proxy_daemon.start(db)
        try:
            base = pxt.create_table(f'pxt://local:{db}/base', {'n': pxt.Int, 's': pxt.String})
            remote_view = pxt.create_view(f'pxt://local:{db}/v', base, additional_columns={'doubled': base.n * 2})
            assert isinstance(remote_view, ViewProxy)
            assert remote_view.get_metadata()['kind'] == 'view'

            # get_table on a view returns a ViewProxy
            fetched_view = pxt.get_table(f'pxt://local:{db}/v')
            assert isinstance(fetched_view, ViewProxy)

            local_base = pxt.create_table('local_base', {'n': pxt.Int, 's': pxt.String})
            local_view = pxt.create_view('local_view', local_base, additional_columns={'doubled': local_base.n * 2})
            assert column_signature(remote_view) == column_signature(local_view)
        finally:
            proxy_daemon.delete(db)

    def test_local_proxy_describe(self, init_env: None) -> None:
        # _path() and describe()/repr render the table's full pxt:// path: the server returns its in-db
        # path and rendered description, and the client rebases the title onto this proxy's catalog.
        from pixeltable.service import proxy_daemon

        db = 'proxydescribe'
        proxy_daemon.start(db)
        try:
            pxt.create_dir(f'pxt://local:{db}/sub')
            t = pxt.create_table(f'pxt://local:{db}/sub/d', {'a': pxt.Int, 's': pxt.String})

            # _path() carries the proxy's org/db together with the in-db components
            path = t._path()
            assert path.org == 'local'
            assert path.db == db
            assert str(path) == f'pxt://local:{db}/sub/d'

            # the rendered description titles the table with its full pxt:// path and lists its columns
            r = repr(t)
            assert f"table 'pxt://local:{db}/sub/d'" in r
            assert 'a' in r and 's' in r
            assert "'sub/d'" not in r  # the bare in-db path never leaks into the title
            assert t._repr_html_() != ''

            # a view's title shows its own full pxt:// path
            v = pxt.create_view(f'pxt://local:{db}/sub/v', t, additional_columns={'a1': t.a + 1})
            assert isinstance(v, ViewProxy)
            assert str(v._path()) == f'pxt://local:{db}/sub/v'
            assert f"view 'pxt://local:{db}/sub/v'" in repr(v)
        finally:
            proxy_daemon.delete(db)

    def test_local_proxy_dirs(self, init_env: None) -> None:
        # create_dir / get_dir_contents / drop_dir over the proxy.
        from pixeltable.service import proxy_daemon

        db = 'proxydirtest'
        proxy_daemon.start(db)
        try:
            d = pxt.create_dir(f'pxt://local:{db}/d')
            assert isinstance(d, Dir)
            pxt.create_table(f'pxt://local:{db}/d/t', {'a': pxt.Int})
            pxt.create_dir(f'pxt://local:{db}/d/sub')

            contents = pxt.get_dir_contents(f'pxt://local:{db}/d', recursive=False)
            assert contents['tables'] == [f'pxt://local:{db}/d/t']
            assert contents['dirs'] == [f'pxt://local:{db}/d/sub']

            # drop_dir removes the subtree; lookups then miss
            pxt.drop_dir(f'pxt://local:{db}/d', force=True)
            assert pxt.get_table(f'pxt://local:{db}/d/t', if_not_exists='ignore') is None
            pxt.drop_dir(f'pxt://local:{db}/d', if_not_exists='ignore')  # no-op, no error
            with pytest.raises(excs.NotFoundError):
                pxt.drop_dir(f'pxt://local:{db}/d')
        finally:
            proxy_daemon.delete(db)

    def test_local_proxy_move(self, init_env: None) -> None:
        # move over the proxy (rename + relocate); pxt.ls exercises the proxy's get_table_by_id.
        from pixeltable.service import proxy_daemon

        db = 'proxymovetest'
        proxy_daemon.start(db)
        try:
            pxt.create_table(f'pxt://local:{db}/foo', {'a': pxt.Int})
            pxt.move(f'pxt://local:{db}/foo', f'pxt://local:{db}/bar')
            assert pxt.get_table(f'pxt://local:{db}/foo', if_not_exists='ignore') is None
            assert pxt.get_table(f'pxt://local:{db}/bar') is not None

            pxt.create_dir(f'pxt://local:{db}/d')
            pxt.move(f'pxt://local:{db}/bar', f'pxt://local:{db}/d/bar')
            assert pxt.get_table(f'pxt://local:{db}/d/bar') is not None

            # pxt.ls() resolves each entry via get_table_by_id over the proxy
            listing = pxt.ls(f'pxt://local:{db}/d')
            bar_rows = listing[listing['Name'] == 'bar']
            assert len(bar_rows) == 1
            assert bar_rows.iloc[0]['Kind'] == 'table'

            with pytest.raises(excs.NotFoundError):
                pxt.move(f'pxt://local:{db}/missing', f'pxt://local:{db}/x')
            pxt.move(f'pxt://local:{db}/missing', f'pxt://local:{db}/x', if_not_exists='ignore')  # no-op
        finally:
            proxy_daemon.delete(db)

    def test_local_proxy_insert(self, init_env: None) -> None:
        # insert over the proxy: list[dict] and list[BaseModel] of scalars
        from pixeltable.service import proxy_daemon

        class Row(pydantic.BaseModel):
            a: int
            s: str

        db = 'proxyinserttest'
        proxy_daemon.start(db)
        try:
            t = pxt.create_table(f'pxt://local:{db}/t', {'a': pxt.Int, 's': pxt.String})

            v0 = t.get_metadata()['version']
            status = t.insert([{'a': 1, 's': 'x'}, {'a': 2, 's': 'y'}])
            assert status.row_count_stats.ins_rows == 2
            # the insert advanced the version; the proxy refreshed its md from the response
            assert t.get_metadata()['version'] > v0

            # return_rows round-trips the inserted rows back to the caller
            status = t.insert([{'a': 3, 's': 'z'}], return_rows=True)
            assert status.row_count_stats.ins_rows == 1
            assert status.rows is not None and status.rows[0]['a'] == 3 and status.rows[0]['s'] == 'z'

            # list[pydantic.BaseModel]
            status = t.insert([Row(a=4, s='w')])
            assert status.row_count_stats.ins_rows == 1
        finally:
            proxy_daemon.delete(db)

    def test_proxy_move_cross_db(self, init_env: None) -> None:
        # cross-catalog moves are rejected before any RPC (no daemon needed)
        with pytest.raises(excs.Error, match='same catalog'):
            pxt.move('pxt://local:db1/t', 'pxt://local:db2/t')
        with pytest.raises(excs.Error, match='same catalog'):
            pxt.move('pxt://local:db/t', 'local_t')  # hosted -> local

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

    def test_ls(self, make_catalog_path: Callable[[str], str]) -> None:
        p = make_catalog_path
        pxt.create_dir(p('test_dir'))
        pxt.create_dir(p('test_dir/subdir'))

        t = pxt.create_table(p('test_dir/tbl'), {'a': pxt.Int})
        t.insert(a=3)
        v1 = pxt.create_view(p('view1'), t)
        t.insert(a=5)
        v1.add_column(b=pxt.Int)
        _s1 = pxt.create_snapshot(p('test_dir/snapshot1'), v1)
        t.insert(a=22)
        v2 = pxt.create_view(p('test_dir/view2'), t)
        _s2 = pxt.create_snapshot(p('test_dir/snapshot2'), v2, additional_columns={'c': pxt.String})
        t.insert(a=4171780)
        df = pxt.ls(p('test_dir'))
        print(repr(df))
        assert dedent(repr(df)) == dedent(
            '''
                 Name      Kind Version              Base
            snapshot1  snapshot                   view1:2
            snapshot2  snapshot          test_dir/view2:0
               subdir       dir                          |
                  tbl     table       4                  |
                view2      view       1      test_dir/tbl
            '''
        ).strip('\n').replace('|', '')  # fmt: skip

    def test_cross_type_replacement(self, make_catalog_path: Callable[[str], str]) -> None:
        """Test that tables, views, and snapshots can replace each other with if_exists='replace'.

        This tests the path collision handling logic: dirs can only collide with dirs,
        but all table subtypes (table, view, snapshot) can collide with each other.
        """
        p = make_catalog_path
        base_table = pxt.create_table(p('base'), {'c1': pxt.Int})

        # One lambda per create_x with expected columns
        creators = {
            'table': (lambda: pxt.create_table(p('target'), {'c2': pxt.String}, if_exists='replace'), ['c2']),
            'view': (
                lambda: pxt.create_view(
                    p('target'), base_table, additional_columns={'c3': pxt.String}, if_exists='replace'
                ),
                ['c3', 'c1'],
            ),
            'snapshot': (lambda: pxt.create_snapshot(p('target'), base_table, if_exists='replace'), ['c1']),
        }

        # Test all permutations: each table subtype can replace any table subtype
        for existing_creator, _ in creators.values():
            for replacing_creator, expected_cols in creators.values():
                existing_creator()
                assert p('target') in pxt.list_tables(p(''))
                result = replacing_creator()
                assert p('target') in pxt.list_tables(p(''))
                assert result.columns() == expected_cols

        # Verify cross-type replacement is blocked in both directions for every table subtype
        pxt.drop_table(p('target'))
        pxt.create_dir(p('target'))
        for creator, _ in creators.values():
            # dirs cannot be replaced by table subtypes
            with pxt_raises(excs.ErrorCode.PATH_ALREADY_EXISTS, match='expected a table, view or snapshot'):
                creator()
            # table subtypes cannot be replaced by dirs
            pxt.drop_dir(p('target'))
            creator()
            with pxt_raises(excs.ErrorCode.PATH_ALREADY_EXISTS, match='expected a directory'):
                pxt.create_dir(p('target'), if_exists='replace')
            pxt.drop_table(p('target'))
            pxt.create_dir(p('target'))

    def test_table_op_from_dict_needs_xact(self) -> None:
        """Verifies that a TableOp can be correctly deserialized from a dict that includes the legacy 'needs_xact'
        field"""
        from pixeltable.catalog.tbl_ops import CreateTableMdOp, TableOp

        # notice needs_xact that is no longer included in the output of to_dict
        # however, for backward compatibility it needs to continue to be accepted
        op = TableOp.from_dict(
            {
                'op_sn': 0,
                'status': 0,
                'tbl_id': 'b8037eea-404d-47c9-97fc-b4976bbb5466',
                'num_ops': 2,
                '_classname': 'CreateTableMdOp',
                'needs_xact': True,
            }
        )
        assert isinstance(op, CreateTableMdOp)
        assert op.needs_xact  # now a ClassVar
        assert 'needs_xact' not in op.to_dict()

    def test_finalize_pending_ops_retriable_error(self, uses_db: None, fault_injection: None) -> None:
        t = pxt.create_table('test', {'a': pxt.Int})
        exc = sql_exc.DBAPIError('', {}, orig=psycopg.errors.SerializationFailure())
        fault = ExceptionFault(exc)
        get_runtime().fault_manager.inject_fault(FaultLocation.CATALOG_FINALIZE_PENDING_OPS_NON_XACT, fault)
        t.add_column(b=pxt.Int)
        fault.assert_count(1)
        _ = t.select(t.b).collect()

    def test_finalize_pending_ops_non_retriable_error(self, uses_db: None, fault_injection: None) -> None:
        t = pxt.create_table('test', {'a': pxt.Int})
        # Inject a non-retriable error into LoadViewOp. LoadViewOp is the last of 3 ops that constitute a view creation.
        # Upon catching the injected error, the catalog should abort view creation, and undo the first two ops that
        # were already executed.
        exc = Exception('injected')
        fault = ExceptionFault(exc, recurring=True)
        get_runtime().fault_manager.inject_fault(FaultLocation.CATALOG_LOAD_VIEW_OP_EXEC, fault)

        with pxt_raises(code=excs.ErrorCode.INTERNAL_ERROR, match=str(exc)):
            _ = pxt.create_view('view', t.where(t.a > 0))
        fault.assert_count(1)

        # Check that view is not in catalog
        ls = pxt.ls()
        assert len(ls) == 1, ls
        assert ls['Name'].iloc[0] == 'test', ls

    def test_concurrent_add_column_insert(self, uses_db: None, fault_injection: None) -> None:
        """Concurrent insert while add_column is blocked mid-finalize"""
        t = pxt.create_table('test', {'a': pxt.Int})
        fault = BlockFault()

        (
            MultiThreadedScenario()
            # Thread 0: arm the fault in pending table ops finalization
            .then_inject_fault(thread_id=0, loc=FaultLocation.CATALOG_FINALIZE_PENDING_OPS_NON_XACT, fault=fault)
            # Thread 0: start adding a computed column, this will block at the fault point
            .then_run_until(
                thread_id=0, name='add column', event=fault.reached, fn=lambda: t.add_computed_column(b=t.a + 1)
            )
            # Thread 1: run an insert concurrently with add column in thread 0 once thread 0 is in finalize pending ops
            # point
            .then_run(thread_id=1, name='insert', fn=lambda: t.insert([{'a': 1}]))
            # Unblock thread 0
            .then_unblock(thread_id=1, fault=fault)
            .execute()
        )

        # Both operations should have completed successfully.
        result = t.select(t.a, t.b).collect()
        assert len(result) == 1
        assert result[0] == {'a': 1, 'b': 2}

    def test_create_view_stale_base_tv_after_txn_failure(self, uses_db: None, fault_injection: None) -> None:
        """
        Verifies bug fix: due to an error in view creation, Catalog would fail to invalidate a modified but not
        persisted TableVersion. Later that would result in Pixeltable acting on that stale TableVersion, which can cause
        various sorts of issues including a data corruption.
        """
        base = pxt.create_table('base', {'a': pxt.Int})

        injected_exc = Exception('injected error')

        def create_view() -> None:
            with pytest.raises(Exception, match='injected'):
                pxt.create_view('va', base)

        (
            MultiThreadedScenario()
            # Thread 0: Warm up its catalog so base's tv is cached.
            .then_run(thread_id=0, name='warm up cache', fn=lambda: pxt.get_table('base'))
            # Thread 0: Arm a non-retriable exception fault inside create_view.
            .then_inject_fault(
                thread_id=0,
                loc=FaultLocation.CATALOG_CREATE_VIEW_BEFORE_MD_COMMITTED,
                fault=ExceptionFault(injected_exc),
            )
            # Thread 0: Run create_view (va) that fails. Before the fix, base_tv was not added to _modified_tvs, so it
            # stays in cache with stale in-memory state, i.e. with view_sn=v+1
            .then_run(thread_id=0, name='create view that fails', fn=create_view)
            # Thread 1: Create view vb on the same base. This also advances the persisted view_sn to v+1, which
            # automatically matches thread 0's stale cached value.
            .then_run(thread_id=1, name='create view', fn=lambda: pxt.create_view('vb', base))
            # Thread 0: insert into base table. Before the fix, Catalog would observe that the cached TableVersion's
            # version and view_sn match the stored values, and based on that decide to skip reloading the table.
            # The outcome of that is the write is not propagated to vb.
            .then_run(thread_id=0, name='insert into base (stale cache)', fn=lambda: base.insert([{'a': 42}]))
            .execute()
        )

        assert base.count() == 1
        # Verify that the insert was propagated to vb.
        assert pxt.get_table('vb').count() == 1

    def test_load_view_concurrent_drop_view(self, uses_db: None, fault_injection: None) -> None:
        """
        Start with a base table and a view. Thread 0 loads the view md, and is about to initialize it when Thread 1
        drops it. Thread 0 then proceeds to initialize the view, which involves loading the base table. At READ
        COMMITTED isolation level, this scenario results in an AssertionError because the base table and
        the view are inconsistent with one another.
        """
        base = pxt.create_table('base', {'a': pxt.Int})
        v = pxt.create_view('v', base)
        block_before_init = BlockFault()

        (
            MultiThreadedScenario()
            .then_inject_fault(
                thread_id=0, loc=FaultLocation.CATALOG_LOAD_TBL_VERSION_BEFORE_INIT, fault=block_before_init
            )
            .then_run_until(thread_id=0, name='access column', event=block_before_init.reached, fn=lambda: v.a)
            # Thread 1: drop v while Thread 0 is waiting to initialize it
            .then_run(thread_id=1, name='drop view', fn=lambda: pxt.drop_table('v'))
            # unblock Thread 0 to continue with v initialization that also loads base
            .then_unblock(thread_id=1, fault=block_before_init)
            .execute()
        )

        assert pxt.get_table('v', if_not_exists='ignore') is None
        base.insert([{'a': 1}])
        assert base.count() == 1

    def test_drop_view_concurrent_insert(self, uses_db: None, fault_injection: None) -> None:
        """
        Start with a base table and a view. Thread 0 begins to drop the view, but pauses inside finalize pending ops
        (without the exclusive lock). Thread 1 swoops in in the meantime to insert a row into the base table, and
        finalizes view drop as a side effect. Before the fix, this would result in the insert failing with "table not
        found" error.
        """
        base = pxt.create_table('base', {'a': pxt.Int})
        _ = pxt.create_view('v', base)
        block_in_finalize = BlockFault()

        (
            MultiThreadedScenario()
            .then_inject_fault(
                thread_id=0, loc=FaultLocation.CATALOG_FINALIZE_PENDING_OPS_NON_XACT, fault=block_in_finalize
            )
            # Thread 0: drop v but block mid-finalize
            .then_run_until(
                thread_id=0, name='drop view', event=block_in_finalize.reached, fn=lambda: pxt.drop_table('v')
            )
            # Thread 1: insert into base
            .then_run(thread_id=1, name='insert into base', fn=lambda: base.insert([{'a': 1}]))
            .then_unblock(thread_id=1, fault=block_in_finalize)
            .execute()
        )

        assert base.count() == 1
        assert pxt.get_table('v', if_not_exists='ignore') is None
