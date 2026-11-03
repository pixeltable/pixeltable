import os
import pathlib
import shutil
import uuid
from typing import Any

import numpy as np
import PIL.Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable import catalog
from pixeltable.catalog import table_copy
from pixeltable.catalog.catalog_proxy import CatalogProxy
from pixeltable.catalog.table_proxy import TableProxy
from pixeltable.runtime import get_runtime
from pixeltable.service.proxy_protocol import LocalFile
from pixeltable.utils.filecache import FileCache
from pixeltable.utils.object_stores import ObjectOps, StorageTarget

from .utils import DatabaseRoot, get_image_files, pxt_raises

# Defined only in this process, so that a hosted database (and the local proxy daemon) cannot load it; see
# test_cloud_errors.py.
if not os.environ.get('PIXELTABLE_PROXY_DAEMON') and not os.environ.get('PXTCLOUD_DB'):

    @pxt.udf
    def local_only_udf(n: int) -> int:
        return n + 1


def _temporary_copies(db_root: DatabaseRoot) -> list[str]:
    """The tables in the destination that a copy fills before moving them to their path."""
    return [path for path in pxt.list_tables(db_root.prefix) if '_copy_' in path]


def _assert_same_rows(rows1: list[dict[str, Any]], rows2: list[dict[str, Any]]) -> None:
    assert len(rows1) == len(rows2)
    for row1, row2 in zip(rows1, rows2):
        assert row1.keys() == row2.keys()
        for name, val1 in row1.items():
            val2 = row2[name]
            if isinstance(val1, np.ndarray):
                assert np.array_equal(val1, val2), name
            elif isinstance(val1, PIL.Image.Image):
                assert np.array_equal(np.asarray(val1), np.asarray(val2)), name
            else:
                assert val1 == val2, name


@pytest.mark.db_roots('proxy', 'cloud', reason='create_table() copies a local table into a hosted database')
class TestCopyTable:
    def test_copy_table(self, db_root: DatabaseRoot, is_data_versioned: bool, local_embed: pxt.Function) -> None:
        p = db_root.make_catalog_path
        img_files = get_image_files()[:6]
        rng = np.random.default_rng(0)
        t = pxt.create_table(
            'src',
            {'id': pxt.Int, 's': pxt.String, 'img': pxt.Image, 'arr': pxt.Array[(32, 32), np.float32], 'j': pxt.Json},
            _is_data_versioned=is_data_versioned,
        )
        # a nondeterministic column: a copy that computed it again would not reproduce its values
        t.add_computed_column(u=pxtf.uuid.uuid4())
        t.add_computed_column(upper=t.s.upper())
        t.add_computed_column(rot=t.img.rotate(90))  # a computed image, stored in the local media dir
        t.add_computed_column(ratio=(t.j.f - 2) / (t.j.f - 2), on_error='ignore')  # an error for id == 2
        t.add_embedding_index(t.s, embedding=local_embed)

        def row(i: int) -> dict[str, Any]:
            # each array is larger than the store's inline limit, so it is stored in a file
            arr = rng.random((32, 32), dtype=np.float32)
            return {'id': i, 's': f'text {i}', 'img': img_files[i % len(img_files)], 'arr': arr, 'j': {'f': i}}

        t.insert((row(i) for i in range(4)), on_error='ignore')
        t.insert((row(i) for i in range(4, 6)), on_error='ignore')
        t.update({'s': 'updated'}, where=t.id == 1)
        t.delete(where=t.id == 0)
        # a dropped column: its values remain part of the history
        t.add_column(note=pxt.String | None)
        t.update({'note': 'a note'}, where=t.id == 3)
        t.drop_column('note')

        cols = ('id', 's', 'u', 'upper', 'arr', 'j', 'img', 'rot', 'ratio')
        select_list = [*(t[col] for col in cols), t.ratio.errortype, t.ratio.errormsg]
        src_rows = t.select(*select_list).order_by(t.id).collect()
        src_versions = t.get_versions()
        src_urls = set(t.select(t.img.fileurl).collect()['img_fileurl']) | set(
            t.select(t.rot.fileurl).collect()['rot_fileurl']
        )

        t2 = pxt.create_table(p('dst'), source=pxt.get_table('src'))

        assert t2.count() == t.count() == 5
        _assert_same_rows(
            list(
                t2.select(*(t2[col] for col in cols), t2.ratio.errortype, t2.ratio.errormsg).order_by(t2.id).collect()
            ),
            list(src_rows),
        )
        assert sum(1 for r in src_rows if r['ratio_errortype'] is not None) == 1
        # every media file was uploaded: the copy's urls are its own
        dst_urls = set(t2.select(t2.img.fileurl).collect()['img_fileurl']) | set(
            t2.select(t2.rot.fileurl).collect()['rot_fileurl']
        )
        assert src_urls.isdisjoint(dst_urls)
        # the embeddings were copied
        sim = t.s.similarity(string='text 3')
        sim2 = t2.s.similarity(string='text 3')
        assert list(t2.select(t2.id, score=sim2).order_by(sim2, asc=False).collect()) == list(
            t.select(t.id, score=sim).order_by(sim, asc=False).collect()
        )

        if is_data_versioned:
            assert t2.get_versions() == src_versions
            for version in (v['version'] for v in src_versions):
                src_version = pxt.get_table(f'src:{version}')
                dst_version = pxt.get_table(f'{p("dst")}:{version}')
                assert dst_version.columns() == src_version.columns()
                _assert_same_rows(
                    list(dst_version.order_by(dst_version.id).collect()),
                    list(src_version.order_by(src_version.id).collect()),
                )

        # the copy doesn't depend on the local table, whose computed images go with it
        pxt.drop_table('src')
        FileCache.get().clear()  # so that reading the images fetches them from the destination
        _assert_same_rows(
            list(t2.select(t2.id, t2.img, t2.rot).order_by(t2.id).collect()),
            [{'id': r['id'], 'img': r['img'], 'rot': r['rot']} for r in src_rows],
        )
        # the copy computes new rows
        t2.insert([row(10)])
        new_row = t2.where(t2.id == 10).select(t2.upper, t2.u, t2.rot).collect()[0]
        assert new_row['upper'] == 'TEXT 10'
        assert new_row['u'] is not None
        assert new_row['rot'] is not None

    def test_copy_table_chunks(self, db_root: DatabaseRoot, monkeypatch: pytest.MonkeyPatch) -> None:
        """Rows in several chunks, which reference media files that earlier chunks uploaded."""
        p = db_root.make_catalog_path
        # a chunk of at most 2 rows, which ends at a row that references a new media file
        monkeypatch.setattr(table_copy, '_ROW_BATCH_SIZE', 2)
        monkeypatch.setattr(table_copy, '_MAX_CHUNK_ROW_BYTES', 1)
        monkeypatch.setattr(table_copy, '_MAX_CHUNK_MEDIA_BYTES', 1)
        num_chunks = 0
        dispatch = TableProxy._dispatch

        def count_chunks(self: TableProxy, method: str, args: dict[str, Any]) -> Any:
            nonlocal num_chunks
            num_chunks += method == 'load_store_rows'
            return dispatch(self, method, args)

        monkeypatch.setattr(TableProxy, '_dispatch', count_chunks)

        img_files = get_image_files()[:2]
        t = pxt.create_table('src', {'id': pxt.Int, 'img': pxt.Image})
        t.add_computed_column(rot=t.img.rotate(90))
        # both versions reference the same two input files
        t.insert({'id': i, 'img': img_files[i % 2]} for i in range(5))
        t.insert({'id': i, 'img': img_files[i % 2]} for i in range(5, 9))

        t2 = pxt.create_table(p('dst'), source=pxt.get_table('src'))
        assert num_chunks > 2
        _assert_same_rows(
            list(t2.select(t2.id, t2.img, t2.rot).order_by(t2.id).collect()),
            list(t.select(t.id, t.img, t.rot).order_by(t.id).collect()),
        )
        # each input file was uploaded once
        assert len(set(t2.select(t2.img.fileurl).collect()['img_fileurl'])) == 2
        # an uploaded file is stored under the earliest version that references it, so reverting the second insert
        # keeps the files of the first
        t2.revert()
        FileCache.get().clear()  # so that reading the images fetches them from the destination
        _assert_same_rows(
            list(t2.select(t2.id, t2.img, t2.rot).order_by(t2.id).collect()),
            list(t.where(t.id < 5).select(t.id, t.img, t.rot).order_by(t.id).collect()),
        )

    def test_copy_table_source_bucket_media(self, db_root: DatabaseRoot, monkeypatch: pytest.MonkeyPatch) -> None:
        """An object that the source stores under its table prefix in a bucket is uploaded, since the source's
        drop_table() and revert() delete that prefix; a remote url outside of it is kept."""
        p = db_root.make_catalog_path
        img_file = get_image_files()[0]
        downloaded: list[tuple[str, bool]] = []  # (url, whether a transaction is open)

        def download(url: str, dest_path: pathlib.Path) -> None:
            downloaded.append((url, get_runtime().in_xact))
            shutil.copy(img_file, dest_path)

        # a bucket whose every object is img_file
        monkeypatch.setattr(ObjectOps, 'copy_object_to_local_file', download)
        t = pxt.create_table('src', {'img': pxt.Image})
        prefix = t._id.hex
        source_url = f's3://bucket/media/{prefix}/ab/abcd/{prefix}_1_1_{uuid.uuid4().hex}.jpg'
        other_url = f's3://bucket/media/{uuid.uuid4().hex}.jpg'
        t.insert([{'img': source_url}, {'img': other_url}])
        downloaded.clear()
        t2 = pxt.create_table(p('dst'), source=pxt.get_table('src'))
        # the export's transaction is closed: a download must not hold its locks and snapshot
        assert downloaded == [(source_url, False)]
        urls = t2.select(t2.img.fileurl).collect()['img_fileurl']
        assert other_url in urls and source_url not in urls
        # the copy reads its own bytes of the object
        pxt.drop_table('src')
        FileCache.get().clear()
        img = t2.where(t2.img.fileurl != other_url).select(t2.img).collect()[0]['img']
        assert np.array_equal(np.asarray(img), np.asarray(PIL.Image.open(img_file)))

    @pytest.mark.very_expensive
    def test_copy_table_bucket_destination(self, db_root: DatabaseRoot) -> None:
        """The copy keeps its own copy of the files that the source wrote to a bucket."""
        from .test_destination import TestDestination  # imported here, so that pytest doesn't collect it twice

        p = db_root.make_catalog_path
        dest_uri = TestDestination.resolve_destination_uri(StorageTarget.S3_STORE)
        t = pxt.create_table('src', {'id': pxt.Int, 'img': pxt.Image})
        t.add_computed_column(rot=t.img.rotate(90), destination=f'{dest_uri}/copy_table')
        t.insert({'id': i, 'img': f} for i, f in enumerate(get_image_files()[:2]))
        src_rows = list(t.select(t.id, t.rot).order_by(t.id).collect())
        t2 = pxt.create_table(p('dst'), source=pxt.get_table('src'))
        pxt.drop_table('src')
        FileCache.get().clear()
        _assert_same_rows(list(t2.select(t2.id, t2.rot).order_by(t2.id).collect()), src_rows)

    def test_copy_empty_table(self, db_root: DatabaseRoot) -> None:
        p = db_root.make_catalog_path
        t = pxt.create_table('src', {'s': pxt.String})
        t.add_computed_column(upper=t.s.upper())
        t2 = pxt.create_table(p('dst'), source=t)
        assert t2.count() == 0
        assert t2.get_versions() == t.get_versions()
        t2.insert([{'s': 'a'}])
        assert t2.select(t2.upper).collect()['upper'] == ['A']

    def test_copy_table_errors(self, db_root: DatabaseRoot, tmp_path: pathlib.Path) -> None:
        p = db_root.make_catalog_path
        kwargs: dict[str, Any]  # a case's table, as source or as schema_source
        t = pxt.create_table('src', {'id': pxt.Int, 'img': pxt.Image})
        pxt.create_view('view', t)
        pxt.create_table(p('existing'), {'id': pxt.Int})

        # a view, as source and as schema_source
        for like_arg in ('source', 'schema_source'):
            kwargs = {like_arg: pxt.get_table('view')}
            with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='must be a base table on this machine'):
                pxt.create_table(p('dst'), **kwargs)
        # a hosted table as schema_source
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='must be a base table on this machine'):
            pxt.create_table(p('dst'), schema_source=pxt.get_table(p('existing')))
        # a query of another database, refused before anything is created (PXT-1718)
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='from a query of another database'):
            pxt.create_table(p('dst'), source=t.select(t.id))
        # arguments that the new table takes from the source table
        for like_arg in ('source', 'schema_source'):
            kwargs = {like_arg: t}
            with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match='cannot be combined with `comment`, `primary_key`'):
                pxt.create_table(p('dst'), primary_key='id', comment='c', **kwargs)
        # schema_source with another schema
        with pxt_raises(pxt.ErrorCode.MISSING_REQUIRED, match='Exactly one of `schema`, `source` or `schema_source`'):
            pxt.create_table(p('dst'), {'id': pxt.Int}, schema_source=t)
        assert pxt.get_table(p('dst'), if_not_exists='ignore') is None

        with pxt_raises(pxt.ErrorCode.PATH_ALREADY_EXISTS, match='already exists'):
            pxt.create_table(p('existing'), source=pxt.get_table('src'))
        with pxt_raises(pxt.ErrorCode.DIRECTORY_NOT_FOUND, match='does not exist'):
            pxt.create_table(p('no_dir/dst'), source=pxt.get_table('src'))

        # a computed column that references another table, which the destination doesn't have, through a @pxt.query
        # that is not the root of its expression
        @pxt.query
        def lookup(n: int) -> pxt.Query:
            return t.where(t.id == n).select(t.id)

        t2 = pxt.create_table('src2', {'n': pxt.Int})
        t2.add_computed_column(r=lookup(t2.n)[0])
        # ... or through a @pxt.query that the where clause of another @pxt.query calls
        t3 = pxt.create_table('src3', {'n': pxt.Int})

        @pxt.query
        def outer(n: int) -> pxt.Query:
            return t3.where(lookup(t3.n)[0] == n).select(t3.n)

        t3.add_computed_column(r=outer(t3.n))
        for src in ('src2', 'src3'):
            for like_arg in ('source', 'schema_source'):
                kwargs = {like_arg: pxt.get_table(src)}
                with pxt_raises(
                    pxt.ErrorCode.UNSUPPORTED_OPERATION, match=r"computed column 'r' references table 'src'"
                ):
                    pxt.create_table(p('dst'), **kwargs)
                assert pxt.get_table(p('dst'), if_not_exists='ignore') is None
                assert _temporary_copies(db_root) == []
        # in this machine's database, a table like src2 can reference src
        like_src2 = pxt.create_table('like_src2', schema_source=t2)
        assert like_src2.columns() == t2.columns()

        # a referenced local media file that no longer exists
        img_path = tmp_path / 'img.jpg'
        shutil.copy(get_image_files()[0], img_path)
        t.insert([{'id': 0, 'img': str(img_path)}])
        img_path.unlink()
        with pxt_raises(pxt.ErrorCode.STORAGE_NOT_FOUND, match='no longer exist'):
            pxt.create_table(p('dst'), source=pxt.get_table('src'))
        assert pxt.get_table(p('dst'), if_not_exists='ignore') is None

    def test_copy_table_missing_udf(self, db_root: DatabaseRoot) -> None:
        p = db_root.make_catalog_path
        t = pxt.create_table('src', {'n': pxt.Int})
        t.add_computed_column(n1=local_only_udf(t.n))
        t.insert([{'n': 1}])
        with pxt_raises(pxt.ErrorCode.FUNCTION_NOT_FOUND, match=r"computed column 'n1'(.|\n)*pxt db update"):
            pxt.create_table(p('dst'), source=pxt.get_table('src'))
        assert pxt.get_table(p('dst'), if_not_exists='ignore') is None
        assert _temporary_copies(db_root) == []
        with pxt_raises(pxt.ErrorCode.FUNCTION_NOT_FOUND):
            pxt.create_table(p('dst'), schema_source=t)
        assert pxt.get_table(p('dst'), if_not_exists='ignore') is None

    def test_copy_table_if_exists(self, db_root: DatabaseRoot, monkeypatch: pytest.MonkeyPatch) -> None:
        p = db_root.make_catalog_path
        t = pxt.create_table('src', {'id': pxt.Int})
        t.insert({'id': i} for i in range(3))
        pxt.create_table(p('dst'), {'other': pxt.String}).insert([{'other': 'a'}])

        # 'ignore' returns the existing table as it is
        t2 = pxt.create_table(p('dst'), source=t, if_exists='ignore')
        assert t2.columns() == ['other']

        # a failed 'replace' leaves the existing table as it is: the copy replaces it only when it finishes
        dispatch = TableProxy._dispatch

        def fail_load(self: TableProxy, method: str, args: dict[str, Any]) -> Any:
            if method == 'load_store_rows':
                raise pxt.RequestError(pxt.ErrorCode.INVALID_STATE, 'injected failure')
            return dispatch(self, method, args)

        with monkeypatch.context() as m:
            m.setattr(TableProxy, '_dispatch', fail_load)
            with pxt_raises(pxt.ErrorCode.INVALID_STATE, match='injected failure'):
                pxt.create_table(p('dst'), source=t, if_exists='replace')
        assert pxt.get_table(p('dst')).collect()['other'] == ['a']
        assert _temporary_copies(db_root) == []

        # 'replace' replaces it with the copy
        t2 = pxt.create_table(p('dst'), source=t, if_exists='replace')
        assert t2.columns() == ['id']
        assert sorted(t2.collect()['id']) == [0, 1, 2]

    def test_create_table_like(self, db_root: DatabaseRoot, is_data_versioned: bool, local_embed: pxt.Function) -> None:
        p = db_root.make_catalog_path
        t = pxt.create_table(
            'src',
            {'id': {'type': pxt.Int, 'primary_key': True}, 's': pxt.String, 'img': pxt.Image | None},
            comment='source table',
            custom_metadata={'k': 'v'},
            media_validation='on_read',
            _is_data_versioned=is_data_versioned,
        )
        t.add_computed_column(upper=t.s.upper(), comment='upper case')
        t.add_computed_column(rot=t.img.rotate(90), stored=False)
        t.add_embedding_index(t.s, embedding=local_embed, metric='ip', idx_name='s_idx')
        t.add_btree_index(t.upper)
        t.insert({'id': i, 's': f'text {i}'} for i in range(3))

        def schema_md(tbl: pxt.Table) -> dict[str, Any]:
            md = tbl.get_metadata()
            # leave out the fields that hold the table's name or history
            columns = {
                name: {
                    **col_md,
                    'version_added': None,
                    'defined_in': None,
                    'depends_on': [col_name for _, col_name in col_md['depends_on'] or []],
                }
                for name, col_md in md['columns'].items()
            }
            settings = {
                'comment': md['comment'],
                'custom_metadata': md['custom_metadata'],
                'media_validation': md['media_validation'],
                'is_data_versioned': md['is_data_versioned'],
            }
            return {'columns': columns, 'indexes': md['indexes'], **settings}

        # in a hosted database, and in this machine's database
        for path in (p('dst'), 'like_src'):
            t2 = pxt.create_table(path, schema_source=t)
            assert schema_md(t2) == schema_md(t)
            assert t2.count() == 0
            assert len(t2.get_versions()) == 1
            # the new table computes its computed columns and indexes its rows
            t2.insert({'id': i, 's': f'text {i}'} for i in range(3))
            assert t2.order_by(t2.id).select(t2.upper).collect()['upper'] == ['TEXT 0', 'TEXT 1', 'TEXT 2']
            sim = t.s.similarity(string='text 1')
            sim2 = t2.s.similarity(string='text 1')
            assert list(t2.select(t2.id).order_by(sim2, asc=False).collect()) == list(
                t.select(t.id).order_by(sim, asc=False).collect()
            )

        # if_exists: 'error' refuses the existing table, 'ignore' returns it, 'replace' creates an empty one
        with pxt_raises(pxt.ErrorCode.PATH_ALREADY_EXISTS, match="'dst' is an existing table"):
            pxt.create_table(p('dst'), schema_source=t)
        assert pxt.create_table(p('dst'), schema_source=t, if_exists='ignore').count() == 3
        assert pxt.create_table(p('dst'), schema_source=t, if_exists='replace').count() == 0

    def test_copy_table_abort(self, db_root: DatabaseRoot, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failed copy leaves nothing behind in the destination."""
        p = db_root.make_catalog_path
        t = pxt.create_table('src', {'id': pxt.Int, 'img': pxt.Image})
        t.insert({'id': i, 'img': f} for i, f in enumerate(get_image_files()[:3]))

        dispatch = TableProxy._dispatch

        def finish_after_conflict(self: TableProxy, method: str, args: dict[str, Any]) -> Any:
            if method == 'finish_copy':
                # someone else creates the destination while the rows are being copied
                pxt.create_table(p('dst'), {'other': pxt.String})
            return dispatch(self, method, args)

        monkeypatch.setattr(TableProxy, '_dispatch', finish_after_conflict)
        with pxt_raises(pxt.ErrorCode.PATH_ALREADY_EXISTS, match='already exists'):
            pxt.create_table(p('dst'), source=pxt.get_table('src'))
        assert pxt.get_table(p('dst')).columns() == ['other']
        assert _temporary_copies(db_root) == []

    def test_copy_table_backfilled_media_version(self, db_root: DatabaseRoot) -> None:
        """The files that a computed column wrote for the rows it filled are stored under the version that added it."""
        p = db_root.make_catalog_path
        t = pxt.create_table('src', {'img': pxt.Image}, _is_data_versioned=True)
        t.insert({'img': f} for f in get_image_files()[:2])
        t.add_computed_column(rot=t.img.rotate(90))  # version 2, which fills the rows of version 1
        t2 = pxt.create_table(p('dst'), source=pxt.get_table('src'))

        def file_versions(tbl: pxt.Table) -> set[int]:
            # a stored file is named <table prefix>_<column id>_<table version>_<suffix>
            urls = tbl.select(tbl.rot.fileurl).collect()['rot_fileurl']
            return {int(pathlib.PurePosixPath(url).name.split('_')[2]) for url in urls}

        assert file_versions(t2) == file_versions(t) == {2}

    def test_copy_table_tmp_path_exists(self, db_root: DatabaseRoot, monkeypatch: pytest.MonkeyPatch) -> None:
        """A copy refuses a temporary path that names an existing table, and leaves that table as it is."""
        p = db_root.make_catalog_path
        t = pxt.create_table('src', {'id': pxt.Int})
        t.insert([{'id': 1}])

        get_table = CatalogProxy.get_table
        occupied: list[str] = []

        def occupy_tmp_path(self: CatalogProxy, path: catalog.Path, *args: Any) -> Any:
            if '_copy_' in path.name and len(occupied) == 0:
                pxt.create_table(str(path), {'other': pxt.String})
                occupied.append(str(path))
            return get_table(self, path, *args)

        monkeypatch.setattr(CatalogProxy, 'get_table', occupy_tmp_path)
        with pxt_raises(pxt.ErrorCode.PATH_ALREADY_EXISTS, match='already exists'):
            pxt.create_table(p('dst'), source=pxt.get_table('src'))
        assert len(occupied) == 1
        assert pxt.get_table(occupied[0]).columns() == ['other']
        assert pxt.get_table(p('dst'), if_not_exists='ignore') is None

    def test_load_store_rows_takes_only_uploaded_files(self, db_root: DatabaseRoot, tmp_path: pathlib.Path) -> None:
        """The destination refuses a rows or media file that a request names by a path of the destination's own."""
        p = db_root.make_catalog_path
        t = pxt.create_table(p('dst'), {'img': pxt.Image})
        assert isinstance(t, TableProxy)
        daemon_file = str(tmp_path / 'daemon_file')
        rows_file = tmp_path / 'rows.parquet'
        rows_file.write_bytes(b'rows')
        for args in (
            {'rows_file': daemon_file, 'new_media': {}, 'prev_media': {}},
            {'rows_file': LocalFile(str(rows_file)), 'new_media': {'file:///img.jpg': daemon_file}, 'prev_media': {}},
        ):
            with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match='must be uploaded files'):
                t._dispatch('load_store_rows', args)

    def test_load_store_rows_only_into_copy_in_progress(self, db_root: DatabaseRoot, tmp_path: pathlib.Path) -> None:
        """A finished copy refuses rows, even rows inside its history."""
        p = db_root.make_catalog_path
        t = pxt.create_table('src', {'id': pxt.Int}, _is_data_versioned=True)
        t.insert({'id': i} for i in range(2))
        t2 = pxt.create_table(p('dst'), source=pxt.get_table('src'))
        assert isinstance(t2, TableProxy)
        assert isinstance(t, catalog.InsertableTable)
        rows = pq.read_table(table_copy._export_table(t, tmp_path).chunks[0].rows_file)
        # a new (rowid, v_min) pair inside the history of the copy
        rows_file = tmp_path / 'rows.parquet'
        pq.write_table(rows.set_column(rows.schema.get_field_index('v_min'), 'v_min', pa.array([0, 0])), rows_file)
        with pxt_raises(pxt.ErrorCode.INVALID_STATE, match='is not a copy in progress'):
            t2._dispatch('load_store_rows', {'rows_file': LocalFile(str(rows_file)), 'new_media': {}, 'prev_media': {}})
        assert t2.count() == 2

    def test_copy_in_progress_refuses_other_writes(
        self, db_root: DatabaseRoot, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """While a copy is in progress, only the copy writes to its table, so the finished copy holds just its rows."""
        p = db_root.make_catalog_path
        t = pxt.create_table('src', {'id': pxt.Int})
        t.insert({'id': i} for i in range(2))
        dispatch = TableProxy._dispatch
        refused: list[bool] = []

        def write_before_finish(self: TableProxy, method: str, args: dict[str, Any]) -> Any:
            if method == 'finish_copy':
                with pxt_raises(pxt.ErrorCode.INVALID_STATE, match=r'is a copy that create_table\(\) has not finished'):
                    self.insert([{'id': 100}])
                with pxt_raises(pxt.ErrorCode.INVALID_STATE, match=r'create_table\(\) has not finished copying it'):
                    pxt.move(str(self._path()), p('moved'))
                refused.append(True)
            return dispatch(self, method, args)

        monkeypatch.setattr(TableProxy, '_dispatch', write_before_finish)
        t2 = pxt.create_table(p('dst'), source=pxt.get_table('src'))
        assert refused == [True]
        assert sorted(t2.collect()['id']) == [0, 1]
        assert pxt.get_table(p('moved'), if_not_exists='ignore') is None

    def test_load_store_rows_refuses_rows_beyond_history(
        self, db_root: DatabaseRoot, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
    ) -> None:
        """A copy refuses a row whose rowid a later insert into the copy would assign."""
        p = db_root.make_catalog_path
        t = pxt.create_table('src', {'id': pxt.Int}, _is_data_versioned=True)
        t.insert({'id': i} for i in range(2))
        dispatch = TableProxy._dispatch

        def shift_rowids(self: TableProxy, method: str, args: dict[str, Any]) -> Any:
            if method == 'load_store_rows':
                rows = pq.read_table(args['rows_file'].path)
                rowid_idx = rows.schema.get_field_index('rowid')
                shifted = pa.array([rowid + 100 for rowid in rows.column(rowid_idx).to_pylist()], pa.int64())
                rows_file = tmp_path / 'shifted_rows.parquet'
                pq.write_table(rows.set_column(rowid_idx, 'rowid', shifted), rows_file)
                args = {**args, 'rows_file': LocalFile(str(rows_file))}
            return dispatch(self, method, args)

        monkeypatch.setattr(TableProxy, '_dispatch', shift_rowids)
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match='is not a row of the history of'):
            pxt.create_table(p('dst'), source=pxt.get_table('src'))
        assert _temporary_copies(db_root) == []

    @pytest.mark.parametrize('tampered_arg', ['new_media', 'prev_media'])
    def test_load_store_rows_refuses_local_urls(
        self, db_root: DatabaseRoot, monkeypatch: pytest.MonkeyPatch, tampered_arg: str
    ) -> None:
        """The destination refuses a row that names one of the destination's own files as its media."""
        p = db_root.make_catalog_path
        # the first chunk ends at the row that uploads the file, so the second chunk references it as prev_media
        monkeypatch.setattr(table_copy, '_MAX_CHUNK_MEDIA_BYTES', 1)
        img_file = get_image_files()[0]
        dispatch = TableProxy._dispatch

        def tamper(self: TableProxy, method: str, args: dict[str, Any]) -> Any:
            if method == 'load_store_rows' and len(args[tampered_arg]) > 0:
                # the local proxy daemon shares this machine's filesystem, so img_file is one of the daemon's own files
                tampered = {} if tampered_arg == 'new_media' else dict.fromkeys(args['prev_media'], img_file)
                args = {**args, tampered_arg: tampered}
            return dispatch(self, method, args)

        monkeypatch.setattr(TableProxy, '_dispatch', tamper)
        t = pxt.create_table('src', {'img': pxt.Image})
        t.insert([{'img': img_file}, {'img': img_file}])
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match='is not a media file of'):
            pxt.create_table(p('dst'), source=pxt.get_table('src'))
        assert _temporary_copies(db_root) == []

    def test_finish_copy_repeated(self, db_root: DatabaseRoot, monkeypatch: pytest.MonkeyPatch) -> None:
        """A repeated finish_copy, as when the client retries a request that committed, finds the copy finished."""
        p = db_root.make_catalog_path
        dispatch = TableProxy._dispatch

        def finish_twice(self: TableProxy, method: str, args: dict[str, Any]) -> Any:
            if method == 'finish_copy':
                dispatch(self, method, args)
            return dispatch(self, method, args)

        monkeypatch.setattr(TableProxy, '_dispatch', finish_twice)
        pxt.create_table('src', {'id': pxt.Int}).insert({'id': i} for i in range(3))
        t2 = pxt.create_table(p('dst'), source=pxt.get_table('src'))
        assert sorted(t2.collect()['id']) == [0, 1, 2]
        assert _temporary_copies(db_root) == []

    def test_load_store_rows_repeated(self, db_root: DatabaseRoot, monkeypatch: pytest.MonkeyPatch) -> None:
        """A chunk that is loaded again, as when the client retries a request that committed, is refused."""
        p = db_root.make_catalog_path
        dispatch = TableProxy._dispatch

        def load_twice(self: TableProxy, method: str, args: dict[str, Any]) -> Any:
            if method == 'load_store_rows':
                dispatch(self, method, args)
            return dispatch(self, method, args)

        monkeypatch.setattr(TableProxy, '_dispatch', load_twice)
        t = pxt.create_table('src', {'id': pxt.Int, 'img': pxt.Image})
        t.insert({'id': i, 'img': f} for i, f in enumerate(get_image_files()[:3]))
        with pxt_raises(pxt.ErrorCode.INVALID_STATE, match='a call was repeated'):
            pxt.create_table(p('dst'), source=pxt.get_table('src'))
        assert pxt.get_table(p('dst'), if_not_exists='ignore') is None
        assert _temporary_copies(db_root) == []
