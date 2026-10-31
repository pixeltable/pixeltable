import os
import pathlib
import shutil
from typing import Any

import numpy as np
import PIL.Image
import pytest

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable import catalog
from pixeltable.catalog import table_copy
from pixeltable.catalog.catalog_proxy import CatalogProxy
from pixeltable.catalog.table_proxy import TableProxy
from pixeltable.service.proxy_protocol import LocalFile
from pixeltable.utils.filecache import FileCache

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


@pytest.mark.db_roots('proxy', 'cloud', reason='copy_table() copies a local table into a hosted database')
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

        t2 = pxt.copy_table('src', p('dst'))

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

        t2 = pxt.copy_table('src', p('dst'))
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

    def test_copy_empty_table(self, db_root: DatabaseRoot) -> None:
        p = db_root.make_catalog_path
        t = pxt.create_table('src', {'s': pxt.String})
        t.add_computed_column(upper=t.s.upper())
        t2 = pxt.copy_table(t, p('dst'))
        assert t2.count() == 0
        assert t2.get_versions() == t.get_versions()
        t2.insert([{'s': 'a'}])
        assert t2.select(t2.upper).collect()['upper'] == ['A']

    def test_copy_table_errors(self, db_root: DatabaseRoot, tmp_path: pathlib.Path) -> None:
        p = db_root.make_catalog_path
        t = pxt.create_table('src', {'id': pxt.Int, 'img': pxt.Image})
        pxt.create_view('view', t)
        pxt.create_table(p('existing'), {'id': pxt.Int})

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='only base tables can be copied'):
            pxt.copy_table('view', p('dst'))
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='is not a path in a hosted database'):
            pxt.copy_table('src', 'dst')
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='is not a local table'):
            pxt.copy_table(p('existing'), p('dst'))
        with pxt_raises(pxt.ErrorCode.PATH_ALREADY_EXISTS, match='already exists'):
            pxt.copy_table('src', p('existing'))
        with pxt_raises(pxt.ErrorCode.DIRECTORY_NOT_FOUND, match='does not exist'):
            pxt.copy_table('src', p('no_dir/dst'))

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
            with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match=r"computed column 'r' references table 'src'"):
                pxt.copy_table(src, p('dst'))
            assert pxt.get_table(p('dst'), if_not_exists='ignore') is None
            assert _temporary_copies(db_root) == []

        # a referenced local media file that no longer exists
        img_path = tmp_path / 'img.jpg'
        shutil.copy(get_image_files()[0], img_path)
        t.insert([{'id': 0, 'img': str(img_path)}])
        img_path.unlink()
        with pxt_raises(pxt.ErrorCode.STORAGE_NOT_FOUND, match='no longer exist'):
            pxt.copy_table('src', p('dst'))
        assert pxt.get_table(p('dst'), if_not_exists='ignore') is None

    def test_copy_table_missing_udf(self, db_root: DatabaseRoot) -> None:
        p = db_root.make_catalog_path
        t = pxt.create_table('src', {'n': pxt.Int})
        t.add_computed_column(n1=local_only_udf(t.n))
        t.insert([{'n': 1}])
        with pxt_raises(pxt.ErrorCode.FUNCTION_NOT_FOUND, match=r"computed column 'n1'(.|\n)*pxt db update"):
            pxt.copy_table('src', p('dst'))
        assert pxt.get_table(p('dst'), if_not_exists='ignore') is None
        assert _temporary_copies(db_root) == []

    def test_copy_table_abort(self, db_root: DatabaseRoot, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failed copy leaves nothing behind in the destination."""
        p = db_root.make_catalog_path
        t = pxt.create_table('src', {'id': pxt.Int, 'img': pxt.Image})
        t.insert({'id': i, 'img': f} for i, f in enumerate(get_image_files()[:3]))

        move = CatalogProxy.move

        def move_after_conflict(self: CatalogProxy, *args: Any, **kwargs: Any) -> None:
            # someone else creates the destination while the rows are being copied
            pxt.create_table(p('dst'), {'other': pxt.String})
            move(self, *args, **kwargs)

        monkeypatch.setattr(CatalogProxy, 'move', move_after_conflict)
        with pxt_raises(pxt.ErrorCode.PATH_ALREADY_EXISTS, match='already exists'):
            pxt.copy_table('src', p('dst'))
        assert pxt.get_table(p('dst')).columns() == ['other']
        assert _temporary_copies(db_root) == []

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
            pxt.copy_table('src', p('dst'))
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
            {'rows_file': daemon_file, 'new_media': {}, 'prev_media': {}, 'num_prev_rows': 0},
            {
                'rows_file': LocalFile(str(rows_file)),
                'new_media': {'file:///img.jpg': daemon_file},
                'prev_media': {},
                'num_prev_rows': 0,
            },
        ):
            with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match='must be uploaded files'):
                t._dispatch('load_store_rows', args)

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
            pxt.copy_table('src', p('dst'))
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
        with pxt_raises(pxt.ErrorCode.INVALID_STATE, match='a call was repeated or skipped'):
            pxt.copy_table('src', p('dst'))
        assert pxt.get_table(p('dst'), if_not_exists='ignore') is None
        assert _temporary_copies(db_root) == []
