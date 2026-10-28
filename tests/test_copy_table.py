import json
import os
import pathlib
import shutil
import uuid
from typing import Any

import numpy as np
import PIL.Image
import pytest
import sqlalchemy as sql

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.catalog import table_copy
from pixeltable.catalog.catalog_proxy import CatalogProxy
from pixeltable.metadata import schema
from pixeltable.runtime import get_runtime
from pixeltable.service import proxy_dispatch, proxy_protocol
from pixeltable.utils.filecache import FileCache

from .utils import DatabaseRoot, get_image_files, pxt_raises

# Defined only in this process, so that a hosted database (and the local proxy daemon) cannot load it; see
# test_cloud_errors.py.
if not os.environ.get('PIXELTABLE_PROXY_DAEMON') and not os.environ.get('PXTCLOUD_DB'):

    @pxt.udf
    def local_only_udf(n: int) -> int:
        return n + 1


def _pending_imports(db_root: DatabaseRoot) -> list[Any]:
    """The ids of incomplete copies in the destination; only the local proxy daemon shares this process's database."""
    assert db_root.id == 'proxy'
    with get_runtime().begin_xact() as conn:
        return list(
            conn.execute(
                sql.select(schema.Table.id).where(schema.Table.additional_md.has_key('pending_import'))
            ).scalars()
        )


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
        import_table_rows = CatalogProxy.import_table_rows

        def count_chunks(self: CatalogProxy, *args: Any, **kwargs: Any) -> dict[str, str]:
            nonlocal num_chunks
            num_chunks += 1
            return import_table_rows(self, *args, **kwargs)

        monkeypatch.setattr(CatalogProxy, 'import_table_rows', count_chunks)

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
        if db_root.id == 'proxy':
            assert _pending_imports(db_root) == []

    def test_copy_table_abort(self, db_root: DatabaseRoot, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failed copy leaves nothing behind in the destination."""
        p = db_root.make_catalog_path
        t = pxt.create_table('src', {'id': pxt.Int, 'img': pxt.Image})
        t.insert({'id': i, 'img': f} for i, f in enumerate(get_image_files()[:3]))

        finish_table_import = CatalogProxy.finish_table_import

        def finish_after_conflict(self: CatalogProxy, *args: Any, **kwargs: Any) -> pxt.Table:
            # someone else creates the destination while the rows are being copied
            pxt.create_table(p('dst'), {'other': pxt.String})
            return finish_table_import(self, *args, **kwargs)

        monkeypatch.setattr(CatalogProxy, 'finish_table_import', finish_after_conflict)
        with pxt_raises(pxt.ErrorCode.PATH_ALREADY_EXISTS, match='already exists'):
            pxt.copy_table('src', p('dst'))
        assert pxt.get_table(p('dst')).columns() == ['other']
        if db_root.id == 'proxy':
            assert _pending_imports(db_root) == []


def test_import_table_rows_takes_only_uploaded_files(init_env: None, tmp_path: pathlib.Path) -> None:
    """The daemon refuses a rows or media file that a request names by a path of the daemon's own."""
    daemon_file = tmp_path / 'daemon_file'
    daemon_file.write_bytes(b'secret')
    uploaded_rows = {'$pxt': 'file', 'name': 'rows.parquet', 'v': 0}
    for rows_file, new_media in ((str(daemon_file), {}), (uploaded_rows, {'file:///img.jpg': str(daemon_file)})):
        request = proxy_protocol.ProxyRequest(
            class_name='CatalogBase',
            method='import_table_rows',
            args={
                'import_id': {'$pxt': 'UUID', 'v': str(uuid.uuid4())},
                'rows_file': rows_file,
                'new_media': new_media,
                'prev_media': {},
            },
        )
        head, _ = proxy_protocol.decode_body(proxy_dispatch.handle(request.model_dump_json(), [b'rows']))
        error = json.loads(head)['error']
        assert error['error_code'] == 'INVALID_ARGUMENT'
        assert 'must be uploaded files' in error['message']
