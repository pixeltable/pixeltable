import os
import pathlib
import shutil
import uuid
from typing import Any

import numpy as np
import PIL.Image
import pytest

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.catalog import table_copy
from pixeltable.config import Config
from pixeltable.env import Env
from pixeltable.runtime import get_runtime
from pixeltable.utils.fault_injection import FaultLocation
from pixeltable.utils.filecache import FileCache
from pixeltable.utils.object_stores import ObjectOps, StorageTarget

from . import test_destination
from .fault_injection import ExceptionFault
from .utils import (
    DatabaseRoot,
    assert_resultset_eq,
    dummy_embedding,
    get_image_files,
    pxt_raises,
    schema_from_tbl_md,
    skip_test_if_not_installed,
)

# These UDFs exist only in the test process, so that a hosted database and the local proxy daemon cannot load them.
if not os.environ.get('PIXELTABLE_PROXY_DAEMON') and not os.environ.get('PXTCLOUD_DB'):

    @pxt.udf
    def local_only_udf(n: int) -> int:
        return n + 1

    @pxt.udf
    def local_only_embed(s: str) -> pxt.Array[(3,), pxt.Float]:
        return np.zeros(3, dtype=np.float32)


class TestCopyTable:
    def test_copy_table(self, db_root: DatabaseRoot, is_data_versioned: bool) -> None:
        skip_test_if_not_installed('imagehash')
        p = db_root.make_catalog_path
        img_files = get_image_files()[:6]
        rng = np.random.default_rng(0)
        t = pxt.create_table(
            'src',
            {'id': pxt.Int, 's': pxt.String, 'img': pxt.Image, 'arr': pxt.Array[(32, 32), np.float32], 'j': pxt.Json},
            comment='source table',
            custom_metadata={'k': 'v'},
            media_validation='on_read',
            _is_data_versioned=is_data_versioned,
        )
        # a nondeterministic column: a copy that computed it again would not reproduce its values
        t.add_computed_column(u=pxtf.uuid.uuid4())
        t.add_computed_column(upper=t.s.upper())
        t.add_computed_column(rot=t.img.rotate(90))  # a computed image, stored in the local media dir
        t.add_computed_column(ratio=(t.j.f - 2) / (t.j.f - 2), on_error='ignore')  # an error for id == 2
        t.add_embedding_index(t.s, embedding=dummy_embedding.using(n=8))

        def row(i: int) -> dict[str, Any]:
            # each array is larger than the store's inline limit, so it is stored in a file
            arr = rng.random((32, 32), dtype=np.float32)
            return {'id': i, 's': f'text {i}', 'img': img_files[i % len(img_files)], 'arr': arr, 'j': {'f': i}}

        t.insert((row(i) for i in range(4)), on_error='ignore')
        t.insert((row(i) for i in range(4, 6)), on_error='ignore')
        t.update({'s': 'updated'}, where=t.id == 1)
        t.delete(where=t.id == 0)
        # the copy succeeds with a dropped column that has a value in a live row
        t.add_column(note=pxt.String | None)
        t.update({'note': 'a note'}, where=t.id == 3)
        t.drop_column('note')

        def rows(tbl: pxt.Table) -> pxt.ResultSet:
            return (
                tbl.select(
                    tbl.id,
                    tbl.s,
                    tbl.u,
                    tbl.upper,
                    tbl.arr,
                    tbl.j,
                    tbl.img,
                    tbl.rot,
                    tbl.ratio,
                    tbl.ratio.errortype,
                    tbl.ratio.errormsg,
                )
                .order_by(tbl.id)
                .collect()
            )

        def media_urls(tbl: pxt.Table) -> set[str]:
            urls = tbl.select(tbl.img.fileurl, tbl.rot.fileurl).collect()
            return set(urls['img_fileurl']) | set(urls['rot_fileurl'])

        src_rows = rows(t)
        t2 = pxt.create_table(p('dst'), source=t)

        assert t2.count() == t.count() == 5
        assert schema_from_tbl_md(t2.get_metadata()) == schema_from_tbl_md(t.get_metadata())
        assert t2.get_metadata()['is_data_versioned'] == is_data_versioned
        assert_resultset_eq(rows(t2), src_rows, compare_col_names=True)
        assert sum(errortype is not None for errortype in src_rows['ratio_errortype']) == 1
        # the copy has its own copy of every media file: no url of t2 is a url of t
        assert media_urls(t).isdisjoint(media_urls(t2))
        # The copy has the embeddings of the source. dummy_embedding returns random vectors, and an embedding that the
        # copy computed again would give other scores.
        query_vec = rng.random(8, dtype=np.float32)
        sim = t.s.similarity(vector=query_vec)
        sim2 = t2.s.similarity(vector=query_vec)
        assert list(t2.select(t2.id, score=sim2).order_by(sim2, asc=False).collect()) == list(
            t.select(t.id, score=sim).order_by(sim, asc=False).collect()
        )
        # Copy t2 to the same database. For a hosted t2, the hosted database makes the copy.
        t3 = pxt.create_table(p('dst2'), source=t2)
        assert_resultset_eq(rows(t3), src_rows, compare_col_names=True)
        assert media_urls(t2).isdisjoint(media_urls(t3))

        # a copy has no history: it has the single version 0
        for copy in (t2, t3):
            assert [v['version'] for v in copy.get_versions()] == [0]
            if is_data_versioned:
                with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='Cannot revert version 0'):
                    copy.revert()

        # the copy still reads its rows after drop_table() deletes the source and its computed images
        pxt.drop_table('src')
        FileCache.get().clear()  # so that reading the images fetches them from the destination
        assert_resultset_eq(rows(t2), src_rows)
        # the copy computes the computed columns of an inserted row
        t2.insert([row(10)])
        new_row = t2.where(t2.id == 10).select(t2.upper, t2.u, t2.rot).collect()[0]
        assert new_row['upper'] == 'TEXT 10'
        assert new_row['u'] is not None
        assert new_row['rot'] is not None

        # an empty table
        empty = pxt.create_table('empty', {'s': pxt.String}, _is_data_versioned=is_data_versioned)
        empty.add_computed_column(upper=empty.s.upper())
        empty2 = pxt.create_table(p('empty_dst'), source=empty)
        assert empty2.count() == 0
        assert [v['version'] for v in empty2.get_versions()] == [0]
        empty2.insert([{'s': 'a'}])
        assert empty2.select(empty2.upper).collect()['upper'] == ['A']

        # a query as source copies only its rows: the computed column becomes a plain one
        rows_only = pxt.create_table(p('rows_only'), source=empty2.select())
        assert rows_only.collect()['upper'] == ['A']
        assert not rows_only.get_metadata()['columns']['upper']['is_computed']

    def test_copy_table_media(self, db_root: DatabaseRoot, monkeypatch: pytest.MonkeyPatch) -> None:
        skip_test_if_not_installed('imagehash')
        p = db_root.make_catalog_path
        img_files = get_image_files()[:2]

        def media_rows(tbl: pxt.Table) -> pxt.ResultSet:
            return tbl.select(tbl.id, tbl.img, tbl.rot).order_by(tbl.id).collect()

        t = pxt.create_table('src', {'id': pxt.Int, 'img': pxt.Image})
        t.add_computed_column(rot=t.img.rotate(90))
        # both versions reference the same two input files
        t.insert({'id': i, 'img': img_files[i % 2]} for i in range(5))
        t.insert({'id': i, 'img': img_files[i % 2]} for i in range(5, 9))
        with monkeypatch.context() as m:
            # The copy exports the rows in several chunks, and a row of a later chunk references a media file of an
            # earlier chunk. With these limits, a chunk ends after 2 rows and at each row with a new media file. The
            # real limits need a table that is too large for a test.
            m.setattr(table_copy, '_ROW_BATCH_SIZE', 2)
            m.setattr(table_copy, '_MAX_CHUNK_ROW_BYTES', 1)
            m.setattr(table_copy, '_MAX_CHUNK_MEDIA_BYTES', 1)
            t2 = pxt.create_table(p('dst'), source=t)
        assert_resultset_eq(media_rows(t2), media_rows(t))
        # the copy stores each input file once
        assert len(set(t2.select(t2.img.fileurl).collect()['img_fileurl'])) == 2
        # revert() deletes the files of each version that it reverts. The files of the copy belong to version 0 and
        # survive a revert to version 0.
        for i in range(3):
            t2.insert([{'id': 10 + i, 'img': img_files[0]}])
        for _ in range(3):
            t2.revert()
        FileCache.get().clear()
        assert_resultset_eq(media_rows(t2), media_rows(t))

        # The copy stores its own copy of each bucket object under the table prefix of the source, because drop_table()
        # and revert() of the source delete that prefix. The copy keeps any other remote url.
        img_file = img_files[0]
        downloaded: list[tuple[str, bool]] = []  # (url, whether a transaction is open)

        def download(url: str, dest_path: pathlib.Path) -> None:
            downloaded.append((url, get_runtime().in_xact))
            shutil.copy(img_file, dest_path)

        with monkeypatch.context() as m:
            # every download returns a copy of img_file
            m.setattr(ObjectOps, 'copy_object_to_local_file', download)
            # each chunk has one object
            m.setattr(table_copy, '_MAX_CHUNK_OBJECTS', 1)
            t = pxt.create_table('bucket_src', {'img': pxt.Image})
            prefix = t._id.hex
            source_urls = [f's3://bucket/media/{prefix}/ab/abcd/{prefix}_1_1_{uuid.uuid4().hex}.jpg' for _ in range(2)]
            other_url = f's3://bucket/media/{uuid.uuid4().hex}.jpg'
            t.insert([{'img': other_url}, {'img': source_urls[0]}, {'img': source_urls[1]}])
            downloaded.clear()
            t2 = pxt.create_table(p('bucket_dst'), source=t)
            # no download runs inside a transaction
            assert sorted(downloaded) == sorted((url, False) for url in source_urls)
            urls = t2.select(t2.img.fileurl).collect()['img_fileurl']
            assert other_url in urls and set(source_urls).isdisjoint(urls)
            # the copy still reads the image after the source is dropped
            pxt.drop_table('bucket_src')
            FileCache.get().clear()
            img = t2.where(t2.img.fileurl != other_url).select(t2.img).collect()[0]['img']
            assert np.array_equal(np.asarray(img), np.asarray(PIL.Image.open(img_file)))

    @pytest.mark.very_expensive
    def test_copy_table_bucket_destination(self, db_root: DatabaseRoot) -> None:
        """The copy keeps its own copy of the files that the source wrote to a bucket."""
        p = db_root.make_catalog_path
        dest_uri = test_destination.TestDestination.resolve_destination_uri(StorageTarget.S3_STORE)
        t = pxt.create_table('src', {'id': pxt.Int, 'img': pxt.Image})
        t.add_computed_column(rot=t.img.rotate(90), destination=f'{dest_uri}/copy_table')
        t.insert({'id': i, 'img': f} for i, f in enumerate(get_image_files()[:2]))
        src_rows = t.select(t.id, t.rot).order_by(t.id).collect()
        t2 = pxt.create_table(p('dst'), source=t)
        pxt.drop_table('src')
        FileCache.get().clear()
        assert_resultset_eq(t2.select(t2.id, t2.rot).order_by(t2.id).collect(), src_rows)

    @pytest.mark.filterwarnings('ignore:.*was not locked for read:pixeltable.exceptions.PixeltableWarning')
    def test_copy_table_errors(self, db_root: DatabaseRoot) -> None:
        p = db_root.make_catalog_path
        is_other_db = db_root.id != 'local'
        # mypy checks the values of an inline `**{like_arg: t}` against every keyword parameter of create_table() and
        # rejects them
        kwargs: dict[str, Any]
        t = pxt.create_table('src', {'id': pxt.Int, 'img': pxt.Image})
        pxt.create_view('view', t)
        pxt.create_table(p('existing'), {'id': pxt.Int})
        pxt.create_view(p('existing_view'), pxt.get_table(p('existing')))

        # a view of each database, as source and as schema_source
        for like_arg in ('source', 'schema_source'):
            for view_path in ('view', p('existing_view')):
                kwargs = {like_arg: pxt.get_table(view_path)}
                with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match=f'`{like_arg}` must be a base table'):
                    pxt.create_table(p('dst'), **kwargs)
        # the error suggests tbl.select() for copying only the rows
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match=r'pass a query such as `tbl.select\(\)`'):
            pxt.create_table(p('dst'), source=pxt.get_table('view'))
        if is_other_db:
            # a hosted table into another database, as source and as schema_source
            for like_arg in ('source', 'schema_source'):
                kwargs = {like_arg: pxt.get_table(p('existing'))}
                with pxt_raises(
                    pxt.ErrorCode.UNSUPPORTED_OPERATION,
                    match=f'`{like_arg}` must be a local table or a table of the database of',
                ):
                    pxt.create_table('dst', **kwargs)
            # create_table() refuses a query of another database before it creates anything
            with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='from a query of another database'):
                pxt.create_table(p('dst'), source=t.select(t.id))
        # arguments that the new table takes from the source table
        for like_arg in ('source', 'schema_source'):
            kwargs = {like_arg: t}
            with pxt_raises(
                pxt.ErrorCode.INVALID_ARGUMENT,
                match='cannot be combined with `comment`, `primary_key`, `media_validation`',
            ):
                pxt.create_table(p('dst'), primary_key='id', comment='c', media_validation='on_read', **kwargs)
        # schema_source with another schema
        with pxt_raises(pxt.ErrorCode.MISSING_REQUIRED, match='Exactly one of `schema`, `source` or `schema_source`'):
            pxt.create_table(p('dst'), {'id': pxt.Int}, schema_source=t)
        assert pxt.get_table(p('dst'), if_not_exists='ignore') is None

        with pxt_raises(pxt.ErrorCode.PATH_ALREADY_EXISTS, match='already exists'):
            pxt.create_table(p('existing'), source=t)
        # 'ignore' refuses a view at the path, as source and as schema_source
        for like_arg in ('source', 'schema_source'):
            kwargs = {like_arg: t}
            with pxt_raises(pxt.ErrorCode.PATH_ALREADY_EXISTS, match='already exists and is not a table'):
                pxt.create_table(p('existing_view'), if_exists='ignore', **kwargs)
        with pxt_raises(pxt.ErrorCode.DIRECTORY_NOT_FOUND, match='does not exist'):
            pxt.create_table(p('no_dir/dst'), source=t)

        # The computed column r calls a @pxt.query that reads src, and the destination database does not have src. The
        # call is inside a larger expression: lookup(...)[0].
        @pxt.query
        def lookup(n: int) -> pxt.Query:
            return t.where(t.id == n).select(t.id)

        t2 = pxt.create_table('src2', {'n': pxt.Int})
        t2.add_computed_column(r=lookup(t2.n)[0])
        # ... or through a @pxt.query called in the where clause of another @pxt.query
        t3 = pxt.create_table('src3', {'n': pxt.Int})

        @pxt.query
        def outer(n: int) -> pxt.Query:
            return t3.where(lookup(t3.n)[0] == n).select(t3.n)

        t3.add_computed_column(r=outer(t3.n))
        if is_other_db:
            for src in (t2, t3):
                for like_arg in ('source', 'schema_source'):
                    kwargs = {like_arg: src}
                    with pxt_raises(
                        pxt.ErrorCode.UNSUPPORTED_OPERATION, match=r"computed column 'r' references table 'src'"
                    ):
                        pxt.create_table(p('dst'), **kwargs)
                    assert pxt.get_table(p('dst'), if_not_exists='ignore') is None
        # In the local database, a copy of src2 and a table like src2 can reference src. r stays computed and reads src.
        t.insert([{'id': 1, 'img': get_image_files()[0]}])
        for like_arg in ('source', 'schema_source'):
            kwargs = {like_arg: t2}
            dst = pxt.create_table(f'{like_arg}_src2', **kwargs)
            assert dst.get_metadata()['columns']['r']['is_computed']
            dst.insert([{'n': 1}])
            assert dst.select(dst.r).collect()['r'] == [1]
        # A copy and a table like src2 succeed after r is dropped. columns() never shows a dropped column, and these
        # asserts show only that create_table() succeeds.
        t2.drop_column('r')
        assert pxt.create_table(p('copy_dropped_r'), source=t2).columns() == ['n']
        assert pxt.create_table(p('like_dropped_r'), schema_source=t2).columns() == ['n']

        if is_other_db:
            # a UDF that the destination cannot load
            t4 = pxt.create_table('src4', {'n': pxt.Int})
            t4.add_computed_column(n1=local_only_udf(t4.n))
            t4.insert([{'n': 1}])
            with pxt_raises(pxt.ErrorCode.FUNCTION_NOT_FOUND, match=r"computed column 'n1'(.|\n)*pxt db update"):
                pxt.create_table(p('dst'), source=t4)
            with pxt_raises(pxt.ErrorCode.FUNCTION_NOT_FOUND, match=r"computed column 'n1'(.|\n)*pxt db update"):
                pxt.create_table(p('dst'), schema_source=t4)
            assert pxt.get_table(p('dst'), if_not_exists='ignore') is None
            # The copy succeeds when a dropped column calls a UDF that the destination cannot load. The input column of
            # the UDF is dropped as well.
            t5 = pxt.create_table('src5', {'n': pxt.Int, 'k': pxt.Int})
            t5.add_computed_column(n1=local_only_udf(t5.n))
            t5.drop_column('n1')
            t5.drop_column('n')
            assert pxt.create_table(p('dst5'), source=t5).columns() == ['k']
            # ... and when a dropped embedding index calls such a UDF
            t6 = pxt.create_table('src6', {'s': pxt.String})
            t6.add_embedding_index('s', idx_name='idx', string_embed=local_only_embed)
            t6.drop_embedding_index(idx_name='idx')
            assert pxt.create_table(p('dst6'), source=t6).get_metadata()['indexes'] == t6.get_metadata()['indexes']
        # the copy succeeds when a dropped column references another dropped column
        t7 = pxt.create_table('src7', {'n': pxt.Int})
        t7.add_computed_column(c1=t7.n + 1)
        t7.add_computed_column(c2=t7.c1 + 1)
        t7.insert([{'n': 1}])
        t7.drop_column('c2')
        t7.drop_column('c1')
        assert list(pxt.create_table(p('dst7'), source=t7).collect()) == [{'n': 1}]

        # A row references a local media file that no longer exists. The file is under the Pixeltable home, so that
        # insert() keeps its url and does not copy it into the media store.
        img_path = Config.get().home / 'missing_img.jpg'
        shutil.copy(get_image_files()[0], img_path)
        t.insert([{'id': 0, 'img': str(img_path)}])
        img_path.unlink()
        with pxt_raises(pxt.ErrorCode.STORAGE_NOT_FOUND, match='no longer exist'):
            pxt.create_table(p('dst'), source=t)
        assert pxt.get_table(p('dst'), if_not_exists='ignore') is None

    def test_copy_table_if_exists(self, db_root: DatabaseRoot, fault_injection: None) -> None:
        p = db_root.make_catalog_path
        t = pxt.create_table('src', {'id': pxt.Int, 'img': pxt.Image})
        t.insert({'id': i, 'img': f} for i, f in enumerate(get_image_files()[:3]))
        pxt.create_table(p('dst'), {'other': pxt.String}).insert([{'other': 'a'}])

        # 'ignore' returns the existing table as it is
        t2 = pxt.create_table(p('dst'), source=t, if_exists='ignore')
        assert t2.columns() == ['other']

        if db_root.id == 'local':
            # A failed copy with 'replace' leaves the existing table unchanged and deletes the media files that it
            # stored. No table references these files, and the test counts all files in the media directory instead.
            num_media_files = sum(f.is_file() for f in Env.get().media_dir.rglob('*'))
            fault = ExceptionFault(pxt.RequestError(pxt.ErrorCode.INVALID_STATE, 'injected failure'))
            get_runtime().fault_manager.inject_fault(FaultLocation.CATALOG_IMPORT_TABLE_BEFORE_MD_COMMITTED, fault)
            with pxt_raises(pxt.ErrorCode.INVALID_STATE, match='injected failure'):
                pxt.create_table(p('dst'), source=t, if_exists='replace')
            fault.assert_count(1)
            assert pxt.get_table(p('dst')).collect()['other'] == ['a']
            assert sum(f.is_file() for f in Env.get().media_dir.rglob('*')) == num_media_files

        # 'replace' replaces an existing table with the copy
        t2 = pxt.create_table(p('dst'), source=t, if_exists='replace')
        assert t2.columns() == ['id', 'img']
        assert sorted(t2.collect()['id']) == [0, 1, 2]

        if db_root.id == 'local':
            # a table is not replaced with a copy of itself
            with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match='with a copy of itself'):
                pxt.create_table('src', source=t, if_exists='replace')
            assert t.count() == 3

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

        t2 = pxt.create_table(p('dst'), schema_source=t)
        # Create a table like t2 in the same database. For a hosted t2, the hosted database creates it.
        dst2 = pxt.create_table(p('dst2'), schema_source=t2)
        for tbl in (t2, dst2):
            assert schema_from_tbl_md(tbl.get_metadata()) == schema_from_tbl_md(t.get_metadata())
            assert tbl.get_metadata()['is_data_versioned'] == is_data_versioned
        assert t2.count() == 0
        assert len(t2.get_versions()) == 1
        # the new table computes its computed columns and indexes its rows
        t2.insert({'id': i, 's': f'text {i}'} for i in range(3))
        assert t2.order_by(t2.id).select(t2.upper).collect()['upper'] == ['TEXT 0', 'TEXT 1', 'TEXT 2']
        sim = t.s.similarity(string='text 1')
        sim2 = t2.s.similarity(string='text 1')
        # id orders the rows with equal scores, which the database returns in any order
        assert list(t2.select(t2.id).order_by(sim2, t2.id, asc=False).collect()) == list(
            t.select(t.id).order_by(sim, t.id, asc=False).collect()
        )

        # if_exists: 'error' refuses the existing table, 'ignore' returns it, 'replace' creates an empty one
        with pxt_raises(pxt.ErrorCode.PATH_ALREADY_EXISTS, match='already exists'):
            pxt.create_table(p('dst'), schema_source=t)
        assert pxt.create_table(p('dst'), schema_source=t, if_exists='ignore').count() == 3
        assert pxt.create_table(p('dst'), schema_source=t, if_exists='replace').count() == 0

        # a @pxt.query that reads the source table reads the new table instead, in any database
        t3 = pxt.create_table('src3', {'n': pxt.Int})

        @pxt.query
        def lookup(n: int) -> pxt.Query:
            return t3.where(t3.n == n).select(t3.n)

        # reads src3 only through its FROM clause
        @pxt.query
        def first_rows(n: int) -> pxt.Query:
            return t3.limit(n)

        @pxt.query
        def other_rows(n: int) -> pxt.Query:
            return t3.where(t3.n != n).select()

        dst3_tbls: list[pxt.Table] = []
        for i, query in enumerate((lookup, first_rows, other_rows)):
            t3.add_computed_column(r=query(t3.n))
            dst3_tbls.append(pxt.create_table(p(f'dst3_{i}'), schema_source=t3))
            t3.drop_column('r')
        # Once src3 is dropped, an insert computes r only if the query reads the new table. The new table has no rows
        # before the insert, and r is empty.
        pxt.drop_table('src3')
        for dst3 in dst3_tbls:
            dst3.insert([{'n': 1}])
            assert dst3.select(dst3.r).collect()['r'] == [[]]
