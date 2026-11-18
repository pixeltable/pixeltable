import datetime
import itertools
import re
import urllib.request
from pathlib import Path
from typing import Any, Callable

import bs4
import numpy as np
import pandas as pd
import PIL.Image
import pydantic
import pytest

import pixeltable as pxt
from pixeltable.functions.string import isalpha, isascii
from pixeltable.functions.video import frame_iterator

from .conftest import SampleFileServer
from .utils import (
    DatabaseRoot,
    ReloadTester,
    create_all_datatypes_tbl,
    get_audio_files,
    get_documents,
    get_image_files,
    get_video_files,
    local_embedding,
    pxt_raises,
    reload_catalog,
    skip_test_if_not_installed,
    validate_repr,
    validate_update_status,
)


class TestQuery:
    @staticmethod
    @pxt.udf
    def is_even_py(x: int) -> bool:
        return x % 2 == 0

    def create_join_tbls(
        self, num_rows: int, p: Callable[[str], str], is_data_versioned: bool = True
    ) -> tuple[pxt.Table, pxt.Table, pxt.Table]:
        t1 = pxt.create_table(
            p(f't1_{num_rows}'),
            {'id': pxt.Int | None, 'i': pxt.Int | None, 'a': pxt.Array | None},
            _is_data_versioned=is_data_versioned,
        )
        validate_update_status(
            t1.insert({'id': i, 'i': i, 'a': np.ones((100, 100), dtype=np.int64) * i} for i in range(num_rows)),
            expected_rows=num_rows,
        )

        t2 = pxt.create_table(
            p(f't2_{num_rows}'),
            {'id': pxt.Int | None, 'f': pxt.Float | None, 'a': pxt.Array | None},
            _is_data_versioned=is_data_versioned,
        )
        # t2 has matching ids
        validate_update_status(
            t2.insert(
                {'id': i, 'f': float(num_rows - i), 'a': np.ones((100, 100), dtype=np.int64) * (num_rows - i)}
                for i in range(num_rows)
            ),
            expected_rows=num_rows,
        )

        # t3:
        # - column i with a different type
        # - only 10% of the ids overlap with t1 and t2
        t3 = pxt.create_table(
            p(f't3_{num_rows}'),
            {'id': pxt.Int | None, 'i': pxt.String | None, 'f': pxt.Float | None},
            _is_data_versioned=is_data_versioned,
        )
        validate_update_status(
            t3.insert({'id': i, 'i': str(i), 'f': float(num_rows - i)} for i in range(0, 10 * num_rows, 10)),
            expected_rows=num_rows,
        )

        return t1, t2, t3

    def test_select_where(self, test_tbl: pxt.Table, db_root: DatabaseRoot, is_data_versioned: bool) -> None:
        p = db_root.make_catalog_path
        t = test_tbl
        res1 = t.collect()
        res2 = t.select().collect()
        assert len(res1) > 0
        assert len(res1) == len(res2)
        pd.testing.assert_frame_equal(
            res1.to_pandas().sort_values(['c1', 'c2']).reset_index(drop=True),
            res2.to_pandas().sort_values(['c1', 'c2']).reset_index(drop=True),
        )

        res1 = t.where(t.c2 < 10).select(t.c1, t.c2, t.c3).collect()
        res3 = t.where(t.c2 < 10).select(c1=t.c1, c2=t.c2, c3=t.c3).collect()
        assert len(res1) == len(res3)
        pd.testing.assert_frame_equal(
            res1.to_pandas().sort_values(['c1', 'c2', 'c3']).reset_index(drop=True),
            res3.to_pandas().sort_values(['c1', 'c2', 'c3']).reset_index(drop=True),
        )

        res4 = t.where(t.c2 < 10).select(t.c1, c2=t.c2, c3=t.c3).collect()
        assert len(res1) == len(res4)
        pd.testing.assert_frame_equal(
            res1.to_pandas().sort_values(['c1', 'c2', 'c3']).reset_index(drop=True),
            res4.to_pandas().sort_values(['c1', 'c2', 'c3']).reset_index(drop=True),
        )

        from pixeltable.functions.string import contains

        _ = t.where(contains(t.c1, 'test')).select(t.c1).collect()
        _ = t.where(contains(t.c1, 'test') & contains(t.c1, '1')).select(t.c1).collect()
        _ = t.where(contains(t.c1, 'test') & (t.c2 >= 10)).select(t.c1).collect()

        _ = t.where(t.c2 < 10).select(t.c2, t.c2).collect()  # repeated name no error

        # where clause needs to be a predicate
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            _ = t.where(t.c1).select(t.c2).collect()
        assert 'needs to return `Bool`' in str(exc_info.value)

        # where clause needs to be a predicate
        with pxt_raises(pxt.ErrorCode.INVALID_EXPRESSION) as exc_info:
            _ = t.where(15).select(t.c2).collect()  # type: ignore[arg-type]
        assert 'where() expects a Pixeltable expression; got: 15' in str(exc_info.value)

        # duplicate select list
        with pxt_raises(pxt.ErrorCode.INVALID_STATE) as exc_info:
            _ = t.select(t.c1).select(t.c2).collect()
        assert 'already specified' in str(exc_info.value)

        # invalid expr in select list: Callable is not a valid literal
        with pxt_raises(pxt.ErrorCode.INVALID_EXPRESSION) as exc_info:
            _ = t.select(datetime.datetime.now).collect()
        assert 'Invalid expression' in str(exc_info.value)

        # catch invalid name in select list from user input
        # only check stuff that's not caught by python kwargs checker
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT) as exc_info:
            _ = t.select(t.c1, **{'c2-1': t.c2}).collect()
        assert 'Invalid name' in str(exc_info.value)

        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT) as exc_info:
            _ = t.select(t.c1, **{'': t.c2}).collect()
        assert 'Invalid name' in str(exc_info.value)

        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT) as exc_info:
            _ = t.select(t.c1, **{'foo.bar': t.c2}).collect()
        assert 'Invalid name' in str(exc_info.value)

        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT) as exc_info:
            _ = t.select(t.c1, _c3=t.c2).collect()
        assert 'Invalid name' in str(exc_info.value)

        # catch repeated name from user input
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            _ = t.select(t.c2, c2=t.c1).collect()
        assert 'Repeated column name' in str(exc_info.value)

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            _ = t.select(t.c2 + 1, col_0=t.c2).collect()
        assert 'Repeated column name' in str(exc_info.value)

        # select list contains invalid references
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            t2 = pxt.create_table(p('t2'), {'c1': pxt.Int | None})
            _ = t.select(t.c1, t2.c1 + t.c2).collect()
        assert 'cannot be evaluated in the context' in str(exc_info.value)

        with pxt_raises(pxt.ErrorCode.INVALID_STATE, match=r'where\(\) clause already specified'):
            _ = t.select(t.c2).where(t.c2 <= 10).where(t.c2 <= 20).count()

    def test_join(self, db_root: DatabaseRoot, is_data_versioned: bool) -> None:
        p = db_root.make_catalog_path
        num_rows = 100 if db_root.is_cloud else 1000
        t1, t2, t3 = self.create_join_tbls(num_rows, p, is_data_versioned)
        # inner join
        query = t1.join(t2, on=t1.id, how='inner').select(t1.i, t2.f, out=t1.i + t2.f).order_by(t2.f)
        pd_df = query.collect().to_pandas()
        assert len(pd_df) == num_rows
        assert pd_df.f.is_monotonic_increasing  # correct ordering
        assert (pd_df.out == float(num_rows)).all()  # correct sum

        # inner join that selects externally-stored arrays
        res = t1.join(t2, on=t1.id, how='inner').select(t1.i, t2.f, a1=t1.a, a2=t2.a).order_by(t2.f).collect()
        assert all(np.array_equal(row['a1'] + row['a2'], np.ones((100, 100), dtype=np.int64) * num_rows) for row in res)

        # the same inner join, but with redundant join predicates
        query = (
            t1.join(t2, on=(t1.id == t2.id) & (t1.i == t2.id), how='inner')
            .select(t1.i, t2.f, out=t1.i + t2.f)
            .order_by(t2.f)
        )
        pd_df2 = query.collect().to_pandas()
        assert pd_df.equals(pd_df2)

        # left outer join
        query = t1.join(t3, on=t1.id, how='left').select(t1.i, t3.f, out=t1.i + t3.f).order_by(t1.i)
        pd_df = query.collect().to_pandas()
        assert len(pd_df) == num_rows
        assert len(pd_df[~pd_df.f.isnull()]) == num_rows // 10  # correct number of nulls
        assert (pd_df[~pd_df.f.isnull()].out == float(num_rows)).all()  # correct sum

        # full outer join:
        # - t1 ids are 0..999, t3 ids are 0,10,...,9990
        # - 100 ids match, 900 are t1-only, and 900 are t3-only
        query = t1.join(t3, on=t1.id == t3.id, how='full_outer').select(left_i=t1.i, right_f=t3.f)
        pd_df = query.collect().to_pandas()
        assert len(pd_df) == num_rows * 19 // 10
        assert len(pd_df[pd_df.left_i.isnull()]) == num_rows * 9 // 10  # t3-only rows
        assert len(pd_df[pd_df.right_f.isnull()]) == num_rows * 9 // 10  # t1-only rows
        assert len(pd_df[pd_df.left_i.notnull() & pd_df.right_f.notnull()]) == num_rows // 10  # matched rows

        # TODO: implement right outer join
        # # right outer join
        # df = (
        #     t1.join(t3, on=t1.id, how='right')
        #     .select(t1.i, t3.f, out=t1.i + t3.f)
        #     .order_by(t1.i)
        # )
        # pd_df = df.collect().to_pandas()
        # assert len(pd_df) == 1000
        # assert len(pd_df[~pd_df.i.isnull()]) == 10  # correct number of nulls
        # assert (pd_df[~pd_df.f.isnull()].out == 1000.0).all()  # correct sum

        # cross join
        num_cj_rows = num_rows // 10
        small_t1, small_t2, _ = self.create_join_tbls(num_cj_rows, p, is_data_versioned)
        query = small_t1.join(small_t2, how='cross').select(small_t1.i, small_t2.f, out=small_t1.i + small_t2.f)
        res = query.collect()
        # TODO: verify result

        # inner join with aggregation and explicit join predicate
        query = t1.join(t2, on=t1.id == t2.id).select(pxt.functions.sum(t1.i + t2.id))
        res = query.collect()[0, 0]
        assert res == sum(range(num_rows)) * 2

        # inner join with grouping aggregation
        query = t1.join(t2, on=t2.id).group_by(t2.id % 10).select(grp=t2.id % 10, val=pxt.functions.sum(t1.i + t2.id))
        res = query.collect()
        pd_df = res.to_pandas()
        # TODO: verify result

    def test_join_errors(self, db_root: DatabaseRoot) -> None:
        p = db_root.make_catalog_path
        t1, t2, t3 = self.create_join_tbls(1000, p)

        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT) as exc_info:
            _ = t1.join(t2, on=(t2.id, 17)).collect()  # type: ignore[arg-type]
        assert 'must be a sequence of column references or a boolean expression' in str(exc_info.value)

        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT) as exc_info:
            _ = t1.join(t2, on=(15, 27)).collect()  # type: ignore[arg-type]
        assert 'must be a sequence of column references or a boolean expression' in str(exc_info.value)

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            _ = t1.join(t2, how='cross', on=t2.id).collect()
        assert '`on` not allowed for cross join' in str(exc_info.value)

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            _ = t1.join(t2).collect()
        assert "`how='inner'` requires `on`" in str(exc_info.value)

        with pxt_raises(pxt.ErrorCode.COLUMN_NOT_FOUND) as exc_info:
            _ = t1.join(t2, on=t2.f).collect()
        assert "'f' not found in any of: t1" in str(exc_info.value)

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            _ = t1.join(t2, on=t3.i).collect()
        assert 'expression cannot be evaluated in the context of the joined tables: i' in str(exc_info.value)

        with pxt_raises(pxt.ErrorCode.TYPE_MISMATCH) as exc_info:
            _ = t1.join(t2, on=t2.id + 1).collect()
        assert '`on` expects an expression of type `Bool`, but got one of type `Int | None`: id + 1' in str(
            exc_info.value
        )

        with pxt_raises(pxt.ErrorCode.INVALID_EXPRESSION, match=r'f\.floor is the method floor\(\)'):
            _ = t1.join(t2, on=t2.f.floor == t1.i).collect()

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            _ = t1.join(t2, on=t2.id).join(t3, on=t3.id).collect()
        assert "ambiguous column reference: 'id'" in str(exc_info.value)

        with pxt_raises(pxt.ErrorCode.COLUMN_NOT_FOUND) as exc_info:
            _ = t1.join(t2, on=t1.i).collect()
        assert "column 'i' not found in joined table" in str(exc_info.value)

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            _ = t1.join(t2, on=t1.id, how='inner').head()
        assert 'head() not supported for joins' in str(exc_info.value)

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            _ = t1.join(t2, on=t1.id, how='inner').tail()
        assert 'tail() not supported for joins' in str(exc_info.value)

    def test_result_set_iterator(self, test_tbl: pxt.Table) -> None:
        t = test_tbl
        res = t.select(t.c1, t.c2, t.c3).collect()
        pd_df = res.to_pandas()

        def check_row(row: dict[str, Any], idx: int) -> None:
            assert len(row) == 3
            assert 'c1' in row
            assert row['c1'] == pd_df['c1'][idx]
            assert 'c2' in row
            assert row['c2'] == pd_df['c2'][idx]
            assert 'c3' in row
            assert row['c3'] == pd_df['c3'][idx]

        # row iteration
        for idx, row in enumerate(res):
            check_row(row, idx)

        # row access
        row = res[0]
        check_row(row, 0)

        # column access
        col_values = res['c2']
        assert col_values == pd_df['c2'].values.tolist()

        # cell access
        assert res[0, 'c2'] == pd_df['c2'][0]
        assert res[0, 'c2'] == res[0, 1]

        with pxt_raises(pxt.ErrorCode.INVALID_COLUMN_NAME) as exc_info:
            _ = res['does_not_exist']
        assert 'Invalid column name' in str(exc_info.value)

        with pxt_raises(pxt.ErrorCode.INVALID_COLUMN_NAME) as exc_info:
            _ = res[0, 'does_not_exist']
        assert 'Invalid column name' in str(exc_info.value)

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            _ = res[0, 0, 0]
        assert 'Bad index' in str(exc_info.value)

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            _ = res['c2', 0]
        assert 'Bad index' in str(exc_info.value)

    def test_order_by(self, test_tbl: pxt.Table) -> None:
        t = test_tbl
        _ = t.select(t.c4, t.c2).order_by(t.c4).order_by(t.c2, asc=False).collect()

        # invalid expr in order_by()
        with pxt_raises(pxt.ErrorCode.INVALID_EXPRESSION) as exc_info:
            _ = t.order_by(datetime.datetime.now()).collect()  # type: ignore[arg-type]
        assert 'Invalid expression' in str(exc_info.value)

        # uncalled method in order_by()
        with pxt_raises(pxt.ErrorCode.INVALID_EXPRESSION, match=r'c1\.upper is the method upper\(\)'):
            _ = t.order_by(t.c1.upper).collect()

    def test_expr_unique_id(self, test_tbl: pxt.Table) -> None:
        t = test_tbl
        # Multiple constants with the same string representation but different types must be unique (Expr._expr_id)
        res = t.select(t.c2, t.c1, t.c1 == '2', t.c1 < '4', t.c2 == 4).limit(4).collect()
        print(res)
        assert len(res) == 4

    def test_limit_basic(self, test_tbl: pxt.Table, is_data_versioned: bool) -> None:
        t = test_tbl

        # Basic return shape: length and schema preserved
        res = t.select(t.c1, t.c2).limit(3).collect()
        assert len(res) == 3
        assert list(res.schema.keys()) == ['c1', 'c2']

        # Array-valued select column
        res = t.select(t.c4, arr=[2, 3, 4]).limit(2).collect()
        assert len(res) == 2
        assert res[0]['arr'] == [2, 3, 4]

        # Computed-expression select column
        res = t.select(t.c2, doubled=t.c2 * 2).order_by(t.c2).limit(4).collect()
        assert [row['doubled'] for row in res] == [0, 2, 4, 6]

        # limit > table size returns all rows
        res = t.select(t.c2).limit(1000).collect()
        assert len(res) == 100

        # offset=None is equivalent to no offset
        res_none = t.select(t.c2).order_by(t.c2).limit(5, offset=None).collect()
        res_no = t.select(t.c2).order_by(t.c2).limit(5).collect()
        assert [r['c2'] for r in res_none] == [r['c2'] for r in res_no]

        # Pagination: basic offsets
        res = t.select(t.c2).order_by(t.c2).limit(5, offset=0).collect()
        assert [row['c2'] for row in res] == [0, 1, 2, 3, 4]
        res = t.select(t.c2).order_by(t.c2).limit(5, offset=5).collect()
        assert [row['c2'] for row in res] == [5, 6, 7, 8, 9]

        # offset=0 equivalent to no offset
        res_zero = t.select(t.c2).order_by(t.c2).limit(5, offset=0).collect()
        assert [r['c2'] for r in res_no] == [r['c2'] for r in res_zero]

        # offset larger than result set
        res = t.select(t.c2).limit(10, offset=200).collect()
        assert len(res) == 0

        # offset with where clause
        res = t.where(t.c2 >= 10).select(t.c2).order_by(t.c2).limit(5, offset=5).collect()
        assert [row['c2'] for row in res] == [15, 16, 17, 18, 19]

        # offset near end of results: fewer rows than limit requested
        res = t.select(t.c2).order_by(t.c2).limit(10, offset=95).collect()
        assert len(res) == 5
        assert [row['c2'] for row in res] == [95, 96, 97, 98, 99]

        # Python-side filter path (FilterNode): a Python UDF can't be translated to SQL,
        # so limit/offset are applied by Python instead of LIMIT/OFFSET in the SQL query.
        # (_force_stored is orthogonal — it just lets us define the UDF locally inside this
        # test method; pxt otherwise rejects nested functions.)

        res = t.where(self.is_even_py(t.c2)).select(t.c2).order_by(t.c2).limit(5, offset=5).collect()
        assert [row['c2'] for row in res] == [10, 12, 14, 16, 18]

        # Python-filter path, offset near end: fewer rows than limit
        res = t.where(self.is_even_py(t.c2)).select(t.c2).order_by(t.c2).limit(10, offset=45).collect()
        assert len(res) == 5
        assert [row['c2'] for row in res] == [90, 92, 94, 96, 98]

        # Python-filter path, offset beyond result: empty
        res = t.where(self.is_even_py(t.c2)).select(t.c2).order_by(t.c2).limit(5, offset=60).collect()
        assert len(res) == 0

    def test_limit_errors(self, test_tbl: pxt.Table) -> None:
        t = test_tbl

        with pxt_raises(pxt.ErrorCode.TYPE_MISMATCH, match='must be of type `Int`'):
            _ = t.limit(5.3)  # type: ignore[arg-type]
        with pxt_raises(pxt.ErrorCode.TYPE_MISMATCH, match='must be of type `Int`'):
            _ = t.limit('5')  # type: ignore[arg-type]
        with pxt_raises(pxt.ErrorCode.MISSING_REQUIRED, match='must be present'):
            _ = t.limit(None)
        with pxt_raises(pxt.ErrorCode.TYPE_MISMATCH, match='must be of type `Int`'):
            _ = t.limit(5, offset='bad')  # type: ignore[arg-type]
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match='must be >= 0'):
            _ = t.limit(-1)
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match='must be >= 0'):
            _ = t.limit(5, offset=-1)
        # limit() after sample() is rejected
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='cannot be used with sample'):
            _ = t.sample(n=10).limit(5)

    def test_limit_joins(self, db_root: DatabaseRoot, is_data_versioned: bool) -> None:
        p = db_root.make_catalog_path
        num_rows = 100
        t1, t2, _ = self.create_join_tbls(num_rows, p, is_data_versioned)

        # inner join + limit: only n rows returned, ordering preserved
        res = t1.join(t2, on=t1.id, how='inner').select(t1.i, t2.f).order_by(t1.i).limit(10).collect()
        assert len(res) == 10
        assert [row['i'] for row in res] == list(range(10))

        # inner join + limit + offset: paging across a join
        res = t1.join(t2, on=t1.id, how='inner').select(t1.i, t2.f).order_by(t1.i).limit(5, offset=10).collect()
        assert len(res) == 5
        assert [row['i'] for row in res] == list(range(10, 15))

        # cross join + limit: caps total output (10 * 10 = 100 → 25)
        small1, small2, _ = self.create_join_tbls(10, p, is_data_versioned)
        res = small1.join(small2, how='cross').select(small1.i, small2.f).limit(25).collect()
        assert len(res) == 25

        # limit greater than join cardinality: returns all rows
        res = small1.join(small2, on=small1.id, how='inner').select(small1.i).limit(1000).collect()
        assert len(res) == 10

    def test_join_unselected_cols(self, db_root: DatabaseRoot, is_data_versioned: bool) -> None:
        """where(), order_by() and group_by() of a join accept columns that select() omits."""
        p = db_root.make_catalog_path
        artists = pxt.create_table(
            p('artists'),
            {'aid': pxt.String, 'grp': pxt.String | None},
            primary_key='aid',
            _is_data_versioned=is_data_versioned,
        )
        posters = pxt.create_table(
            p('posters'),
            {'pid': pxt.String, 'aid': pxt.String | None, 'year': pxt.Int | None, 'title': pxt.String | None},
            primary_key='pid',
            _is_data_versioned=is_data_versioned,
        )
        # inv fails for p0
        posters.add_computed_column(inv=100 / (posters.year - 1900))
        artists.insert({'aid': f'a{i}', 'grp': 'x' if i < 2 else 'y'} for i in range(4))
        titles = ['abc', 'def', 'ghi', 'jkl', 'mno', 'pqr', 'stu', 'vwx']
        posters.insert(
            ({'pid': f'p{i}', 'aid': f'a{i % 4}', 'year': 1900 + i, 'title': titles[i]} for i in range(8)),
            on_error='ignore',
        )
        posters.add_embedding_index('title', embedding=local_embedding)
        sim = posters.title.similarity(string=titles[3])

        for how in ('inner', 'left', 'full_outer'):
            # every poster has an artist and vice versa, so all join types return the same rows
            j = posters.join(artists, on=posters.aid == artists.aid, how=how)

            # where()
            res = j.where(posters.year < 1904).select(posters.pid).collect()
            assert sorted(res['pid']) == ['p0', 'p1', 'p2', 'p3']
            res = j.where(artists.grp == 'x').select(posters.pid).collect()
            assert sorted(res['pid']) == ['p0', 'p1', 'p4', 'p5']
            res = j.where((posters.year >= 1904) & (artists.grp == 'x')).select(artists.aid).collect()
            assert sorted(res['aid']) == ['a0', 'a1']
            res = j.where(posters.year % 2 == artists.grp.len()).select(posters.pid).collect()
            assert sorted(res['pid']) == ['p1', 'p3', 'p5', 'p7']
            res = j.where(sim > 0.99).select(posters.pid).collect()
            assert res['pid'] == ['p3']
            res = j.where(posters.inv.errortype != None).select(artists.aid).collect()
            assert res['aid'] == ['a0']
            assert j.where(posters.year < 1904).count() == 4
            res = j.where(posters.year < 1904).select(n=pxt.functions.count(posters.pid)).collect()
            assert res['n'] == [4]
            res = (
                j.where(posters.year < 1906).group_by(artists.grp).select(n=pxt.functions.count(posters.pid)).collect()
            )
            assert sorted(res['n']) == [2, 4]
            res = j.group_by(artists.grp).select(pids=pxt.functions.json.make_list(posters.pid)).collect()
            assert sorted(sorted(pids) for pids in res['pids']) == [['p0', 'p1', 'p4', 'p5'], ['p2', 'p3', 'p6', 'p7']]
            # window function
            res = j.select(
                posters.pid, total=pxt.functions.sum(posters.year - 1900, group_by=artists.grp, order_by=posters.year)
            ).collect()
            assert dict(zip(res['pid'], res['total'])) == {
                'p0': 0, 'p1': 1, 'p4': 5, 'p5': 10, 'p2': 2, 'p3': 5, 'p6': 11, 'p7': 18
            }  # fmt: skip

            # order_by()
            res = j.order_by(posters.year, asc=False).select(posters.pid).collect()
            assert res['pid'] == [f'p{i}' for i in range(7, -1, -1)]
            res = j.order_by(artists.grp, posters.year % 4, posters.year).select(posters.pid).collect()
            assert res['pid'] == ['p0', 'p4', 'p1', 'p5', 'p2', 'p6', 'p3', 'p7']
            res = j.order_by(sim, asc=False).select(posters.pid).limit(1).collect()
            assert res['pid'] == ['p3']
            res = j.order_by(sim, asc=False).select(posters.pid, sim=sim).limit(1).collect()
            assert res['pid'] == ['p3']

            # where() and order_by() combined
            res = j.where(artists.grp == 'y').order_by(posters.year, asc=False).select(posters.pid).limit(3).collect()
            assert res['pid'] == ['p7', 'p6', 'p3']

        # cross join
        j = posters.join(artists, how='cross')
        res = j.where((posters.year == 1900) & (artists.grp == 'x')).select(artists.aid).collect()
        assert sorted(res['aid']) == ['a0', 'a1']

        # three tables
        grps = pxt.create_table(
            p('grps'), {'grp': pxt.String, 'label': pxt.String | None}, _is_data_versioned=is_data_versioned
        )
        grps.insert([{'grp': 'x', 'label': 'first'}, {'grp': 'y', 'label': 'second'}])
        j = posters.join(artists, on=posters.aid == artists.aid).join(grps, on=artists.grp == grps.grp)
        res = j.where(grps.label == 'second').order_by(posters.year).select(artists.aid).collect()
        assert res['aid'] == ['a2', 'a3', 'a2', 'a3']

        if is_data_versioned:
            # view and snapshot
            recent = pxt.create_view(p('recent'), posters.where(posters.year >= 1904))
            snap = pxt.create_snapshot(p('artists_snap'), artists)
            j = recent.join(snap, on=recent.aid == snap.aid)
            res = j.where(snap.grp == 'x').order_by(recent.year, asc=False).select(recent.pid).collect()
            assert res['pid'] == ['p5', 'p4']

    def test_outer_join_nulls(self, db_root: DatabaseRoot) -> None:
        """Expressions over an outer join see nulls in the columns of unmatched rows."""
        p = db_root.make_catalog_path
        people = pxt.create_table(p('people'), {'pid': pxt.String, 'name': pxt.String | None}, primary_key='pid')
        visits = pxt.create_table(
            p('visits'),
            {'vid': pxt.String, 'pid': pxt.String | None, 'note': pxt.String | None, 'scan': pxt.Array | None},
            primary_key='vid',
        )
        people.insert([{'pid': 'p1', 'name': 'Ann'}, {'pid': 'p2', 'name': 'Bob'}])
        visits.insert(
            [
                {'vid': 'v1', 'pid': 'p1', 'note': 'hello', 'scan': np.ones((100, 100), dtype=np.int64)},
                {'vid': 'v2', 'pid': 'p3', 'note': None, 'scan': None},
            ]
        )

        q = people.join(visits, on=people.pid == visits.pid, how='left')
        # where() on the nullable side, with and without the column in select()
        assert q.where(visits.vid == None).select(people.pid).collect()['pid'] == ['p2']
        assert q.where(visits.vid == None).select(people.pid, visits.vid).collect()['pid'] == ['p2']
        assert q.where(visits.note == 'hello').select(people.pid).collect()['pid'] == ['p1']
        assert q.where(visits.pid == None).select(people.pid).collect()['pid'] == ['p2']
        assert q.where(visits.vid != None).select(people.pid).collect()['pid'] == ['p1']
        assert q.where(visits.scan == None).select(people.pid).collect()['pid'] == ['p2']
        assert q.where(visits.scan != None).select(people.pid, visits.scan).collect()['pid'] == ['p1']
        assert q.where((visits.note == None) | (visits.note == 'hello')).order_by(people.pid).select(
            people.pid
        ).collect()['pid'] == ['p1', 'p2']
        # select() and order_by()
        res = q.select(people.pid, no_visit=visits.vid == None).order_by(people.pid).collect()
        assert res['no_visit'] == [False, True]
        res = q.select(people.pid).order_by(visits.vid == None, asc=False).collect()
        assert res['pid'] == ['p2', 'p1']

        q = people.join(visits, on=people.pid == visits.pid, how='full_outer')
        assert q.where(visits.vid == None).select(people.pid).collect()['pid'] == ['p2']
        assert q.where(people.name == None).select(visits.vid).collect()['vid'] == ['v2']
        assert q.where((people.pid == None) | (visits.vid == None)).count() == 2
        res = q.select(visits.vid, no_person=people.name == None, no_visit=visits.vid == None).collect()
        assert sorted(zip(res['no_person'], res['no_visit'])) == [(False, False), (False, True), (True, False)]

    def test_limit_iterator_views(self, db_root: DatabaseRoot) -> None:
        p = db_root.make_catalog_path
        base_t = pxt.create_table(p('lim_base'), {'video': pxt.Video | None})
        view_t = pxt.create_view(p('lim_frames'), base_t, iterator=frame_iterator(base_t.video, fps=1))
        base_t.insert(video=get_video_files()[0])

        total = view_t.count()
        assert total > 10, f'expected > 10 frames, got {total}'

        # basic limit on iterator view
        res = view_t.select(view_t.pos).order_by(view_t.pos).limit(5).collect()
        assert len(res) == 5
        assert [row['pos'] for row in res] == [0, 1, 2, 3, 4]

        # limit + offset on iterator view
        res = view_t.select(view_t.pos).order_by(view_t.pos).limit(5, offset=3).collect()
        assert len(res) == 5
        assert [row['pos'] for row in res] == [3, 4, 5, 6, 7]

        # limit larger than frame count: returns all
        res = view_t.select(view_t.pos).limit(total + 100).collect()
        assert len(res) == total

    @pxt.uda
    class py_agg(pxt.Aggregator):
        def __init__(self) -> None:
            pass

        def update(self, val: int) -> None:
            pass

        def value(self) -> int:
            return 0

    @pxt.uda
    class py_sum(pxt.Aggregator):
        total: int

        def __init__(self) -> None:
            self.total = 0

        def update(self, val: int) -> None:
            self.total += val

        def value(self) -> int:
            return self.total

    def test_limit_0(self, test_tbl: pxt.Table, is_data_versioned: bool) -> None:
        t = test_tbl

        def check(query: pxt.Query, expected_cols: list[str]) -> None:
            assert list(query.schema.keys()) == expected_cols
            res = query.collect()
            assert len(res) == 0
            assert list(res.schema.keys()) == expected_cols

        # plain SQL scan
        check(t.select(t.c1, t.c2).order_by(t.c2).limit(0), ['c1', 'c2'])

        # SQL scan with offset
        check(t.select(t.c1, t.c2).order_by(t.c2).limit(0, offset=5), ['c1', 'c2'])

        # SQL aggregation (SqlAggregationNode): all exprs translatable to SQL
        check(t.group_by(t.c1).select(t.c1, cnt=pxt.functions.count(t.c2)).limit(0), ['c1', 'cnt'])

        # Python filter (FilterNode path): UDF forces Python-side filter
        @pxt.udf(_force_stored=True)
        def is_positive(x: int) -> bool:
            return x > 0

        check(t.where(is_positive(t.c2)).select(t.c1, t.c2).order_by(t.c2).limit(0), ['c1', 'c2'])

        # Python filter with offset
        check(t.where(is_positive(t.c2)).select(t.c1, t.c2).order_by(t.c2).limit(0, offset=3), ['c1', 'c2'])

        # in-memory aggregation (AggregationNode): Python UDA cannot be pushed to SQL
        check(t.group_by(t.c1).select(t.c1, s=self.py_agg(t.c2)).limit(0), ['c1', 's'])

        # cursor: schema accessible before and after iteration, no rows yielded
        query = t.select(t.c1, t.c2).order_by(t.c2).limit(0)
        cur = query.cursor()
        # TODO: add ResultCursor.keys(), analogously to Sqlalchemy's CursorResult.keys()?
        assert list(cur.schema.keys()) == ['c1', 'c2']
        rows = list(cur)
        assert rows == []
        assert list(cur.schema.keys()) == ['c1', 'c2']

    def test_head_tail(self, test_tbl: pxt.Table, db_root: DatabaseRoot, is_data_versioned: bool) -> None:
        p = db_root.make_catalog_path
        t = test_tbl
        res = t.head(10).to_pandas()
        assert np.all(res.c2 == list(range(10)))
        reload_catalog()
        t = pxt.get_table(p('test_tbl'))
        res = t.head(10).to_pandas()
        # Where is applied
        res = t.where(t.c2 > 9).head(10).to_pandas()
        assert np.all(res.c2 == list(range(10, 20)))
        # order_by() is an error
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            _ = t.order_by(t.c2).head(10)
        assert 'cannot be used with order_by' in str(exc_info.value)
        # group_by() is an error
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            _ = t.group_by(t.c2).head(10)
        assert 'cannot be used with group_by' in str(exc_info.value)

        res = t.tail().to_pandas()
        assert np.all(res.c2 == list(range(90, 100)))
        res = t.where(t.c2 < 90).tail().to_pandas()
        assert np.all(res.c2 == list(range(80, 90)))
        # order_by() is an error
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            _ = t.order_by(t.c2).tail(10)
        assert 'cannot be used with order_by' in str(exc_info.value)
        # group_by() is an error
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            _ = t.group_by(t.c2).tail(10)
        assert 'cannot be used with group_by' in str(exc_info.value)

    def test_repr(self, test_tbl: pxt.Table) -> None:
        t = test_tbl
        query = t.select(t.c1, t.c1.upper(), t.c2 + 5).where(t.c2 < 10).group_by(t.c1).order_by(t.c3).limit(10)
        query.describe()

        validate_repr(
            query,
            """   Name    Type  Expression
               ---------------------------
                    c1  String          c1
                 upper  String  c1.upper()
                 col_2     Int      c2 + 5

               From      test_tbl
               Where      c2 < 10
               Group By        c1
               Order By    c3 asc
               Limit           10""",
        )

    def test_count(self, test_tbl: pxt.Table, is_data_versioned: bool) -> None:
        t = test_tbl
        cnt = t.count()
        assert cnt == 100

        cnt = t.where(t.c2 < 10).count()
        assert cnt == 10

    def test_count_errors(self, test_tbl: pxt.Table, small_img_tbl: pxt.Table) -> None:
        t = small_img_tbl
        # Python-only filter forces a non-SQL plan
        with pxt_raises(
            pxt.ErrorCode.UNSUPPORTED_OPERATION,
            match=re.escape('count() cannot be used: query plan contains a non-SQL node'),
        ):
            _ = t.where(t.img.width > 100).count()

        t = test_tbl
        with pxt_raises(
            pxt.ErrorCode.UNSUPPORTED_OPERATION, match=re.escape('count() cannot be used with limit() or offset()')
        ):
            _ = t.limit(5).count()
        with pxt_raises(
            pxt.ErrorCode.UNSUPPORTED_OPERATION, match=re.escape('count() cannot be used with limit() or offset()')
        ):
            _ = t.limit(5, offset=5).count()

    def test_count_with_group_by(self, test_tbl: pxt.Table, is_data_versioned: bool) -> None:
        """Test that count() works with group_by()."""
        t = test_tbl
        # Count with group_by should return the number of groups
        cnt = t.group_by(t.c1).count()
        # Should return the number of distinct c1 values
        distinct_c1 = len(t.select(t.c1).distinct().collect())
        assert cnt == distinct_c1

        # Count with group_by and where clause
        cnt = t.where(t.c2 < 10).group_by(t.c1).count()
        # Should return the number of distinct c1 values in filtered rows
        distinct_c1_filtered = len(t.where(t.c2 < 10).select(t.c1).distinct().collect())
        assert cnt == distinct_c1_filtered

    def test_group_by_python_exprs(self, db_root: DatabaseRoot, is_data_versioned: bool) -> None:
        """group_by() and distinct() return one row per key when part of the query is evaluated in Python."""
        p = db_root.make_catalog_path
        t = pxt.create_table(
            p('test_group_by'),
            {'acct': pxt.String, 'v': pxt.Int, 'info': pxt.Json},
            _is_data_versioned=is_data_versioned,
        )
        # consecutive rows have different keys
        validate_update_status(
            t.insert({'acct': f'a{i % 2}', 'v': i, 'info': {'v': i}} for i in range(6)), expected_rows=6
        )
        t2 = pxt.create_table(
            p('test_group_by_2'), {'acct': pxt.String, 'w': pxt.Int}, _is_data_versioned=is_data_versioned
        )
        validate_update_status(t2.insert({'acct': f'a{i % 2}', 'w': i} for i in range(4)), expected_rows=4)
        t3 = pxt.create_table(
            p('test_group_by_3'),
            {'a': pxt.String | None, 'b': pxt.Int, 'v': pxt.Int},
            _is_data_versioned=is_data_versioned,
        )
        validate_update_status(
            t3.insert({'a': None if i % 2 == 0 else 'x', 'b': (i // 2) % 2, 'v': i} for i in range(8)), expected_rows=8
        )

        @pxt.udf(_force_stored=True)
        def is_small(x: int) -> bool:
            return x < 5

        @pxt.udf(_force_stored=True)
        def tag(x: int) -> str:
            return f'<{x}>'

        @pxt.udf(_force_stored=True)
        def upper(s: str) -> str:
            return s.upper()

        count = pxt.functions.count

        def rows(query: pxt.Query) -> list[tuple]:
            # key=repr: None is not comparable to str
            return sorted((tuple(row.values()) for row in query.collect()), key=repr)

        # where() in SQL
        assert rows(t.where(t.v < 5).group_by(t.acct).select(t.acct, n=count(t.v), s=pxt.functions.sum(t.v))) == [
            ('a0', 3, 6),
            ('a1', 2, 4),
        ]
        # where() in Python, alone and with a conjunct in SQL
        assert rows(t.where(is_small(t.v)).group_by(t.acct).select(t.acct, n=count(t.v), s=pxt.functions.sum(t.v))) == [
            ('a0', 3, 6),
            ('a1', 2, 4),
        ]
        assert rows(t.where(is_small(t.v) & (t.v > 0)).group_by(t.acct).select(t.acct, n=count(t.v))) == [
            ('a0', 2),
            ('a1', 2),
        ]
        assert rows(t.where(is_small(t.v)).select(n=count(t.v))) == [(5,)]
        assert rows(t.where(is_small(t.v)).select(t.acct).distinct()) == [('a0',), ('a1',)]

        # select list in Python, over an aggregate and over a grouping expr
        assert rows(t.group_by(t.acct).select(t.acct, n=tag(count(t.v)))) == [('a0', '<3>'), ('a1', '<3>')]
        assert rows(t.group_by(t.acct).select(a=upper(t.acct), n=count(t.v))) == [('A0', 3), ('A1', 3)]

        # user-defined aggregate
        assert rows(t.where(is_small(t.v)).group_by(t.acct).select(t.acct, s=self.py_sum(t.v))) == [
            ('a0', 6),
            ('a1', 4),
        ]

        # grouping expr other than a column
        assert rows(t.where(is_small(t.v)).group_by(t.v % 2).select(k=t.v % 2, n=count(t.v))) == [(0, 3), (1, 2)]

        # several grouping exprs, one with nulls
        assert rows(t3.where(is_small(t3.v)).group_by(t3.a, t3.b).select(t3.a, t3.b, n=count(t3.v))) == [
            ('x', 0, 1),
            ('x', 1, 1),
            (None, 0, 2),
            (None, 1, 1),
        ]
        assert rows(t3.where(is_small(t3.v)).group_by(t3.a).select(t3.a, n=count(t3.v))) == [('x', 2), (None, 3)]
        assert rows(t3.where(is_small(t3.v)).select(t3.a, t3.b).distinct()) == [
            ('x', 0),
            ('x', 1),
            (None, 0),
            (None, 1),
        ]

        # aggregate over a Json column
        assert rows(t.group_by(t.acct).select(t.acct, n=count(t.info))) == [('a0', 3), ('a1', 3)]

        query = t.join(t2, on=t.acct == t2.acct).where(is_small(t.v)).group_by(t.acct).select(t.acct, n=count(t2.w))
        assert rows(query) == [('a0', 6), ('a1', 4)]

        # order_by() on the grouping expr, with limit()
        query = t.where(is_small(t.v)).group_by(t.acct).select(t.acct, n=count(t.v)).order_by(t.acct, asc=False)
        assert [tuple(row.values()) for row in query.collect()] == [('a1', 2), ('a0', 3)]
        assert rows(query.limit(1)) == [('a1', 2)]

        # order_by() on the grouping exprs in another order, also followed by another expr
        expected = [('x', 0, 1), (None, 0, 2), ('x', 1, 1), (None, 1, 1)]
        query = t3.where(is_small(t3.v)).group_by(t3.a, t3.b).select(t3.a, t3.b, n=count(t3.v))
        for ordering in ([t3.b, t3.a], [t3.b], [t3.b, t3.a, t3.v]):
            assert [tuple(row.values()) for row in query.order_by(*ordering).collect()] == expected
        res = t3.where(is_small(t3.v)).select(t3.a, t3.b).distinct().order_by(t3.b, t3.a).collect()
        assert [tuple(row.values()) for row in res] == [row[:2] for row in expected]
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match=r'limited to the grouping expressions \(a, b\)'):
            _ = query.order_by(t3.b, t3.v).collect()

        # order_by() on anything else conflicts with the ordering on the grouping exprs
        for ordering, ordering_text in ((t.v, '(v, asc=True)'), (count(t.v), '(globals.count(v), asc=True)')):
            with pxt_raises(
                pxt.ErrorCode.UNSUPPORTED_OPERATION,
                match=r'order_by\(\) is limited to the grouping expressions \(acct\) when part of an aggregate query '
                r'cannot run in SQL',
            ) as exc_info:
                _ = t.where(is_small(t.v)).group_by(t.acct).select(t.acct, n=count(t.v)).order_by(ordering).collect()
            message = str(exc_info.value)
            assert ordering_text in message
            assert 'sort the collected rows' in message

    def test_order_by_python_exprs(self, db_root: DatabaseRoot, is_data_versioned: bool) -> None:
        """order_by() on an expression without a SQL translation raises an error."""
        p = db_root.make_catalog_path
        t = pxt.create_table(
            p('test_order_by'), {'id': pxt.Int, 'x': pxt.Int, 'info': pxt.Json}, _is_data_versioned=is_data_versioned
        )
        validate_update_status(
            t.insert({'id': i, 'x': x, 'info': {'k': x}} for i, x in enumerate([3, 1, 4, 2])), expected_rows=4
        )
        validate_update_status(t.add_computed_column(live=t.info.k.astype(pxt.Int), stored=False), expected_rows=4)

        @pxt.udf(_force_stored=True)
        def double(x: int) -> int:
            return x * 2

        @pxt.query
        def ranked(offset: int) -> pxt.Query:
            return t.order_by(double(t.x + offset)).select(t.id)

        store = (
            'As a workaround, you can add the value as a stored computed column of a scalar type '
            '(cast with .astype() if needed) and use that column with order_by().'
        )
        sort = 'As a workaround, you can sort the collected rows.'
        rewrite = (
            'so a stored computed column would not have the current value. As a workaround, you can rewrite the '
            'expression with functions that have a SQL translation, or sort the collected rows.'
        )
        count = pxt.functions.count
        posix_timestamp, now = pxt.functions.timestamp.posix_timestamp, pxt.functions.timestamp.now
        # (query, order_by() expr, its part without a SQL translation, workaround)
        queries: list[tuple[Callable[[], Any], str, str, str]] = [
            (t.order_by(double(t.x)).select(t.id).collect, 'double(x)', 'double(x)', store),
            (t.order_by(double(t.x), asc=False).select(t.id).collect, 'double(x)', 'double(x)', store),
            (
                t.order_by(double(t.x) + double(t.x) + t.x).select(t.id).collect,
                '(double(x) + double(x)) + x',
                'double(x)',
                store,
            ),
            (t.order_by(t.live).select(t.id).collect, 'live', 'info.k', store),
            (t.order_by(t.live, asc=False).select(t.id).collect, 'live', 'info.k', store),
            (t.order_by(t.info.k).select(t.id).collect, 'info.k', 'info.k', store),
            (t.order_by(t.info.k, asc=False).select(t.id).collect, 'info.k', 'info.k', store),
            (t.order_by(t.info.k.astype(pxt.Int)).select(t.id).collect, 'info.k.astype(Int)', 'info.k', store),
            # with other clauses
            (t.order_by(t.x).order_by(double(t.x)).select(t.id).collect, 'double(x)', 'double(x)', store),
            (t.where(t.x > 1).order_by(double(t.x)).select(t.id).limit(2).collect, 'double(x)', 'double(x)', store),
            (t.where(double(t.x) > 2).order_by(double(t.x)).select(t.id).collect, 'double(x)', 'double(x)', store),
            # an aggregate cannot be stored in a computed column
            (
                t.group_by(t.x).select(t.x, n=count(t.id)).order_by(double(count(t.id))).collect,
                'double(globals.count(id))',
                'double(globals.count(id))',
                sort,
            ),
            # a stored computed column cannot track now() or a query parameter
            (
                t.order_by(posix_timestamp(now()) - t.x).select(t.id).collect,
                'posix_timestamp(now()) - x',
                'posix_timestamp(now())',
                f'The value depends on now(), {rewrite}',
            ),
            (
                lambda: pxt.eval_udf(ranked, 1),
                'double(x + offset)',
                'double(x + offset)',
                f'The value depends on the parameter offset, {rewrite}',
            ),
        ]
        for run_query, expr_text, no_sql_text, workaround in queries:
            with pxt_raises(pxt.ErrorCode.INVALID_EXPRESSION, match=r'Invalid order_by\(\) expression') as exc_info:
                _ = run_query()
            assert str(exc_info.value) == (
                f'Invalid order_by() expression, needs to be expressible in SQL: {expr_text}. '
                f'No SQL translation: {no_sql_text}. {workaround}'
            )

        # a where() in Python does not affect an order_by() in SQL
        res = t.where(double(t.x) > 2).order_by(t.x, asc=False).select(t.x).collect()
        assert res['x'] == [4, 3, 2]

        # the advice works
        validate_update_status(t.add_computed_column(k=t.info.k.astype(pxt.Int)), expected_rows=4)
        assert t.order_by(t.k).select(t.id).collect()['id'] == [1, 3, 0, 2]
        assert t.order_by(t.k, asc=False).select(t.id).collect()['id'] == [2, 0, 3, 1]

    def test_order_by_read_validated_media(self, db_root: DatabaseRoot, sample_file_server: SampleFileServer) -> None:
        """order_by() on a media column with media_validation='on_read' orders by the stored file url."""
        p = db_root.make_catalog_path
        t = pxt.create_table(p('test_order_by_media'), {'id': pxt.Int, 'img': pxt.Image}, media_validation='on_read')
        urls = sorted((sample_file_server.url(f, db_root) for f in get_image_files()[:4]), reverse=True)
        validate_update_status(t.insert({'id': i, 'img': url} for i, url in enumerate(urls)), expected_rows=4)

        assert t.order_by(t.img).select(t.id).collect()['id'] == [3, 2, 1, 0]
        assert t.order_by(t.img, asc=False).select(t.id).collect()['id'] == [0, 1, 2, 3]
        assert t.order_by(t.img).select(t.id, t.img.width).collect()['id'] == [3, 2, 1, 0]

    def test_count_join(self, db_root: DatabaseRoot, is_data_versioned: bool) -> None:
        """Tests for join...count queries"""
        # TODO(PXT-1108): when the join+where bug is fixed, add some queries with where().
        p = db_root.make_catalog_path
        num_rows = 100
        t1, t2, t3 = self.create_join_tbls(num_rows, p, is_data_versioned)

        # Inner join + count. Note: t1 and t2 have matching ids.
        cnt = t1.join(t2, on=(t1.id == t2.id), how='inner').count()
        assert cnt == num_rows

        # Inner join + count with a different table. Note: only 10% of t3's ids are present in t1.
        cnt = t1.join(t3, on=(t1.id == t3.id), how='inner').count()
        assert cnt == num_rows // 10

        # Left join + count. Since it's left join, all t1 rows should be present.
        cnt = t1.join(t3, on=(t1.id == t3.id), how='left').count()
        assert cnt == num_rows

    def test_select_literal(self, test_tbl: pxt.Table) -> None:
        t = test_tbl
        res = t.select(1.0).where(t.c2 < 10).collect()
        assert res[next(iter(res.schema.keys()))] == [1.0] * 10

    def test_html_media_url(self, db_root: DatabaseRoot) -> None:
        p = db_root.make_catalog_path
        tab = pxt.create_table(
            p('test_html_repr'), {'video': pxt.Video | None, 'audio': pxt.Audio | None, 'doc': pxt.Document | None}
        )

        pdf_doc = next(f for f in get_documents() if f.endswith('.pdf'))
        status = tab.insert(video=get_video_files()[0], audio=get_audio_files()[0], doc=pdf_doc)
        assert status.num_rows == 1
        assert status.num_excs == 0

        res = tab.select(tab.video, tab.audio, tab.doc).collect()
        doc = bs4.BeautifulSoup(res._repr_html_(), features='html.parser')
        video_tags = doc.find_all('video')
        assert len(video_tags) == 1
        audio_tags = doc.find_all('audio')
        assert len(audio_tags) == 1
        # get the source elements and test their src link are valid and can be retrieved
        # from running web-server
        for tag in video_tags + audio_tags:
            sources = tag.find_all('source')
            assert len(sources) == 1
            for src in sources:
                op = urllib.request.urlopen(src['src'])
                assert op.getcode() == 200

        document_tags = doc.find_all('div', attrs={'class': 'pxt_document'})
        assert len(document_tags) == 1
        res0 = document_tags[0]
        href = res0.find('a')['href']
        thumb = res0.find('img')['src']
        # check link is valid and server is running
        href_op = urllib.request.urlopen(url=href)
        assert href_op.getcode() == 200
        # check thumbnail is well formed image
        opurl_img = urllib.request.urlopen(url=thumb)
        PIL.Image.open(opurl_img)

    def test_update_delete_where(self, test_tbl: pxt.Table, is_data_versioned: bool) -> None:
        t = test_tbl

        # Update with where
        validate_update_status(t.where(t.c2 >= 50).update({'c3': 4171780.0}), expected_rows=50)
        assert t.where((t.c2 >= 50) & (t.c3 > 1000000.0)).count() == 50
        assert t.where((t.c2 < 50) & (t.c3 < 1000.0)).count() == 50

        # Update without where
        validate_update_status(t.select().update({'c3': 94.0}))
        new: list[int] = t.select(t.c3).collect()['c3']
        assert all(new[i] == 94.0 for i in range(len(new)))

        # Delete with where
        validate_update_status(t.where((t.c2 >= 50) & (t.c2 < 75)).delete())
        assert t.count() == 75
        # Delete without where
        validate_update_status(t.select().delete())
        assert t.count() == 0

    def test_mutation_op_restrictions(self, test_tbl: pxt.Table, db_root: DatabaseRoot) -> None:
        p = db_root.make_catalog_path
        t = test_tbl

        # select_list
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            t.select(t.c2).update({'c3': 0.0})
        assert 'Cannot use `update` after `select`' in str(exc_info.value)

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            t.select(t.c2).delete()
        assert 'Cannot use `delete` after `select`' in str(exc_info.value)

        # group_by

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            t.group_by(t.c2).update({'c3': 0.0})
        assert 'Cannot use `update` after `group_by`' in str(exc_info.value)

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            t.group_by(t.c2).delete()
        assert 'Cannot use `delete` after `group_by`' in str(exc_info.value)

        # order_by

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            t.order_by(t.c2).update({'c3': 0.0})
        assert 'Cannot use `update` after `order_by`' in str(exc_info.value)

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            t.order_by(t.c2).delete()
        assert 'Cannot use `delete` after `order_by`' in str(exc_info.value)

        # limit

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            t.limit(10).update({'c3': 0.0})
        assert 'Cannot use `update` after `limit`' in str(exc_info.value)

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            t.limit(10).delete()
        assert 'Cannot use `delete` after `limit`' in str(exc_info.value)

        # grouping_tbl

        t2 = pxt.create_table(p('test_tbl_2'), {'name': pxt.String | None, 'video': pxt.Video | None})
        v2 = pxt.create_view(p('test_view_2'), t2, iterator=frame_iterator(t2.video, fps=1))
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            v2.select(pxt.functions.video.make_video(v2.pos, v2.frame)).group_by(t2).update({'name': 'test'})
        assert 'Cannot use `update` after `group_by`' in str(exc_info.value)

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            v2.select(pxt.functions.video.make_video(v2.pos, v2.frame)).group_by(t2).delete()
        assert 'Cannot use `delete` after `group_by`' in str(exc_info.value)

        # delete from view
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            v2.where(t.c2 < 10).delete()
        assert 'Cannot use `delete` on a view.' in str(exc_info.value)

        # update snapshot
        snap = pxt.create_snapshot(p('test_snapshot'), t)
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            snap.where(t.c2 < 10).update({'c3': 0.0})
        assert 'Cannot use `update` on a snapshot.' in str(exc_info.value)

        # delete from snapshot
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            snap.where(t.c2 < 10).delete()
        assert 'Cannot use `delete` on a snapshot.' in str(exc_info.value)

        # join+update
        with pytest.raises(pxt.Error, match='Cannot use `update` after `join`'):
            t.join(t2, how='cross').update({'c3': 0.0})

        # join+delete
        with pytest.raises(pxt.Error, match='Cannot use `delete` after `join`'):
            t.join(t2, how='cross').delete()

        # recompute_columns: op-sequence restrictions
        with pytest.raises(pxt.Error, match='Cannot use `recompute_columns` after `select`'):
            t.select(t.c2).recompute_columns(t.c8)

        with pytest.raises(pxt.Error, match='Cannot use `recompute_columns` after `group_by`'):
            t.group_by(t.c2).recompute_columns(t.c8)

        with pytest.raises(pxt.Error, match='Cannot use `recompute_columns` after `order_by`'):
            t.order_by(t.c2).recompute_columns(t.c8)

        with pytest.raises(pxt.Error, match='Cannot use `recompute_columns` after `limit`'):
            t.limit(10).recompute_columns(t.c8)

        with pytest.raises(pxt.Error, match='Cannot use `recompute_columns` after `join`'):
            t.join(t2, how='cross').recompute_columns(t.c8)

        with pytest.raises(pxt.Error, match='Cannot use `recompute_columns` on a snapshot'):
            snap.where(t.c2 < 10).recompute_columns(t.c8)

    def __check_constant_query(self, query: pxt.Query, v: Any) -> None:
        r = query.limit(5).collect()
        assert all(r[i, 0] == v for i in range(len(r)))

    def test_select_constant(self, all_datatypes_tbl: pxt.Table) -> None:
        t = all_datatypes_tbl
        self.__check_constant_query(t.select(5), 5)
        self.__check_constant_query(t.select(None), None)
        self.__check_constant_query(t.select(foo=5), 5)
        self.__check_constant_query(t.select(foo=None), None)

    @pytest.mark.db_roots('local', reason='TODO: convert')
    def test_to_pytorch_dataset(self, all_datatypes_tbl: pxt.Table) -> None:
        """tests all types are handled correctly in this conversion"""
        skip_test_if_not_installed('torch', 'torchvision', 'pyarrow')
        import torch

        t = all_datatypes_tbl
        query = t.where(t.row_id < 1)
        assert query.count() > 0
        ds = query.to_pytorch_dataset()
        for tup in ds:
            for col in query.schema:
                assert col in tup

            arrval = tup['c_array']
            assert isinstance(arrval, np.ndarray)
            # c_array is declared pxt.Array[(10,), pxt.Float]
            assert arrval.shape == (10,)
            assert arrval.dtype == np.float32
            assert arrval.dtype == np.float32
            assert arrval.flags['WRITEABLE'], 'required by pytorch collate function'

            assert isinstance(tup['c_bool'], bool)
            assert isinstance(tup['c_int'], int)
            assert isinstance(tup['c_float'], float)
            assert isinstance(tup['c_timestamp'], float)
            assert torch.is_tensor(tup['c_image'])
            assert isinstance(tup['c_video'], str)
            assert isinstance(tup['c_json'], dict)

    @pytest.mark.db_roots('local', reason='TODO: convert')
    def test_to_pytorch_image_format(self, all_datatypes_tbl: pxt.Table) -> None:
        """tests the image_format parameter is honored"""
        skip_test_if_not_installed('torch', 'torchvision', 'pyarrow')
        import torch
        import torchvision.transforms  # type: ignore[import-untyped]

        w, h = 220, 224  # make different from each other
        t = all_datatypes_tbl
        query = t.select(t.row_id, t.c_image, c_image_xformed=t.c_image.resize([w, h]).convert('RGB')).where(
            t.row_id < 1
        )

        pandas_df = query.show().to_pandas()
        im_plain = pandas_df['c_image'].values[0]
        im_xformed = pandas_df['c_image_xformed'].values[0]
        assert pandas_df.shape[0] == 1

        ds = query.to_pytorch_dataset(image_format='np')
        ds_ptformat = query.to_pytorch_dataset(image_format='pt')

        elt_count = 0
        for elt, elt_pt in zip(ds, ds_ptformat):
            arr_plain = elt['c_image']
            assert isinstance(arr_plain, np.ndarray)
            assert arr_plain.flags['WRITEABLE'], 'required by pytorch collate function'

            # compare numpy array bc PIL.Image object itself is not using same file.
            assert (arr_plain == np.array(im_plain)).all(), 'numpy image should be the same as the original'
            arr_xformed = elt['c_image_xformed']
            assert isinstance(arr_xformed, np.ndarray)
            assert arr_xformed.flags['WRITEABLE'], 'required by pytorch collate function'

            assert arr_xformed.shape == (h, w, 3)
            assert arr_xformed.dtype == np.uint8
            # same as above, compare numpy array bc PIL.Image object itself is not using same file.
            assert (arr_xformed == np.array(im_xformed)).all(), (
                'numpy image array for xformed image should be the same as the original'
            )

            # now compare pytorch version
            arr_pt = elt_pt['c_image']
            assert torch.is_tensor(arr_pt)
            arr_pt = elt_pt['c_image_xformed']
            assert torch.is_tensor(arr_pt)
            assert arr_pt.shape == (3, h, w)
            assert arr_pt.dtype == torch.float32
            assert (arr_pt >= 0.0).all()
            assert (arr_pt <= 1.0).all()
            assert torch.isclose(torchvision.transforms.ToTensor()(arr_xformed), arr_pt).all(), (
                'pytorch image should be consistent with numpy image'
            )
            elt_count += 1
        assert elt_count == 1

    @pytest.mark.db_roots('local', reason='to_pytorch_dataset caching keyed on the local dataset cache')
    def test_pytorch_dataset_caching(self, uses_db: None) -> None:
        """Tests that dataset caching works
        1. using the same dataset twice in a row uses the cache
        2. adding a row to the table invalidates the cached version
        3. changing the select list invalidates the cached version
        """
        skip_test_if_not_installed('torch', 'torchvision', 'pyarrow')
        from pixeltable.utils.pytorch import PixeltablePytorchDataset

        # to_pytorch_dataset goes through arrow codepath via export_parquet
        t = create_all_datatypes_tbl(name='all_datatype_tbl', arrow_compatible_json=True)

        t.drop_column('c_video')  # null value video column triggers internal assertions in DataRow
        # see https://github.com/pixeltable/pixeltable/issues/38

        t.drop_column('c_array')  # no support yet for null array values in the pytorch dataset

        def _get_mtimes(dir: Path) -> dict[str, float]:
            return {p.name: p.stat().st_mtime for p in dir.iterdir()}

        #  check result cached
        ds1 = t.to_pytorch_dataset(image_format='pt')
        assert isinstance(ds1, PixeltablePytorchDataset)
        ds1_mtimes = _get_mtimes(ds1.path)

        ds2 = t.to_pytorch_dataset(image_format='pt')
        assert isinstance(ds2, PixeltablePytorchDataset)
        ds2_mtimes = _get_mtimes(ds2.path)
        assert ds2.path == ds1.path, 'result should be cached'
        assert ds2_mtimes == ds1_mtimes, 'no extra file system work should have occurred'

        # check invalidation on insert
        t_size = t.count()
        t.insert(row_id=t_size)
        ds3 = t.to_pytorch_dataset(image_format='pt')
        assert isinstance(ds3, PixeltablePytorchDataset)
        assert ds3.path != ds1.path, 'different path should be used'

        # check select list invalidation
        ds4 = t.select(t.row_id).to_pytorch_dataset(image_format='pt')
        assert isinstance(ds4, PixeltablePytorchDataset)
        assert ds4.path != ds3.path, 'different select list, hence different path should be used'

    @pytest.mark.db_roots('local', reason='exports a COCO dataset to the local filesystem')
    def test_to_coco(self, uses_db: None) -> None:
        pytest.importorskip('pycocotools')
        from pycocotools.coco import COCO

        image_files = get_image_files()[:5]
        # each row gets its own category, so that the exported annotations can be matched back to their input row
        rows: list[dict[str, Any]] = [
            {
                'i': i,
                'image': file,
                'annotations': [
                    {'bbox': [i, i + 1, 10 + i, 20 + i], 'category': f'category_{i}'},
                    {'bbox': [2 * i, 3 * i, 5, 7], 'category': f'category_{i}'},
                ],
            }
            for i, file in enumerate(image_files)
        ]
        t = pxt.create_table('images', {'i': pxt.Int, 'image': pxt.Image, 'annotations': pxt.Json})
        validate_update_status(t.insert(rows), expected_rows=len(rows))

        query = t.select({'image': t.image, 'annotations': t.annotations}).order_by(t.i)
        path = query.to_coco_dataset()

        # we get a valid COCO dataset that reproduces the input rows
        coco_ds = COCO(path)
        assert len(coco_ds.imgs) == len(rows)
        assert {cat['name'] for cat in coco_ds.cats.values()} == {f'category_{i}' for i in range(len(rows))}
        for img_id, img_info in coco_ds.imgs.items():
            anns = [coco_ds.anns[ann_id] for ann_id in coco_ds.getAnnIds(imgIds=[img_id])]
            categories = {coco_ds.cats[ann['category_id']]['name'] for ann in anns}
            assert len(categories) == 1, categories
            row = rows[int(categories.pop().removeprefix('category_'))]
            with PIL.Image.open(row['image']) as img:
                assert (img_info['width'], img_info['height']) == img.size
            assert Path(img_info['file_name']).exists()
            assert sorted(ann['bbox'] for ann in anns) == sorted(ann['bbox'] for ann in row['annotations'])
            assert all(ann['area'] == ann['bbox'][2] * ann['bbox'][3] for ann in anns)
            assert all(ann['iscrowd'] == 0 for ann in anns)

        # we call to_coco_dataset() again and get the cached dataset
        new_path = query.to_coco_dataset()
        assert path == new_path

        # the cache is invalidated when we add more data
        validate_update_status(t.insert(rows[:1]), expected_rows=1)
        new_path = query.to_coco_dataset()
        assert path != new_path
        coco_ds = COCO(new_path)
        assert len(coco_ds.imgs) == len(rows) + 1

        # incorrect select list
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            _ = t.select({'image': t.image, 'annotations': t.annotations[0]}).to_coco_dataset()
        assert '"annotations" is not a list' in str(exc_info.value)

        with pxt_raises(pxt.ErrorCode.MISSING_REQUIRED) as exc_info:
            _ = t.select(t.annotations[0]).to_coco_dataset()
        assert 'missing key "image"' in str(exc_info.value).lower()

    def test_distinct(self, db_root: DatabaseRoot, reload_tester: ReloadTester, is_data_versioned: bool) -> None:
        p = db_root.make_catalog_path
        schema: dict[str, Any] = {
            'c1': pxt.String | None,
            'c2': pxt.Int | None,
            'c3': pxt.Float | None,
            'c4': pxt.Timestamp | None,
            'c5': pxt.Json | None,
        }
        t = pxt.create_table(p('test_distinct'), schema, _is_data_versioned=is_data_versioned)
        results = t.distinct().collect()
        assert len(results) == 0
        rows = [
            {'c1': 'SF', 'c2': 100, 'c3': 3.14, 'c4': datetime.datetime(2024, 7, 2), 'c5': {'k1': 'v1'}},
            {'c1': 'SF', 'c2': 100, 'c3': 3.14, 'c4': datetime.datetime(2024, 7, 2), 'c5': {'k1': 'v1'}},
            {'c1': 'SF', 'c2': 101, 'c3': 3.14, 'c4': datetime.datetime(2024, 7, 21), 'c5': {'k2': 'v2'}},
            {'c1': 'SF', 'c2': 101, 'c3': 3.15, 'c4': datetime.datetime.now(), 'c5': {'k3': 'v3'}},
            {'c1': 'LA', 'c2': 101, 'c3': 3.16, 'c4': datetime.datetime(2024, 7, 20), 'c5': {'k1': 'v1'}},
            {'c1': 'LA', 'c2': 104, 'c3': 3.13, 'c4': datetime.datetime(2024, 7, 22), 'c5': {'k2': 'v2'}},
            {'c1': 'LA', 'c2': 104, 'c3': 3.13, 'c4': datetime.datetime(2024, 7, 22), 'c5': {'k1': 'v1'}},
        ]
        t.insert(rows)

        # select all columns
        results = t.select().distinct().collect()
        assert len(results) == 6

        # Test single-column selects
        assert len(t.select(t.c1).distinct().collect()) == 2
        assert len(t.select(t.c2).distinct().collect()) == 3
        assert len(t.select(t.c3).distinct().collect()) == 4
        assert len(t.select(t.c4).distinct().collect()) == 5
        # TODO: fix grouping by json columns; this is currently broken due to CellReconstructionNodes getting
        # inserted into the plan in the wrong place
        # TODO: the same is true for grouping by media columns
        # assert len(t.select(t.c5).distinct().collect()) == 3

        # Test select columns clauses
        assert len(t.select(t.c1, t.c3).distinct().collect()) == 4
        assert len(t.select(t.c1, t.c2).distinct().collect()) == 4
        assert len(t.select(t.c2, t.c3).distinct().collect()) == 5
        assert len(t.select(t.c1, t.c4).distinct().collect()) == 5
        # assert len(t.select(t.c1, t.c5).distinct().collect()) == 5
        # assert len(t.select(t.c4, t.c5).distinct().collect()) == 6

        # Test expressions
        assert len(t.select(t.c2 // 10).distinct().collect()) == 1
        assert len(t.select(t.c2 % 10).distinct().collect()) == 3
        assert len(t.select((t.c2 + 100 * t.c3) // 100).distinct().collect()) == 1
        assert len(t.select(t.c1 + '.pxt', t.c2 - 100).distinct().collect()) == 4

        # Test with filtering
        results = t.select(t.c1).where(t.c2 == 101).distinct().collect()
        assert len(results) == 2
        results = t.select(t.c4).where(t.c2 == 101).distinct().collect()
        assert len(results) == 3

        # Test order_by, limit
        results = t.select().distinct().limit(2).collect()
        assert len(results) == 2
        results = t.select(t.c1).distinct().order_by(t.c1).limit(1).collect()
        assert len(results) == 1
        assert results[0]['c1'] == 'LA'
        results = t.select(t.c1).distinct().order_by(t.c1, asc=False).limit(1).collect()
        assert len(results) == 1
        assert results[0]['c1'] == 'SF'
        results = t.select(t.c4).distinct().show()
        assert len(results) == 5

        # Test head, tail, group by - which will not work
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            _ = t.select(t.c1, t.c3).distinct().head(2)
        assert 'head() cannot be used with group_by' in str(exc_info.value)
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            _ = t.select(t.c1, t.c3).distinct().tail(2)
        assert 'tail() cannot be used with group_by' in str(exc_info.value)
        with pxt_raises(pxt.ErrorCode.INVALID_STATE) as exc_info:
            t.select(t.c1, t.c3).group_by(t.c2).distinct()
        assert 'group_by() already specified' in str(exc_info.value)

        # count() with distinct()
        cnt = t.select(t.c1, t.c3).distinct().count()
        assert cnt == len(t.select(t.c1, t.c3).distinct().collect())

        with pxt_raises(pxt.ErrorCode.INVALID_STATE) as exc_info:
            t.distinct().distinct()
        assert 'group_by() already specified' in str(exc_info.value)

        # select after distinct
        results = t.distinct().select(t.c1).collect()
        assert len(results) == 6
        assert len(results.schema) == 1
        assert 'c1' in results.schema
        results = t.distinct().select(t.c1, t.c3, t.c4).collect()
        assert len(results) == 6
        assert len(results.schema) == 3
        assert 'c1' in results.schema
        assert 'c3' in results.schema
        assert 'c4' in results.schema

    def test_distinct_non_sql_groupable(self, db_root: DatabaseRoot, is_data_versioned: bool) -> None:
        """distinct() must not silently return duplicates when a select item cannot be grouped in SQL."""
        p = db_root.make_catalog_path
        t = pxt.create_table(
            p('test_distinct_json'), {'n': pxt.Int, 'info': pxt.Json}, _is_data_versioned=is_data_versioned
        )
        validate_update_status(
            t.insert(
                [
                    {'n': i, 'info': {'depot': ['north', 'south'][i % 2], 'stops': [{'stop': ['a', 'b'][i % 2]}]}}
                    for i in range(6)
                ]
            ),
            expected_rows=6,
        )

        # Stored scalar columns, including where/order_by/limit, are unchanged.
        assert len(t.select(t.n).distinct().collect()) == 6
        assert len(t.select(t.n).where(t.n >= 3).distinct().collect()) == 3
        first = t.select(t.n).distinct().order_by(t.n).limit(1).collect()
        assert first[0]['n'] == 0
        last = t.select(t.n).where(t.n >= 1).distinct().order_by(t.n, asc=False).limit(1).collect()
        assert last[0]['n'] == 5

        def assert_invalid_grouping(query: pxt.Query, expr_text: str) -> None:
            with pxt_raises(pxt.ErrorCode.INVALID_EXPRESSION, match='needs to be expressible in SQL') as exc_info:
                _ = query.collect()
            message = str(exc_info.value)
            assert expr_text in message
            assert 'computed column' in message

        # JSON path, cast JSON path, and a stored column mixed with a JSON path.
        assert_invalid_grouping(t.select(depot=t.info.depot).distinct(), 'info.depot')
        assert_invalid_grouping(t.select(depot=t.info.depot.astype(pxt.String)).distinct(), 'info.depot')
        assert_invalid_grouping(t.select(t.n, depot=t.info.depot).distinct(), 'info.depot')
        # List index, the shape reported as worst_stops[0].stop.
        assert_invalid_grouping(t.select(stop=t.info.stops[0].stop).distinct(), 'info.stops[0].stop')

        # where / order_by / limit must not bypass the check and return duplicate rows.
        assert_invalid_grouping(t.select(depot=t.info.depot).where(t.n >= 1).distinct(), 'info.depot')
        assert_invalid_grouping(t.select(depot=t.info.depot).distinct().order_by(t.n), 'info.depot')
        assert_invalid_grouping(t.select(depot=t.info.depot).distinct().limit(2), 'info.depot')
        assert_invalid_grouping(
            t.select(depot=t.info.depot).where(t.n >= 1).distinct().order_by(t.n).limit(2), 'info.depot'
        )
        with pxt_raises(pxt.ErrorCode.INVALID_EXPRESSION, match='needs to be expressible in SQL'):
            _ = t.select(depot=t.info.depot).distinct().count()

        # An explicit group_by of the same expression raises the same error.
        assert_invalid_grouping(
            t.group_by(t.info.depot).select(depot=t.info.depot, n=pxt.functions.count(t.n)), 'info.depot'
        )

        # Workaround: store a scalar computed column, then distinct() deduplicates.
        validate_update_status(t.add_computed_column(depot=t.info.depot.astype(pxt.String)), expected_rows=6)
        depots = t.select(t.depot).distinct().order_by(t.depot).collect()
        assert [row['depot'] for row in depots] == ['north', 'south']
        filtered = t.select(t.depot).where(t.n >= 1).distinct().order_by(t.depot, asc=False).limit(1).collect()
        assert filtered[0]['depot'] == 'south'
        assert len(t.select(parity=t.n % 2, depot=t.depot).distinct().collect()) == 2

    def test_distinct_docstring_states_execution_raises_request_error(self) -> None:
        """Query.distinct() documents that execution raises RequestError.

        The SQL-groupable check runs in Analyzer._analyze_agg() when the query is planned, so
        distinct() itself returns a Query. The public docstring must say that, and name RequestError.
        """
        doc = pxt.Query.distinct.__doc__
        assert doc is not None
        assert 'Executing the query raises `RequestError` rather than returning duplicate rows.' in doc
        assert 'distinct() raises `Error`' not in doc

    def test_to_pydantic(self, db_root: DatabaseRoot) -> None:
        p = db_root.make_catalog_path

        class TestModel(pydantic.BaseModel):
            i: int
            s: str
            f: float
            b: bool
            ts: datetime.datetime
            d: dict

        class StrictTestModel(pydantic.BaseModel):
            i: int
            s: str
            f: float
            b: bool
            ts: datetime.datetime
            d: dict
            model_config = pydantic.ConfigDict(extra='forbid')

        schema: dict[str, Any] = {
            'i': pxt.Int | None,
            's': pxt.String | None,
            'f': pxt.Float | None,
            'b': pxt.Bool | None,
            'ts': pxt.Timestamp | None,
            'd': pxt.Json | None,
        }
        t = pxt.create_table(p('pydantic_tbl'), schema)
        t.insert(
            [
                {'i': 1, 's': 'one', 'f': 1.0, 'b': True, 'ts': datetime.datetime(2024, 7, 2), 'd': {'k1': 'v1'}},
                {'i': 2, 's': 'two', 'f': 2.0, 'b': False, 'ts': datetime.datetime(2024, 7, 3), 'd': {'k2': 'v2'}},
                {'i': 3, 's': 'three', 'f': 3.0, 'b': True, 'ts': datetime.datetime(2024, 7, 4), 'd': {'k3': 'v3'}},
            ]
        )
        results = t.collect()
        models = list(results.to_pydantic(TestModel))
        assert len(models) == len(results)
        models = list(results.to_pydantic(StrictTestModel))
        assert len(models) == len(results)

        def extract_fields(exc_info: pytest.ExceptionInfo) -> set[str]:
            # extract missing fields from error message
            fields_str = re.search(r'\{[^}]+\}', str(exc_info.value)).group()
            return set(re.findall(r"'([^']*)'", fields_str))

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match=r'Required model fields .* are missing') as exc_info:
            _ = list(t.select(t.i).collect().to_pydantic(TestModel))
        assert extract_fields(exc_info) == {'s', 'f', 'b', 'ts', 'd'}

        # (s?): dotall mode, needed to match the embedded \n's
        with pxt_raises(
            pxt.ErrorCode.UNSUPPORTED_OPERATION, match=r'(?s)1 validation error .* Input should be a valid integer'
        ):
            _ = list(t.select(t.f, t.s, t.b, t.ts, t.d, i=t.i + 0.1).collect().to_pydantic(TestModel))

        # extra fields
        _ = list(t.select(t.i, t.s, t.f, t.b, t.ts, t.d, extra=t.i + t.f).collect().to_pydantic(TestModel))
        with pxt_raises(
            pxt.ErrorCode.UNSUPPORTED_OPERATION, match=r'Extra fields .* are not allowed in model'
        ) as exc_info:
            _ = list(t.select(t.i, t.s, t.f, t.b, t.ts, t.d, extra=t.i + t.f).collect().to_pydantic(StrictTestModel))
        assert extract_fields(exc_info) == {'extra'}

    def test_to_pydantic_case_insensitive(self, db_root: DatabaseRoot) -> None:
        """Model field names are matched against the folded result column names."""
        p = db_root.make_catalog_path
        t = pxt.create_table(p('t'), {'i': pxt.Int | None, 's': pxt.String | None})
        t.insert([{'i': 1, 's': 'one'}, {'i': 2, 's': 'two'}])

        class MixedCaseModel(pydantic.BaseModel):
            MyInt: int
            MyStr: str

        mixed = list(t.select(MyInt=t.i, MYSTR=t.s).order_by(t.i).collect().to_pydantic(MixedCaseModel))
        assert [m.MyInt for m in mixed] == [1, 2]
        assert [m.MyStr for m in mixed] == ['one', 'two']

        # two fields that fold onto each other cannot both be matched
        class CollidingModel(pydantic.BaseModel):
            Val: int = -1
            val: int = -1

        with pxt_raises(
            pxt.ErrorCode.UNSUPPORTED_OPERATION, match=r"CollidingModel has fields 'Val' and 'val' which are the same"
        ):
            _ = list(t.select(val=t.i).collect().to_pydantic(CollidingModel))

    def test_cursor_lifecycle(self, test_tbl: pxt.Table) -> None:
        query = test_tbl.select(test_tbl.c1, test_tbl.c2, test_tbl.c3).order_by(test_tbl.c2)

        # repr reflects state transitions
        cur = query.cursor()
        assert 'pending' in repr(cur)
        cur.open()
        assert 'open' in repr(cur)
        # double open raises
        with pxt_raises(pxt.ErrorCode.INVALID_STATE, match='Cursor is already open'):
            cur.open()
        cur.close()
        assert 'closed' in repr(cur)
        # double close is a no-op
        cur.close()
        # reopen after close raises
        with pxt_raises(pxt.ErrorCode.INVALID_STATE, match='Cursor is closed and cannot be reopened'):
            cur.open()
        # iterating a closed cursor raises
        with pxt_raises(pxt.ErrorCode.INVALID_STATE, match='Cursor is closed and cannot be iterated upon'):
            list(cur)

        # context manager: partial consumption cleans up
        with query.cursor() as cur:
            list(itertools.islice(cur, 10))
        assert cur._closed

        # auto-open and auto-close on exhaustion without context manager
        cur = query.cursor()
        rows = list(cur)
        assert cur._closed
        assert len(rows) == 100

        # schema
        cur = query.cursor()
        assert list(cur.schema.keys()) == ['c1', 'c2', 'c3']

    def test_cursor_row(self, test_tbl: pxt.Table) -> None:
        query = test_tbl.select(test_tbl.c1, test_tbl.c2, test_tbl.c3).order_by(test_tbl.c2)
        collected = query.collect()

        with query.cursor() as cur:
            for i, row in enumerate(cur):
                # dict-like access
                assert row['c1'] == collected[i]['c1']
                assert row['c2'] == collected[i]['c2']
                assert row['c3'] == collected[i]['c3']
                # keys / values / items
                assert list(row.keys()) == ['c1', 'c2', 'c3']
                assert len(list(row.values())) == 3
                items = list(row.items())
                assert items[0] == ('c1', row['c1'])
                # contains / len
                assert 'c1' in row
                assert 'nonexistent' not in row
                assert len(row) == 3
                # missing key
                with pxt_raises(pxt.ErrorCode.COLUMN_NOT_FOUND, match='does not exist'):
                    row['nonexistent']

    def test_table_cursor(self, db_root: DatabaseRoot) -> None:
        p = db_root.make_catalog_path
        tbl = pxt.create_table(p('cursor_tbl'), {'a': pxt.Int | None, 'b': pxt.String | None})
        tbl.insert([{'a': i, 'b': f'val_{i}'} for i in range(5)])
        rows = list(tbl.cursor())
        assert len(rows) == 5
        assert all('a' in row and 'b' in row for row in rows)

    @pytest.mark.benchmark(group='select_inexpensive')
    @pytest.mark.db_roots('local', reason='Benchmark test is intended for local only')
    def test_select_inexpensive(self, db_root: DatabaseRoot, benchmark: Any) -> None:
        p = db_root.make_catalog_path
        t = pxt.create_table(p('test_inexpensive'), {'c1': pxt.Int | None, 'c2': pxt.String | None})

        row_count = 100000

        t.insert({'c1': i, 'c2': f'str_{i}'} for i in range(row_count))

        def select_inexpensive() -> None:
            res = t.select(t.c1, t.c2, isascii(t.c2), isalpha(t.c2)).collect()
            assert len(res) == row_count

        benchmark(select_inexpensive)

    def test_query_after_column_drop(self, db_root: DatabaseRoot, is_data_versioned: bool) -> None:
        p = db_root.make_catalog_path
        t = pxt.create_table(p('t_drop'), {'a': pxt.Int, 'b': pxt.Int}, _is_data_versioned=is_data_versioned)
        validate_update_status(t.insert([{'a': i, 'b': i * 10} for i in range(10)]), expected_rows=10)
        q = t.select(t.a, t.b)
        assert len(q.collect()) == 10

        t.drop_column('b')

        with pxt_raises(pxt.ErrorCode.COLUMN_NOT_FOUND, match='dropped'):
            q.collect()

    def test_query_after_column_drop_and_add(self, db_root: DatabaseRoot, is_data_versioned: bool) -> None:
        p = db_root.make_catalog_path
        t = pxt.create_table(p('t_readd'), {'a': pxt.Int, 'keep': pxt.Int}, _is_data_versioned=is_data_versioned)
        validate_update_status(t.insert([{'a': 1, 'keep': 0}]), expected_rows=1)
        q = t.select(t.a)
        assert len(q.collect()) == 1

        t.drop_column('a')
        t.add_column(a=pxt.String | None)

        with pxt_raises(pxt.ErrorCode.COLUMN_NOT_FOUND, match='dropped'):
            q.collect()

    def test_query_after_schema_change(self, db_root: DatabaseRoot, is_data_versioned: bool) -> None:
        p = db_root.make_catalog_path
        t = pxt.create_table(p('t_add'), {'c1': pxt.Int | None}, _is_data_versioned=is_data_versioned)
        q_c1 = t.where(t.c1 > 1).select(t.c1)
        q_where = t.where(t.c1 > 1)
        q_select = t.where(t.c1 > 1).select()
        t.insert([{'c1': 1}, {'c1': 2}])

        t.add_column(c2=pxt.Int | None)
        t.add_computed_column(c3=t.c1 * 10)

        for q in (q_where, q_select):
            res = q.collect()
            assert list(res.schema.keys()) == ['c1', 'c2', 'c3']
            assert len(res) == 1
            assert res[0] == {'c1': 2, 'c2': None, 'c3': 20}

        res = q_c1.collect()
        assert list(res.schema.keys()) == ['c1']
        assert len(res) == 1
        assert res[0] == {'c1': 2}

    def test_order_by_after_schema_change(self, db_root: DatabaseRoot, is_data_versioned: bool) -> None:
        p = db_root.make_catalog_path
        # Confirm where/order_by/limit clauses don't capture stale select-list state.
        t = pxt.create_table(
            p('t_add_ob'), {'c1': pxt.Int | None, 'c2': pxt.Int | None}, _is_data_versioned=is_data_versioned
        )
        t.insert([{'c1': i, 'c2': 5 - i} for i in range(5)])
        q = t.where(t.c1 >= 1).order_by(t.c2)
        assert list(q.schema.keys()) == ['c1', 'c2']

        t.add_computed_column(c3=t.c1 * 10)
        res = q.collect()
        assert list(res.schema.keys()) == ['c1', 'c2', 'c3']
        assert len(res) == 4

        t.drop_column(t.c2)
        with pxt_raises(pxt.ErrorCode.COLUMN_NOT_FOUND, match='dropped'):
            _ = q.collect()

    def test_case_insensitive_result_access(self, db_root: DatabaseRoot) -> None:
        p = db_root.make_catalog_path
        t = pxt.create_table(p('t'), {'MyCol': pxt.Int | None})
        t.insert([{'mycol': 1}])

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='Repeated column name'):
            _ = t.select(A=t.MyCol, a=t.MyCol)

        rs = t.select(t.MyCol).collect()
        assert list(rs.schema.keys()) == ['mycol']
        assert rs['MyCol'] == [1]
        assert rs[0, 'MYCOL'] == 1

        # a Row is keyed by the folded column names
        row = next(iter(t.select(t.MyCol).cursor()))
        assert list(row.keys()) == ['mycol']
        assert row['mycol'] == 1

        # same with Row.errors
        t.add_computed_column(Inv=1 // t.MyCol)
        out = t.compute([{'MYCOL': 0}], on_error='ignore')
        assert list(out[0].errors.keys()) == ['inv']
        assert out[0].errors['inv']['errortype'] == 'ZeroDivisionError'
