import datetime
import pickle
import urllib.request
from pathlib import Path
from typing import Any

import bs4
import numpy as np
import PIL.Image
import pytest

import pixeltable as pxt
from pixeltable import catalog, exceptions as excs
from pixeltable.iterators import FrameIterator

from .utils import (
    ReloadTester,
    get_audio_files,
    get_documents,
    get_video_files,
    skip_test_if_not_installed,
    strip_lines,
    validate_update_status,
)


class TestDataFrame:
    def create_join_tbls(self, num_rows: int) -> tuple[catalog.Table, catalog.Table, catalog.Table]:
        t1 = pxt.create_table(f't1_{num_rows}', {'id': pxt.Int, 'i': pxt.Int})
        t2 = pxt.create_table(f't2_{num_rows}', {'id': pxt.Int, 'f': pxt.Float})
        validate_update_status(t1.insert({'id': i, 'i': i} for i in range(num_rows)), expected_rows=num_rows)
        # t2 has matching ids
        validate_update_status(
            t2.insert({'id': i, 'f': float(num_rows - i)} for i in range(num_rows)), expected_rows=num_rows
        )

        # t3:
        # - column i with a different type
        # - only 10% of the ids overlap with t1 and t2
        t3 = pxt.create_table(f't3_{num_rows}', {'id': pxt.Int, 'i': pxt.String, 'f': pxt.Float})
        validate_update_status(
            t3.insert({'id': i, 'i': str(i), 'f': float(num_rows - i)} for i in range(0, 10 * num_rows, 10)),
            expected_rows=num_rows,
        )

        return t1, t2, t3

    def test_select_where(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        res1 = t.collect()
        res2 = t.select().collect()
        assert len(res1) > 0 and res1 == res2

        res1 = t.where(t.c2 < 10).select(t.c1, t.c2, t.c3).collect()

        res3 = t.where(t.c2 < 10).select(c1=t.c1, c2=t.c2, c3=t.c3).collect()
        assert res1 == res3

        res4 = t.where(t.c2 < 10).select(t.c1, c2=t.c2, c3=t.c3).collect()
        assert res1 == res4

        from pixeltable.functions.string import contains

        _ = t.where(contains(t.c1, 'test')).select(t.c1).collect()
        _ = t.where(contains(t.c1, 'test') & contains(t.c1, '1')).select(t.c1).collect()
        _ = t.where(contains(t.c1, 'test') & (t.c2 >= 10)).select(t.c1).collect()

        _ = t.where(t.c2 < 10).select(t.c2, t.c2).collect()  # repeated name no error

        # where clause needs to be a predicate
        with pytest.raises(excs.Error) as exc_info:
            _ = t.where(t.c1).select(t.c2).collect()
        assert 'needs to return bool' in str(exc_info.value)

        # where clause needs to be a predicate
        with pytest.raises(excs.Error) as exc_info:
            _ = t.where(15).select(t.c2).collect()  # type: ignore[arg-type]
        assert 'requires a pixeltable expression' in str(exc_info.value).lower()

        # duplicate select list
        with pytest.raises(excs.Error) as exc_info:
            _ = t.select(t.c1).select(t.c2).collect()
        assert 'already specified' in str(exc_info.value)

        # invalid expr in select list: Callable is not a valid literal
        with pytest.raises(TypeError) as exc_info:
            _ = t.select(datetime.datetime.now).collect()
        assert 'Not a valid literal' in str(exc_info.value)

        # catch invalid name in select list from user input
        # only check stuff that's not caught by python kwargs checker
        with pytest.raises(excs.Error) as exc_info:
            _ = t.select(t.c1, **{'c2-1': t.c2}).collect()
        assert 'Invalid name' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            _ = t.select(t.c1, **{'': t.c2}).collect()
        assert 'Invalid name' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            _ = t.select(t.c1, **{'foo.bar': t.c2}).collect()
        assert 'Invalid name' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            _ = t.select(t.c1, _c3=t.c2).collect()
        assert 'Invalid name' in str(exc_info.value)

        # catch repeated name from user input
        with pytest.raises(excs.Error) as exc_info:
            _ = t.select(t.c2, c2=t.c1).collect()
        assert 'Repeated column name' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            _ = t.select(t.c2 + 1, col_0=t.c2).collect()
        assert 'Repeated column name' in str(exc_info.value)

        # select list contains invalid references
        with pytest.raises(excs.Error) as exc_info:
            t2 = pxt.create_table('t2', {'c1': pxt.Int})
            _ = t.select(t.c1, t2.c1 + t.c2).collect()
        assert 'cannot be evaluated in the context' in str(exc_info.value)

    def test_join(self, reset_db) -> None:
        t1, t2, t3 = self.create_join_tbls(1000)
        # inner join
        df = t1.join(t2, on=t1.id, how='inner').select(t1.i, t2.f, out=t1.i + t2.f).order_by(t2.f)
        pd_df = df.collect().to_pandas()
        assert len(pd_df) == 1000
        assert pd_df.f.is_monotonic_increasing  # correct ordering
        assert (pd_df.out == 1000.0).all()  # correct sum

        # the same inner join, but with redundant join predicates
        df = (
            t1.join(t2, on=(t1.id == t2.id) & (t1.i == t2.id), how='inner')
            .select(t1.i, t2.f, out=t1.i + t2.f)
            .order_by(t2.f)
        )
        pd_df2 = df.collect().to_pandas()
        assert pd_df.equals(pd_df2)

        # left outer join
        df = t1.join(t3, on=t1.id, how='left').select(t1.i, t3.f, out=t1.i + t3.f).order_by(t1.i)
        pd_df = df.collect().to_pandas()
        assert len(pd_df) == 1000
        assert len(pd_df[~pd_df.f.isnull()]) == 100  # correct number of nulls
        assert (pd_df[~pd_df.f.isnull()].out == 1000.0).all()  # correct sum

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
        small_t1, small_t2, _ = self.create_join_tbls(100)
        df = small_t1.join(small_t2, how='cross').select(small_t1.i, small_t2.f, out=small_t1.i + small_t2.f)
        res = df.collect()
        # TODO: verify result

        # inner join with aggregation and explicit join predicate
        df = t1.join(t2, on=t1.id == t2.id).select(pxt.functions.sum(t1.i + t2.id))
        res = df.collect()[0, 0]
        assert res == sum(range(1000)) * 2

        # inner join with grouping aggregation
        df = t1.join(t2, on=t2.id).group_by(t2.id % 10).select(grp=t2.id % 10, val=pxt.functions.sum(t1.i + t2.id))
        res = df.collect()
        pd_df = res.to_pandas()
        # TODO: verify result

    def test_join_errors(self, reset_db) -> None:
        t1, t2, t3 = self.create_join_tbls(1000)

        with pytest.raises(excs.Error) as exc_info:
            _ = t1.join(t2, on=(t2.id, 17)).collect()  # type: ignore[arg-type]
        assert 'must be a sequence of column references or a boolean expression' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            _ = t1.join(t2, on=(15, 27)).collect()  # type: ignore[arg-type]
        assert 'must be a sequence of column references or a boolean expression' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            _ = t1.join(t2, how='cross', on=t2.id).collect()
        assert "'on' not allowed for cross join" in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            _ = t1.join(t2).collect()
        assert "how='inner' requires 'on'" in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            _ = t1.join(t2, on=t2.f).collect()
        assert "'f' not found in any of: t1" in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            _ = t1.join(t2, on=t3.i).collect()
        assert 'expression cannot be evaluated in the context of the joined tables: i' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            _ = t1.join(t2, on=t2.id + 1).collect()
        assert 'boolean expression expected, but got Optional[Int]: id + 1' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            _ = t1.join(t2, on=t2.id).join(t3, on=t3.id).collect()
        assert "ambiguous column reference: 'id'" in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            _ = t1.join(t2, on=t1.i).collect()
        assert "column 'i' not found in joined table" in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            _ = t1.join(t2, on=t1.id, how='inner').head()
        assert 'head() not supported for joins' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            _ = t1.join(t2, on=t1.id, how='inner').tail()
        assert 'tail() not supported for joins' in str(exc_info.value)

    def test_result_set_iterator(self, test_tbl: catalog.Table) -> None:
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

        with pytest.raises(excs.Error) as exc_info:
            _ = res['does_not_exist']
        assert 'Invalid column name' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            _ = res[0, 'does_not_exist']
        assert 'Invalid column name' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            _ = res[0, 0, 0]
        assert 'Bad index' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            _ = res['c2', 0]
        assert 'Bad index' in str(exc_info.value)

    def test_order_by(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        res = t.select(t.c4, t.c2).order_by(t.c4).order_by(t.c2, asc=False).collect()

        # invalid expr in order_by()
        with pytest.raises(excs.Error) as exc_info:
            _ = t.order_by(datetime.datetime.now()).collect()  # type: ignore[arg-type]
        assert 'Invalid expression' in str(exc_info.value)

    def test_expr_unique_id(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        # Multiple constants with the same string representation but different types must be unique (expr.id)
        res = t.select(t.c2, t.c1, t.c1 =='2', t.c1 < '4', t.c2 == 4).limit(4).collect()
        print(res)
        assert len(res) == 4

    def test_limit(self, test_tbl: catalog.Table, reload_tester: ReloadTester) -> None:
        t = test_tbl
        nrows = 3
        res = t.select(t.c4).limit(nrows).collect()
        assert len(res) == nrows

        @pxt.query
        def get_lim(n: int):
            return t.select(t.c4).limit(n)

        res = t.select(t.c4, get_lim(2)).collect()
        print(res)
        print(res[0]['get_lim'])
        assert res[0]['get_lim'] == [{'c4': False}, {'c4': True}]

        with pytest.raises(excs.Error, match='must be of type int'):
            _ = t.limit(5.3).collect()

        results = reload_tester.run_query(
            t.select(t.c4, get_lim(3)).limit(3)
        )
        print(results.schema)
        reload_tester.run_reload_test()

    def test_head_tail(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        res = t.head(10).to_pandas()
        assert np.all(res.c2 == list(range(10)))
        # Where is applied
        res = t.where(t.c2 > 9).head(10).to_pandas()
        assert np.all(res.c2 == list(range(10, 20)))
        # order_by() is an error
        with pytest.raises(excs.Error) as exc_info:
            _ = t.order_by(t.c2).head(10)
        assert 'cannot be used with order_by' in str(exc_info.value)

        res = t.tail().to_pandas()
        assert np.all(res.c2 == list(range(90, 100)))
        res = t.where(t.c2 < 90).tail().to_pandas()
        assert np.all(res.c2 == list(range(80, 90)))
        # order_by() is an error
        with pytest.raises(excs.Error) as exc_info:
            _ = t.order_by(t.c2).tail(10)
        assert 'cannot be used with order_by' in str(exc_info.value)

    def test_repr(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        df = t.select(t.c1, t.c1.upper(), t.c2 + 5).where(t.c2 < 10).group_by(t.c1).order_by(t.c3).limit(10)
        df.describe()

        r = df.__repr__()
        assert strip_lines(r) == strip_lines(
            """Name              Type  Expression
               c1  Required[String]          c1
            upper  Required[String]  c1.upper()
            col_2     Required[Int]      c2 + 5

            From      test_tbl
            Where      c2 < 10
            Group By        c1
            Order By    c3 asc
            Limit           10"""
        )
        _ = df._repr_html_()  # TODO: Is there a good way to test this output?

    def test_count(self, test_tbl: catalog.Table, small_img_tbl) -> None:
        t = test_tbl
        cnt = t.count()
        assert cnt == 100

        cnt = t.where(t.c2 < 10).count()
        assert cnt == 10

        # for now, count() doesn't work with non-SQL Where clauses
        t = small_img_tbl
        with pytest.raises(excs.Error):
            _ = t.where(t.img.width > 100).count()

    def test_select_literal(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        res = t.select(1.0).where(t.c2 < 10).collect()
        assert res[next(iter(res.schema.keys()))] == [1.0] * 10

    def test_html_media_url(self, reset_db) -> None:
        tab = pxt.create_table('test_html_repr', {'video': pxt.Video, 'audio': pxt.Audio, 'doc': pxt.Document})

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

    def test_update_delete_where(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        old: list[int] = t.select(t.c3).collect()['c3']

        # Update with where
        validate_update_status(t.where(t.c2 >= 50).update({'c3': 4171780.0}), expected_rows=50)
        new: list[int] = t.select(t.c3).collect()['c3']
        assert new[:50] == old[:50]
        assert all(new[i] == 4171780.0 for i in range(51, len(new)))

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

        # select_list

        with pytest.raises(excs.Error) as exc_info:
            t.select(t.c2).update({'c3': 0.0})
        assert 'Cannot use `update` after `select`' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            t.select(t.c2).delete()
        assert 'Cannot use `delete` after `select`' in str(exc_info.value)

        # group_by

        with pytest.raises(excs.Error) as exc_info:
            t.group_by(t.c2).update({'c3': 0.0})
        assert 'Cannot use `update` after `group_by`' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            t.group_by(t.c2).delete()
        assert 'Cannot use `delete` after `group_by`' in str(exc_info.value)

        # order_by

        with pytest.raises(excs.Error) as exc_info:
            t.order_by(t.c2).update({'c3': 0.0})
        assert 'Cannot use `update` after `order_by`' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            t.order_by(t.c2).delete()
        assert 'Cannot use `delete` after `order_by`' in str(exc_info.value)

        # limit

        with pytest.raises(excs.Error) as exc_info:
            t.limit(10).update({'c3': 0.0})
        assert 'Cannot use `update` after `limit`' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            t.limit(10).delete()
        assert 'Cannot use `delete` after `limit`' in str(exc_info.value)

        # grouping_tbl

        t2 = pxt.create_table('test_tbl_2', {'name': pxt.StringType(), 'video': pxt.VideoType()})
        v2 = pxt.create_view('test_view_2', t2, iterator=FrameIterator.create(video=t2.video, fps=1))
        with pytest.raises(excs.Error) as exc_info:
            v2.select(pxt.functions.video.make_video(v2.pos, v2.frame)).group_by(t2).update({'name': 'test'})
        assert 'Cannot use `update` after `group_by`' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            v2.select(pxt.functions.video.make_video(v2.pos, v2.frame)).group_by(t2).delete()
        assert 'Cannot use `delete` after `group_by`' in str(exc_info.value)

        # delete from view
        with pytest.raises(excs.Error) as exc_info:
            v2.where(t.c2 < 10).delete()
        assert 'Cannot delete from view' in str(exc_info.value)

        # update snapshot
        snap = pxt.create_snapshot('test_snapshot', t)
        with pytest.raises(excs.Error) as exc_info:
            snap.where(t.c2 < 10).update({'c3': 0.0})
        assert 'Cannot update a snapshot' in str(exc_info.value)

        # delete from snapshot
        with pytest.raises(excs.Error) as exc_info:
            snap.where(t.c2 < 10).delete()
        assert 'Cannot delete from view' in str(exc_info.value)

    def test_to_pytorch_dataset(self, all_datatypes_tbl: catalog.Table) -> None:
        """tests all types are handled correctly in this conversion"""
        skip_test_if_not_installed('torch')
        skip_test_if_not_installed('pyarrow')
        import torch

        t = all_datatypes_tbl
        df = t.where(t.row_id < 1)
        assert df.count() > 0
        ds = df.to_pytorch_dataset()
        for tup in ds:
            for col in df.schema.keys():
                assert col in tup

            arrval = tup['c_array']
            assert isinstance(arrval, np.ndarray)
            col_type = df.schema['c_array']
            assert isinstance(col_type, pxt.ArrayType)
            assert arrval.dtype == col_type.numpy_dtype()
            assert arrval.shape == col_type.shape
            assert arrval.dtype == np.float32
            assert arrval.flags['WRITEABLE'], 'required by pytorch collate function'

            assert isinstance(tup['c_bool'], bool)
            assert isinstance(tup['c_int'], int)
            assert isinstance(tup['c_float'], float)
            assert isinstance(tup['c_timestamp'], float)
            assert torch.is_tensor(tup['c_image'])
            assert isinstance(tup['c_video'], str)
            assert isinstance(tup['c_json'], dict)

    def test_to_pytorch_image_format(self, all_datatypes_tbl: catalog.Table) -> None:
        """tests the image_format parameter is honored"""
        skip_test_if_not_installed('torch')
        skip_test_if_not_installed('torchvision')
        skip_test_if_not_installed('pyarrow')
        import torch
        import torchvision.transforms as T  # type: ignore[import-untyped]

        W, H = 220, 224  # make different from each other
        t = all_datatypes_tbl
        df = t.select(t.row_id, t.c_image, c_image_xformed=t.c_image.resize([W, H]).convert('RGB')).where(t.row_id < 1)

        pandas_df = df.show().to_pandas()
        im_plain = pandas_df['c_image'].values[0]
        im_xformed = pandas_df['c_image_xformed'].values[0]
        assert pandas_df.shape[0] == 1

        ds = df.to_pytorch_dataset(image_format='np')
        ds_ptformat = df.to_pytorch_dataset(image_format='pt')

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

            assert arr_xformed.shape == (H, W, 3)
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
            assert arr_pt.shape == (3, H, W)
            assert arr_pt.dtype == torch.float32
            assert (0.0 <= arr_pt).all()
            assert (arr_pt <= 1.0).all()
            assert torch.isclose(T.ToTensor()(arr_xformed), arr_pt).all(), (
                'pytorch image should be consistent with numpy image'
            )
            elt_count += 1
        assert elt_count == 1

    @pytest.mark.skip('Flaky test (fails intermittently)')
    def test_to_pytorch_dataloader(self, all_datatypes_tbl: catalog.Table) -> None:
        """Tests the dataset works well with pytorch dataloader:
        1. compatibility with multiprocessing
        2. compatibility of all types with default collate_fn
        """
        skip_test_if_not_installed('torch')
        import torch.utils.data

        @pxt.udf
        def restrict_json_for_default_collate(obj: pxt.Json) -> pxt.Json:
            keys = ['id', 'label', 'iscrowd', 'bounding_box']
            return {k: obj[k] for k in keys}

        t = all_datatypes_tbl
        df = t.select(
            t.row_id,
            t.c_int,
            t.c_float,
            t.c_bool,
            t.c_timestamp,
            t.c_array,
            t.c_video,
            # default collate_fn doesnt support null values, nor lists of different lengths
            # but does allow some dictionaries if they are uniform
            c_json=restrict_json_for_default_collate(t.c_json.detections[0]),
            # images must be uniform shape for pytorch collate_fn to not fail
            c_image=t.c_image.resize([220, 224]).convert('RGB'),
        )
        df_size = df.count()
        ds = df.to_pytorch_dataset(image_format='pt')
        # test serialization:
        #  - pickle.dumps() and pickle.loads() must work so that
        #   we can use num_workers > 0
        x = pickle.dumps(ds)
        _ = pickle.loads(x)

        # test we get all rows
        def check_recover_all_rows(ds, size: int, **kwargs):
            dl = torch.utils.data.DataLoader(ds, **kwargs)
            loaded_ids = set()
            for batch in dl:
                for row_id in batch['row_id']:
                    val = int(row_id)  # np.int -> int or will fail set equality test below.
                    assert val not in loaded_ids, val
                    loaded_ids.add(val)

            assert loaded_ids == set(range(size))

        # check different number of workers
        check_recover_all_rows(ds, size=df_size, batch_size=3, num_workers=0)  # within this process
        check_recover_all_rows(ds, size=df_size, batch_size=3, num_workers=2)  # two separate processes

        # check edge case where some workers get no rows
        short_size = 1
        df_short = df.where(t.row_id < short_size)
        ds_short = df_short.to_pytorch_dataset(image_format='pt')
        check_recover_all_rows(ds_short, size=short_size, batch_size=13, num_workers=short_size + 1)

    def test_pytorch_dataset_caching(self, all_datatypes_tbl: catalog.Table) -> None:
        """Tests that dataset caching works
        1. using the same dataset twice in a row uses the cache
        2. adding a row to the table invalidates the cached version
        3. changing the select list invalidates the cached version
        """
        skip_test_if_not_installed('torch')
        skip_test_if_not_installed('pyarrow')
        from pixeltable.utils.pytorch import PixeltablePytorchDataset

        t = all_datatypes_tbl

        t.drop_column('c_video')  # null value video column triggers internal assertions in DataRow
        # see https://github.com/pixeltable/pixeltable/issues/38

        t.drop_column('c_array')  # no support yet for null array values in the pytorch dataset

        def _get_mtimes(dir: Path):
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

    def test_to_coco(self, reset_db) -> None:
        skip_test_if_not_installed('yolox')
        from pycocotools.coco import COCO

        from pixeltable.ext.functions.yolox import yolo_to_coco, yolox

        base_t = pxt.create_table('videos', {'video': pxt.VideoType()})
        view_t = pxt.create_view('frames', base_t, iterator=FrameIterator.create(video=base_t.video, fps=1))
        view_t.add_computed_column(detections=yolox(view_t.frame, model_id='yolox_m'))
        base_t.insert(video=get_video_files()[0])

        query = view_t.select({'image': view_t.frame, 'annotations': yolo_to_coco(view_t.detections)})
        path = query.to_coco_dataset()
        # we get a valid COCO dataset
        coco_ds = COCO(path)
        assert len(coco_ds.imgs) == view_t.count()

        # we call to_coco_dataset() again and get the cached dataset
        new_path = query.to_coco_dataset()
        assert path == new_path

        # the cache is invalidated when we add more data
        base_t.insert(video=get_video_files()[1])
        new_path = query.to_coco_dataset()
        assert path != new_path
        coco_ds = COCO(new_path)
        assert len(coco_ds.imgs) == view_t.count()

        # incorrect select list
        with pytest.raises(excs.Error) as exc_info:
            _ = view_t.select({'image': view_t.frame, 'annotations': view_t.detections}).to_coco_dataset()
        assert '"annotations" is not a list' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            _ = view_t.select(view_t.detections).to_coco_dataset()
        assert 'missing key "image"' in str(exc_info.value).lower()
