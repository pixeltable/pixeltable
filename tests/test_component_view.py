from typing import Any

import numpy as np
import pandas as pd
import PIL
import pytest

import pixeltable as pxt
import pixeltable.type_system as ts
from pixeltable.iterators import ComponentIterator
from pixeltable.iterators.video import FrameIterator

from .utils import assert_resultset_eq, get_test_video_files, reload_catalog, validate_update_status


class ConstantImgIterator(ComponentIterator):
    """Component iterator that generates a fixed number of all-black 1280x720 images."""

    def __init__(self, video: str, *, num_frames: int = 10):
        self.img = PIL.Image.new('RGB', (1280, 720))
        self.next_frame_idx = 0
        self.num_frames = num_frames
        self.pos_msec = 0.0
        self.pos_frame = 0.0

    @classmethod
    def input_schema(cls) -> dict[str, ts.ColumnType]:
        return {'video': ts.VideoType(nullable=False), 'fps': ts.FloatType()}

    @classmethod
    def output_schema(cls, *args: Any, **kwargs: Any) -> tuple[dict[str, ts.ColumnType], list[str]]:
        return {
            'frame_idx': ts.IntType(),
            'pos_msec': ts.FloatType(),
            'pos_frame': ts.FloatType(),
            'frame': ts.ImageType(),
        }, ['frame']

    def __next__(self) -> dict[str, Any]:
        while True:
            if self.next_frame_idx == self.num_frames:
                raise StopIteration
            result = {
                'frame_idx': self.next_frame_idx,
                'pos_msec': self.pos_msec,
                'pos_frame': self.pos_frame,
                'frame': self.img,
            }
            self.next_frame_idx += 1
            return result

    def close(self) -> None:
        pass

    def set_pos(self, pos: int) -> None:
        if pos == self.next_frame_idx:
            return
        self.next_frame_idx = pos


class TestComponentView:
    def test_basic(self, reset_db: None) -> None:
        # create video table
        schema = {'video': pxt.Video, 'angle': pxt.Int, 'other_angle': pxt.Int}
        video_t = pxt.create_table('video_tbl', schema)
        video_filepaths = get_test_video_files()

        # cannot add 'pos' column
        with pytest.raises(pxt.Error) as excinfo:
            video_t.add_column(pos=pxt.Int)
        assert 'reserved' in str(excinfo.value)

        # parameter missing
        with pytest.raises(pxt.Error) as excinfo:
            _ = pxt.create_view('test_view', video_t, iterator=FrameIterator.create(fps=1))
        assert 'missing a required argument' in str(excinfo.value)

        # bad parameter type
        with pytest.raises(pxt.Error) as excinfo:
            _ = pxt.create_view('test_view', video_t, iterator=FrameIterator.create(video=video_t.video, fps='1'))
        assert 'argument type String does not match parameter type Optional[Float]' in str(excinfo.value)

        # bad parameter type
        with pytest.raises(pxt.Error) as excinfo:
            _ = pxt.create_view('test_view', video_t, iterator=FrameIterator.create(video=1, fps=1))
        assert 'argument type Int does not match parameter type Video' in str(excinfo.value)

        # create frame view
        view_t = pxt.create_view('test_view', video_t, iterator=FrameIterator.create(video=video_t.video, fps=1))
        # computed column that references a column from the base
        view_t.add_computed_column(angle2=view_t.angle + 1)
        # computed column that references an unstored and a stored computed view column
        view_t.add_computed_column(v1=view_t.frame.rotate(view_t.angle2), stored=True)
        # computed column that references a stored computed column from the view
        view_t.add_computed_column(v2=view_t.frame_idx - 1)
        # computed column that references an unstored view column and a column from the base; the stored value
        # cannot be materialized in SQL directly
        view_t.add_computed_column(v3=view_t.frame.rotate(video_t.other_angle), stored=True)

        # and load data
        rows = [{'video': p, 'angle': 30, 'other_angle': -30} for p in video_filepaths]
        status = video_t.insert(rows)
        assert status.num_excs == 0
        reload_catalog()
        view_t = pxt.get_table('test_view')
        # pos and frame_idx are identical
        res = view_t.select(view_t.pos, view_t.frame_idx).collect().to_pandas()
        assert np.all(res['pos'] == res['frame_idx'])

        video_url = video_t.select(video_t.video.fileurl).collect()[0, 0]
        result = (
            view_t.where(view_t.video == video_url)
            .select(view_t.frame_idx)
            .order_by(view_t.frame_idx)
            .collect()
            .to_pandas()
        )
        assert len(result) > 0
        assert np.all(result['frame_idx'] == pd.Series(range(len(result))))

    def test_add_column(self, reset_db: None) -> None:
        # create video table
        video_t = pxt.create_table('video_tbl', {'video': pxt.Video})
        video_filepaths = get_test_video_files()
        # create frame view
        view_t = pxt.create_view('test_view', video_t, iterator=FrameIterator.create(video=video_t.video, fps=1))

        rows = [{'video': p} for p in video_filepaths]
        validate_update_status(video_t.insert(rows))
        # adding a non-computed column backfills it with nulls
        validate_update_status(view_t.add_column(annotation=pxt.Json))
        assert view_t.count() == view_t.where(view_t.annotation == None).count()
        # adding more data via the base table sets the column values to null
        validate_update_status(video_t.insert(rows))
        assert view_t.count() == view_t.where(view_t.annotation == None).count()

        with pytest.raises(pxt.Error) as excinfo:
            view_t.add_column(annotation=pxt.Required[pxt.Json])
        assert "Duplicate column name: 'annotation'" in str(excinfo.value)

    def test_update(self, reset_db: None) -> None:
        # create video table
        video_t = pxt.create_table('video_tbl', {'video': pxt.Video})
        # create frame view with manually updated column
        view_t = pxt.create_view(
            'test_view',
            video_t,
            additional_columns={'annotation': pxt.Json},
            iterator=FrameIterator.create(video=video_t.video, fps=1),
        )

        video_filepaths = get_test_video_files()
        rows = [{'video': p} for p in video_filepaths]
        status = video_t.insert(rows)
        assert status.num_excs == 0
        import urllib

        video_url = urllib.parse.urljoin('file:', urllib.request.pathname2url(video_filepaths[0]))
        validate_update_status(
            view_t.update({'annotation': {'a': 1}}, where=view_t.video == video_url),
            expected_rows=view_t.where(view_t.video == video_url).count(),
        )
        assert view_t.where(view_t.annotation != None).count() == view_t.where(view_t.video == video_url).count()

        # batch update with _rowid works
        validate_update_status(
            view_t.batch_update(
                [{'annotation': {'a': 1}, '_rowid': (1, 0)}, {'annotation': {'a': 1}, '_rowid': (1, 1)}]
            ),
            expected_rows=2,
        )
        with pytest.raises(AssertionError):
            # malformed _rowid
            view_t.batch_update([{'annotation': {'a': 1}, '_rowid': (1,)}])

        with pytest.raises(pxt.Error) as excinfo:
            _ = pxt.create_view(
                'bad_view',
                video_t,
                additional_columns={'annotation': pxt.Required[pxt.Json]},
                iterator=FrameIterator.create(video=video_t.video, fps=1),
            )
        assert 'must be nullable' in str(excinfo.value)

    @pytest.mark.parametrize('has_column,has_filter', [(False, False), (True, False), (False, True), (True, True)])
    def test_snapshot(self, has_column: bool, has_filter: bool, reset_db: None) -> None:
        for reload_md in [False, True]:
            reload_catalog()
            self.run_snapshot_test(has_column=has_column, has_filter=has_filter, reload_md=reload_md)

    def run_snapshot_test(self, has_column: bool, has_filter: bool, reload_md: bool) -> None:
        base_path = 'video_tbl'
        view_path = 'test_view'
        snap_path = 'test_snap'

        # create video table
        video_t = pxt.create_table(base_path, {'video': pxt.Video, 'margin': pxt.Int})
        video_filepaths = get_test_video_files()
        rows = [{'video': path, 'margin': i * 10} for i, path in enumerate(video_filepaths)]
        status = video_t.insert(rows)
        assert status.num_rows == len(rows)
        assert status.num_excs == 0

        # create frame view with a computed column
        view_t = pxt.create_view(view_path, video_t, iterator=ConstantImgIterator.create(video=video_t.video))
        view_t.add_computed_column(
            cropped=view_t.frame.crop([view_t.margin, view_t.margin, view_t.frame.width, view_t.frame.height]),
            stored=True,
        )
        snap_col_expr = [view_t.cropped.width * view_t.cropped.height] if has_column else []
        view_query = view_t.select(
            view_t.margin,
            view_t.frame.width,
            view_t.frame.height,
            view_t.cropped.width,
            view_t.cropped.height,
            *snap_col_expr,
        ).order_by(view_t.video, view_t.pos)
        if has_filter:
            view_query = view_query.where(view_t.frame_idx < 10)
        orig_resultset = view_query.collect()

        # create snapshot of view
        query: pxt.Table | pxt.DataFrame = view_t.where(view_t.frame_idx < 10) if has_filter else view_t
        schema = {'c1': view_t.cropped.width * view_t.cropped.height} if has_column else {}
        snap_t = pxt.create_snapshot(snap_path, query, additional_columns=schema)
        snap_cols = [snap_t.c1] if has_column else []
        snap_query = snap_t.select(
            snap_t.margin,
            snap_t.frame.width,
            snap_t.frame.height,
            snap_t.cropped.width,
            snap_t.cropped.height,
            *snap_cols,
        ).order_by(snap_t.video, snap_t.pos)
        assert_resultset_eq(snap_query.collect(), orig_resultset)

        if reload_md:
            reload_catalog()
            video_t = pxt.get_table(base_path)
            snap_t = pxt.get_table(snap_path)
            snap_cols = [snap_t.c1] if has_column else []
            snap_query = snap_t.select(
                snap_t.margin,
                snap_t.frame.width,
                snap_t.frame.height,
                snap_t.cropped.width,
                snap_t.cropped.height,
                *snap_cols,
            ).order_by(snap_t.video, snap_t.pos)

        # snapshot is unaffected by base insert()
        status = video_t.insert(rows)
        assert status.num_excs == 0
        assert_resultset_eq(snap_query.collect(), orig_resultset)

        # snapshot is unaffected by base update()
        status = video_t.update({'margin': video_t.margin + 1})
        assert status.num_excs == 0
        assert_resultset_eq(snap_query.collect(), orig_resultset)

        # snapshot is unaffected by base delete()
        status = video_t.delete()
        assert status.num_excs == 0
        assert_resultset_eq(snap_query.collect(), orig_resultset)

        pxt.drop_table(snap_path)
        pxt.drop_table(view_path)
        pxt.drop_table(base_path)

    def test_chained_views(self, reset_db: None) -> None:
        """Component view followed by a standard view"""
        # create video table
        schema = {'video': pxt.Video, 'int1': pxt.Int, 'int2': pxt.Int}
        video_t = pxt.create_table('video_tbl', schema)
        video_filepaths = get_test_video_files()

        # create first view
        v1 = pxt.create_view('test_view', video_t, iterator=ConstantImgIterator.create(video=video_t.video))
        # computed column that references stored base column
        v1.add_computed_column(int3=v1.int1 + 1)
        # stored computed column that references an unstored and a stored computed view column
        v1.add_computed_column(img1=v1.frame.crop([v1.int3, v1.int3, v1.frame.width, v1.frame.height]), stored=True)
        # computed column that references a stored computed view column
        v1.add_computed_column(int4=v1.frame_idx + 1)
        # unstored computed column that references an unstored and a stored computed view column
        v1.add_computed_column(img2=v1.frame.crop([v1.int4, v1.int4, v1.frame.width, v1.frame.height]), stored=False)

        # create second view
        v2 = pxt.create_view('chained_view', v1)
        # computed column that references stored video_t column
        v2.add_computed_column(int5=v2.int1 + 1)
        v2.add_computed_column(int6=v2.int2 + 1)
        # stored computed column that references a stored base column and a stored computed view column;
        # indirectly references int1
        v2.add_computed_column(img3=v2.img1.crop([v2.int5, v2.int5, v2.img1.width, v2.img1.height]), stored=True)
        # stored computed column that references an unstored base column and a manually updated column from video_t;
        # indirectly references int2
        v2.add_computed_column(img4=v2.img2.crop([v2.int6, v2.int6, v2.img2.width, v2.img2.height]), stored=True)
        # comuted column that indirectly references int1 and int2
        v2.add_computed_column(int7=v2.img3.width + v2.img4.width)

        def check_view() -> None:
            assert_resultset_eq(
                v1.select(v1.int3).order_by(v1.video, v1.pos).collect(),
                v1.select(v1.int1 + 1).order_by(v1.video, v1.pos).collect(),
            )
            assert_resultset_eq(
                v1.select(v1.int4).order_by(v1.video, v1.pos).collect(),
                v1.select(v1.frame_idx + 1).order_by(v1.video, v1.pos).collect(),
            )
            assert_resultset_eq(
                v1.select(v1.video, v1.img1.width, v1.img1.height).order_by(v1.video, v1.pos).collect(),
                v1.select(v1.video, v1.frame.width - v1.int1 - 1, v1.frame.height - v1.int1 - 1)
                .order_by(v1.video, v1.pos)
                .collect(),
            )
            assert_resultset_eq(
                v2.select(v2.int5).order_by(v2.video, v2.pos).collect(),
                v2.select(v2.int1 + 1).order_by(v2.video, v2.pos).collect(),
            )
            assert_resultset_eq(
                v2.select(v2.int6).order_by(v2.video, v2.pos).collect(),
                v2.select(v2.int2 + 1).order_by(v2.video, v2.pos).collect(),
            )
            assert_resultset_eq(
                v2.select(v2.video, v2.img3.width, v2.img3.height).order_by(v2.video, v2.pos).collect(),
                v2.select(v2.video, v2.frame.width - v2.int1 * 2 - 2, v2.frame.height - v2.int1 * 2 - 2)
                .order_by(v2.video, v2.pos)
                .collect(),
            )
            assert_resultset_eq(
                v2.select(v2.video, v2.img4.width, v2.img4.height).order_by(v2.video, v2.pos).collect(),
                v2.select(
                    v2.video, v2.frame.width - v2.frame_idx - v2.int2 - 2, v2.frame.height - v2.frame_idx - v2.int2 - 2
                )
                .order_by(v2.video, v2.pos)
                .collect(),
            )
            assert_resultset_eq(
                v2.select(v2.int7).order_by(v2.video, v2.pos).collect(),
                v2.select(v2.img3.width + v2.img4.width).order_by(v2.video, v2.pos).collect(),
            )
            assert_resultset_eq(
                v2.select(v2.int7).order_by(v2.video, v2.pos).collect(),
                v2.select(v2.frame.width - v2.int1 * 2 - 2 + v2.frame.width - v2.frame_idx - v2.int2 - 2)
                .order_by(v2.video, v2.pos)
                .collect(),
            )

        # load data
        rows = [{'video': p, 'int1': i, 'int2': len(video_filepaths) - i} for i, p in enumerate(video_filepaths)]
        status = video_t.insert(rows)
        assert status.num_rows == video_t.count() + v1.count() + v2.count()
        check_view()

        # update int1: propagates to int3, img1, int5, img3, int7
        # TODO: how to test that img4 doesn't get recomputed as part of the computation of int7?
        # need to collect more runtime stats (eg, called functions)
        import urllib

        video_url = urllib.parse.urljoin('file:', urllib.request.pathname2url(video_filepaths[0]))
        status = video_t.update({'int1': video_t.int1 + 1}, where=video_t.video == video_url)
        assert status.num_rows == 1 + v1.where(v1.video == video_url).count() + v2.where(v2.video == video_url).count()
        assert sorted(str.split('.')[1] for str in status.updated_cols) == [
            'img1',
            'img3',
            'int1',
            'int3',
            'int5',
            'int7',
        ]
        check_view()

        # update int2: propagates to img4, int6, int7
        status = video_t.update({'int2': video_t.int2 + 1}, where=video_t.video == video_url)
        assert status.num_rows == 1 + v2.where(v2.video == video_url).count()
        assert sorted(str.split('.')[1] for str in status.updated_cols) == ['img4', 'int2', 'int6', 'int7']
        check_view()
