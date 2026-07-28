import threading
import time
from typing import Callable

import numpy as np
import PIL.Image
import pytest

import pixeltable as pxt
from pixeltable.functions.video import frame_iterator

from .utils import get_image_files, get_video_files, reload_catalog, validate_update_status

# Module-level UDFs: computed columns require module UDFs (serialized by path).


@pxt.udf(run_in_thread=True)
def threaded_inc(x: int) -> int:
    return x + 1


@pxt.udf(run_in_thread=True)
def threaded_thread_name(x: int) -> str:
    return threading.current_thread().name


@pxt.udf
def inline_thread_name(x: int) -> str:
    return threading.current_thread().name


# set by tests that need worker-side coordination/observation
_barriers: dict[str, threading.Barrier] = {}
_call_log: list[int] = []


@pxt.udf(run_in_thread=True)
def threaded_barrier_wait(x: int) -> int:
    _barriers['b'].wait(timeout=10)
    return x


@pxt.udf(run_in_thread=True)
def threaded_log_call(x: int) -> int:
    _call_log.append(x)
    return x


@pxt.udf(run_in_thread=True)
def threaded_sleep_inv(x: int, n: int) -> int:
    # earlier rows sleep longer, so completion order inverts input order
    time.sleep((n - x) * 0.01)
    return x


@pxt.udf(run_in_thread=True)
def threaded_raise_on_2(x: int) -> int:
    if x == 2:
        raise ValueError(f'boom {x}')
    return x


@pxt.udf
def inline_raise_on_2(x: int) -> int:
    if x == 2:
        raise ValueError(f'boom {x}')
    return x


class TestThreadedEval:
    def test_routing(self, make_catalog_path: Callable[[str], str]) -> None:
        # a run_in_thread UDF executes on the cpu pool, an unflagged UDF stays on the caller thread
        t = pxt.create_table(make_catalog_path('test'), {'c1': pxt.Int})
        validate_update_status(t.insert({'c1': i} for i in range(10)), expected_rows=10)
        res = t.select(threaded=threaded_thread_name(t.c1), inline=inline_thread_name(t.c1)).collect()
        assert all(row['threaded'].startswith('pxt-cpu') for row in res)
        assert all(not row['inline'].startswith('pxt-cpu') for row in res)

    @pytest.mark.local('coordinates with the in-process barrier')
    def test_parallelism(self, make_catalog_path: Callable[[str], str]) -> None:
        # the barrier only releases if both rows execute concurrently; serial execution would
        # time out the first waiter and surface a BrokenBarrierError
        _barriers['b'] = threading.Barrier(2)
        t = pxt.create_table(make_catalog_path('test'), {'c1': pxt.Int})
        validate_update_status(t.insert([{'c1': 0}, {'c1': 1}]), expected_rows=2)
        res = t.select(out=threaded_barrier_wait(t.c1)).collect()
        assert sorted(row['out'] for row in res) == [0, 1]

    def test_ordering(self, make_catalog_path: Callable[[str], str]) -> None:
        # results come back in input order even though completion order is inverted
        n = 20
        t = pxt.create_table(make_catalog_path('test'), {'c1': pxt.Int})
        validate_update_status(t.insert({'c1': i} for i in range(n)), expected_rows=n)
        res = t.order_by(t.c1).select(t.c1, out=threaded_sleep_inv(t.c1, n)).collect()
        assert [row['out'] for row in res] == [row['c1'] for row in res]

    @pytest.mark.local('observes the in-process call log')
    def test_none_short_circuit(self, make_catalog_path: Callable[[str], str]) -> None:
        # a None arg to a non-nullable parameter yields None without invoking the UDF
        _call_log.clear()
        t = pxt.create_table(make_catalog_path('test'), {'c1': pxt.Int})
        validate_update_status(t.insert([{'c1': 1}, {'c1': None}, {'c1': 3}]), expected_rows=3)
        res = t.select(t.c1, out=threaded_log_call(t.c1)).collect()
        assert [row['out'] for row in res] == [row['c1'] for row in res]
        assert sorted(_call_log) == [1, 3]

    def test_error_parity(self, make_catalog_path: Callable[[str], str]) -> None:
        # a raising thread-pool UDF surfaces the same per-cell errors as its inline twin
        t1 = pxt.create_table(make_catalog_path('threaded'), {'c1': pxt.Int})
        t1.add_computed_column(out=threaded_raise_on_2(t1.c1))
        status1 = t1.insert(({'c1': i} for i in range(4)), on_error='ignore')
        t2 = pxt.create_table(make_catalog_path('inline'), {'c1': pxt.Int})
        t2.add_computed_column(out=inline_raise_on_2(t2.c1))
        status2 = t2.insert(({'c1': i} for i in range(4)), on_error='ignore')

        assert status1.num_excs == status2.num_excs
        res1 = t1.order_by(t1.c1).select(err=t1.out.errortype, msg=t1.out.errormsg).collect()
        res2 = t2.order_by(t2.c1).select(err=t2.out.errortype, msg=t2.out.errormsg).collect()
        assert list(res1) == list(res2)
        assert res1[2]['err'] == 'ValueError'
        assert res1[2]['msg'] == 'boom 2'

    def test_computed_column_reload(self, make_catalog_path: Callable[[str], str]) -> None:
        # a run_in_thread UDF in a computed column survives a catalog reload
        path = make_catalog_path('test')
        t = pxt.create_table(path, {'c1': pxt.Int})
        t.add_computed_column(out=threaded_inc(t.c1))
        validate_update_status(t.insert({'c1': i} for i in range(5)), expected_rows=5)
        reload_catalog()
        t = pxt.get_table(path)
        validate_update_status(t.insert({'c1': i} for i in range(5, 10)), expected_rows=5)
        assert sorted(row['out'] for row in t.collect()) == list(range(1, 11))


class TestImageLoad:
    def test_correctness(self, make_catalog_path: Callable[[str], str]) -> None:
        # decode routed through the injected _load_image call yields the same images as direct PIL
        files = get_image_files()[:4]
        t = pxt.create_table(make_catalog_path('test'), {'img': pxt.Image})
        validate_update_status(t.insert({'img': f} for f in files), expected_rows=len(files))
        res = t.select(path=t.img.localpath, out=t.img.rotate(90)).collect()
        for row in res:
            ref = PIL.Image.open(row['path'])
            ref.load()
            assert np.array_equal(np.asarray(row['out']), np.asarray(ref.rotate(90)))

    @pytest.mark.local('patches in-process PIL to observe decode threads')
    def test_decode_on_pool(self, make_catalog_path: Callable[[str], str], monkeypatch: pytest.MonkeyPatch) -> None:
        # image args of UDF calls decode on the cpu pool, not on the event loop
        files = get_image_files()[:8]
        t = pxt.create_table(make_catalog_path('test'), {'img': pxt.Image})
        validate_update_status(t.insert({'img': f} for f in files), expected_rows=len(files))

        decode_threads: list[str] = []
        orig_open = PIL.Image.open

        def tracking_open(*args: object, **kwargs: object) -> PIL.Image.Image:
            decode_threads.append(threading.current_thread().name)
            return orig_open(*args, **kwargs)

        monkeypatch.setattr(PIL.Image, 'open', tracking_open)
        res = t.select(out=t.img.rotate(90)).collect()
        assert len(res) == len(files)
        assert len(decode_threads) == len(files)
        assert all(name.startswith('pxt-cpu') for name in decode_threads)

    @pytest.mark.local('patches in-process PIL to count decodes')
    def test_single_load_slot(self, make_catalog_path: Callable[[str], str], monkeypatch: pytest.MonkeyPatch) -> None:
        # multiple UDF consumers of the same image column share one load slot: one decode per row
        files = get_image_files()[:4]
        t = pxt.create_table(make_catalog_path('test'), {'img': pxt.Image})
        validate_update_status(t.insert({'img': f} for f in files), expected_rows=len(files))

        num_opens = [0]
        orig_open = PIL.Image.open

        def counting_open(*args: object, **kwargs: object) -> PIL.Image.Image:
            num_opens[0] += 1
            return orig_open(*args, **kwargs)

        monkeypatch.setattr(PIL.Image, 'open', counting_open)
        res = t.select(out1=t.img.rotate(90), out2=t.img.rotate(180)).collect()
        assert len(res) == len(files)
        assert num_opens[0] == len(files)

    def test_none_image(self, make_catalog_path: Callable[[str], str]) -> None:
        # a NULL image cell yields None without decoding
        files = get_image_files()[:1]
        t = pxt.create_table(make_catalog_path('test'), {'img': pxt.Image})
        validate_update_status(t.insert([{'img': files[0]}, {'img': None}]), expected_rows=2)
        res = t.select(out=t.img.rotate(90)).collect()
        assert sum(row['out'] is None for row in res) == 1

    def test_iterator_frames(self, make_catalog_path: Callable[[str], str]) -> None:
        # iterator-produced image columns have no backing file; they keep the in-memory path
        video = get_video_files(include_mpgs=False)[0]
        t = pxt.create_table(make_catalog_path('videos'), {'video': pxt.Video})
        v = pxt.create_view(make_catalog_path('frames'), t, iterator=frame_iterator(t.video, fps=1))
        v.add_computed_column(rot=v.frame.rotate(90))
        validate_update_status(t.insert([{'video': video}]))
        res = v.select(v.rot).collect()
        assert len(res) > 0
        assert all(row['rot'] is not None for row in res)


@pxt.udf
def passthrough_image_path(path: str) -> PIL.Image.Image:
    # a path-returning image UDF: the stored file must be used as-is, not re-encoded
    return path  # type: ignore[return-value]


@pxt.udf
def make_float_image(x: int) -> PIL.Image.Image:
    # mode F cannot be written as JPEG, so encoding this value fails
    return PIL.Image.new('F', (8, 8), x)


class TestMediaEncode:
    """Media encode-on-save runs on the cpu pool inside ObjectStoreSaveNode."""

    def test_computed_image_stored(self, make_catalog_path: Callable[[str], str]) -> None:
        # a computed image column encodes into the media store and reads back
        files = get_image_files()[:4]
        t = pxt.create_table(make_catalog_path('test'), {'img': pxt.Image})
        t.add_computed_column(rot=t.img.rotate(90))
        validate_update_status(t.insert({'img': f} for f in files), expected_rows=len(files))
        res = t.select(t.rot, url=t.rot.fileurl, path=t.rot.localpath).collect()
        for row in res:
            assert row['url'] is not None
            reread = PIL.Image.open(row['path'])
            assert reread.size == row['rot'].size

    def test_pil_insert(self, make_catalog_path: Callable[[str], str]) -> None:
        # inserting an in-memory PIL image into a non-computed column encodes it
        t = pxt.create_table(make_catalog_path('test'), {'img': pxt.Image})
        validate_update_status(t.insert([{'img': PIL.Image.new('RGB', (32, 16), (255, 0, 0))}]), expected_rows=1)
        res = t.select(t.img, url=t.img.fileurl).collect()
        assert res[0]['url'] is not None
        assert res[0]['img'].size == (32, 16)

    def test_downstream_consumer(self, make_catalog_path: Callable[[str], str]) -> None:
        # the source slot keeps its in-memory value for downstream computed columns
        files = get_image_files()[:2]
        t = pxt.create_table(make_catalog_path('test'), {'img': pxt.Image})
        t.add_computed_column(rot=t.img.rotate(90))
        t.add_computed_column(rot2=t.rot.rotate(90))
        status = t.insert({'img': f} for f in files)
        assert status.num_excs == 0
        res = t.select(t.img, t.rot2).collect()
        for row in res:
            assert row['rot2'].size == row['img'].size

    @pytest.mark.local('patches in-process PIL to count encodes')
    def test_no_reencode_for_path_udf(
        self, make_catalog_path: Callable[[str], str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # a UDF that returns a file path stores that file as-is: no encode, identical bytes
        files = get_image_files()[:2]
        t = pxt.create_table(make_catalog_path('test'), {'img': pxt.Image})
        t.add_computed_column(ret=passthrough_image_path(t.img.localpath))

        num_saves = [0]
        orig_save = PIL.Image.Image.save

        def counting_save(self: PIL.Image.Image, *args: object, **kwargs: object) -> None:
            num_saves[0] += 1
            orig_save(self, *args, **kwargs)

        monkeypatch.setattr(PIL.Image.Image, 'save', counting_save)
        validate_update_status(t.insert({'img': f} for f in files), expected_rows=len(files))
        assert num_saves[0] == 0

        res = t.select(src=t.img.localpath, ret=t.ret.localpath).collect()
        for row in res:
            with open(row['src'], 'rb') as f1, open(row['ret'], 'rb') as f2:
                assert f1.read() == f2.read()

    @pytest.mark.local('asserts the raw encode error message')
    def test_encode_failure(self, make_catalog_path: Callable[[str], str]) -> None:
        # an unencodable value fails the operation (encode errors are not per-cell errors)
        t = pxt.create_table(make_catalog_path('test'), {'c1': pxt.Int})
        t.add_computed_column(img=make_float_image(t.c1))
        with pytest.raises(Exception, match='JPEG'):
            t.insert([{'c1': 1}], on_error='ignore')

    def test_two_cols_one_expr(self, make_catalog_path: Callable[[str], str]) -> None:
        # two media columns sharing one value expr share one slot: encoded once, same file url for both
        # (btree index value columns rely on this: they must store the same url as the indexed column)
        files = get_image_files()[:2]
        t = pxt.create_table(make_catalog_path('test'), {'img': pxt.Image})
        t.add_computed_column(a=t.img.rotate(90))
        t.add_computed_column(b=t.img.rotate(90))
        validate_update_status(t.insert({'img': f} for f in files), expected_rows=len(files))
        res = t.select(a_url=t.a.fileurl, b_url=t.b.fileurl).collect()
        for row in res:
            assert row['a_url'] is not None
            assert row['a_url'] == row['b_url']


class TestMediaLoad:
    """Validating image ColumnRefs (write-validated inserts, on_read selects) decode on the cpu pool."""

    @pytest.mark.local('patches in-process PIL to observe decode threads')
    def test_insert_decode_on_pool(
        self, make_catalog_path: Callable[[str], str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # on insert, validation and decode run as one cpu-pool job: exactly one decode per row,
        # none of them on the event loop
        files = get_image_files()[:8]
        t = pxt.create_table(make_catalog_path('test'), {'img': pxt.Image})
        t.add_computed_column(rot=t.img.rotate(90))

        decode_threads: list[str] = []
        orig_open = PIL.Image.open

        def tracking_open(*args: object, **kwargs: object) -> PIL.Image.Image:
            decode_threads.append(threading.current_thread().name)
            return orig_open(*args, **kwargs)

        monkeypatch.setattr(PIL.Image, 'open', tracking_open)
        validate_update_status(t.insert({'img': f} for f in files), expected_rows=len(files))
        # two opens per row: validate_media's header check plus the decode, both in the same pool job
        assert len(decode_threads) == 2 * len(files)
        assert all(name.startswith('pxt-cpu') for name in decode_threads)

    def test_validation_error_per_cell(self, make_catalog_path: Callable[[str], str]) -> None:
        # a file that fails media validation on the pool stays a per-cell error, valid rows insert
        files = get_image_files(include_bad_image=True)
        bad = next(f for f in files if 'bad_image' in f)
        good = [f for f in files if 'bad_image' not in f][:2]
        t = pxt.create_table(make_catalog_path('test'), {'img': pxt.Image})
        t.add_computed_column(rot=t.img.rotate(90))
        status = t.insert(({'img': f} for f in [*good, bad]), on_error='ignore')
        assert status.num_excs > 0
        res = t.select(err=t.img.errortype, out=t.rot).collect()
        assert sum(row['err'] is not None for row in res) == 1
        assert sum(row['out'] is not None for row in res) == len(good)

    def test_on_read_validation(self, make_catalog_path: Callable[[str], str]) -> None:
        # on_read-validated columns validate and decode on the pool at query time
        files = get_image_files()[:4]
        t = pxt.create_table(make_catalog_path('test'), {'img': pxt.Image}, media_validation='on_read')
        validate_update_status(t.insert({'img': f} for f in files), expected_rows=len(files))
        res = t.select(out=t.img.rotate(90)).collect()
        assert all(row['out'] is not None for row in res)
