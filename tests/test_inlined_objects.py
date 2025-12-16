import time
from typing import Any

import numpy as np
import PIL.Image
import random

import pixeltable as pxt
from pixeltable.env import Env
from pixeltable.utils.local_store import LocalStore

from .utils import assert_columns_eq, assert_json_eq, inf_array_iterator, inf_image_iterator, validate_update_status


class TestInlinedObjects:
    def test_null_arrays(self, reset_db: None) -> None:
        t = pxt.create_table('test_tbl', {'i': pxt.Int, 'data': pxt.Array})
        validate_update_status(
            t.insert(
                {'i': i, 'data': np.random.rand(256, 256, 3).astype(np.float32) if i % 2 == 0 else None}
                for i in range(10)
            ),
            expected_rows=10,
        )
        res = t.where(t.data == None).collect()
        assert all(row['data'] is None for row in res)
        assert all(row['i'] % 2 == 1 for row in res)
        res = t.where(t.data != None).collect()
        assert all(row['data'] is not None for row in res)
        assert all(row['i'] % 2 == 0 for row in res)

    def test_insert_arrays(self, reset_db: None) -> None:
        """Test storing arrays of various sizes and dtypes."""
        # 5 columns: cycle through different shapes and sizes in each row
        t = pxt.create_table(
            'test', {'ar1': pxt.Array, 'ar2': pxt.Array, 'ar3': pxt.Array, 'ar4': pxt.Array, 'ar5': pxt.Array}
        )

        vals = inf_array_iterator(
            shapes=[(4, 4), (40, 40), (500, 500), (1000, 2000)], dtypes=[np.int64, np.float32, np.bool_]
        )
        rows = [
            {'ar1': next(vals), 'ar2': next(vals), 'ar3': next(vals), 'ar4': next(vals), 'ar5': next(vals)}
            for _ in range(60)
        ]
        total_bytes = sum(
            row['ar1'].nbytes + row['ar2'].nbytes + row['ar3'].nbytes + row['ar4'].nbytes + row['ar5'].nbytes
            for row in rows
        )
        start = time.monotonic()
        status = t.insert(rows)
        end = time.monotonic()
        print(
            f'inserted {total_bytes / 2**20:.2f}MB in {end - start:.2f}s, '
            f'{total_bytes / (end - start) / 2**20:.2f} MB/s'
        )
        assert status.num_excs == 0
        tbl_id = t._id
        assert LocalStore(Env.get().media_dir).count(tbl_id) > 0

        res = t.head(100)  # head(): return in insertion order
        assert all(np.array_equal(row['ar1'], rows[i]['ar1']) for i, row in enumerate(res))
        assert all(np.array_equal(row['ar2'], rows[i]['ar2']) for i, row in enumerate(res))
        assert all(np.array_equal(row['ar3'], rows[i]['ar3']) for i, row in enumerate(res))
        assert all(np.array_equal(row['ar4'], rows[i]['ar4']) for i, row in enumerate(res))
        assert all(np.array_equal(row['ar5'], rows[i]['ar5']) for i, row in enumerate(res))

        pxt.drop_table('test')
        assert LocalStore(Env.get().media_dir).count(tbl_id) == 0

    def test_insert_binary(self, reset_db: None) -> None:
        """Test storing binary data of various sizes."""
        t = pxt.create_table('test', {'data': pxt.Binary})

        rnd = random.Random(4171780)
        data = [rnd.randbytes(size) for size in (0, 2**10, 2**5, 2**20, 2**8)]
        validate_update_status(t.insert({'data': d} for d in data), expected_rows=len(data))
        tbl_id = t._id
        assert LocalStore(Env.get().media_dir).count(tbl_id) > 0

        res = t.head(100)  # head(): return in insertion order
        assert_columns_eq('data', res.schema['data'], data, res['data'])

        pxt.drop_table('test')
        assert LocalStore(Env.get().media_dir).count(tbl_id) == 0

    def test_insert_inlined_objects(self, reset_db: None) -> None:
        """Test storing lists and dicts with arrays of various sizes and dtypes."""
        rnd = random.Random(4171780)

        schema = {
            'array_list': pxt.Json,
            'array_dict': pxt.Json,
            'img1': pxt.Image,
            'img2': pxt.Image,
            'img3': pxt.Image,
            'img_list': pxt.Json,
            'img_dict': pxt.Json,
            'bytes_list': pxt.Json,
            'bytes_dict': pxt.Json,
        }
        t = pxt.create_table('test', schema)

        array_vals = inf_array_iterator(
            shapes=[(4, 4), (100, 100), (500, 500), (1000, 2000)], dtypes=[np.int64, np.float32, np.bool_]
        )
        imgs = inf_image_iterator()
        rng = np.random.default_rng(0)
        rows: list[dict[str, Any]] = []
        for _ in range(10):
            img1 = next(imgs)
            img2 = next(imgs)
            img3 = next(imgs)
            rows.append(
                {
                    'array_list': [next(array_vals) for _ in range(rng.integers(1, 10, endpoint=True, dtype=int))],
                    'array_dict': {
                        str(i): next(array_vals) for i in range(rng.integers(1, 10, endpoint=True, dtype=int))
                    },
                    'img1': img1,
                    'img2': img2,
                    'img3': img3,
                    'img_list': [img1, img2, img3],
                    'img_dict': {'img1': img1, 'img2': img2, 'img3': img3},
                    'bytes_list': [rnd.randbytes(size) for size in (0, 2**10, 2**5, 2**20, 2**8)],
                    'bytes_dict': {str(size): rnd.randbytes(size) for size in (0, 2**10, 2**5, 2**20, 2**8)},
                }
            )
        status = t.insert(rows)
        assert status.num_excs == 0
        tbl_id = t._id
        assert LocalStore(Env.get().media_dir).count(tbl_id) > 0

        res = t.head(10)  # head(): return in insertion order
        for i, row in enumerate(res):
            assert_json_eq(row['array_list'], rows[i]['array_list'])
            assert_json_eq(row['array_dict'], rows[i]['array_dict'])
            assert_json_eq(row['img_list'], [row['img1'], row['img2'], row['img3']])
            assert_json_eq(row['img_dict'], {'img1': row['img1'], 'img2': row['img2'], 'img3': row['img3']})
            assert_json_eq(row['bytes_list'], rows[i]['bytes_list'])
            assert_json_eq(row['bytes_dict'], rows[i]['bytes_dict'])

        pxt.drop_table('test')
        assert LocalStore(Env.get().media_dir).count(tbl_id) == 0

    def test_nonstandard_json_construction(self, reset_db: None) -> None:
        # test list/dict construction
        # use 5 arrays to ensure every row sees a different combination of shapes and dtypes
        schema = {
            'id': pxt.Int,
            'a1': pxt.Array,
            'a2': pxt.Array,
            'a3': pxt.Array,
            'a4': pxt.Array,
            'a5': pxt.Array,
            'img1': pxt.Image,
            'img2': pxt.Image,
            'img3': pxt.Image,
            'img4': pxt.Image,
        }
        t = pxt.create_table('test', schema)
        t.add_computed_column(l1=[t.a1, t.img1, t.a2, t.img2, t.a3, t.img3, t.a4, t.img4, t.a5])
        t.add_computed_column(
            d1={
                'a': t.a1,
                'z': t.img1,
                'b': t.a2,
                'y': t.img2,
                'c': t.a3,
                'x': t.img3,
                'd': t.a4,
                'w': t.img4,
                'e': t.a5,
            }
        )

        array_vals = inf_array_iterator(
            shapes=[(4, 4), (100, 100), (500, 500), (1000, 2000)], dtypes=[np.int64, np.float32, np.bool_]
        )
        imgs = inf_image_iterator()
        rows = [
            {
                'id': i,
                'a1': next(array_vals),
                'a2': next(array_vals),
                'a3': next(array_vals),
                'a4': next(array_vals),
                'a5': next(array_vals),
                'img1': next(imgs),
                'img2': next(imgs),
                'img3': next(imgs),
                'img4': next(imgs),
            }
            for i in range(100)
        ]
        status = t.insert(rows)
        assert status.num_excs == 0
        tbl_id = t._id
        assert LocalStore(Env.get().media_dir).count(tbl_id) > 0

        # list construction
        res = t.select(t.l1, l2=[t.a1, t.img1, t.a2, t.img2, t.a3, t.img3, t.a4, t.img4, t.a5]).order_by(t.id).collect()
        for i, row in enumerate(res):
            assert_json_eq(row['l1'], row['l2'], context=f'row {i}')

        # dict construction
        res = (
            t.select(
                t.d1,
                d2={
                    'a': t.a1,
                    'z': t.img1,
                    'b': t.a2,
                    'y': t.img2,
                    'c': t.a3,
                    'x': t.img3,
                    'd': t.a4,
                    'w': t.img4,
                    'e': t.a5,
                },
            )
            .order_by(t.id)
            .collect()
        )
        for i, row in enumerate(res):
            assert_json_eq(row['d1'], row['d2'], context=f'row {i}')

        # test json path materialization (instead of full reconstruction of l1/d1)
        # TODO: collect runtime information to verify that we're only reconstructing l1[0], not the entire cell
        res = t.select(t.a1, l_a1=t.l1[0]).order_by(t.id).collect()
        for i, row in enumerate(res):
            assert_json_eq(row['a1'], row['l_a1'], context=f'row {i}')
        res = t.select(t.img1, l_img1=t.l1[1]).order_by(t.id).collect()
        for i, row in enumerate(res):
            assert_json_eq(row['img1'], row['l_img1'], context=f'row {i}')
        res = t.select(t.a2, d_a2=t.d1['b']).order_by(t.id).collect()
        for i, row in enumerate(res):
            assert_json_eq(row['a2'], row['d_a2'], context=f'row {i}')
        res = t.select(t.img2, d_img2=t.d1['y']).order_by(t.id).collect()
        for i, row in enumerate(res):
            assert_json_eq(row['img2'], row['d_img2'], context=f'row {i}')

        pxt.drop_table('test')
        assert LocalStore(Env.get().media_dir).count(tbl_id) == 0

    def test_samples(self, reset_db: None) -> None:
        schema = {'id': pxt.Int, 'c': pxt.Int, 'a': pxt.Array, 'd': pxt.Json}
        t = pxt.create_table('test', schema)

        rows = [
            {
                'id': i,
                'c': i % 10,
                'a': np.ones((256, 256, 3), dtype=np.float32) * i if i % 2 == 0 else None,
                'd': {
                    'a': np.ones((256, 256, 3), dtype=np.float32) * i,
                    'b': PIL.Image.new('RGB', (i + 10, i + 10), color=(0, 0, 0)),
                }
                if i % 2 == 1
                else None,
            }
            for i in range(100)
        ]
        status = t.insert(rows)
        assert status.num_excs == 0

        def check_sample(q: pxt.Query, n: int, rel_error: float = 0.0) -> None:
            res = q.collect()
            if n is not None:
                assert len(res) >= n * (1.0 - rel_error)
                assert len(res) <= n * (1.0 + rel_error)
            assert all(row['a'] is None for row in res if row['id'] % 2 == 1)
            assert all(
                np.array_equal(row['a'], np.ones((256, 256, 3), dtype=np.float32) * row['id'])
                for row in res
                if row['id'] % 2 == 0
            )
            assert all(row['d'] is None for row in res if row['id'] % 2 == 0)
            assert all(
                np.array_equal(row['d']['a'], np.ones((256, 256, 3), dtype=np.float32) * row['id'])
                for row in res
                if row['id'] % 2 == 1
            )
            assert all(row['d']['b'].size == (row['id'] + 10, row['id'] + 10) for row in res if row['id'] % 2 == 1)

        q_n = t.sample(n=10, seed=1)
        check_sample(q_n, 10)
        q_stratified_n = t.sample(n=10, seed=1, stratify_by=[t.c])
        check_sample(q_stratified_n, 10)
        q_per_stratum = t.sample(n_per_stratum=2, seed=1, stratify_by=[t.c])
        check_sample(q_per_stratum, 20)
        q_fraction = t.sample(fraction=0.1, seed=1)
        check_sample(q_fraction, 10, rel_error=0.1)
        q_stratified_fraction = t.sample(fraction=0.1, seed=1, stratify_by=[t.c])
        check_sample(q_stratified_fraction, 10, rel_error=0.1)
