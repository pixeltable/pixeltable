import datetime
import io
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import PIL.Image
import pytest

import pixeltable as pxt

from ..utils import DatabaseRoot, open_async_resources, pxt_raises, sigint_at_pos, skip_test_if_not_installed


@pxt.udf
def udf_with_exc(i: int, val: int) -> int:
    if i == val:
        raise ValueError(f'Error for row {i}')
    return i


@pytest.mark.db_roots('local', 'proxy', reason='lancedb dependencies are not installed on cloud tests')
class TestLanceDb:
    def test_export(self, db_root: DatabaseRoot, tmp_path: Path) -> None:
        skip_test_if_not_installed('lance', 'lancedb')
        import lancedb

        p = db_root.make_catalog_path
        n_rows = 1000
        schema = {
            'row_id': pxt.Int | None,
            'c_int': pxt.Int | None,
            'c_float': pxt.Float | None,
            'c_bool': pxt.Bool | None,
            'c_string': pxt.String | None,
            'c_timestamp': pxt.Timestamp | None,
            'c_date': pxt.Date | None,
            'c_json': pxt.Json | None,
            'c_array': pxt.Array[(10,), pxt.Float] | None,
            'c_image': pxt.Image | None,
        }
        t = pxt.create_table(p('test_export'), schema)

        rows = [
            {
                'row_id': i,
                'c_int': i + 1 if i % 10 != 0 else None,
                'c_float': i * 10.0,
                'c_bool': bool(i % 2),
                'c_string': f'string_{i}',
                'c_timestamp': datetime.datetime.now() - datetime.timedelta(seconds=i),
                'c_date': datetime.date.today() - datetime.timedelta(days=i),
                'c_json': {'key': i, 'value': f'val_{i}', 'nested': {'data': i * 2}},
                'c_array': np.array([i] * 10, dtype=np.float32),
                'c_image': PIL.Image.new('RGB', (100, 100), color=(i % 256, (i * 2) % 256, (i * 3) % 256)),
            }
            for i in range(n_rows)
        ]
        t.insert(rows)

        db_path = tmp_path / 'test_lancedb'

        def validate_data(lance_table_name: str, rows: list[dict[str, Any]]) -> None:
            db = lancedb.connect(str(db_path))
            lance_tbl = db.open_table(lance_table_name)
            lance_df = lance_tbl.to_pandas().sort_values('row_id').reset_index(drop=True)
            assert len(lance_df) == len(rows)
            assert lance_df['row_id'].tolist() == [row['row_id'] for row in rows]
            assert [None if pd.isna(i) else i for i in lance_df['c_int'].tolist()] == [row['c_int'] for row in rows]
            assert lance_df['c_float'].tolist() == [row['c_float'] for row in rows]
            assert lance_df['c_bool'].tolist() == [row['c_bool'] for row in rows]
            assert lance_df['c_string'].tolist() == [row['c_string'] for row in rows]
            assert lance_df['c_timestamp'].tolist() == [row['c_timestamp'] for row in rows]
            assert lance_df['c_date'].tolist() == [row['c_date'] for row in rows]
            assert lance_df['c_json'].tolist() == [row['c_json'] for row in rows]
            all(np.array_equal(arr1, arr2) for arr1, arr2 in zip(lance_df['c_array'], [r['c_array'] for r in rows]))
            for lance_img_bytes, row in zip(lance_df['c_image'], rows):
                lance_img = PIL.Image.open(io.BytesIO(lance_img_bytes))
                original_img = row['c_image']
                assert lance_img.size == original_img.size
                assert lance_img.mode == original_img.mode

        pxt.io.export_lancedb(t, db_path, 'test')
        validate_data('test', list(t.order_by(t.row_id).collect()))

        with pxt_raises(pxt.ErrorCode.PATH_ALREADY_EXISTS, match='already exists in'):
            pxt.io.export_lancedb(t, db_path, 'test', if_exists='error')

        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match='must be one of'):
            pxt.io.export_lancedb(t, db_path, 'test', if_exists='badval')  # type: ignore[arg-type]

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='exists and is not a directory'):
            pxt.io.export_lancedb(t, Path(__file__), 'test', if_exists='overwrite')

        # export query result containing PIL image, with if_exists='overwrite'
        t2 = pxt.create_table(p('test2'), schema)
        t2.insert(rows[:100])
        query = t2.order_by(t2.row_id, asc=True).select(
            t2.row_id,
            t2.c_int,
            t2.c_float,
            t2.c_bool,
            t2.c_string,
            t2.c_timestamp,
            t2.c_date,
            t2.c_json,
            t2.c_array,
            c_image=t2.c_image.rotate(180),
        )
        pxt.io.export_lancedb(query, db_path, 'test', if_exists='overwrite')
        validate_data('test', list(query.collect()))

        # if_exists='append'
        pxt.io.export_lancedb(t, db_path, 'test', if_exists='append', batch_size_bytes=1024)
        validate_data('test', sorted(list(query.collect()) + list(t.collect()), key=lambda r: r['row_id']))

        # lancedb fails while the query is still producing batches
        with pytest.raises(ValueError, match="field 'k' does not exist"):
            pxt.io.export_lancedb(t.select(k=t.row_id), db_path, 'test', if_exists='append', batch_size_bytes=1024)

        # error during export
        error_db_path = tmp_path / 'error_db'
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION):
            pxt.io.export_lancedb(t.select(t.c_int, udf_with_exc(t.c_int, 100)), error_db_path, 'test')
        assert not error_db_path.exists()

    def test_export_query_udf(self, db_root: DatabaseRoot, tmp_path: Path) -> None:
        """An export that calls a query UDF in every batch succeeds and leaves no event loop open."""
        skip_test_if_not_installed('lance', 'lancedb')
        import lancedb

        p = db_root.make_catalog_path
        n_rows = 5000
        t = pxt.create_table(p('test_query_udf'), {'i': pxt.Int})
        t.insert({'i': i} for i in range(n_rows))

        @pxt.query
        def rows_with_i(i: int) -> pxt.Query:
            return t.where(t.i == i).select(t.i)

        query = t.select(t.i, matches=rows_with_i(t.i))
        db_path = tmp_path / 'test_lancedb'
        open_before = open_async_resources()
        pxt.io.export_lancedb(query, db_path, 'test', batch_size_bytes=1024)
        pxt.io.export_lancedb(query, db_path, 'test', if_exists='append', batch_size_bytes=1024)
        assert open_async_resources() == open_before

        rows = lancedb.connect(str(db_path)).open_table('test').to_arrow().to_pylist()
        expected = [{'i': i, 'matches': [i]} for i in range(n_rows)]
        assert sorted(rows, key=lambda row: row['i']) == [row for row in expected for _ in range(2)]

    @pytest.mark.skipif(platform.system() == 'Windows', reason='signal.pthread_kill() is POSIX only')
    @pytest.mark.db_roots(
        'local', reason='the UDF signals the main thread of this process; a proxy query runs in the server'
    )
    def test_export_interrupt(self, db_root: DatabaseRoot, tmp_path: Path) -> None:
        """Ctrl-C during an export raises KeyboardInterrupt and stops the query."""
        skip_test_if_not_installed('lance', 'lancedb')
        p = db_root.make_catalog_path
        t = pxt.create_table(p('test_interrupt'), {'i': pxt.Int})
        t.insert({'i': i} for i in range(5000))
        open_before = open_async_resources()
        with pytest.raises(KeyboardInterrupt):
            pxt.io.export_lancedb(t.select(t.i, j=sigint_at_pos(t.i, 3000)), tmp_path / 'db', 'test')
        assert open_async_resources() == open_before
