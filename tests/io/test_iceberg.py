import json
import pathlib
from typing import Any

import numpy as np
import pytest

import pixeltable as pxt

from ..utils import create_all_datatypes_tbl, pxt_raises, skip_test_if_not_installed, validate_update_status


@pxt.udf
def array_to_list(arr: pxt.Array[(10,), pxt.Float]) -> pxt.Json:
    return arr.tolist()


class TestIceberg:
    def _catalog(self, tmp_path: pathlib.Path) -> Any:
        from pixeltable.utils.iceberg import sqlite_catalog

        return sqlite_catalog(tmp_path / 'warehouse')

    def test_export_all_types(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Export a table with every supported type and verify the Iceberg output."""
        skip_test_if_not_installed('pyiceberg')
        t = create_all_datatypes_tbl()

        # Iceberg has no fixed-shape tensor type; the caller is expected to project the
        # column to a list before exporting.
        query = t.select(
            t.row_id,
            t.c_audio,
            t.c_bool,
            t.c_date,
            t.c_float,
            t.c_image,
            t.c_int,
            t.c_json,
            t.c_string,
            t.c_timestamp,
            t.c_uuid,
            t.c_binary,
            t.c_video,
            t.c_document,
            c_array=array_to_list(t.c_array),
        )

        rows = query.collect()
        catalog = self._catalog(tmp_path)
        pxt.io.export_iceberg(query, catalog, 'pxt.all_types')

        iceberg_tbl = catalog.load_table('pxt.all_types')
        exported = iceberg_tbl.scan().to_arrow().to_pylist()

        assert len(exported) == len(rows)

        for exp_row, orig_row in zip(exported, rows):
            assert exp_row['c_string'] == orig_row['c_string']
            assert exp_row['c_int'] == orig_row['c_int']
            assert exp_row['c_float'] == pytest.approx(orig_row['c_float'])
            assert exp_row['c_bool'] == orig_row['c_bool']
            assert exp_row['c_date'] == orig_row['c_date']
            assert exp_row['c_uuid'] == orig_row['c_uuid']
            assert exp_row['c_binary'] == orig_row['c_binary']

            # JSON columns are unwrapped to their underlying string storage on export.
            assert json.loads(exp_row['c_json']) == orig_row['c_json']

            assert exp_row['c_array'] == orig_row['c_array']

            for col in ['c_video', 'c_audio', 'c_document']:
                assert isinstance(exp_row[col], str), f'{col} should be a string'
                assert exp_row[col] != '', f'{col} should not be empty'

            # Image bytes are inlined into the Iceberg row as `pa.binary()`.
            assert isinstance(exp_row['c_image'], bytes)
            assert len(exp_row['c_image']) > 0

    def test_export_fixed_shape_tensor_errors(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Exporting a fixed-shape tensor column should raise; Iceberg has no analogous type."""
        skip_test_if_not_installed('pyiceberg')
        t = pxt.create_table('test_iceberg_tensor', {'c_array': pxt.Array[(4,), pxt.Float]})  # type: ignore[misc]
        t.insert([{'c_array': np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)}])

        catalog = self._catalog(tmp_path)
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='fixed-shape tensor'):
            pxt.io.export_iceberg(t, catalog, 'pxt.tensor')

    def test_export_with_nulls(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Verify null handling across multiple types."""
        skip_test_if_not_installed('pyiceberg')
        t = pxt.create_table(
            'test_iceberg_nulls',
            {
                'c_int': pxt.Int,
                'c_string': pxt.String,
                'c_float': pxt.Float,
                'c_json': pxt.Json,
                'c_timestamp': pxt.Timestamp,
            },
        )
        t.insert(
            [
                {'c_int': 1, 'c_string': None, 'c_float': None, 'c_json': None, 'c_timestamp': None},
                {'c_int': None, 'c_string': 'hello', 'c_float': 1.5, 'c_json': {'a': 1}, 'c_timestamp': None},
            ]
        )

        catalog = self._catalog(tmp_path)
        pxt.io.export_iceberg(t, catalog, 'pxt.nulls')

        iceberg_tbl = catalog.load_table('pxt.nulls')
        exported = iceberg_tbl.scan().to_arrow().sort_by([('c_int', 'ascending')]).to_pylist()

        # First row: c_int=1, others null
        first = next(r for r in exported if r['c_int'] == 1)
        assert first['c_string'] is None
        assert first['c_float'] is None
        assert first['c_json'] is None
        assert first['c_timestamp'] is None

        # Second row: c_int=None, c_string='hello'
        second = next(r for r in exported if r['c_int'] is None)
        assert second['c_string'] == 'hello'

    def test_export_with_query(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Test export with filtering and column selection."""
        skip_test_if_not_installed('pyiceberg')
        t = pxt.create_table('test_iceberg_query', {'c_int': pxt.Int, 'c_string': pxt.String})
        rows = [{'c_int': i, 'c_string': f'row_{i}'} for i in range(10)]
        validate_update_status(t.insert(rows), expected_rows=10)

        catalog = self._catalog(tmp_path)

        # Filtered
        pxt.io.export_iceberg(t.where(t.c_int < 5), catalog, 'pxt.filtered')
        filtered = catalog.load_table('pxt.filtered').scan().to_arrow()
        assert filtered.num_rows == 5

        # Column subset
        pxt.io.export_iceberg(t.select(t.c_string), catalog, 'pxt.subset')
        subset = catalog.load_table('pxt.subset').scan().to_arrow()
        assert subset.num_rows == 10
        assert subset.column_names == ['c_string']

    def test_if_exists(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Verify error/overwrite/append branches of if_exists."""
        skip_test_if_not_installed('pyiceberg')
        t = pxt.create_table('test_iceberg_if_exists', {'c_int': pxt.Int, 'c_string': pxt.String})
        t.insert([{'c_int': i, 'c_string': f'row_{i}'} for i in range(5)])

        catalog = self._catalog(tmp_path)

        pxt.io.export_iceberg(t, catalog, 'pxt.if_exists')
        assert catalog.load_table('pxt.if_exists').scan().to_arrow().num_rows == 5

        # Default: error
        with pxt_raises(pxt.ErrorCode.PATH_ALREADY_EXISTS, match='already exists'):
            pxt.io.export_iceberg(t, catalog, 'pxt.if_exists')

        # Overwrite: drops + recreates, ends with same row count
        pxt.io.export_iceberg(t.where(t.c_int < 3), catalog, 'pxt.if_exists', if_exists='overwrite')
        assert catalog.load_table('pxt.if_exists').scan().to_arrow().num_rows == 3

        # Append: doubles the row count
        pxt.io.export_iceberg(t.where(t.c_int < 3), catalog, 'pxt.if_exists', if_exists='append')
        assert catalog.load_table('pxt.if_exists').scan().to_arrow().num_rows == 6

        # Invalid if_exists value
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match='must be one of'):
            pxt.io.export_iceberg(t, catalog, 'pxt.if_exists', if_exists='badval')  # type: ignore[arg-type]

    def test_append_schema_mismatch(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Appending a query whose schema doesn't match the existing Iceberg table should raise."""
        skip_test_if_not_installed('pyiceberg')
        t = pxt.create_table('test_iceberg_mismatch', {'c_int': pxt.Int, 'c_string': pxt.String, 'c_float': pxt.Float})
        t.insert([{'c_int': 1, 'c_string': 'a', 'c_float': 1.0}])

        catalog = self._catalog(tmp_path)
        pxt.io.export_iceberg(t, catalog, 'pxt.mismatch')

        # Subset of columns: missing 'c_float'
        with pxt_raises(pxt.ErrorCode.TYPE_MISMATCH, match='not compatible'):
            pxt.io.export_iceberg(t.select(t.c_int, t.c_string), catalog, 'pxt.mismatch', if_exists='append')

    def test_namespace_auto_create(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """A non-existent namespace in the table identifier should be created automatically."""
        skip_test_if_not_installed('pyiceberg')
        t = pxt.create_table('test_iceberg_ns', {'c_int': pxt.Int})
        t.insert([{'c_int': 1}, {'c_int': 2}])

        catalog = self._catalog(tmp_path)
        pxt.io.export_iceberg(t, catalog, 'fresh_ns.tbl')

        assert catalog.load_table('fresh_ns.tbl').scan().to_arrow().num_rows == 2
