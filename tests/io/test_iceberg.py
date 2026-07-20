import pathlib
from typing import Any, Callable

import numpy as np
import pyarrow as pa
import pytest

import pixeltable as pxt
from pixeltable.runtime import get_runtime
from pixeltable.utils.fault_injection import FaultLocation

from ..fault_injection import ExceptionFault
from ..utils import (
    create_all_datatypes_tbl,
    iceberg_catalog,
    pxt_raises,
    skip_test_if_not_installed,
    validate_update_status,
)


class TestIceberg:
    @classmethod
    def _catalog(cls, tmp_path: pathlib.Path) -> Any:
        return iceberg_catalog(tmp_path / 'warehouse')

    def test_export_all_types(self, make_catalog_path: Callable[[str], str], tmp_path: pathlib.Path) -> None:
        """Export a table with every supported type and verify the Iceberg output."""
        skip_test_if_not_installed('pyiceberg')
        p = make_catalog_path
        t = create_all_datatypes_tbl(name=p('all_datatype_tbl'), arrow_compatible_json=True)

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
            c_array=t.c_array.to_list(),
        )

        rows = query.collect()
        catalog = TestIceberg._catalog(tmp_path)
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

            # JSON columns map to a unified struct in Iceberg; keys absent from a given source row
            # are filled with None (or [] for list-typed fields), so compare only the keys actually
            # present in the original row.
            for k, v in orig_row['c_json'].items():
                assert exp_row['c_json'][k] == v, k

            assert exp_row['c_array'] == orig_row['c_array']

            for col in ['c_video', 'c_audio', 'c_document']:
                assert isinstance(exp_row[col], str), f'{col} should be a string'
                assert exp_row[col] != '', f'{col} should not be empty'

            # Image bytes are inlined into the Iceberg row as `pa.binary()`.
            assert isinstance(exp_row['c_image'], bytes)
            assert len(exp_row['c_image']) > 0

    def test_export_fixed_shape_tensor_errors(
        self, make_catalog_path: Callable[[str], str], tmp_path: pathlib.Path
    ) -> None:
        """Fixed-shape array columns should raise; Iceberg has no analogous type."""
        skip_test_if_not_installed('pyiceberg')
        p = make_catalog_path
        catalog = TestIceberg._catalog(tmp_path)

        fixed = pxt.create_table(p('test_iceberg_tensor'), {'c_array': pxt.Array[(4,), pxt.Float]})
        fixed.insert([{'c_array': np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)}])
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='fixed-shape tensor'):
            pxt.io.export_iceberg(fixed, catalog, 'pxt.tensor_fixed')

    def test_export_variable_shape_array(self, make_catalog_path: Callable[[str], str], tmp_path: pathlib.Path) -> None:
        """Variable-shape arrays map to pa.list_(...) and are exported as Iceberg lists."""
        skip_test_if_not_installed('pyiceberg')
        p = make_catalog_path
        variable = pxt.create_table(p('test_iceberg_tensor_var'), {'c_array': pxt.Array[(None,), pxt.Float]})
        variable.insert(
            [
                {'c_array': np.array([1.0, 2.0, 3.0], dtype=np.float32)},
                {'c_array': np.array([4.0, 5.0], dtype=np.float32)},
            ]
        )

        catalog = TestIceberg._catalog(tmp_path)
        pxt.io.export_iceberg(variable, catalog, 'pxt.tensor_var')

        exported = catalog.load_table('pxt.tensor_var').scan().to_arrow().to_pylist()
        assert sorted(r['c_array'] for r in exported) == [[1.0, 2.0, 3.0], [4.0, 5.0]]

    def test_export_with_nulls(self, make_catalog_path: Callable[[str], str], tmp_path: pathlib.Path) -> None:
        """Verify null handling across multiple types."""
        skip_test_if_not_installed('pyiceberg')
        p = make_catalog_path
        t = pxt.create_table(
            p('test_iceberg_nulls'),
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

        catalog = TestIceberg._catalog(tmp_path)
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

    def test_export_with_query(self, make_catalog_path: Callable[[str], str], tmp_path: pathlib.Path) -> None:
        """Test export with filtering and column selection."""
        skip_test_if_not_installed('pyiceberg')
        p = make_catalog_path
        t = pxt.create_table(p('test_iceberg_query'), {'c_int': pxt.Int, 'c_string': pxt.String})
        rows = [{'c_int': i, 'c_string': f'row_{i}'} for i in range(10)]
        validate_update_status(t.insert(rows), expected_rows=10)

        catalog = TestIceberg._catalog(tmp_path)

        # Filtered
        pxt.io.export_iceberg(t.where(t.c_int < 5), catalog, 'pxt.filtered')
        filtered = catalog.load_table('pxt.filtered').scan().to_arrow()
        assert filtered.num_rows == 5

        # Column subset
        pxt.io.export_iceberg(t.select(t.c_string), catalog, 'pxt.subset')
        subset = catalog.load_table('pxt.subset').scan().to_arrow()
        assert subset.num_rows == 10
        assert subset.column_names == ['c_string']

    def test_if_exists(self, make_catalog_path: Callable[[str], str], tmp_path: pathlib.Path) -> None:
        """Verify error/replace/append branches of if_exists."""
        skip_test_if_not_installed('pyiceberg')
        p = make_catalog_path

        t = pxt.create_table(p('test_iceberg_if_exists'), {'c_int': pxt.Int, 'c_string': pxt.String})
        t.insert([{'c_int': i, 'c_string': f'row_{i}'} for i in range(5)])

        catalog = TestIceberg._catalog(tmp_path)

        pxt.io.export_iceberg(t, catalog, 'pxt.if_exists')
        assert catalog.load_table('pxt.if_exists').scan().to_arrow().num_rows == 5

        # Default: error
        with pxt_raises(pxt.ErrorCode.PATH_ALREADY_EXISTS, match='already exists'):
            pxt.io.export_iceberg(t, catalog, 'pxt.if_exists')

        # Replace: drops + recreates, ends with same row count
        pxt.io.export_iceberg(t.where(t.c_int < 3), catalog, 'pxt.if_exists', if_exists='replace')
        assert catalog.load_table('pxt.if_exists').scan().to_arrow().num_rows == 3

        # Append: doubles the row count
        pxt.io.export_iceberg(t.where(t.c_int < 3), catalog, 'pxt.if_exists', if_exists='append')
        assert catalog.load_table('pxt.if_exists').scan().to_arrow().num_rows == 6

        # Invalid if_exists value
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match='must be one of'):
            pxt.io.export_iceberg(t, catalog, 'pxt.if_exists', if_exists='badval')  # type: ignore[arg-type]

        # Replace + preflight failure: existing table must be preserved.
        bad = pxt.create_table(p('test_iceberg_replace_bad'), {'c_array': pxt.Array[(4,), pxt.Float]})
        bad.insert([{'c_array': np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)}])
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='fixed-shape tensor'):
            pxt.io.export_iceberg(bad, catalog, 'pxt.if_exists', if_exists='replace')
        assert catalog.load_table('pxt.if_exists').scan().to_arrow().num_rows == 6

        # Invalid table_name: empty namespace or empty name segments must be rejected up front.
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match='namespace-qualified'):
            pxt.io.export_iceberg(t, catalog, '.tbl')
        with pxt_raises(pxt.ErrorCode.INVALID_ARGUMENT, match='namespace-qualified'):
            pxt.io.export_iceberg(t, catalog, 'ns.')

    def test_failure_handling(
        self, make_catalog_path: Callable[[str], str], fault_injection: None, tmp_path: pathlib.Path
    ) -> None:
        """Inject faults at the instrumented points in export_iceberg() and verify that the existing
        table is left intact and no temp table is leaked."""
        skip_test_if_not_installed('pyiceberg')
        p = make_catalog_path

        t = pxt.create_table(p('test_iceberg_faults'), {'c_int': pxt.Int, 'c_string': pxt.String})
        t.insert([{'c_int': i, 'c_string': f'row_{i}'} for i in range(5)])

        catalog = TestIceberg._catalog(tmp_path)
        pxt.io.export_iceberg(t, catalog, 'pxt.faults')
        assert catalog.load_table('pxt.faults').scan().to_arrow().num_rows == 5

        # Replace path: a failure while writing the temp table or just before the swap must preserve
        # the existing table and drop the temp table.
        for loc in (FaultLocation.IO_EXPORT_ICEBERG_WRITE_TEMP, FaultLocation.IO_EXPORT_ICEBERG_BEFORE_SWAP):
            fault = ExceptionFault(RuntimeError('synthetic failure'))
            get_runtime().fault_manager.inject_fault(loc, fault)
            with pxt_raises(pxt.ErrorCode.INTERNAL_ERROR, match='failed to write'):
                pxt.io.export_iceberg(t.where(t.c_int < 3), catalog, 'pxt.faults', if_exists='replace')
            fault.assert_count(1)
            assert catalog.load_table('pxt.faults').scan().to_arrow().num_rows == 5
            leftover = [name for _, name in catalog.list_tables('pxt') if '__pxt_tmp_' in name]
            assert leftover == []

        # Append path: a mid-append failure must roll back, leaving the existing table unchanged.
        fault = ExceptionFault(RuntimeError('synthetic failure'))
        get_runtime().fault_manager.inject_fault(FaultLocation.IO_EXPORT_ICEBERG_APPEND, fault)
        with pxt_raises(pxt.ErrorCode.INTERNAL_ERROR, match='failed to append'):
            pxt.io.export_iceberg(t, catalog, 'pxt.faults', if_exists='append')
        fault.assert_count(1)
        assert catalog.load_table('pxt.faults').scan().to_arrow().num_rows == 5

    def test_append_schema_mismatch(self, make_catalog_path: Callable[[str], str], tmp_path: pathlib.Path) -> None:
        """Appending a query whose schema doesn't match the existing Iceberg table should raise."""
        skip_test_if_not_installed('pyiceberg')
        p = make_catalog_path
        t = pxt.create_table(
            p('test_iceberg_mismatch'), {'c_int': pxt.Int, 'c_string': pxt.String, 'c_float': pxt.Float}
        )
        t.insert([{'c_int': 1, 'c_string': 'a', 'c_float': 1.0}])

        catalog = TestIceberg._catalog(tmp_path)
        pxt.io.export_iceberg(t, catalog, 'pxt.mismatch')

        # Subset of columns: missing 'c_float'
        with pxt_raises(pxt.ErrorCode.TYPE_MISMATCH, match='not compatible'):
            pxt.io.export_iceberg(t.select(t.c_int, t.c_string), catalog, 'pxt.mismatch', if_exists='append')

    def test_export_json_invalid_rejected(
        self, make_catalog_path: Callable[[str], str], tmp_path: pathlib.Path
    ) -> None:
        """JSON columns whose values cannot be reduced to a single arrow type must be rejected."""
        skip_test_if_not_installed('pyiceberg')
        p = make_catalog_path
        t = pxt.create_table(p('test_iceberg_bad_json'), {'c_json': pxt.Json})
        catalog = TestIceberg._catalog(tmp_path)

        # Mixed struct and list shapes across rows: pa.infer_type() can't unify them.
        t.insert([{'c_json': {'a': 1}}, {'c_json': [1, 2, 3]}])
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='mixed types'):
            pxt.io.export_iceberg(t, catalog, 'pxt.bad_json')
        t.delete()

        # List with incompatible element types: infer succeeds but coercion fails at batch build.
        t.insert([{'c_json': [1, 'a', 2]}])
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='cannot be coerced'):
            pxt.io.export_iceberg(t, catalog, 'pxt.bad_json')
        t.delete()

        # Struct field with no non-None values: Iceberg has no null-only type.
        t.insert([{'c_json': {'a': None}}, {'c_json': {'a': None}}])
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='every sampled'):
            pxt.io.export_iceberg(t, catalog, 'pxt.bad_json')
        t.delete()

    def test_schema_override(self, make_catalog_path: Callable[[str], str], tmp_path: pathlib.Path) -> None:
        """`schema_overrides` pins arrow types for specified columns."""
        skip_test_if_not_installed('pyiceberg')
        p = make_catalog_path
        catalog = TestIceberg._catalog(tmp_path)

        # Null-only JSON field: an explicit struct override pins the column type so the export succeeds.
        t = pxt.create_table(p('test_iceberg_override_null'), {'c_json': pxt.Json})
        t.insert([{'c_json': {'a': None}}, {'c_json': {'a': None}}])
        override = {'c_json': pa.struct([pa.field('a', pa.string())])}
        pxt.io.export_iceberg(t, catalog, 'pxt.override_null', schema_overrides=override)
        exported = catalog.load_table('pxt.override_null').scan().to_arrow()
        assert exported.num_rows == 2
        assert all(r['c_json']['a'] is None for r in exported.to_pylist())

        # Scalar downcast: int64 -> int32 round-trips with matching values.
        t2 = pxt.create_table(p('test_iceberg_override_int'), {'c_int': pxt.Int})
        t2.insert([{'c_int': i} for i in range(3)])
        pxt.io.export_iceberg(t2, catalog, 'pxt.override_int', schema_overrides={'c_int': pa.int32()})
        loaded = catalog.load_table('pxt.override_int').scan().to_arrow()
        assert loaded.schema.field('c_int').type == pa.int32()
        assert sorted(r['c_int'] for r in loaded.to_pylist()) == [0, 1, 2]

        # Override that does not fit the data: string values cannot be cast to int64.
        t3 = pxt.create_table(p('test_iceberg_override_bad'), {'c_string': pxt.String})
        t3.insert([{'c_string': 'hello'}])
        with pxt_raises(pxt.ErrorCode.TYPE_MISMATCH, match='schema_overrides'):
            pxt.io.export_iceberg(t3, catalog, 'pxt.override_bad', schema_overrides={'c_string': pa.int64()})

        # Override key that is not present in the source schema is rejected.
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='not present in the source'):
            pxt.io.export_iceberg(
                t3, catalog, 'pxt.override_extraneous', schema_overrides={'does_not_exist': pa.int64()}
            )

    def test_namespace_auto_create(self, make_catalog_path: Callable[[str], str], tmp_path: pathlib.Path) -> None:
        """A non-existent namespace in the table identifier should be created automatically."""
        skip_test_if_not_installed('pyiceberg')
        p = make_catalog_path
        t = pxt.create_table(p('test_iceberg_ns'), {'c_int': pxt.Int})
        t.insert([{'c_int': 1}, {'c_int': 2}])

        catalog = TestIceberg._catalog(tmp_path)
        pxt.io.export_iceberg(t, catalog, 'fresh_ns.tbl')

        assert catalog.load_table('fresh_ns.tbl').scan().to_arrow().num_rows == 2
