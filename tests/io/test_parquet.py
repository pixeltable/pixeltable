import datetime
import pathlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pa_parquet
import pytest

import pixeltable as pxt
from pixeltable.utils.arrow import to_pydict

from ..utils import (
    ensure_s3_pytest_resources_access,
    get_image_files,
    make_test_arrow_table,
    skip_test_if_not_installed,
    validate_update_status,
)


def validate_parquet_files(path: pathlib.Path, rows: list[dict]) -> None:
    """Validate that the parquet files in the directory match the rows."""
    pq_table = pa_parquet.read_table(str(path))
    assert pq_table.num_rows == len(rows)

    col_names = list(rows[0].keys())
    assert set(pq_table.column_names) == set(col_names)

    pydict = to_pydict(pq_table)
    for col_name in col_names:
        for i, row in enumerate(rows):
            expected = row[col_name]
            actual = pydict[col_name][i]
            if isinstance(expected, np.ndarray):
                array_val: np.ndarray
                if isinstance(actual, np.ndarray):
                    array_val = actual
                else:
                    assert isinstance(actual, list)
                    array_val = np.array(actual)
                assert np.array_equal(array_val, expected), f'Row {i}, column {col_name}: arrays differ'
            else:
                assert actual == expected, f'Row {i}, column {col_name}: {actual} != {expected}'


class TestParquet:
    def test_import_parquet_examples(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        skip_test_if_not_installed('pyarrow')

        pdts = []
        pqts = []
        test_path = 'tests/data/datasets/'
        file_list = [
            'bank_failures.parquet',
            'gold_vs_bitcoin.parquet',
            'iris.parquet',
            'search_trends.parquet',
            'table.parquet',
            'taq.parquet',
            'titanic.parquet',
            'userdata.parquet',
            'v0.7.1.parquet',
            # This one fails on Pandas 3.0 (bug in Pandas?)
            # 'v0.7.1.column-metadata-handling-2.parquet',
            'v0.7.1.some-named-index.parquet',
            'v0.7.1.all-named-index.parquet',
            # 'transactions-t4-20.parquet'
        ]
        for i, fn in enumerate(file_list):
            xfile = test_path + fn
            try:
                pddf = pd.read_parquet(xfile)
            except Exception as e:
                raise RuntimeError(f'Error reading {fn} with pandas: {e}') from e
            print(len(pddf))
            print(pddf.dtypes)
            print(pddf.head())
            pdname = 'pdtable_' + str(i)
            pdts.append(pxt.io.import_pandas(pdname, pddf))
            pqname = 'pqtable_' + str(i)
            pqts.append(pxt.io.import_parquet(pqname, parquet_path=xfile))

        for fn, pdt, pqt in zip(file_list, pdts, pqts):
            print(fn, 'row count: ', pqt.count())
            if pdt.columns != pqt.columns:
                print(pdt.columns)
                print(pqt.columns)
            assert pdt.count() == pqt.count()

        for fn, pqt in zip(file_list, pqts):
            len1 = pqt.count()
            xfile = test_path + fn
            pqt.insert(xfile)
            assert pqt.count() == len1 * 2

    @pytest.mark.parametrize(
        'source',
        [
            'https://raw.githubusercontent.com/apache/parquet-testing/master/data/alltypes_plain.parquet',
            's3://pxt-test/pytest-resources/alltypes_plain.parquet',
        ],
    )
    def test_import_parquet_from_remote(self, uses_db: None, source: str) -> None:
        skip_test_if_not_installed('pyarrow')
        if source.startswith('s3://'):
            ensure_s3_pytest_resources_access()
        tab = pxt.create_table('from_remote_parquet', source=source, source_format='parquet')
        assert tab.count() == 8
        for col in (
            'id',
            'bool_col',
            'tinyint_col',
            'smallint_col',
            'int_col',
            'bigint_col',
            'float_col',
            'double_col',
        ):
            assert col in tab.columns()
        rows = tab.order_by(tab.id).collect()
        r0, r1 = rows[0], rows[1]
        assert r0['id'] == 0 and r0['bool_col'] is True
        assert r0['tinyint_col'] == 0 and r0['smallint_col'] == 0 and r0['int_col'] == 0
        assert r0['bigint_col'] == 0 and r0['float_col'] == 0.0 and r0['double_col'] == 0.0
        assert r1['id'] == 1 and r1['bool_col'] is False
        assert r1['tinyint_col'] == 1 and r1['smallint_col'] == 1 and r1['int_col'] == 1
        assert r1['bigint_col'] == 10 and r1['double_col'] == 10.1
        assert abs(r1['float_col'] - 1.1) < 1e-5  # float32 precision

    def test_import_parquet(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        skip_test_if_not_installed('pyarrow')
        import pyarrow as pa
        from pyarrow import parquet

        from pixeltable.utils.arrow import iter_tuples

        parquet_dir = tmp_path / 'test_data'
        parquet_dir.mkdir()
        _ = make_test_arrow_table(parquet_dir)

        # This test passes only a directory to the parquet reader. The source_format must be explicit in this case.
        tab = pxt.io.import_parquet('test_parquet', parquet_path=str(parquet_dir), source_format='parquet')
        assert 'test_parquet' in pxt.list_tables()
        assert tab is not None
        num_elts = tab.count()
        arrow_tab: pa.Table = parquet.read_table(str(parquet_dir))
        assert num_elts == arrow_tab.num_rows
        assert set(tab._get_schema().keys()) == set(arrow_tab.column_names)

        result_set = tab.order_by(tab.c_id).collect()
        column_types = tab._get_schema()

        for tup, arrow_tup in zip(result_set, iter_tuples(arrow_tab)):
            assert tup['c_id'] == arrow_tup['c_id']
            for col, val in tup.items():
                if val is None:
                    assert arrow_tup[col] is None
                    continue

                if column_types[col].is_array_type():
                    assert (val == arrow_tup[col]).all()
                elif column_types[col].is_timestamp_type():
                    assert val == arrow_tup[col].astimezone(None)
                else:
                    assert val == arrow_tup[col]

    def test_insert_parquet(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        skip_test_if_not_installed('pyarrow')

        parquet_dir = tmp_path / 'test_data'
        parquet_dir.mkdir()
        _ = make_test_arrow_table(parquet_dir)

        # This test passes only a directory to the parquet reader. The source_format must be explicit in this case.
        tab = pxt.io.import_parquet('test_parquet', parquet_path=str(parquet_dir), source_format='parquet')
        len1 = tab.count()
        tab.insert(str(parquet_dir), source_format='parquet')
        assert tab.count() == len1 * 2

    def test_export_parquet_simple(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        skip_test_if_not_installed('pyarrow')
        from zoneinfo import ZoneInfo

        import pyarrow as pa
        from pyarrow import parquet

        t = pxt.create_table('test1', {'c1': pxt.Int, 'c2': pxt.String, 'c3': pxt.Timestamp, 'c4': pxt.Json})

        tz = ZoneInfo('America/Anchorage')
        ts1 = datetime.datetime(2012, 1, 1, 12, 0, 0, 25, tz)
        ts2 = datetime.datetime(2012, 2, 1, 12, 0, 0, 25, tz)
        json_val1 = {'int_val': 100, 'str_val': 'hello'}
        json_val2 = {'int_val': 200, 'str_val': 'world'}
        t.insert(
            [{'c1': 1, 'c2': 'row1', 'c3': ts1, 'c4': json_val1}, {'c1': 2, 'c2': 'row2', 'c3': ts2, 'c4': json_val2}]
        )

        export_file1 = tmp_path / 'test1.pq'
        pxt.io.export_parquet(t, export_file1)
        assert export_file1.exists()
        pq1 = parquet.read_table(str(export_file1))
        assert pq1.num_rows == 2
        assert pq1.column_names == ['c1', 'c2', 'c3', 'c4']
        assert pq1.schema.types[:3] == [pa.int64(), pa.string(), pa.timestamp('us', tz='UTC')]
        assert pa.types.is_struct(pq1.schema.types[3])
        assert pa.array(pq1.column('c1'), type='int64').equals(pa.array([1, 2]))
        assert pa.array(pq1.column('c2'), type='str').equals(pa.array(['row1', 'row2']))
        assert pa.array(pq1.column('c3')).equals(  # type: ignore[call-overload]
            pa.array([ts1.astimezone(datetime.timezone.utc), ts2.astimezone(datetime.timezone.utc)])
        )
        c4_data = pq1.column('c4').to_pylist()
        assert c4_data == [json_val1, json_val2]

        export_file2 = tmp_path / 'test2.pq'
        pxt.io.export_parquet(t.select(t.c1, t.c2), export_file2)
        assert export_file2.exists()
        pq2 = parquet.read_table(str(export_file2))
        assert pq2.num_rows == 2
        assert pq2.column_names == ['c1', 'c2']
        assert pa.array(pq2.column('c1'), type='int64').equals(pa.array([1, 2]))
        assert pa.array(pq2.column('c2'), type='str').equals(pa.array(['row1', 'row2']))

        export_file3 = tmp_path / 'test3.pq'
        pxt.io.export_parquet(t.where(t.c1 == 1), export_file3)
        assert export_file3.exists()
        pq3 = parquet.read_table(str(export_file3))
        assert pq3.num_rows == 1
        assert pq3.column_names == ['c1', 'c2', 'c3', 'c4']
        assert pa.array(pq3.column('c1'), type='int64').equals(pa.array([1]))
        assert pa.array(pq3.column('c2'), type='str').equals(pa.array(['row1']))
        assert pa.array(pq3.column('c3')).equals(  # type: ignore[call-overload]
            pa.array([ts1.astimezone(datetime.timezone.utc)])
        )
        c4_data = pq3.column('c4').to_pylist()
        assert c4_data == [json_val1]

        it = pxt.io.import_parquet('imported_test1', parquet_path=str(export_file1))
        assert it.count() == t.count()
        assert it._get_schema() == t._get_schema()
        assert it.select(it.c1).collect() == t.select(t.c1).collect()
        assert it.select(it.c2).collect() == t.select(t.c2).collect()
        assert it.select(it.c3).collect() == t.select(t.c3).collect(), it.select(it.c3).collect()
        assert it.select(it.c4).collect() == t.select(t.c4).collect()

        it = pxt.io.import_parquet('imported_test2', parquet_path=str(export_file2))
        assert it.count() == t.count()
        assert it.columns() == ['c1', 'c2']
        assert it.c1.col_type == t.c1.col_type
        assert it.c2.col_type == t.c2.col_type
        assert it.select(it.c1).collect() == t.select(t.c1).collect()
        assert it.select(it.c2).collect() == t.select(t.c2).collect()

        it = pxt.io.import_parquet('imported_test3', parquet_path=str(export_file3))
        assert it.count() == 1
        assert it._get_schema() == t._get_schema()
        assert it.select(it.c1).collect() == t.where(t.c1 == 1).select(t.c1).collect()
        assert it.select(it.c2).collect() == t.where(t.c1 == 1).select(t.c2).collect()
        assert it.select(it.c3).collect() == t.where(t.c1 == 1).select(t.c3).collect()
        assert it.select(it.c4).collect() == t.where(t.c1 == 1).select(t.c4).collect()

    def test_export_parquet(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        skip_test_if_not_installed('pyarrow')
        import pyarrow as pa
        from pyarrow import parquet

        parquet_dir = tmp_path / 'test_data'
        parquet_dir.mkdir()
        orig_file_path = make_test_arrow_table(parquet_dir)

        tab = pxt.io.import_parquet('test_parquet', parquet_path=str(parquet_dir), source_format='parquet')
        assert 'test_parquet' in pxt.list_tables()
        assert tab is not None
        result_before = tab.order_by(tab.c_id).collect()

        export_path = tmp_path / 'exported.parquet'
        pxt.io.export_parquet(tab.select(), export_path)
        assert export_path.exists()

        # verify the data is same by reading it back into pixeltable
        exported_tab = pxt.io.import_parquet('exported', parquet_path=str(export_path))
        assert 'exported' in pxt.list_tables()
        assert exported_tab is not None
        assert tab.count() == exported_tab.count()
        assert tab._get_schema() == exported_tab._get_schema()
        result_after = exported_tab.order_by(exported_tab.c_id).collect()
        for tup1, tup2 in zip(result_before, result_after):
            for (col1, val1), (col2, val2) in zip(tup1.items(), tup2.items()):
                assert col1 == col2
                assert tab._get_schema()[col1] == exported_tab._get_schema()[col2]
                if tab._get_schema()[col1].is_array_type():
                    assert val1.all() == val2.all()
                else:
                    assert val1 == val2

        # verify the data is same by reading it back into pyarrow table
        exported_arrow_tab: pa.Table = parquet.read_table(str(export_path))
        orig_arrow_tab: pa.Table = parquet.read_table(orig_file_path)
        assert exported_arrow_tab.num_rows == orig_arrow_tab.num_rows
        assert exported_arrow_tab.column_names == orig_arrow_tab.column_names
        # assert exported_arrow_tab.equals(orig_arrow_tab)
        # and
        # assert exported_arrow_tab.schema == orig_arrow_tab.schema
        # Doesn't work because of two differences:
        # - c_id and c_int32 is DataType(int32) in pyarrow which maps
        #   to DataType(int64) in pixeltable (_pa_to_pt). So schema for
        #   these columns differ (values dont).
        # - c_timestamp has timezone=None. pyarrow interprets it as
        #   DataType(timestamp(us)) whereas pixeltable interprets is
        #   as DataType(timestamp(us, 'default timezone'))
        #   So the schema and value of that column differ.
        #

    def test_export_parquet_image(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        skip_test_if_not_installed('pyarrow')

        tab = pxt.create_table('test_image', {'c1': pxt.Image})
        tab.insert([{'c1': get_image_files()[0]}])

        export_path = tmp_path / 'exported_image.parquet'
        with pytest.raises(pxt.Error, match="Cannot export image column 'c1'"):
            pxt.io.export_parquet(tab.select(), export_path)

        pxt.io.export_parquet(tab.select(), export_path, inline_images=True)
        assert export_path.exists()

        # Test that we can reimport the image (it will come back as bytes)
        _ = pxt.io.import_parquet('imported_image', parquet_path=str(export_path))

    def test_export_array(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        t = pxt.create_table('test_array1', {'idx': pxt.Int, 'a1': pxt.Array[(10, 10), np.int64]})  # type: ignore[misc]
        rows = [{'idx': i, 'a1': np.ones((10, 10), dtype=np.int64) * i} for i in range(1000)]
        validate_update_status(t.insert(rows), expected_rows=len(rows))

        export_path = tmp_path / 'export.pq'
        pxt.io.export_parquet(t.order_by(t.idx), export_path)
        validate_parquet_files(export_path, rows)

    def test_export_ragged_array(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        t = pxt.create_table('test_array1', {'idx': pxt.Int, 'a1': pxt.Array[(None, None), np.int64]})  # type: ignore[misc]
        rng = np.random.default_rng(0)
        rows = [
            {'idx': i, 'a1': np.ones((rng.integers(1, 10) + 1, rng.integers(1, 10) + 1), dtype=np.int64) * i}
            for i in range(1000)
        ]
        validate_update_status(t.insert(rows), expected_rows=len(rows))

        export_path = tmp_path / 'export.pq'
        pxt.io.export_parquet(t.order_by(t.idx), export_path)
        validate_parquet_files(export_path, rows)

        with pytest.raises(pxt.Error, match='Cannot export array column'):
            u = pxt.create_table('test_array2', {'idx': pxt.Int, 'a1': pxt.Array[np.int64]})  # type: ignore[misc]
            validate_update_status(u.insert(rows), expected_rows=len(rows))
            export_path = tmp_path / 'error.pq'
            pxt.io.export_parquet(u.order_by(u.idx), export_path)
