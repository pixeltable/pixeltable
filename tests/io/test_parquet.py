import datetime
import pathlib
import pytest
import pixeltable as pxt

from pixeltable.env import Env
from pixeltable import exceptions as excs

from ..utils import get_image_files, make_test_arrow_table, skip_test_if_not_installed


class TestParquet:
    def test_import_parquet(self, reset_db, tmp_path: pathlib.Path) -> None:
        skip_test_if_not_installed('pyarrow')
        import pyarrow as pa
        from pyarrow import parquet
        from pixeltable.utils.arrow import iter_tuples

        parquet_dir = tmp_path / 'test_data'
        parquet_dir.mkdir()
        _ = make_test_arrow_table(parquet_dir)

        tab = pxt.io.import_parquet('test_parquet', parquet_path=str(parquet_dir))
        assert 'test_parquet' in pxt.list_tables()
        assert tab is not None
        num_elts = tab.count()
        arrow_tab: pa.Table = parquet.read_table(str(parquet_dir))
        assert num_elts == arrow_tab.num_rows
        assert set(tab._schema.keys()) == set(arrow_tab.column_names)

        result_set = tab.order_by(tab.c_id).collect()
        column_types = tab._schema

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

    def test_export_parquet_simple(self, reset_db, tmp_path: pathlib.Path) -> None:
        skip_test_if_not_installed('pyarrow')
        import pyarrow as pa
        from pyarrow import parquet

        t = pxt.create_table('test1', {'c1': pxt.Int, 'c2': pxt.String, 'c3': pxt.Timestamp})
        from zoneinfo import ZoneInfo
        tz = ZoneInfo('America/Anchorage')
        t.insert([  {'c1': 1, 'c2': 'row1', 'c3': datetime.datetime(2012, 1, 1, 12, 0, 0, 25, tz)},
                    {'c1': 2, 'c2': 'row2', 'c3': datetime.datetime(2012, 2, 1, 12, 0, 0, 25, tz)}])

        tz_default = Env().get().default_time_zone

        print("test_export_parquet_simple with tz: ", tz, "and default tz: ", tz_default)

        export_file1 = tmp_path / 'test1.pq'
        pxt.io.export_parquet(t, export_file1)
        assert export_file1.exists()
        ptest1 = parquet.read_table(str(export_file1))
        assert ptest1.num_rows == 2
        assert ptest1.column_names == ['c1', 'c2', 'c3']
        assert ptest1.schema.types == [pa.int64(), pa.string(), pa.timestamp('us', tz=datetime.timezone.utc)]
        assert pa.array(ptest1.column('c1')).equals(pa.array([1, 2]))
        assert pa.array(ptest1.column('c2')).equals(pa.array(['row1', 'row2']))
        assert pa.array(ptest1.column('c3')).equals(pa.array([datetime.datetime(2012, 1, 1, 12, 0, 0, 25, tz).astimezone(datetime.timezone.utc),
                                                              datetime.datetime(2012, 2, 1, 12, 0, 0, 25, tz).astimezone(datetime.timezone.utc)]))

        export_file2 = tmp_path / 'test2.pq'
        pxt.io.export_parquet(t.select(t.c1, t.c2), export_file2)
        assert export_file2.exists()
        ptest2 = parquet.read_table(str(export_file2))
        assert ptest2.num_rows == 2
        assert ptest2.column_names == ['c1', 'c2']
        assert pa.array(ptest2.column('c1')).equals(pa.array([1, 2]))
        assert pa.array(ptest2.column('c2')).equals(pa.array(['row1', 'row2']))

        export_file3 = tmp_path / 'test3.pq'
        pxt.io.export_parquet(t.where(t.c1 == 1), export_file3)
        assert export_file3.exists()
        ptest3 = parquet.read_table(str(export_file3))
        assert ptest3.num_rows == 1
        assert ptest3.column_names == ['c1', 'c2', 'c3']
        assert pa.array(ptest3.column('c1')).equals(pa.array([1]))
        assert pa.array(ptest3.column('c2')).equals(pa.array(['row1']))
        assert pa.array(ptest3.column('c3')).equals(pa.array([datetime.datetime(2012, 1, 1, 12, 0, 0, 25, tz).astimezone(datetime.timezone.utc)]))

        it = pxt.io.import_parquet('imported_test1', parquet_path=str(export_file1))
        assert it.count() == t.count()
        assert it._schema == t._schema
        assert it.select(it.c1).collect() == t.select(t.c1).collect()
        assert it.select(it.c2).collect() == t.select(t.c2).collect()
        assert it.select(it.c3).collect() == t.select(t.c3).collect(), it.select(it.c3).collect()

        it = pxt.io.import_parquet('imported_test2', parquet_path=str(export_file2))
        assert it.count() == t.count()
        assert it.columns == ['c1', 'c2']
        assert it.c1.col_type == t.c1.col_type
        assert it.c2.col_type == t.c2.col_type
        assert it.select(it.c1).collect() == t.select(t.c1).collect()
        assert it.select(it.c2).collect() == t.select(t.c2).collect()


        it = pxt.io.import_parquet('imported_test3', parquet_path=str(export_file3))
        assert it.count() == 1
        assert it._schema == t._schema
        assert it.select(it.c1).collect() == t.where(t.c1 == 1).select(t.c1).collect()
        assert it.select(it.c2).collect() == t.where(t.c1 == 1).select(t.c2).collect()
        assert it.select(it.c3).collect() == t.where(t.c1 == 1).select(t.c3).collect()


    def test_export_parquet(self, reset_db, tmp_path: pathlib.Path) -> None:
        skip_test_if_not_installed('pyarrow')
        import pyarrow as pa
        from pyarrow import parquet

        parquet_dir = tmp_path / 'test_data'
        parquet_dir.mkdir()
        orig_file_path = make_test_arrow_table(parquet_dir)

        tab = pxt.io.import_parquet('test_parquet', parquet_path=str(parquet_dir))
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
        assert tab._schema == exported_tab._schema
        result_after = exported_tab.order_by(exported_tab.c_id).collect()
        for tup1, tup2 in zip(result_before, result_after):
            for (col1, val1), (col2, val2) in zip(tup1.items(), tup2.items()):
                assert col1 == col2
                assert tab._schema[col1] == exported_tab._schema[col2]
                if tab._schema[col1].is_array_type():
                    assert val1.all() == val2.all()
                else:
                    assert val1 == val2
                assert None == None

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

    def test_export_parquet_image(self, reset_db, tmp_path: pathlib.Path) -> None:
        skip_test_if_not_installed('pyarrow')

        tab = pxt.create_table('test_image', {'c1': pxt.Image})
        tab.insert([{'c1': get_image_files()[0]}])

        export_path = tmp_path / 'exported_image.parquet'
        with pytest.raises(excs.Error) as exc_info:
            pxt.io.export_parquet(tab._df(), export_path)
        assert 'Cannot export Dataframe with image columns' in str(exc_info.value)

        pxt.io.export_parquet(tab._df(), export_path, inline_images=True)
        assert export_path.exists()

        # Right now we cannot import a table with inlined image back into pixeltable
        with pytest.raises(excs.Error) as exc_info:
            imported_tab = pxt.io.import_parquet('imported_image', parquet_path=str(export_path))
        assert 'Could not infer pixeltable type for column c1 from parquet file' in str(exc_info.value)
