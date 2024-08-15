import pathlib

import pixeltable as pxt

from ..utils import make_test_arrow_table, skip_test_if_not_installed


class TestParquet:
    def test_import_parquet(self, reset_db, tmp_path: pathlib.Path) -> None:
        skip_test_if_not_installed('pyarrow')
        import pyarrow as pa
        import pyarrow.parquet

        from pixeltable.utils.arrow import iter_tuples

        parquet_dir = tmp_path / 'test_data'
        parquet_dir.mkdir()
        make_test_arrow_table(parquet_dir)

        tab = pxt.io.import_parquet('test_parquet', parquet_path=str(parquet_dir))
        assert 'test_parquet' in pxt.list_tables()
        assert tab is not None
        num_elts = tab.count()
        arrow_tab: pa.Table = pyarrow.parquet.read_table(str(parquet_dir))
        assert num_elts == arrow_tab.num_rows
        assert set(tab.column_names()) == set(arrow_tab.column_names)

        result_set = tab.order_by(tab.c_id).collect()
        column_types = tab.column_types()

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
