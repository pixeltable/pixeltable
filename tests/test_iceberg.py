import json
import pixeltable as pxt
from pixeltable.env import Env
from pixeltable.io.iceberg import export_iceberg, sqlite_catalog
from pyiceberg.table import Table as IcebergTable

class TestIceberg:

    def test_iceberg(self, test_tbl: pxt.Table):
        catalog_path = Env.get().create_tmp_path()
        catalog = sqlite_catalog(catalog_path)
        test_tbl.drop_column('c8')  # Arrays are not working yet
        export_iceberg(test_tbl, catalog)

        # Reinstantiate a catalog to test reads from scratch
        catalog = sqlite_catalog(catalog_path)
        assert catalog.list_tables('pxt') == [('pxt', 'test_tbl')]
        iceberg_tbl = catalog.load_table('pxt.test_tbl')
        self.__check_iceberg_tbl(test_tbl, iceberg_tbl)

    def __check_iceberg_tbl(self, tbl: pxt.Table, iceberg_tbl: IcebergTable):
        iceberg_data = iceberg_tbl.scan().to_pandas()
        # Only check columns defined in the table (not ancestors)
        col_refs = [tbl[col] for col in tbl._tbl_version.cols_by_name]
        pxt_data = tbl.select(*col_refs).collect()
        for col in tbl._tbl_version.cols_by_name:
            print(f'Checking column: {col}')
            pxt_values = pxt_data[col]
            iceberg_values = list(iceberg_data[col])
            if tbl._schema[col].is_json_type():
                # JSON columns were exported as strings; check that they parse properly
                iceberg_values = [json.loads(val) for val in iceberg_values]
            assert pxt_values == iceberg_values

    def test_iceberg_views(self, test_tbl: pxt.Table):
        catalog_path = Env.get().create_tmp_path()
        catalog = sqlite_catalog(catalog_path)
        test_tbl.drop_column('c8')  # Arrays are not working yet

        pxt.create_dir('iceberg_dir')
        pxt.create_dir('iceberg_dir.subdir')
        view = pxt.create_view('iceberg_dir.subdir.test_view', test_tbl)
        view.add_computed_column(vc2=(view.c2 + 1))
        subview = pxt.create_view('iceberg_dir.subdir.test_subview', view)
        subview.add_computed_column(vvc2=(subview.vc2 + 1))
        export_iceberg(subview, catalog)

        catalog = sqlite_catalog(catalog_path)
        assert catalog.list_tables('pxt') == [('pxt', 'test_tbl')]
        assert set(catalog.list_tables('pxt.iceberg_dir.subdir')) == {
            ('pxt', 'iceberg_dir', 'subdir', 'test_view'),
            ('pxt', 'iceberg_dir', 'subdir', 'test_subview')
        }
        self.__check_iceberg_tbl(test_tbl, catalog.load_table('pxt.test_tbl'))
        self.__check_iceberg_tbl(view, catalog.load_table('pxt.iceberg_dir.subdir.test_view'))
        self.__check_iceberg_tbl(subview, catalog.load_table('pxt.iceberg_dir.subdir.test_subview'))
