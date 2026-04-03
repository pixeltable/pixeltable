import sqlalchemy as sql

import pixeltable as pxt
from pixeltable.runtime import get_runtime

from .utils import ReloadTester, validate_update_status


class TestUnversionedTable:
    def test_basic_ops(self, uses_db: None, reload_tester: ReloadTester) -> None:
        schema = {'c0': pxt.Int, 'c1': pxt.String}
        tbl = pxt.create_table('test', schema, _versioned=False)
        validate_update_status(tbl.insert([{'c0': 0, 'c1': 'a'}, {'c0': 1, 'c1': 'b'}, {'c0': 2, 'c1': 'c'}]), 3)
        assert tbl.count() == 3

        rows = reload_tester.run_query(tbl.order_by(tbl.c0))
        assert len(rows) == 3
        assert rows[0]['c0'] == 0 and rows[0]['c1'] == 'a'
        assert rows[1]['c0'] == 1 and rows[1]['c1'] == 'b'
        assert rows[2]['c0'] == 2 and rows[2]['c1'] == 'c'
        reload_tester.run_reload_test()

        validate_update_status(tbl.delete(where=tbl.c0 == 0), 1)
        assert tbl.count() == 2

        rows = reload_tester.run_query(tbl.order_by(tbl.c0, asc=False))
        assert len(rows) == 2
        assert rows[0]['c0'] == 2 and rows[0]['c1'] == 'c'
        assert rows[1]['c0'] == 1 and rows[1]['c1'] == 'b'
        reload_tester.run_reload_test()

        # Verify the store table state. Check for unexpected rows and columns.
        store_name = tbl._tbl_version.get().store_tbl.sa_tbl.name

        with get_runtime().begin_xact() as conn:
            inspector = sql.inspect(conn)
            col_names = {col['name'] for col in inspector.get_columns(store_name)}
            col_names.remove('rowid')
            assert all(name.startswith('col_') for name in col_names), col_names

            row_count = conn.execute(sql.text(f'SELECT COUNT(*) FROM "{store_name}"')).scalar()
            assert row_count == 2

            # Verify that we did not generate unexpected version entries for this table
            for system_tbl in ('tableversions', 'tableschemaversions'):
                row_count = conn.execute(
                    sql.text(f"SELECT COUNT(*) FROM {system_tbl} where tbl_id = '{tbl._id}'")
                ).scalar()
                assert row_count == 1, (system_tbl, row_count)
