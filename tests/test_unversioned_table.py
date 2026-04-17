import pytest
import sqlalchemy as sql

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable.runtime import get_runtime

from .utils import ReloadTester, validate_update_status


class TestUnversionedTable:
    def test_basic_ops(self, uses_db: None, reload_tester: ReloadTester) -> None:
        schema = {'c0': pxt.Int, 'c1': pxt.String}
        tbl = pxt.create_table('test', schema, _is_versioned=False)
        md = tbl.get_metadata()
        assert not md['is_versioned']

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

            pk_constraint = inspector.get_pk_constraint(store_name)
            assert pk_constraint['constrained_columns'] == ['rowid'], pk_constraint

            row_count = conn.execute(sql.text(f'SELECT COUNT(*) FROM "{store_name}"')).scalar()
            assert row_count == 2

            # Verify that we did not generate unexpected version entries for this table
            for system_tbl in ('tableversions', 'tableschemaversions'):
                row_count = conn.execute(
                    sql.text(f"SELECT COUNT(*) FROM {system_tbl} where tbl_id = '{tbl._id}'")
                ).scalar()
                assert row_count == 1, (system_tbl, row_count)

    def test_select_where(self, uses_db: None) -> None:
        schema = {'c_int': pxt.Int, 'c_str': pxt.String, 'c_float': pxt.Float, 'c_bool': pxt.Bool}
        tbl = pxt.create_table('test', schema, _is_versioned=False)
        validate_update_status(
            tbl.insert(
                [
                    {'c_int': 0, 'c_str': 'alpha', 'c_float': 0.0, 'c_bool': True},
                    {'c_int': 1, 'c_str': 'beta', 'c_float': 1.5, 'c_bool': False},
                    {'c_int': 2, 'c_str': 'gamma', 'c_float': 2.7, 'c_bool': True},
                    {'c_int': 3, 'c_str': 'delta', 'c_float': -1.0, 'c_bool': False},
                    {'c_int': 4, 'c_str': 'epsilon', 'c_float': 3.14, 'c_bool': True},
                    {'c_int': 5, 'c_str': 'zeta', 'c_float': 0.5, 'c_bool': False},
                    {'c_int': 10, 'c_str': 'eta', 'c_float': 9.9, 'c_bool': True},
                ]
            ),
            7,
        )

        rows = tbl.select(tbl.c_int).where(tbl.c_int > 3).order_by(tbl.c_int).collect()
        assert list(rows['c_int']) == [4, 5, 10]

        rows = tbl.where(tbl.c_bool).order_by(tbl.c_int).select(tbl.c_int).collect()
        assert list(rows['c_int']) == [0, 2, 4, 10]

        rows = tbl.select(tbl.c_int, tbl.c_str).where(tbl.c_float < 0).collect()
        assert list(rows['c_int']) == [3]
        assert list(rows['c_str']) == ['delta']

        rows = tbl.select(tbl.c_str).where(tbl.c_str.contains('ta')).collect()
        assert set(rows['c_str']) == {'beta', 'delta', 'zeta', 'eta'}

        rows = tbl.select(tbl.c_int).where(~tbl.c_bool & (tbl.c_int < 4)).order_by(tbl.c_int).collect()
        assert list(rows['c_int']) == [1, 3]

        rows = tbl.where(tbl.c_int > 100).collect()
        assert len(rows) == 0

    def test_select_limit_offset(self, uses_db: None) -> None:
        tbl = pxt.create_table('test', {'n': pxt.Int}, _is_versioned=False)
        validate_update_status(tbl.insert([{'n': i} for i in range(10)]), 10)

        rows = tbl.select(tbl.n).order_by(tbl.n).limit(3).collect()
        assert list(rows['n']) == [0, 1, 2]

        for limit in (10, 100):
            rows = tbl.select(tbl.n).order_by(tbl.n).limit(limit).collect()
            assert list(rows['n']) == list(range(10))

        rows = tbl.select(tbl.n).order_by(tbl.n).limit(3, offset=4).collect()
        assert list(rows['n']) == [4, 5, 6]

        rows = tbl.select(tbl.n).order_by(tbl.n).limit(10, offset=10).collect()
        assert len(rows) == 0

    def test_unsupported_joins(self, uses_db: None) -> None:
        """Joins between versioned and unversioned tables are not supported."""
        unversioned_tbl = pxt.create_table('t0', {'n': pxt.Int}, _is_versioned=False)
        versioned_tbl = pxt.create_table('t1', {'n': pxt.Int}, _is_versioned=True)
        with pytest.raises(excs.Error, match='join is not supported between versioned and unversioned tables'):
            versioned_tbl.select().join(unversioned_tbl, on=(versioned_tbl.n == unversioned_tbl.n))
        with pytest.raises(excs.Error, match='join is not supported between versioned and unversioned tables'):
            unversioned_tbl.select().join(versioned_tbl, on=(versioned_tbl.n == unversioned_tbl.n))

    def test_unsupported_ops(self, uses_db: None) -> None:
        tbl = pxt.create_table('test', {'n': pxt.Int}, _is_versioned=False)
        validate_update_status(tbl.insert([{'n': i} for i in range(10)]), 10)

        with pytest.raises(excs.Error, match='Revert is supported on versioned tables only'):
            tbl.revert()
