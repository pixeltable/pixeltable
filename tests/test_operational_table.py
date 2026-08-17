import random
import string
from typing import Any

import pytest

import pixeltable as pxt
import pixeltable.exceptions as excs

from .utils import ReloadTester, btree_idxs, local_embedding, pxt_raises, reload_catalog, validate_update_status

pytestmark = pytest.mark.local('TODO: convert; operational-table feature')


class TestOperationalTable:
    def test_basic_ops(self, uses_db: None, reload_tester: ReloadTester) -> None:
        schema: dict[str, Any] = {'c0': pxt.Int | None, 'c1': pxt.String | None}
        tbl = pxt.create_table('test', schema, _is_data_versioned=False)
        md = tbl.get_metadata()
        assert not md['is_data_versioned']
        assert md['version'] is None

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

        pxt.drop_table(tbl)

    def test_select_where(self, uses_db: None) -> None:
        schema: dict[str, Any] = {
            'c_int': pxt.Int | None,
            'c_str': pxt.String | None,
            'c_float': pxt.Float | None,
            'c_bool': pxt.Bool | None,
        }
        tbl = pxt.create_table('test', schema, _is_data_versioned=False)
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
        tbl = pxt.create_table('test', {'n': pxt.Int | None}, _is_data_versioned=False)
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

    def test_default_btree_indexes(self, uses_db: None) -> None:
        tbl = pxt.create_table(
            'test',
            {'c_int': pxt.Int, 'c_str': pxt.String, 'c_bool': pxt.Bool},
            _is_data_versioned=False,
            has_default_idxs=True,
        )
        # bools aren't eligible for a B-tree index
        assert btree_idxs(tbl) == {'idx0': 'c_int', 'idx1': 'c_str'}

        validate_update_status(tbl.insert([{'c_int': i, 'c_str': f'str{i}', 'c_bool': True} for i in range(3)]), 3)
        assert tbl.where(tbl.c_str == 'str1').count() == 1

    @pytest.mark.parametrize('do_reload_catalog', [False, True], ids=['no_reload_catalog', 'reload_catalog'])
    def test_added_btree_index(self, uses_db: None, do_reload_catalog: bool) -> None:
        tbl = pxt.create_table('test', {'c_int': pxt.Int, 'c_str': pxt.String}, _is_data_versioned=False)
        assert btree_idxs(tbl) == {}

        validate_update_status(tbl.insert([{'c_int': i, 'c_str': f'str{i}'} for i in range(5)]), 5)

        tbl.add_btree_index('c_int')
        tbl.add_btree_index(tbl.c_str, idx_name='str_idx')

        reload_catalog(do_reload_catalog)
        assert btree_idxs(tbl) == {'idx0': 'c_int', 'str_idx': 'c_str'}

        assert tbl.where(tbl.c_int == 2).count() == 1
        assert tbl.where(tbl.c_int < 2).order_by(tbl.c_int).collect()['c_int'] == [0, 1]
        assert tbl.where(tbl.c_str > 'str2').order_by(tbl.c_int).collect()['c_int'] == [3, 4]

        assert tbl.where(tbl.c_int == 0).count() == 1
        validate_update_status(tbl.delete(where=tbl.c_str < 'str1'), 1)
        assert tbl.where(tbl.c_int == 0).count() == 0

        tbl.drop_index(column=tbl.c_int)
        reload_catalog(do_reload_catalog)
        assert btree_idxs(tbl) == {'str_idx': 'c_str'}
        assert tbl.where(tbl.c_int == 0).count() == 0

    def test_oversized_index_key(self, uses_db: None) -> None:
        """Indexed string value exceeds the B-tree limit imposed by Postgresql."""
        # Note: the value has to be incompressible to exceed the limit
        rng = random.Random(0)
        long_str = ''.join(rng.choices(string.ascii_letters + string.digits, k=4000))

        tbl = pxt.create_table('test', {'c_str': pxt.String}, _is_data_versioned=False)
        tbl.add_btree_index('c_str')
        with pxt_raises(
            pxt.ErrorCode.CONSTRAINT_VIOLATION, match="Value too large for the btree index on column 'c_str'"
        ):
            tbl.insert([{'c_str': long_str}])

        # the same limit applies when the index is built over existing rows
        tbl.drop_index(column='c_str')
        validate_update_status(tbl.insert([{'c_str': long_str}]), 1)
        with pxt_raises(
            pxt.ErrorCode.CONSTRAINT_VIOLATION, match="Value too large for the btree index on column 'c_str'"
        ):
            tbl.add_btree_index('c_str')

    @pytest.mark.parametrize('do_reload_catalog', [False, True], ids=['no_reload_catalog', 'reload_catalog'])
    def test_embedding_index(self, uses_db: None, do_reload_catalog: bool) -> None:
        tbl = pxt.create_table('test', {'id': pxt.Int, 'text': pxt.String}, _is_data_versioned=False)
        validate_update_status(
            tbl.insert(
                [
                    {'id': 0, 'text': 'The cat dozed on the warm windowsill and watched the birds outside.'},
                    {'id': 1, 'text': 'Volcanic eruptions can reshape an entire coastline within days.'},
                ]
            ),
            2,
        )

        tbl.add_embedding_index('text', idx_name='text_idx', embedding=local_embedding.using(dim=512))
        reload_catalog(do_reload_catalog)
        assert 'text_idx' in tbl.get_metadata()['indexes']

        validate_update_status(
            tbl.insert(
                [
                    {'id': 2, 'text': 'An espresso machine builds up pressure to extract coffee.'},
                    {'id': 3, 'text': 'The quarterly earnings report exceeded every analyst forecast.'},
                    {'id': 4, 'text': 'Migratory whales navigate by sensing the magnetic field of the earth.'},
                ]
            ),
            3,
        )
        assert tbl.count() == 5

        sim = tbl.text.similarity(string='Volcanic eruptions reshape entire coastlines in a matter of days.')
        assert tbl.select(tbl.id).order_by(sim, asc=False).limit(1).collect()['id'] == [1]

        validate_update_status(tbl.delete(where=tbl.id == 4), 1)

        sim = tbl.text.similarity(string='Espresso machines build pressure in order to extract the coffee.')
        res = tbl.select(tbl.id).order_by(sim, asc=False).collect()['id']
        assert res[0] == 2, res
        assert 4 not in res, res

        tbl.drop_embedding_index(idx_name='text_idx')
        reload_catalog(do_reload_catalog)
        assert 'text_idx' not in tbl.get_metadata()['indexes']
        with pxt_raises(pxt.ErrorCode.INDEX_NOT_FOUND):
            _ = tbl.text.similarity(string='Espresso machines build pressure in order to extract the coffee.')

        validate_update_status(tbl.insert([{'id': 5, 'text': 'A new row inserted after the index was dropped.'}]), 1)
        assert tbl.count() == 5

    def test_unsupported_ops(self, uses_db: None) -> None:
        operational_tbl = pxt.create_table('t0', {'n': pxt.Int | None}, _is_data_versioned=False)
        data_versioned_tbl = pxt.create_table('t1', {'n': pxt.Int | None}, _is_data_versioned=True)

        # Joins between data-versioned and operational tables are not supported.
        with pytest.raises(excs.Error, match='join is not supported between data-versioned and operational tables'):
            data_versioned_tbl.select().join(operational_tbl, on=(data_versioned_tbl.n == operational_tbl.n))
        with pytest.raises(excs.Error, match='join is not supported between data-versioned and operational tables'):
            operational_tbl.select().join(data_versioned_tbl, on=(data_versioned_tbl.n == operational_tbl.n))

        with pytest.raises(excs.Error, match='Revert is supported on data-versioned tables only'):
            operational_tbl.revert()
