from typing import Callable

import pytest

import pixeltable as pxt

from .utils import pxt_raises, reload_catalog, validate_update_status


class TestAlterColumn:
    @pytest.mark.parametrize('do_reload_catalog', [False, True], ids=['no_reload_catalog', 'reload_catalog'])
    def test_alter_column(self, make_catalog_path: Callable[[str], str], do_reload_catalog: bool) -> None:
        t = pxt.create_table(make_catalog_path('test_tbl'), {'c1': pxt.Required[pxt.String]})
        validate_update_status(t.insert(c1='a'), 1)

        # before type widening, inserting a null into the non-nullable column is rejected
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='expected non-None'):
            t.insert(c1=None)

        t.alter_column('c1', set_type=pxt.String)
        reload_catalog(do_reload_catalog)

        # a null can now be inserted
        validate_update_status(t.insert(c1=None), 1)
        res = t.select(t.c1).order_by(t.c1).collect()
        assert res['c1'] == ['a', None]

        # revert restores the non-nullable type
        t.revert()
        t.revert()
        reload_catalog(do_reload_catalog)
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='expected non-None'):
            t.insert(c1=None)

    def test_alter_column_via_reference(self, make_catalog_path: Callable[[str], str]) -> None:
        t = pxt.create_table(make_catalog_path('test_tbl'), {'c1': pxt.Required[pxt.Float]})
        t.add_column(c2=pxt.Required[pxt.Float])
        t.alter_column(t.c1, set_type=pxt.Float)
        t.alter_column(t.c2, set_type=pxt.Float)
        validate_update_status(t.insert(c1=None, c2=None), 1)

    def test_alter_column_same_type(self, make_catalog_path: Callable[[str], str]) -> None:
        t = pxt.create_table(make_catalog_path('test_tbl'), {'c1': pxt.Int})
        vers_before = len(t.get_versions())
        # alter c1, new type is the same as old type
        t.alter_column('c1', set_type=pxt.Int)
        vers_after = len(t.get_versions())
        assert vers_before == vers_after

    @pytest.mark.parametrize('do_reload_catalog', [False, True], ids=['no_reload_catalog', 'reload_catalog'])
    def test_alter_column_history(self, make_catalog_path: Callable[[str], str], do_reload_catalog: bool) -> None:
        t = pxt.create_table(make_catalog_path('test_tbl'), {'c1': pxt.Required[pxt.String]})
        t.alter_column('c1', set_type=pxt.String)
        reload_catalog(do_reload_catalog)

        versions = t.get_versions()
        assert versions[0]['change_type'] == 'schema'
        schema_change = versions[0]['schema_change']
        assert schema_change == 'Altered: c1 (type changed to String | None)'

        hist = str(t.history())
        assert schema_change in hist

    def test_alter_column_errors(self, make_catalog_path: Callable[[str], str]) -> None:
        t = pxt.create_table(
            make_catalog_path('test_tbl'),
            {'c1': pxt.Required[pxt.String], 'c2': pxt.Int, 'c3': pxt.Required[pxt.Float], 'c5': pxt.Required[pxt.Int]},
            primary_key='c5',
        )
        t.add_computed_column(c4=t.c3 + 1)

        # computed column
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='Cannot alter the type of computed column'):
            t.alter_column('c4', set_type=pxt.Float)

        # primary key column
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='Cannot alter the type of primary key column'):
            t.alter_column('c5', set_type=pxt.Int)

        # column with dependent computed columns
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match=r'Cannot alter.+c3.+columns depend on it.+c4'):
            t.alter_column('c3', set_type=pxt.Float)

        # changing the base type
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='cannot be changed from'):
            t.alter_column('c1', set_type=pxt.Int)

        # narrowing (nullable -> non-nullable)
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='cannot be changed from'):
            t.alter_column('c2', set_type=pxt.Required[pxt.Int])

        # unknown column
        with pxt_raises(pxt.ErrorCode.COLUMN_NOT_FOUND, match='Unknown column'):
            t.alter_column('unknown', set_type=pxt.String)

        # column of a different table
        t2 = pxt.create_table(make_catalog_path('test_tbl2'), {'c1': pxt.Required[pxt.String]})
        with pxt_raises(pxt.ErrorCode.COLUMN_NOT_FOUND, match='Unknown column'):
            t2.alter_column(t.c1, set_type=pxt.String)

        # not allowed on a snapshot
        s = pxt.create_snapshot(make_catalog_path('snap'), t)
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='Cannot alter columns of a snapshot'):
            s.alter_column('c1', set_type=pxt.String)

        # not allowed on a base table column via a view
        v = pxt.create_view(make_catalog_path('view'), t)
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='Cannot alter base table column'):
            v.alter_column('c1', set_type=pxt.String)
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='Cannot alter base table column'):
            v.alter_column(t.c1, set_type=pxt.String)
