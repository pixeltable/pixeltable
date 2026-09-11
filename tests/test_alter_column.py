import pytest

import pixeltable as pxt
import pixeltable.functions as pxtf

from .utils import DatabaseRoot, pxt_raises, reload_catalog, validate_update_status


class TestAlterColumn:
    @pytest.mark.parametrize('do_reload_catalog', [False, True], ids=['no_reload_catalog', 'reload_catalog'])
    def test_alter_column(self, db_root: DatabaseRoot, do_reload_catalog: bool, is_data_versioned: bool) -> None:
        t = pxt.create_table(
            db_root.make_catalog_path('test_tbl'), {'c1': pxt.String}, _is_data_versioned=is_data_versioned
        )
        validate_update_status(t.insert(c1='a'), 1)

        # before type widening, inserting a null into the non-nullable column is rejected
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='expected non-None'):
            t.insert(c1=None)

        t.alter_column('c1', type_=pxt.String | None)
        reload_catalog(do_reload_catalog)

        # a null can now be inserted
        validate_update_status(t.insert(c1=None), 1)
        res = t.select(t.c1).order_by(t.c1).collect()
        assert res['c1'] == ['a', None]

        if is_data_versioned:
            t.revert()
            t.revert()
            reload_catalog(do_reload_catalog)
            with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='expected non-None'):
                t.insert(c1=None)

    def test_alter_column_via_reference(self, db_root: DatabaseRoot, is_data_versioned: bool) -> None:
        t = pxt.create_table(
            db_root.make_catalog_path('test_tbl'), {'c1': pxt.Float}, _is_data_versioned=is_data_versioned
        )
        t.add_column(c2=pxt.Float)
        t.alter_column(t.c1, type_=pxt.Float | None)
        t.alter_column(t.c2, type_=pxt.Float | None)
        validate_update_status(t.insert(c1=None, c2=None), 1)

    def test_alter_column_same_type(self, db_root: DatabaseRoot, is_data_versioned: bool) -> None:
        t = pxt.create_table(
            db_root.make_catalog_path('test_tbl'), {'c1': pxt.Int | None}, _is_data_versioned=is_data_versioned
        )
        vers_before = len(t.get_versions())
        # alter c1, new type is the same as old type
        t.alter_column('c1', type_=pxt.Int | None)
        vers_after = len(t.get_versions())
        assert vers_before == vers_after

    @pytest.mark.parametrize('do_reload_catalog', [False, True], ids=['no_reload_catalog', 'reload_catalog'])
    def test_alter_column_history(
        self, db_root: DatabaseRoot, do_reload_catalog: bool, is_data_versioned: bool
    ) -> None:
        t = pxt.create_table(
            db_root.make_catalog_path('test_tbl'), {'c1': pxt.String}, _is_data_versioned=is_data_versioned
        )
        t.alter_column('c1', type_=pxt.String | None)
        reload_catalog(do_reload_catalog)

        versions = t.get_versions()
        assert versions[0]['change_type'] == 'schema'
        schema_change = versions[0]['schema_change']
        assert schema_change == 'Altered: c1 (type changed to String | None)'

        hist = str(t.history())
        assert schema_change in hist

    def test_alter_column_errors(self, db_root: DatabaseRoot) -> None:
        t = pxt.create_table(
            db_root.make_catalog_path('test_tbl'),
            {'c1': pxt.String, 'c2': pxt.Int | None, 'c3': pxt.Float, 'c5': pxt.Int},
            primary_key='c5',
        )
        t.add_computed_column(c4=t.c3 + 1)

        # computed column
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='Cannot alter the type of computed column'):
            t.alter_column('c4', type_=pxt.Float | None)

        # primary key column
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='Cannot alter the type of primary key column'):
            t.alter_column('c5', type_=pxt.Int | None)

        # column with dependent computed columns
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match=r'Cannot alter.+c3.+columns depend on it.+c4'):
            t.alter_column('c3', type_=pxt.Float | None)

        # changing the base type
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='cannot be changed from'):
            t.alter_column('c1', type_=pxt.Int | None)

        # narrowing (nullable -> non-nullable)
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='cannot be changed from'):
            t.alter_column('c2', type_=pxt.Int)

        # unknown column
        with pxt_raises(pxt.ErrorCode.COLUMN_NOT_FOUND, match='Unknown column'):
            t.alter_column('unknown', type_=pxt.String | None)

        # column of a different table
        t2 = pxt.create_table(db_root.make_catalog_path('test_tbl2'), {'c1': pxt.String})
        with pxt_raises(pxt.ErrorCode.COLUMN_NOT_FOUND, match='Unknown column'):
            t2.alter_column(t.c1, type_=pxt.String | None)

        # not allowed on a snapshot
        s = pxt.create_snapshot(db_root.make_catalog_path('snap'), t)
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='Cannot alter columns of a snapshot'):
            s.alter_column('c1', type_=pxt.String | None)

        # not allowed on a base table column via a view
        v = pxt.create_view(db_root.make_catalog_path('view'), t)
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='Cannot alter base table column'):
            v.alter_column('c1', type_=pxt.String | None)
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='Cannot alter base table column'):
            v.alter_column(t.c1, type_=pxt.String | None)

        # alter_computed_column() takes the column as the single keyword argument
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='exactly one keyword argument'):
            t.alter_computed_column()
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='exactly one keyword argument'):
            t.alter_computed_column(c3=t.c3, c4=t.c3)
        with pxt_raises(pxt.ErrorCode.COLUMN_NOT_FOUND, match='Unknown column'):
            t.alter_computed_column(unknown=t.c3 + 1)
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='is not a computed column'):
            t.alter_computed_column(c1=t.c3)

        # a different output type
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='has type `Int`, but the column has type `Float`'):
            t.alter_computed_column(c4=t.c5 + 1)

        # a reference that makes the column depend on itself, directly or indirectly
        t.add_computed_column(c6=t.c4 * 2)
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='circular dependency'):
            t.alter_computed_column(c4=t.c4 + 1)
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='circular dependency'):
            t.alter_computed_column(c4=t.c3 + t.c6)

        # a reference to a cell metadata property
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match="'errortype' property"):
            t.alter_computed_column(c4=t.c3 + (t.c4.errortype != None).astype(pxt.Float))

        # a ColumnRef of a snapshot
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='is not bound by'):
            t.alter_computed_column(c4=s.c3 + 1)
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='Cannot alter columns of a snapshot'):
            s.alter_computed_column(c4=s.c3 + 1)
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='Cannot alter base table column'):
            v.alter_computed_column(c4=v.c3 + 1)

        # a column produced by a view's iterator: the view owns it, but it holds no value expression
        component_v = pxt.create_view(
            db_root.make_catalog_path('component_view'),
            t,
            iterator=pxtf.string.string_splitter(text=t.c1, separators='sentence'),
        )
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match="Column 'text' is not a computed column"):
            component_v.alter_computed_column(text=t.c1)

    @pytest.mark.parametrize('do_reload_catalog', [False, True], ids=['no_reload_catalog', 'reload_catalog'])
    def test_alter_computed_column(
        self, db_root: DatabaseRoot, do_reload_catalog: bool, is_data_versioned: bool
    ) -> None:
        t = pxt.create_table(
            db_root.make_catalog_path('test_tbl'), {'n': pxt.Int}, _is_data_versioned=is_data_versioned
        )
        t.add_computed_column(c=t.n * 2)
        validate_update_status(t.insert([{'n': 1}, {'n': 2}]), 2)
        assert t.select(t.c).order_by(t.n).collect()['c'] == [2, 4]
        num_versions = len(t.get_versions())

        t.alter_computed_column(c=t.n * 10, recompute=False)
        reload_catalog(do_reload_catalog)
        version = t.get_versions()[0]
        assert version['change_type'] == 'schema'
        assert version['schema_change'] == 'Altered: c (value expression changed)'

        # no recompute -- the stored values didn't change
        assert t.select(t.c).order_by(t.n).collect()['c'] == [2, 4]
        assert len(t.get_versions()) == num_versions + 1
        # newly inserted rows do use the new expression
        validate_update_status(t.insert(n=3), 1)
        assert t.select(t.c).where(t.n == 3).collect()['c'] == [30]

        if not is_data_versioned:
            # TODO(PXT-1101): an operational table has no recompute path yet
            return

        num_versions = len(t.get_versions())
        t.alter_computed_column(c=t.n * 100)
        reload_catalog(do_reload_catalog)
        assert t.select(t.c).order_by(t.n).collect()['c'] == [100, 200, 300]
        # the alteration and the recompute are in the same version
        assert len(t.get_versions()) == num_versions + 1

        # and a single revert undoes both
        t.revert()
        reload_catalog(do_reload_catalog)
        assert t.select(t.c).order_by(t.n).collect()['c'] == [2, 4, 30]
        validate_update_status(t.insert(n=6), 1)
        assert t.select(t.c).where(t.n == 6).collect()['c'] == [60]

        # altering to the same expression is a no-op, and creates no version
        num_versions = len(t.get_versions())
        t.alter_computed_column(c=t.n * 10)
        assert len(t.get_versions()) == num_versions

    @pytest.mark.parametrize('do_reload_catalog', [False, True], ids=['no_reload_catalog', 'reload_catalog'])
    def test_alter_computed_column_refs(self, db_root: DatabaseRoot, do_reload_catalog: bool) -> None:
        t = pxt.create_table(db_root.make_catalog_path('test_tbl'), {'n': pxt.Int, 'm': pxt.Int})
        t.add_computed_column(c=t.n + t.m)
        validate_update_status(t.insert(n=1, m=5), 1)
        assert t.select(t.c).collect()['c'] == [6]

        # drop c's dependency on m, then add it back
        t.alter_computed_column(c=t.n)
        reload_catalog(do_reload_catalog)
        assert t.select(t.c).collect()['c'] == [1]
        t.alter_computed_column(c=t.n + t.m)
        reload_catalog(do_reload_catalog)
        assert t.select(t.c).collect()['c'] == [6]

        # depend on a column that was added after c
        t.add_computed_column(later=t.m * 100)
        t.alter_computed_column(c=t.n + t.later)
        reload_catalog(do_reload_catalog)
        assert t.select(t.c).collect()['c'] == [501]

        # updating the new dependency cascades to c
        validate_update_status(t.update({'m': 2}), 1)
        assert t.select(t.c).collect()['c'] == [201]

        # columns that are no longer referenced by c can be dropped
        t.alter_computed_column(c=t.n)
        reload_catalog(do_reload_catalog)
        t.drop_column('later')
        t.drop_column('m')
        reload_catalog(do_reload_catalog)
        assert t.columns() == ['n', 'c']

        # but a live dependency cannot be dropped
        with pxt_raises(
            pxt.ErrorCode.UNSUPPORTED_OPERATION, match="Cannot drop column 'n' because the following columns depend"
        ):
            t.drop_column('n')
        assert t.columns() == ['n', 'c']

    @pytest.mark.parametrize('cascade', [False, True], ids=['no_cascade', 'cascade'])
    def test_alter_computed_column_cascade(self, db_root: DatabaseRoot, cascade: bool) -> None:
        t = pxt.create_table(db_root.make_catalog_path('test_tbl'), {'n': pxt.Int})
        t.add_computed_column(c=t.n * 2)
        t.add_btree_index('c')
        t.add_computed_column(d=t.c + 1)
        # a view whose additional column depend on t.d
        v = pxt.create_view(db_root.make_catalog_path('test_view'), t, additional_columns={'e': t.d * 10})
        # a view whose filter depends on t.c
        filtered_v = pxt.create_view(db_root.make_catalog_path('filter_view'), t.where(t.c > 50))
        validate_update_status(t.insert(n=1), 1 + 1)
        assert filtered_v.count() == 0

        status = t.alter_computed_column(c=t.n * 100, cascade=cascade)
        if cascade:
            assert set(status.updated_cols) == {'test_tbl.c', 'test_tbl.d', 'test_view.e'}
            # the row satisfies the filter now, so it joins the view
            assert filtered_v.count() == 1
        else:
            assert set(status.updated_cols) == {'test_tbl.c'}
            # the dependents keep the values computed from the previous version of c
            assert t.select(t.c, t.d).collect()[0] == {'c': 100, 'd': 3}
            assert v.select(v.e).collect()['e'] == [30]
            # the views aren't revisited at all, so the filter's membership is stale as well
            assert filtered_v.count() == 0
            # recomputing them explicitly brings them in sync
            validate_update_status(t.recompute_columns('d'), 2)
            validate_update_status(t.recompute_columns('c'), 3)
            assert filtered_v.count() == 1

        # verify that the computed columns of t and of both views are now up to date with their deps
        assert t.select(t.c, t.d).collect()[0] == {'c': 100, 'd': 101}
        assert v.select(v.e).collect()['e'] == [1010]
        assert filtered_v.select(filtered_v.c, filtered_v.d).collect()[0] == {'c': 100, 'd': 101}

        # the B-tree has the correct values
        assert t.where(t.c == 100).count() == 1
        assert t.where(t.c == 2).count() == 0

    @pytest.mark.parametrize('do_reload_catalog', [False, True], ids=['no_reload_catalog', 'reload_catalog'])
    def test_alter_computed_column_unstored(self, db_root: DatabaseRoot, do_reload_catalog: bool) -> None:
        t = pxt.create_table(db_root.make_catalog_path('test_tbl'), {'n': pxt.Int})
        t.add_computed_column(u=t.n * 2, stored=False)
        t.add_computed_column(s=t.u + 1)
        validate_update_status(t.insert(n=1), 1)
        assert t.select(t.u, t.s).collect()[0] == {'u': 2, 's': 3}

        # recompute unstored column with cascade
        t.alter_computed_column(u=t.n * 10, cascade=True)
        reload_catalog(do_reload_catalog)
        # reading u evaluates its expression, so this is also what says the altered one was stored correctly
        assert t.select(t.u, t.s).collect()[0] == {'u': 10, 's': 11}

        # same without cascade
        t.alter_computed_column(u=t.n * 100, cascade=False)
        reload_catalog(do_reload_catalog)
        # u is computed on demand (using the new expr), s keeps the old value
        assert t.select(t.u, t.s).collect()[0] == {'u': 100, 's': 11}
        validate_update_status(t.recompute_columns('s'), 1)
        assert t.select(t.s).collect()['s'] == [101]

    def test_alter_computed_column_on_view(self, db_root: DatabaseRoot) -> None:
        t = pxt.create_table(db_root.make_catalog_path('test_tbl'), {'n': pxt.Int})
        t.add_computed_column(c=t.n * 2)
        v = pxt.create_view(db_root.make_catalog_path('test_view'), t, additional_columns={'d': t.c + 1})
        validate_update_status(t.insert(n=1), 2)

        # a view's computed column can be altered
        v.alter_computed_column(d=v.c + 100)
        assert v.select(v.d).collect()['d'] == [102]

        # a base table's column cannot be altered through the view
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='Cannot alter base table column'):
            v.alter_computed_column(c=v.n * 3)

        # a view's computed column can pick up a new dependency on a base column
        t.add_column(extra=pxt.Int | None)
        validate_update_status(t.update({'extra': 5}), 1)
        v.alter_computed_column(d=(v.c + t.extra).astype(pxt.Int))
        assert v.select(v.d).collect()['d'] == [7]
        validate_update_status(t.update({'extra': 10}), 2)
        assert v.select(v.d).collect()['d'] == [12]

        # a base table's column cannot be made to depend on one of its views
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='is not bound by'):
            t.alter_computed_column(c=v.d + 1)

        # a base query's select list are the view's own column and can be altered just as well
        projected = pxt.create_view(db_root.make_catalog_path('projected_view'), t.select(proj=t.n * 2))
        assert projected.select(projected.proj).collect()['proj'] == [2]
        projected.alter_computed_column(proj=t.n * 10)
        assert projected.select(projected.proj).collect()['proj'] == [10]
