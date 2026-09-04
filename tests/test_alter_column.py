import pytest

import pixeltable as pxt

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

    def test_alter_computed_column_subset_refs(self, db_root: DatabaseRoot) -> None:
        t = pxt.create_table(db_root.make_catalog_path('test_tbl'), {'n': pxt.Int, 'm': pxt.Int})
        t.add_computed_column(c=t.n + t.m)
        validate_update_status(t.insert(n=1, m=5), 1)
        assert t.select(t.c).collect()['c'] == [6]

        # alter t.c to drop dependency on t.m
        t.alter_computed_column(c=t.n)
        assert t.select(t.c).collect()['c'] == [1]
        t.drop_column('m')
        assert 'm' not in t.columns()

    @pytest.mark.parametrize('cascade', [False, True], ids=['no_cascade', 'cascade'])
    def test_alter_computed_column_cascade(self, db_root: DatabaseRoot, cascade: bool) -> None:
        t = pxt.create_table(db_root.make_catalog_path('test_tbl'), {'n': pxt.Int})
        t.add_computed_column(c=t.n * 2)
        t.add_btree_index('c')
        t.add_computed_column(d=t.c + 1)
        v = pxt.create_view(db_root.make_catalog_path('test_view'), t, additional_columns={'e': t.d * 10})
        validate_update_status(t.insert(n=1), 1 + 1)

        status = t.alter_computed_column(c=t.n * 100, cascade=cascade)
        if cascade:
            assert set(status.updated_cols) == {'test_tbl.c', 'test_tbl.d', 'test_view.e'}
        else:
            assert set(status.updated_cols) == {'test_tbl.c'}
            # the dependents keep the values computed from the previous version of c
            assert t.select(t.c, t.d).collect()[0] == {'c': 100, 'd': 3}
            assert v.select(v.e).collect()['e'] == [30]
            # recomputing them explicitly brings them in sync
            validate_update_status(t.recompute_columns('d'), 2)

        # verify that t and v's computed columns are now up to date with their deps
        assert t.select(t.c, t.d).collect()[0] == {'c': 100, 'd': 101}
        assert v.select(v.e).collect()['e'] == [1010]

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

    def test_alter_computed_column_errors(self, db_root: DatabaseRoot) -> None:
        t = pxt.create_table(db_root.make_catalog_path('test_tbl'), {'n': pxt.Int, 'm': pxt.Int})
        t.add_computed_column(c=t.n + t.m)
        t.add_computed_column(d=t.c * 2)
        validate_update_status(t.insert(n=1, m=2), 1)

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='exactly one keyword argument'):
            t.alter_computed_column()
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='exactly one keyword argument'):
            t.alter_computed_column(c=t.n, d=t.m)
        with pxt_raises(pxt.ErrorCode.COLUMN_NOT_FOUND, match='Unknown column'):
            t.alter_computed_column(unknown=t.n + 1)
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='is not a computed column'):
            t.alter_computed_column(n=t.m + 1)

        # a different output type
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='has type Float, but the column has type Int'):
            t.alter_computed_column(c=(t.n + t.m) / 2)
        # a reference the current expression doesn't have
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='may only reference columns'):
            t.alter_computed_column(c=t.n + t.m + t.d)
        # a reference to a cell metadata property
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match="'errortype' property"):
            t.alter_computed_column(c=t.n + (t.c.errortype != None).astype(pxt.Int))

        # use a ColumnRef of a snapshot
        s = pxt.create_snapshot(db_root.make_catalog_path('snap'), t)
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='is not bound by'):
            t.alter_computed_column(c=s.n + s.m)

        # not allowed on a snapshot
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='Cannot alter columns of a snapshot'):
            s.alter_computed_column(c=s.n + s.m)
