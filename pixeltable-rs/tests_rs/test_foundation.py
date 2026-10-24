"""End-to-end tests of the foundation slice: directories, tables, inserts, queries, persistence, types, errors."""

import datetime
import uuid

import numpy as np
import pytest

import pixeltable as pxt
import pixeltable.functions as pxtf
import pixeltable.type_system as ts
from pixeltable import exceptions as excs
from pixeltable._testing import reload_catalog


def raises(code: pxt.ErrorCode, match: str | None = None) -> pytest.RaisesExc:
    return pytest.raises(excs._error_class(code), match=match)


class TestExceptions:
    def test_classes(self) -> None:
        assert excs._error_class(pxt.ErrorCode.PATH_NOT_FOUND) is pxt.NotFoundError
        assert excs._error_class(pxt.ErrorCode.INVALID_PATH) is pxt.RequestError
        with pytest.raises(AssertionError):
            pxt.RequestError(pxt.ErrorCode.PATH_NOT_FOUND, 'mismatched')


class TestTypes:
    def test_repr_and_algebra(self) -> None:
        assert repr(ts.ColumnType.normalize_type(pxt.Int | None)) == 'Int | None'
        assert repr(ts.ColumnType.normalize_type(pxt.Array[(None,), np.float32])) == 'Array[(None,), float32]'
        assert ts.IntType().supertype(ts.FloatType(nullable=True)) == ts.FloatType(nullable=True)
        assert ts.StringType().supertype(ts.IntType(), for_inference=True) is None
        with pytest.warns(excs.PixeltableDeprecationWarning):
            assert ts.ColumnType.normalize_type(pxt.Required[pxt.String]) == ts.StringType()
        t = ts.ColumnType.normalize_type(pxt.Json[{'a': pxt.Int, 'b': [pxt.String]}])
        assert repr(t) == "Json[{'a': Int, 'b': Json[(String, ...)]}]"
        assert ts.ColumnType.from_dict(t.as_dict()) == t
        assert ts.ColumnType.Type(1) == ts.ColumnType.Type.INT
        assert ts.IntType().is_int_type() and ts.StringType(nullable=True).nullable

    def test_validation(self) -> None:
        with pytest.raises(TypeError, match='Expected int, got str'):
            ts.IntType().validate_literal('x')
        assert ts.FloatType().create_literal(1) == 1.0
        with pytest.raises(TypeError, match='Expected non-None value'):
            ts.IntType().validate_literal(None)
        with pytest.raises(TypeError, match=r'Expected numpy.ndarray\(\(2,\), dtype=int64\)'):
            ts.ArrayType((2,), ts.IntType()).validate_literal(np.zeros(3, dtype=np.int64))

    def test_invalid_type(self) -> None:
        with raises(pxt.ErrorCode.INVALID_TYPE, match='Standard Python type `str` cannot be used here'):
            pxt.create_table('t', {'c': str})


class TestDirs:
    def test_create_list_drop(self) -> None:
        pxt.create_dir('a/b/c', parents=True)
        assert pxt.list_dirs('', recursive=True) == ['a', 'a/b', 'a/b/c']
        with raises(pxt.ErrorCode.PATH_ALREADY_EXISTS, match="'a' is an existing directory"):
            pxt.create_dir('A')
        with raises(pxt.ErrorCode.INVALID_PATH, match='Invalid path: 1x'):
            pxt.create_dir('1x')
        with raises(pxt.ErrorCode.DIRECTORY_NOT_FOUND, match='Create it first with'):
            pxt.create_dir('x/y')
        with raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='is not empty'):
            pxt.drop_dir('a')
        pxt.drop_dir('a', force=True)
        assert pxt.list_dirs() == []

    def test_move_and_tree(self) -> None:
        pxt.create_dir('d1')
        pxt.create_table('d1/t', {'c': pxt.Int | None})
        pxt.move('d1', 'd2')
        assert pxt.list_tables() == ['d2/t']
        tree = pxt.get_dir_tree()
        assert tree[0]['path'] == 'd2' and tree[0]['entries'][0]['kind'] == 'table'
        assert pxt.get_dir_contents('d2') == {'dirs': [], 'tables': ['d2/t']}
        assert list(pxt.ls('d2')['Name']) == ['t']


class TestTables:
    def test_create_insert_query(self) -> None:
        t = pxt.create_table(
            'tbl',
            {'s': pxt.String, 'i': pxt.Int | None, 'f': pxt.Float | None, 'ts': pxt.Timestamp | None, 'j': pxt.Json | None},
            primary_key='s',
            comment='a table',
        )
        assert t.columns() == ['s', 'i', 'f', 'ts', 'j']
        now = datetime.datetime(2024, 7, 1, 12, 0)
        status = t.insert([{'s': f'r{n}', 'i': n, 'f': n / 2, 'ts': now, 'j': {'n': n}} for n in range(5)])
        assert status.num_rows == 5 and status.num_excs == 0
        t.insert(s='r5', i=None)
        assert t.count() == 6
        assert t.where(t.i == None).count() == 1
        res = t.where((t.i >= 2) & ~(t.i == 4)).select(t.s, x=t.i * 10 + 1).order_by(t.i, asc=False).collect()
        assert res['s'] == ['r3', 'r2']
        assert res['x'] == [31, 21]
        assert res.schema == {'s': 'String', 'x': 'Int | None'}
        assert res[0] == {'s': 'r3', 'x': 31}
        assert res[1, 'x'] == 21
        row0 = t.select(t.ts, t.j).head(1)[0]
        assert row0['ts'] == now.replace(tzinfo=row0['ts'].tzinfo) and row0['j'] == {'n': 0}
        assert t.tail(1)['s'] == ['r5']
        assert t.order_by(t.i).limit(2, offset=1).collect()['i'] == [1, 2]
        df = t.select(t.s, t.i).collect().to_pandas()
        assert list(df.columns) == ['s', 'i'] and len(df) == 6
        with raises(pxt.ErrorCode.CONSTRAINT_VIOLATION, match='Duplicate primary key value'):
            t.insert(s='r1')
        assert t.get_metadata()['comment'] == 'a table'
        assert t.get_metadata()['primary_key'] == ['s']

    def test_insert_errors(self) -> None:
        t = pxt.create_table('t', {'c1': pxt.String, 'c2': pxt.Int | None})
        with raises(pxt.ErrorCode.MISSING_REQUIRED, match=r'Missing required column\(s\) \(c1\)'):
            t.insert(c2=1)
        with raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='Error in column c2: expected int, got str'):
            t.insert(c1='a', c2='x')
        with raises(pxt.ErrorCode.COLUMN_NOT_FOUND, match='Unknown column name c3'):
            t.insert(c1='a', c3=1)
        with raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='Cannot insert an empty sequence'):
            t.insert([])

    def test_expression_errors(self) -> None:
        t = pxt.create_table('t', {'s': pxt.String | None, 'i': pxt.Int | None})
        with raises(pxt.ErrorCode.TYPE_MISMATCH, match=r'requires numeric types'):
            _ = t.s - 1
        with pytest.raises(TypeError, match='cannot be used in conjunction with Python boolean operators'):
            _ = (t.i > 1) and (t.i < 3)
        with raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='needs to return `Bool`'):
            t.where(t.i)
        assert repr((t.i + 1) * 2) == '(i + 1) * 2'
        assert repr(t.s.upper()) == 's.upper()'

    def test_builtins_and_computed_columns(self) -> None:
        t = pxt.create_table('t', {'s': pxt.String | None, 'i': pxt.Int | None})
        t.insert([{'s': 'Ab', 'i': 1}, {'s': None, 'i': 2}])
        t.add_computed_column(up=t.s.upper())
        t.add_computed_column(twice=t.i * 2, stored=False)
        t.insert(s='cd', i=3)
        res = t.select(t.up, t.twice).collect()
        assert res['up'] == ['AB', None, 'CD']
        assert res['twice'] == [2, 4, 6]
        from pixeltable.functions.string import contains

        assert t.where(contains(t.s, 'b', case=False)).count() == 1

    def test_schema_changes_and_versions(self) -> None:
        t = pxt.create_table('t', {'a': pxt.Int | None, 'b': pxt.Int | None})
        t.insert(a=1, b=2)
        q = t.select(t.a, t.b)
        t.add_column(c=pxt.String | None)
        t.rename_column('b', 'bb')
        assert t.columns() == ['a', 'bb', 'c']
        t.drop_column('bb')
        with raises(pxt.ErrorCode.COLUMN_NOT_FOUND, match='dropped'):
            q.collect()
        assert t.get_metadata()['version'] == 4
        v1 = pxt.get_table('t:1')
        assert v1.collect()['b'] == [2]
        assert [v['version'] for v in t.get_versions()] == [4, 3, 2, 1, 0]

    def test_views_and_snapshots(self) -> None:
        t = pxt.create_table('t', {'a': pxt.Int | None})
        t.insert([{'a': n} for n in range(5)])
        v = pxt.create_view('v', t.where(t.a >= 3))
        s = pxt.create_snapshot('s', t)
        t.insert(a=10)
        assert v.count() == 3
        assert s.count() == 5
        assert v.get_metadata()['base'] == 't'
        with raises(pxt.ErrorCode.CONSTRAINT_VIOLATION, match='cannot be dropped'):
            pxt.drop_table('t')
        pxt.drop_table('t', force=True)
        assert pxt.list_tables() == []

    def test_window_columns_on_inserts_and_updates(self) -> None:
        # as upstream's plans do, a window column is computed over the rows an insert adds or an update changes
        t = pxt.create_table('t', {'g': pxt.Int, 'o': pxt.Int, 'x': pxt.Int})
        t.add_computed_column(s=pxtf.sum(t.x, group_by=t.g, order_by=t.o))
        v = pxt.create_view('v', t)
        v.add_computed_column(vs=pxtf.sum(t.x, group_by=t.g, order_by=t.o))
        status = t.insert({'g': i % 2, 'o': i, 'x': i} for i in range(6))
        assert status.num_excs == 0
        assert list(t.order_by(t.o).collect()['s']) == [0, 1, 2, 4, 6, 9]
        assert list(v.order_by(v.o).collect()['vs']) == [0, 1, 2, 4, 6, 9]
        status = t.update({'x': t.x + 10}, where=t.o < 3)
        assert status.num_excs == 0
        assert list(t.order_by(t.o).collect()['s']) == [10, 11, 22, 4, 6, 9]
        assert list(v.order_by(v.o).collect()['vs']) == [10, 11, 22, 4, 6, 9]


class TestPersistence:
    def test_reload(self) -> None:
        pxt.create_dir('d')
        t = pxt.create_table('d/t', {'u': pxt.UUID | None, 'a': pxt.Array[(2,), pxt.Float] | None, 'b': pxt.Bool | None})
        u = uuid.uuid4()
        t.insert(u=u, a=np.array([1.0, 2.0], dtype=np.float32), b=True)
        q = t.where(t.b == True).select(t.u, t.a)
        d = q.as_dict()
        reload_catalog()
        res = pxt.Query.from_dict(d).collect()
        assert res['u'] == [u]
        assert np.array_equal(res['a'][0], np.array([1.0, 2.0], dtype=np.float32))
        # handles held from before the reload still work
        assert t.count() == 1
        assert pxt.get_table('D/T') == t

    def test_cursor(self) -> None:
        t = pxt.create_table('t', {'a': pxt.Int | None})
        t.insert([{'a': n} for n in range(3)])
        with t.order_by(t.a).cursor() as cur:
            assert [row['a'] for row in cur] == [0, 1, 2]
        assert cur._closed
