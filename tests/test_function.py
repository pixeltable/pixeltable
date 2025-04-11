import re
import typing
from datetime import datetime
from textwrap import dedent
from typing import Optional

import numpy as np
import pytest

import pixeltable as pxt
import pixeltable.exceptions as excs
import pixeltable.functions as pxtf
from pixeltable import catalog, func
from pixeltable.func import Batch, Function, FunctionRegistry

from .utils import SAMPLE_IMAGE_URL, ReloadTester, assert_resultset_eq, reload_catalog, validate_update_status


def dummy_fn(i: int) -> int:
    return i


T = typing.TypeVar('T')


class TestFunction:
    @staticmethod
    @pxt.udf
    def func(x: int) -> int:
        """A UDF."""
        return x + 1

    @pxt.uda
    class agg(pxt.Aggregator):
        """An aggregator."""

        def __init__(self) -> None:
            self.sum = 0

        def update(self, val: int) -> None:
            if val is not None:
                self.sum += val

        def value(self) -> int:
            return self.sum

    def test_serialize_anonymous(self, init_env: None) -> None:
        d = self.func.as_dict()
        FunctionRegistry.get().clear_cache()
        deserialized = Function.from_dict(d)
        assert isinstance(deserialized, func.CallableFunction)
        # TODO: add Function.exec() and then use that
        assert deserialized.py_fn(1) == 2

    def test_list(self, reset_db: None) -> None:
        _ = FunctionRegistry.get().list_functions()
        print(_)

    def test_list_functions(self, init_env: None) -> None:
        _ = pxt.list_functions()
        print(_)

    def test_stored_udf(self, reset_db: None) -> None:
        t = pxt.create_table('test', {'c1': pxt.IntType(), 'c2': pxt.FloatType()})
        rows = [{'c1': i, 'c2': i + 0.5} for i in range(100)]
        status = t.insert(rows)
        assert status.num_rows == len(rows)
        assert status.num_excs == 0

        @pxt.udf(_force_stored=True)
        def f1(a: int, b: float) -> float:
            return a + b

        t.add_computed_column(f1=f1(t.c1, t.c2))

        func.FunctionRegistry.get().clear_cache()
        reload_catalog()
        t = pxt.get_table('test')
        status = t.insert(rows)
        assert status.num_rows == len(rows)
        assert status.num_excs == 0

    @staticmethod
    @pxt.udf
    def f1(a: int, b: float, c: float = 0.0, d: float = 1.0) -> float:
        return a + b + c + d

    @staticmethod
    @pxt.udf
    def f2(a: Optional[int], b: float = 0.0, c: Optional[float] = 1.0) -> float:
        return (0.0 if a is None else a) + b + (0.0 if c is None else c)

    def test_call(self, test_tbl: catalog.Table) -> None:
        t = test_tbl

        r0 = t.select(t.c2, t.c3).collect().to_pandas()
        # positional params with default args
        r1 = t.select(self.f1(t.c2, t.c3)).collect().to_pandas()['f1']
        assert np.all(r1 == r0.c2 + r0.c3 + 1.0)
        # kw args only
        r2 = t.select(self.f1(c=0.0, b=t.c3, a=t.c2)).collect().to_pandas()['f1']
        assert np.all(r1 == r2)
        # overriding default args
        r3 = t.select(self.f1(d=0.0, c=1.0, b=t.c3, a=t.c2)).collect().to_pandas()['f1']
        assert np.all(r2 == r3)
        # overriding default with positional arg
        r4 = t.select(self.f1(t.c2, t.c3, 0.0)).collect().to_pandas()['f1']
        assert np.all(r3 == r4)
        # overriding default with positional arg and kw arg
        r5 = t.select(self.f1(t.c2, t.c3, 1.0, d=0.0)).collect().to_pandas()['f1']
        assert np.all(r4 == r5)
        # d is kwarg
        r6 = t.select(self.f1(t.c2, d=1.0, b=t.c3)).collect().to_pandas()['f1']
        assert np.all(r5 == r6)
        # d is Expr kwarg
        r6 = t.select(self.f1(1, d=t.c3, b=t.c3)).collect().to_pandas()['f1']
        assert np.all(r5 == r6)

        # test handling of Nones
        r0 = t.select(self.f2(1, t.c3)).collect().to_pandas()['f2']
        r1 = t.select(self.f2(None, t.c3, 2.0)).collect().to_pandas()['f2']
        assert np.all(r0 == r1)
        r2 = t.select(self.f2(2, t.c3, None)).collect().to_pandas()['f2']
        assert np.all(r1 == r2)
        # kwarg with None
        r3 = t.select(self.f2(c=None, a=t.c2)).collect().to_pandas()['f2']
        # kwarg with Expr
        r4 = t.select(self.f2(c=t.c3, a=None)).collect().to_pandas()['f2']
        assert np.all(r3 == r4)

        with pytest.raises(TypeError) as exc_info:
            _ = t.select(self.f1(t.c2, c=0.0)).collect()
        assert "'b'" in str(exc_info.value)
        with pytest.raises(TypeError) as exc_info:
            _ = t.select(self.f1(t.c2)).collect()
        assert "'b'" in str(exc_info.value)
        with pytest.raises(TypeError) as exc_info:
            _ = t.select(self.f1(c=1.0, a=t.c2)).collect()
        assert "'b'" in str(exc_info.value)

        # bad default value
        with pytest.raises(excs.Error) as exc_info:

            @pxt.udf
            def f1(a: int, b: float, c: float = '') -> float:  # type: ignore[assignment]
                return a + b + c

        assert 'default value' in str(exc_info.value).lower()
        # missing param type
        with pytest.raises(excs.Error) as exc_info:

            @pxt.udf
            def f1(a: int, b: float, c='') -> float:  # type: ignore[no-untyped-def]
                return a + b + c

        assert "cannot infer pixeltable type for parameter 'c'" in str(exc_info.value).lower()
        # bad parameter name
        with pytest.raises(excs.Error) as exc_info:

            @pxt.udf
            def f1(group_by: int) -> int:
                return group_by

        assert 'reserved' in str(exc_info.value)
        # bad parameter name
        with pytest.raises(excs.Error) as exc_info:

            @pxt.udf
            def f1(order_by: int) -> int:
                return order_by

        assert 'reserved' in str(exc_info.value)

    @staticmethod
    @pxt.udf(is_method=True)
    def increment(n: int) -> int:
        return n + 1

    @staticmethod
    @pxt.udf(is_property=True)
    def successor(n: int) -> int:
        return n + 1

    @staticmethod
    @pxt.udf(is_method=True)
    def append(s: str, suffix: str) -> str:
        return s + suffix

    def test_member_access_udf(self, reset_db: None) -> None:
        t = pxt.create_table('test', {'c1': pxt.String, 'c2': pxt.Int})
        rows = [{'c1': 'a', 'c2': 1}, {'c1': 'b', 'c2': 2}]
        validate_update_status(t.insert(rows))
        result = t.select(t.c2.increment(), t.c2.successor, t.c1.append('x')).collect()
        # properties have a default column name; methods do not
        assert result[0] == {'increment': 2, 'successor': 2, 'append': 'ax'}
        assert result[1] == {'increment': 3, 'successor': 3, 'append': 'bx'}

        with pytest.raises(excs.Error) as exc_info:

            @pxt.udf(is_method=True, is_property=True)
            def udf7(n: int) -> int:
                return n + 1

        assert 'Cannot specify both `is_method` and `is_property` (in function `udf7`)' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:

            @pxt.udf(is_property=True)
            def udf8(a: int, b: int) -> int:
                return a + b

        assert '`is_property=True` expects a UDF with exactly 1 parameter, but `udf8` has 2' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:

            @pxt.udf(is_method=True, _force_stored=True)
            def udf9(n: int) -> int:
                return n + 1

        assert 'Stored functions cannot be declared using `is_method` or `is_property`' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:

            @pxt.udf(is_property=True, _force_stored=True)
            def udf10(n: int) -> int:
                return n + 1

        assert 'Stored functions cannot be declared using `is_method` or `is_property`' in str(exc_info.value)

    def test_query(self, reset_db: None, reload_tester: ReloadTester) -> None:
        t = pxt.create_table('test', {'c1': pxt.Int, 'c2': pxt.Float})
        name = t._name
        rows = [{'c1': i, 'c2': i + 0.5} for i in range(100)]
        validate_update_status(t.insert(rows))

        @pxt.query
        def lt_x(x: int) -> pxt.DataFrame:
            return t.where(t.c2 < x).select(t.c2, t.c1).order_by(t.c1)

        @pxt.query
        def lt_x_with_default(x: int, mult: int = 2) -> pxt.DataFrame:
            return t.where(t.c2 < x * mult).select(t.c2, t.c1).order_by(t.c1)

        @pxt.query
        def lt_x_with_unused_default(x: int, mult: int = 2) -> pxt.DataFrame:
            return t.where(t.c2 < x).select(t.c2, t.c1).order_by(t.c1)

        res1 = reload_tester.run_query(t.select(out=lt_x(t.c1)).order_by(t.c1))
        for i in range(100):
            assert res1[i] == {'out': [{'c2': j + 0.5, 'c1': j} for j in range(i)]}

        res2 = reload_tester.run_query(t.select(out=lt_x_with_default(t.c1)).order_by(t.c1))
        for i in range(100):
            assert res2[i] == {'out': [{'c2': j + 0.5, 'c1': j} for j in range(min(i * 2, 100))]}

        res3 = reload_tester.run_query(t.select(out=lt_x_with_unused_default(t.c1)).order_by(t.c1))
        for i in range(100):
            assert res3[i] == {'out': [{'c2': j + 0.5, 'c1': j} for j in range(i)]}

        # As computed columns
        validate_update_status(t.add_computed_column(query1=lt_x(t.c1)))
        validate_update_status(t.add_computed_column(query2=lt_x_with_default(t.c1)))
        validate_update_status(t.add_computed_column(query3=lt_x_with_unused_default(t.c1)))
        reload_tester.run_query(t.select(t.query1, t.query2, t.query3).order_by(t.c1))

        reload_tester.run_reload_test()

        # insert more rows in order to verify that lt_x() is still executable after catalog reload
        t = pxt.get_table(name)
        validate_update_status(t.insert(rows))

    def test_query2(self, reset_db: None) -> None:
        schema = {'query_text': pxt.String, 'i': pxt.Int}
        queries = pxt.create_table('queries', schema)
        query_rows = [
            {'query_text': 'how much is the stock of AI companies up?', 'i': 1},
            {'query_text': 'what happened to the term machine learning?', 'i': 2},
        ]
        validate_update_status(queries.insert(query_rows), expected_rows=len(query_rows))

        chunks = pxt.create_table('test_doc_chunks', {'text': pxt.String})
        chunks.insert(
            [
                {'text': 'the stock of artificial intelligence companies is up 1000%'},
                {
                    'text': (
                        'the term machine learning has fallen out of fashion now that AI has been '
                        'rehabilitated and is now the new hotness'
                    )
                },
                {'text': 'machine learning is a subset of artificial intelligence'},
                {'text': 'gas car companies are in danger of being left behind by electric car companies'},
            ]
        )

        @pxt.query
        def retrieval(s: str, n: int) -> pxt.DataFrame:
            """simply returns 2 passages from the table"""
            return chunks.select(chunks.text).limit(2)

        res = queries.select(queries.i, out=retrieval(queries.query_text, queries.i)).collect()
        assert all(len(out) == 2 for out in res['out'])
        validate_update_status(queries.add_computed_column(chunks=retrieval(queries.query_text, queries.i)))
        res = queries.select(queries.i, queries.chunks).collect()
        assert all(len(c) == 2 for c in res['chunks'])

        reload_catalog()
        queries = pxt.get_table('queries')
        res = queries.select(queries.chunks).collect()
        assert all(len(c) == 2 for c in res['chunks'])
        validate_update_status(queries.insert(query_rows), expected_rows=len(query_rows))
        res = queries.select(queries.chunks).collect()
        assert all(len(c) == 2 for c in res['chunks'])

    def test_query_over_view(self, reset_db: None) -> None:
        pxt.create_dir('test')
        t = pxt.create_table('test.tbl', {'a': pxt.String})
        v = pxt.create_view('test.view', t, additional_columns={'text': pxt.String})

        @pxt.query
        def retrieve() -> pxt.DataFrame:
            return v.select(v.text).limit(20)

        t = pxt.create_table('test.retrieval', {'n': pxt.Int})
        t.add_computed_column(result=retrieve())

        # This tests a specific edge case where calling drop_dir() as the first action after a catalog reload can lead
        # to a circular initialization failure.
        reload_catalog()
        pxt.drop_dir('test', force=True)

    def test_query_json_mapper(self, reset_db: None, reload_tester: ReloadTester) -> None:
        t = pxt.create_table('test', {'c1': pxt.Int, 'c2': pxt.Float})
        t_rows = [{'c1': i, 'c2': i + 0.5} for i in range(100)]
        validate_update_status(t.insert(t_rows), 100)

        @pxt.query
        def lt_x(x: int) -> pxt.DataFrame:
            return t.where(t.c2 < x).select(t.c2, t.c1).order_by(t.c1)

        u = pxt.create_table('test2', {'c': pxt.Json})
        u.add_computed_column(out=pxtf.map(u.c['*'], lambda x: lt_x(x)))
        u_rows = [{'c': [i, i + 1, i + 2]} for i in range(10)]
        validate_update_status(u.insert(u_rows), len(u_rows))
        _ = u.select(u.out).collect()

    def test_query_errors(self, reset_db: None) -> None:
        schema = {'a': pxt.Int, 'b': pxt.Int}
        t = pxt.create_table('test', schema)
        rows = [{'a': i, 'b': i + 1} for i in range(100)]
        validate_update_status(t.insert(rows), expected_rows=len(rows))

        @pxt.query
        def c(x: int, y: int) -> pxt.DataFrame:
            return t.order_by(t.a).where(t.a > x).select(c=t.a + y).limit(10)

    @staticmethod
    @pxt.udf
    def binding_test_udf(p1: str, p2: str, p3: str, p4: str = 'default') -> str:
        return f'{p1} {p2} {p3} {p4}'

    def test_partial_binding(self, reset_db: None) -> None:
        pb1 = self.binding_test_udf.using(p2='y')
        pb2 = self.binding_test_udf.using(p1='x', p3='z')
        pb3 = self.binding_test_udf.using(p1='x', p2='y', p3='z')
        assert pb1.arity == 3
        assert pb2.arity == 2
        assert pb3.arity == 1
        assert len(pb1.signatures[0].required_parameters) == 2
        assert len(pb2.signatures[0].required_parameters) == 1
        assert len(pb3.signatures[0].required_parameters) == 0
        assert pb2.signatures[0].required_parameters[0].name == 'p2'

        t = pxt.create_table('test', {'c1': pxt.String, 'c2': pxt.String, 'c3': pxt.String})
        t.insert(c1='a', c2='b', c3='c')
        t.add_computed_column(pb1=pb1(t.c1, t.c3))
        t.add_computed_column(pb2=pb2(t.c2))
        t.add_computed_column(pb3=pb3(p4='changed'))
        res = t.select(t.pb1, t.pb2, t.pb3).collect()
        assert res[0] == {'pb1': 'a y c default', 'pb2': 'x b z default', 'pb3': 'x y z changed'}

        with pytest.raises(excs.Error) as exc_info:
            self.binding_test_udf.using(non_param='a')
        assert 'Unknown parameter: non_param' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            self.binding_test_udf.using(p1=t.c1)
        assert "Expected a constant value for parameter 'p1' in call to .using()" in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:
            self.binding_test_udf.using(p1=5)
        assert "Expected type `String` for parameter 'p1'; got `Int`" in str(exc_info.value)

        with pytest.raises(TypeError) as exc_info:
            _ = pb1(p1='a')
        assert 'missing a required argument' in str(exc_info.value).lower()

    def test_nested_partial_binding(self, reset_db) -> None:
        pb1 = self.binding_test_udf.using(p2='y')
        pb2 = pb1.using(p1='x')
        pb3 = pb2.using(p3='z')
        assert pb1.arity == 3
        assert pb2.arity == 2
        assert pb3.arity == 1
        assert len(pb1.signatures[0].required_parameters) == 2
        assert len(pb2.signatures[0].required_parameters) == 1
        assert len(pb3.signatures[0].required_parameters) == 0
        assert pb2.signatures[0].required_parameters[0].name == 'p3'

        t = pxt.create_table('test', {'c1': pxt.String, 'c2': pxt.String, 'c3': pxt.String})
        t.insert(c1='a', c2='b', c3='c')
        t.add_computed_column(pb1=pb1(t.c1, t.c3))
        t.add_computed_column(pb2=pb2(t.c3))
        t.add_computed_column(pb3=pb3(p4='changed'))
        res = t.select(t.pb1, t.pb2, t.pb3).collect()
        assert res[0] == {'pb1': 'a y c default', 'pb2': 'x y c default', 'pb3': 'x y z changed'}

    @staticmethod
    @pxt.expr_udf
    def add1(x: int) -> int:
        return x + 1

    @staticmethod
    @pxt.expr_udf
    def add2(x: int, y: int) -> int:
        return x + y

    @staticmethod
    @pxt.expr_udf
    def add2_with_default(x: int, y: int = 1) -> int:
        return x + y

    def test_expr_udf(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        t.add_computed_column(other_int=t.c2 + 5)

        res1 = t.select(out=self.add1(t.c2)).order_by(t.c2).collect()
        res2 = t.select(t.c2 + 1).order_by(t.c2).collect()
        assert_resultset_eq(res1, res2)

        # return type inferred from expression
        res1 = t.select(out=self.add2(t.c2, t.c2)).order_by(t.c2).collect()
        res2 = t.select(t.c2 * 2).order_by(t.c2).collect()
        assert_resultset_eq(res1, res2)

        # multiple evaluations of the same expr_udf in the same computation
        res1 = t.select(out1=self.add1(t.c2), out2=self.add1(t.other_int)).order_by(t.c2).collect()
        res2 = t.select(t.c2 + 1, t.other_int + 1).order_by(t.c2).collect()
        assert_resultset_eq(res1, res2)

        with pytest.raises(TypeError) as exc_info:
            _ = t.select(self.add1(y=t.c2)).collect()
        assert 'missing a required argument' in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            # parameter types cannot be inferred
            @pxt.expr_udf
            def add1(x, y) -> int:  # type: ignore[no-untyped-def]
                return x + y

        assert 'cannot infer pixeltable type' in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            # missing param types
            @pxt.expr_udf(param_types=[pxt.IntType()])
            def add1(x, y) -> int:  # type: ignore[no-untyped-def]
                return x + y

        assert "missing type for parameter 'y'" in str(exc_info.value).lower()

        with pytest.raises(TypeError) as t_exc_info:
            # signature has correct parameter kind
            @pxt.expr_udf
            def add1(*, x: int) -> int:
                return x

            _ = t.select(add1(t.c2)).collect()
        assert 'takes 0 positional arguments' in str(t_exc_info.value).lower()

        res1 = t.select(out=self.add2_with_default(t.c2)).order_by(t.c2).collect()
        res2 = t.select(out=self.add2(t.c2, 1)).order_by(t.c2).collect()
        assert_resultset_eq(res1, res2)

    # Test that various invalid udf definitions generate
    # correct error messages.
    def test_invalid_udfs(self) -> None:
        with pytest.raises(excs.Error) as exc_info:

            @pxt.udf
            def udf1(name: Batch[str]) -> str:
                return ''

        assert 'batched parameters in udf, but no `batch_size` given' in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:

            @pxt.udf(batch_size=32)
            def udf2(name: Batch[str]) -> str:
                return ''

        assert 'batch_size is specified; Python return type must be a `Batch`' in str(exc_info.value)

        with pytest.raises(excs.Error) as exc_info:

            @pxt.udf
            def udf3(name: str) -> Optional[np.ndarray]:
                return None

        assert 'cannot infer pixeltable return type' in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:

            @pxt.udf
            def udf4(array: np.ndarray) -> str:
                return ''

        assert "cannot infer pixeltable type for parameter 'array'" in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:

            @pxt.udf
            def udf5(name: str, untyped) -> str:  # type: ignore[no-untyped-def]
                return ''

        assert "cannot infer pixeltable type for parameter 'untyped'" in str(exc_info.value).lower()

        with pytest.raises(ValueError) as v_exc_info:

            @udf6.conditional_return_type
            def _(wrong_param: str) -> pxt.ColumnType:
                return pxt.StringType()

        assert '`wrong_param` that is not in a signature' in str(v_exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            from .module_with_duplicate_udf import duplicate_udf
        assert 'A UDF with that name already exists: tests.module_with_duplicate_udf.duplicate_udf' in str(
            exc_info.value
        )

    def test_udf_docstring(self) -> None:
        assert self.func.__doc__ == 'A UDF.'
        assert self.agg.__doc__ == 'An aggregator.'

    @pxt.udf
    def overloaded_udf(x: str, y: str, z: str = 'a') -> str:  # type: ignore[misc]
        return x + y

    @staticmethod
    @overloaded_udf.overload
    def _(x: int, y: int, z: str = 'a') -> int:
        return x + y + 1

    @staticmethod
    @overloaded_udf.overload
    def _(x: float, y: float) -> float:
        return x + y + 2.0

    @pxt.udf(type_substitutions=({T: str}, {T: int}, {T: float}))
    def typevar_udf(x: T, y: T, z: str = 'a') -> T:
        return x + y  # type: ignore[operator]

    def test_overloaded_udf(self, test_tbl: pxt.Table, reload_tester: ReloadTester) -> None:
        t = test_tbl
        fc_str = self.overloaded_udf(t.c1, t.c1)
        fc_int = self.overloaded_udf(t.c2, t.c2)
        fc_float = self.overloaded_udf(t.c3, t.c3)

        # Check that the correct signature is selected for various argument types
        assert len(self.overloaded_udf.signatures) == 3
        assert fc_str.fn.signature == self.overloaded_udf.signatures[0]
        assert fc_str.col_type.is_string_type()
        assert fc_int.fn.signature == self.overloaded_udf.signatures[1]
        assert fc_int.col_type.is_int_type()
        assert fc_float.fn.signature == self.overloaded_udf.signatures[2]
        assert fc_float.col_type.is_float_type()

        from pixeltable.functions.string import format

        # Check that the correct Python function is invoked for each signature
        res = t.select(fc_str, fc_int, fc_float).order_by(t.c2).collect()
        res_direct = t.select(format('{0}{1}', t.c1, t.c1), t.c2 + t.c2 + 1, t.c3 + t.c3 + 2.0).order_by(t.c2).collect()
        assert_resultset_eq(res, res_direct)

        validate_update_status(t.add_computed_column(fc_str=fc_str))
        validate_update_status(t.add_computed_column(fc_int=fc_int))
        validate_update_status(t.add_computed_column(fc_float=fc_float))
        res_cc = reload_tester.run_query(t.select(t.fc_str, t.fc_int, t.fc_float).order_by(t.c2))
        assert_resultset_eq(res_cc, res_direct)

        # Check that .using() works correctly with overloaded UDFs: it should keep only the
        # signatures for which the substitution is valid
        fn = self.overloaded_udf.using(z='b')
        fc_str2 = fn(t.c1, t.c1)
        fc_int2 = fn(t.c2, t.c2)

        assert len(fn.signatures) == 2
        assert fc_str2.fn.signature == fn.signatures[0]
        assert fc_str2.col_type.is_string_type()
        assert fc_int2.fn.signature == fn.signatures[1]
        assert fc_int2.col_type.is_int_type()

        with pytest.raises(excs.Error, match='has no matching signature') as exc_info:
            fn(t.c3, t.c3)

        res = t.select(fc_str2, fc_int2).order_by(t.c2).collect()
        res_direct = t.select(format('{0}{1}', t.c1, t.c1), t.c2 + t.c2 + 1).order_by(t.c2).collect()
        assert_resultset_eq(res, res_direct)

        validate_update_status(t.add_computed_column(fc_str2=fc_str2))
        validate_update_status(t.add_computed_column(fc_int2=fc_int2))
        res_cc = reload_tester.run_query(t.select(t.fc_str2, t.fc_int2).order_by(t.c2))
        assert_resultset_eq(res_cc, res_direct)

        fc_str3 = self.typevar_udf(t.c1, t.c1)
        fc_int3 = self.typevar_udf(t.c2, t.c2)
        fc_float3 = self.typevar_udf(t.c3, t.c3)

        assert len(self.typevar_udf.signatures) == 3
        assert fc_str3.fn.signature == self.typevar_udf.signatures[0]
        assert fc_str3.col_type.is_string_type()
        assert fc_int3.fn.signature == self.typevar_udf.signatures[1]
        assert fc_int3.col_type.is_int_type()
        assert fc_float3.fn.signature == self.typevar_udf.signatures[2]
        assert fc_float3.col_type.is_float_type()

        res = t.select(fc_str3, fc_int3, fc_float3).order_by(t.c2).collect()
        res_direct = t.select(format('{0}{1}', t.c1, t.c1), t.c2 + t.c2, t.c3 + t.c3).order_by(t.c2).collect()
        assert_resultset_eq(res, res_direct)

        validate_update_status(t.add_computed_column(fc_str3=fc_str3))
        validate_update_status(t.add_computed_column(fc_int3=fc_int3))
        validate_update_status(t.add_computed_column(fc_float3=fc_float3))
        res_cc = reload_tester.run_query(t.select(t.fc_str3, t.fc_int3, t.fc_float3).order_by(t.c2))
        assert_resultset_eq(res_cc, res_direct)

        reload_tester.run_reload_test()

    @pxt.uda
    class overloaded_uda(pxt.Aggregator):
        def __init__(self) -> None:
            self.sum = ''

        def update(self, x: str) -> None:
            self.sum += x

        def value(self) -> str:
            return self.sum

    @overloaded_uda.overload
    class _(pxt.Aggregator):  # type: ignore[no-redef]
        def __init__(self) -> None:
            self.sum = 0

        def update(self, x: int) -> None:
            self.sum += x + 1

        def value(self) -> int:
            return self.sum

    @overloaded_uda.overload
    class _(pxt.Aggregator):  # type: ignore[no-redef]
        def __init__(self) -> None:
            self.sum = 0.0

        def update(self, x: float) -> None:
            self.sum += x + 2.0

        def value(self) -> float:
            return self.sum

    @pxt.uda(type_substitutions=({T: str}, {T: int}, {T: float}))  # type: ignore[misc]
    class typevar_uda(pxt.Aggregator, typing.Generic[T]):
        max: Optional[T]

        def __init__(self) -> None:
            self.max = None

        def update(self, x: T) -> None:
            if self.max is None or x > self.max:  # type: ignore[operator]
                self.max = x

        def value(self) -> T:
            return self.max

    def test_overloaded_uda(self, test_tbl: pxt.Table) -> None:
        t = test_tbl
        fc_str = self.overloaded_uda(t.c1)
        fc_int = self.overloaded_uda(t.c2)
        fc_float = self.overloaded_uda(t.c3)

        # Check that the correct signature is selected for various argument types
        assert len(self.overloaded_uda.signatures) == 3
        assert fc_str.fn.signature == self.overloaded_uda.signatures[0]
        assert fc_str.col_type.is_string_type()
        assert fc_int.fn.signature == self.overloaded_uda.signatures[1]
        assert fc_int.col_type.is_int_type()
        assert fc_float.fn.signature == self.overloaded_uda.signatures[2]
        assert fc_float.col_type.is_float_type()

        res = t.order_by(t.c2).select(c1=fc_str, c2=fc_int, c3=fc_float).collect()
        res_direct = t.order_by(t.c2).select(t.c1, t.c2, t.c3).collect()
        assert len(res) == 1
        assert res[0] == {
            'c1': ''.join(res_direct['c1']),
            'c2': sum(res_direct['c2']) + len(res_direct['c2']),
            'c3': sum(res_direct['c3']) + 2.0 * len(res_direct['c3']),
        }

        fc_str2 = self.typevar_uda(t.c1)
        fc_int2 = self.typevar_uda(t.c2)
        fc_float2 = self.typevar_uda(t.c3)

        assert len(self.typevar_uda.signatures) == 3
        assert fc_str2.fn.signature == self.typevar_uda.signatures[0]
        assert fc_str2.col_type.is_string_type()
        assert fc_int2.fn.signature == self.typevar_uda.signatures[1]
        assert fc_int2.col_type.is_int_type()
        assert fc_float2.fn.signature == self.typevar_uda.signatures[2]
        assert fc_float2.col_type.is_float_type()

        res = t.order_by(t.c2).select(c1=fc_str2, c2=fc_int2, c3=fc_float2).collect()
        res_direct = t.order_by(t.c2).select(t.c1, t.c2, t.c3).collect()
        assert len(res) == 1
        assert res[0] == {'c1': max(res_direct['c1']), 'c2': max(res_direct['c2']), 'c3': max(res_direct['c3'])}

    def test_constants(self, reset_db: None) -> None:
        """
        Test UDFs with default values and/or constant arguments that are not JSON serializable.
        """

        @pxt.udf(_force_stored=True)
        def udf_with_timestamp_constants(ts1: datetime, ts2: datetime = datetime.fromtimestamp(0)) -> float:
            return (ts1 - ts2).seconds

        t = pxt.create_table('test1', {'ts1': pxt.Timestamp})
        t.add_computed_column(seconds_since_epoch=udf_with_timestamp_constants(t.ts1))
        t.add_computed_column(seconds_since_2000=udf_with_timestamp_constants(t.ts1, ts2=datetime(2000, 1, 1)))

        @pxt.udf(_force_stored=True)
        def udf_with_array_constants(
            a: pxt.Array[pxt.Float, (6,)], b: pxt.Array[pxt.Float, (6,)] = np.ones(6, dtype=np.float32)
        ) -> pxt.Array[pxt.Float, (6,)]:
            return a + b

        t = pxt.create_table('test2', {'a': pxt.Array[pxt.Float, (6,)]})  # type: ignore[misc]
        t.add_computed_column(add_one=udf_with_array_constants(t.a))
        t.add_computed_column(add_zeros=udf_with_array_constants(t.a, b=np.zeros(6, dtype=np.float32)))

        reload_catalog()

    @pytest.mark.parametrize('as_kwarg', [False, True])
    def test_udf_evolution(self, as_kwarg: bool, reset_db: None) -> None:
        """
        Tests that code changes to UDFs that are backward-compatible with the code pattern in a stored computed
        column are accepted by Pixeltable.

        The test operates by instantiating a computed column with the UDF `evolving_udf`, then repeatedly
        monkey-patching `evolving_udf` with different signatures and checking that they new signatures are
        accepted by Pixeltable.

        The test runs two ways:
        - with the UDF invoked using a positional argument: `evolving_udf(t.c1)`
        - with the UDF invoked using a keyword argument: `evolving_udf(a=t.c1)`

        We also test that backward-incompatible changes raise appropriate warnings and errors. Because the
        error messages are lengthy and complex, we match against the entire fully-baked error string, to ensure
        that they remain comprehensible after future refactorings.
        """
        import tests.test_function

        t = pxt.create_table('test', {'c1': pxt.String})
        t.insert(c1='xyz')

        def mimic(fn: func.CallableFunction) -> None:
            """Monkey-patches `tests.test_function.evolving_udf` with the given function."""
            tests.test_function.evolving_udf = func.CallableFunction(
                fn.signatures, fn.py_fns, 'tests.test_function.evolving_udf'
            )

        def reload_and_validate_table(validation_error: Optional[str] = None) -> None:
            reload_catalog()

            # Ensure a warning is generated when the table is accessed, if appropriate
            if validation_error is None:
                t = pxt.get_table('test')
            else:
                with pytest.warns(pxt.PixeltableWarning, match=warning_regex(validation_error)):
                    t = pxt.get_table('test')
                _ = pxt.get_table('test')  # Ensure the warning is only displayed once

            # Ensure the table can be queried even if there are invalid columns
            assert list(t.head()) == [{'c1': 'xyz', 'result': None}]

            # Ensure that inserting or updating raises an error if there is an invalid column
            if validation_error is None:
                t.insert(c1='abc')
                t.where(t.c1 == 'abc').update({'c1': 'def'})
                t.where(t.c1 == 'def').delete()
            else:
                with pytest.raises(excs.Error, match=insert_error_regex(validation_error)):
                    t.insert(c1='abc')
                with pytest.raises(excs.Error, match=update_error_regex(validation_error)):
                    t.where(t.c1 == 'xyz').update({'c1': 'def'})

        def warning_regex(msg: str) -> str:
            regex = '\n'.join(
                [
                    re.escape("The computed column 'result' in table 'test' is no longer valid."),
                    re.escape(msg),
                    re.escape(
                        'You can continue to query existing data from this column, but evaluating it on new data will raise an error.'
                    ),
                ]
            )
            return '(?s)' + regex

        def insert_error_regex(msg: str) -> str:
            regex = '\n'.join(
                [
                    re.escape(
                        "Data cannot be inserted into the table 'test',\nbecause the column 'result' is currently invalid:"
                    ),
                    re.escape(msg),
                ]
            )
            return '(?s)' + regex

        def update_error_regex(msg: str) -> str:
            regex = '.*'.join(
                [
                    re.escape(
                        "Data cannot be updated in the table 'test',\nbecause the column 'result' is currently invalid:"
                    ),
                    re.escape(msg),
                ]
            )
            return '(?s)' + regex

        db_params = '(a: Optional[String])' if as_kwarg else '(Optional[String])'
        signature_error = dedent(
            f"""
            The signature stored in the database for a UDF call to 'tests.test_function.evolving_udf' no longer
            matches its signature as currently defined in the code. This probably means that the
            code for 'tests.test_function.evolving_udf' has changed in a backward-incompatible way.
            Signature of UDF call in the database: {db_params} -> Optional[Array[Float]]
            Signature of UDF as currently defined in code: {{params}} -> Optional[Array[Float]]
            """
        ).strip()
        return_type_error = dedent(
            """
            The return type stored in the database for a UDF call to 'tests.test_function.evolving_udf' no longer
            matches its return type as currently defined in the code. This probably means that the
            code for 'tests.test_function.evolving_udf' has changed in a backward-incompatible way.
            Return type of UDF call in the database: Optional[Array[Float]]
            Return type of UDF as currently defined in code: {return_type}
            """
        ).strip()

        @pxt.udf(_force_stored=True)
        def udf_base_version(a: str, b: int = 3) -> Optional[pxt.Array[pxt.Float]]:
            return None

        mimic(udf_base_version)
        if as_kwarg:
            t.add_computed_column(result=tests.test_function.evolving_udf(a=t.c1))
        else:
            t.add_computed_column(result=tests.test_function.evolving_udf(t.c1))

        # Change type of an unused optional parameter; this works in all cases
        @pxt.udf(_force_stored=True)
        def udf_version_2(a: str, b: str = 'x') -> Optional[pxt.Array[pxt.Float]]:
            return None

        mimic(udf_version_2)
        reload_and_validate_table()

        # Rename the parameter; this works only if the UDF was invoked with a positional argument
        @pxt.udf(_force_stored=True)
        def udf_version_3(c: str, b: str = 'x') -> Optional[pxt.Array[pxt.Float]]:
            return None

        mimic(udf_version_3)
        if as_kwarg:
            reload_and_validate_table(validation_error=signature_error.format(params='(c: String, b: String)'))
        else:
            reload_and_validate_table()

        # Change the parameter from fixed to variable; this works only if the UDF was invoked with a positional
        # argument
        @pxt.udf(_force_stored=True)
        def udf_version_4(*a: str) -> Optional[pxt.Array[pxt.Float]]:
            return None

        mimic(udf_version_4)
        if as_kwarg:
            reload_and_validate_table(validation_error=signature_error.format(params='(*a)'))
        else:
            reload_and_validate_table()

        # Narrow the return type; this works in all cases
        @pxt.udf(_force_stored=True)
        def udf_version_5(a: str, b: int = 3) -> Optional[pxt.Array[pxt.Float, (512,)]]:
            return None

        mimic(udf_version_5)
        reload_and_validate_table()

        # Change the type of the parameter to something incompatible; this fails in all cases
        @pxt.udf(_force_stored=True)
        def udf_version_6(a: float, b: int = 3) -> Optional[pxt.Array[pxt.Float]]:
            return None

        mimic(udf_version_6)
        reload_and_validate_table(validation_error=signature_error.format(params='(a: Float, b: Int)'))

        # Widen the return type; this fails in all cases
        @pxt.udf(_force_stored=True)
        def udf_version_7(a: str, b: int = 3) -> Optional[pxt.Array]:
            return None

        mimic(udf_version_7)
        reload_and_validate_table(validation_error=return_type_error.format(return_type='Optional[Array]'))

        # Add a poison parameter; this works only if the UDF was invoked with a keyword argument
        @pxt.udf(_force_stored=True)
        def udf_version_8(c: float = 5.0, a: str = '', b: int = 3) -> Optional[pxt.Array[pxt.Float]]:
            return None

        mimic(udf_version_8)
        if as_kwarg:
            reload_and_validate_table()
        else:
            reload_and_validate_table(validation_error=signature_error.format(params='(c: Float, a: String, b: Int)'))

        # Make the function into a non-UDF
        tests.test_function.evolving_udf = lambda x: x  # type: ignore[assignment]
        validation_error = (
            "The UDF 'tests.test_function.evolving_udf' cannot be located, because\n"
            "the symbol 'tests.test_function.evolving_udf' is no longer a UDF. (Was the `@pxt.udf` decorator removed?)"
        )
        reload_and_validate_table(validation_error=validation_error)

        # Remove the function entirely
        del tests.test_function.evolving_udf
        validation_error = (
            "The UDF 'tests.test_function.evolving_udf' cannot be located, because\n"
            "the symbol 'tests.test_function.evolving_udf' no longer exists. (Was the UDF moved or renamed?)"
        )
        reload_and_validate_table(validation_error=validation_error)

    def test_tool_errors(self) -> None:
        with pytest.raises(excs.Error) as exc_info:
            pxt.tools(pxt.functions.sum)  # type: ignore[arg-type]
        assert 'Aggregator UDFs cannot be used as tools' in str(exc_info.value)

    def test_from_table(self, reset_db: None) -> None:
        schema = {'in1': pxt.Required[pxt.Int], 'in2': pxt.Required[pxt.String], 'in3': pxt.Float, 'in4': pxt.Image}
        t = pxt.create_table('test', schema)
        t.add_computed_column(out1=(t.in1 + 5))
        t.add_computed_column(out2=(t.in3 + t.out1))
        t.add_computed_column(out3=pxtf.string.format('xyz {0}', t.in2))
        t.add_computed_column(out4=pxtf.string.format('{0} {1}', t.in1, t.out3))

        fn = pxt.udf(t)
        assert fn.arity == 4
        assert len(fn.signature.required_parameters) == 2
        assert list(fn.signature.parameters.keys()) == ['in1', 'in2', 'in3', 'in4']
        assert fn.__doc__ == dedent(
            """
            UDF for table 'test'

            Args:
                in1: of type `Int`
                in2: of type `String`
                in3: of type `Optional[Float]`
                in4: of type `Optional[Image]`
            """
        ).strip()  # fmt: skip

        u = pxt.create_table('udf_test', {'a': pxt.String, 'b': pxt.Image})
        u.insert(a='grapefruit')
        u.insert(a='canteloupe')
        u.add_computed_column(result=fn(19, u.a, in3=11.0))
        res = u.select(u.result).collect()['result']
        assert res == [
            {
                'in1': 19,
                'in2': 'grapefruit',
                'in3': 11.0,
                'in4': None,
                'out1': 24,
                'out2': 35.0,
                'out3': 'xyz grapefruit',
                'out4': '19 xyz grapefruit',
            },
            {
                'in1': 19,
                'in2': 'canteloupe',
                'in3': 11.0,
                'in4': None,
                'out1': 24,
                'out2': 35.0,
                'out3': 'xyz canteloupe',
                'out4': '19 xyz canteloupe',
            },
        ]

        # table_as_udf on a view
        v = pxt.create_view('test_view', t)
        v.add_column(in5=pxt.Json)
        v.add_computed_column(out5=(v.out1 + v.in3 + v.in5.number))

        vv = pxt.create_view('test_subview', v, comment='This is an example table comment.')
        vv.add_column(in6=pxt.Json)
        vv.add_computed_column(out6=(vv.out5 + v.out1 + t.in3 + vv.in6.number))

        fn2 = pxt.udf(vv)
        res = u.select(result=fn2(22, 'jackfruit', in3=28.0, in5={'number': 33})).collect()['result']
        assert res == [
            {
                'in1': 22,
                'in2': 'jackfruit',
                'in3': 28.0,
                'in4': None,
                'out1': 27,
                'out2': 55.0,
                'out3': 'xyz jackfruit',
                'out4': '22 xyz jackfruit',
                'in5': {'number': 33},
                'out5': 88.0,
                'in6': None,
                'out6': None,
            },
            {
                'in1': 22,
                'in2': 'jackfruit',
                'in3': 28.0,
                'in4': None,
                'out1': 27,
                'out2': 55.0,
                'out3': 'xyz jackfruit',
                'out4': '22 xyz jackfruit',
                'in5': {'number': 33},
                'out5': 88.0,
                'in6': None,
                'out6': None,
            },
        ]
        assert fn2.__doc__ == dedent(
            """
            This is an example table comment.

            Args:
                in1: of type `Int`
                in2: of type `String`
                in3: of type `Optional[Float]`
                in4: of type `Optional[Image]`
                in5: of type `Optional[Json]`
                in6: of type `Optional[Json]`
            """
        ).strip()  # fmt: skip

        # Explicit return_value and description
        fn3 = pxt.udf(t, return_value=t.out3.upper(), description='An overriden UDF description.')
        res = u.select(result=fn3(22, u.a)).collect()['result']
        assert res == ['XYZ GRAPEFRUIT', 'XYZ CANTELOUPE']
        assert fn3.__doc__ == dedent(
            """
            An overriden UDF description.

            Args:
                in1: of type `Int`
                in2: of type `String`
                in3: of type `Optional[Float]`
                in4: of type `Optional[Image]`
            """
        ).strip()  # fmt: skip

        # return_value is a direct ColumnRef
        fn4 = pxt.udf(t, return_value=t.out3)
        res = u.select(result=fn4(22, u.a)).collect()['result']
        assert res == ['xyz grapefruit', 'xyz canteloupe']

        # return value is a custom dict
        fn5 = pxt.udf(
            t, return_value={'plus_five': t.out1, 'xyz': t.out3, 'abcxyz': pxtf.string.format('abc {0}', t.out3)}
        )
        res = u.select(result=fn5(22, u.a)).collect()['result']
        assert res == [
            {'plus_five': 27, 'xyz': 'xyz grapefruit', 'abcxyz': 'abc xyz grapefruit'},
            {'plus_five': 27, 'xyz': 'xyz canteloupe', 'abcxyz': 'abc xyz canteloupe'},
        ]

        fn6 = pxt.udf(t, return_value=t.in4.rotate(t.in1))
        u.select(fn6(22, 'starfruit', in4=u.b)).collect()


@pxt.udf
def udf6(name: str) -> str:
    return ''


evolving_udf: Optional[func.CallableFunction] = None
