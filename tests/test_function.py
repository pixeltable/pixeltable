from typing import Optional
import typing

import numpy as np
import pytest

import pixeltable as pxt
import pixeltable.exceptions as excs
import pixeltable.func as func
from pixeltable import catalog
from pixeltable.func import Batch, Function, FunctionRegistry

from .utils import assert_resultset_eq, reload_catalog, validate_update_status, ReloadTester


def dummy_fn(i: int) -> int:
    return i


T = typing.TypeVar('T')

class TestFunction:
    @pxt.udf
    def func(x: int) -> int:
        """A UDF."""
        return x + 1

    @pxt.uda
    class agg:
        """An aggregator."""
        def __init__(self):
            self.sum = 0
        def update(self, val: int) -> None:
            if val is not None:
                self.sum += val
        def value(self) -> int:
            return self.sum

    def test_serialize_anonymous(self, init_env) -> None:
        d = self.func.as_dict()
        FunctionRegistry.get().clear_cache()
        deserialized = Function.from_dict(d)
        assert isinstance(deserialized, func.CallableFunction)
        # TODO: add Function.exec() and then use that
        assert deserialized.py_fn(1) == 2

    def test_list(self, reset_db) -> None:
        _ = FunctionRegistry.get().list_functions()
        print(_)

    def test_list_functions(self, init_env) -> None:
        _ = pxt.list_functions()
        print(_)

    def test_stored_udf(self, reset_db) -> None:
        t = pxt.create_table('test', {'c1': pxt.IntType(), 'c2': pxt.FloatType()})
        rows = [{'c1': i, 'c2': i + 0.5} for i in range(100)]
        status = t.insert(rows)
        assert status.num_rows == len(rows)
        assert status.num_excs == 0

        @pxt.udf(_force_stored=True)
        def f1(a: int, b: float) -> float:
            return a + b
        t['f1'] = f1(t.c1, t.c2)

        func.FunctionRegistry.get().clear_cache()
        reload_catalog()
        t = pxt.get_table('test')
        status = t.insert(rows)
        assert status.num_rows == len(rows)
        assert status.num_excs == 0

    @pxt.udf
    def f1(a: int, b: float, c: float = 0.0, d: float = 1.0) -> float:
        return a + b + c + d

    @pxt.udf
    def f2(a: Optional[int], b: float = 0.0, c: Optional[float] = 1.0) -> int:
        return (0.0 if a is None else a) + b + (0.0 if c is None else c)

    def test_call(self, test_tbl: catalog.Table) -> None:
        t = test_tbl

        r0 = t[t.c2, t.c3].collect().to_pandas()
        # positional params with default args
        r1 = t[self.f1(t.c2, t.c3)].collect().to_pandas()['f1']
        assert np.all(r1 == r0.c2 + r0.c3 + 1.0)
        # kw args only
        r2 = t[self.f1(c=0.0, b=t.c3, a=t.c2)].collect().to_pandas()['f1']
        assert np.all(r1 == r2)
        # overriding default args
        r3 = t[self.f1(d=0.0, c=1.0, b=t.c3, a=t.c2)].collect().to_pandas()['f1']
        assert np.all(r2 == r3)
        # overriding default with positional arg
        r4 = t[self.f1(t.c2, t.c3, 0.0)].collect().to_pandas()['f1']
        assert np.all(r3 == r4)
        # overriding default with positional arg and kw arg
        r5 = t[self.f1(t.c2, t.c3, 1.0, d=0.0)].collect().to_pandas()['f1']
        assert np.all(r4 == r5)
        # d is kwarg
        r6 = t[self.f1(t.c2, d=1.0, b=t.c3)].collect().to_pandas()['f1']
        assert np.all(r5 == r6)
        # d is Expr kwarg
        r6 = t[self.f1(1, d=t.c3, b=t.c3)].collect().to_pandas()['f1']
        assert np.all(r5 == r6)

        # test handling of Nones
        r0 = t[self.f2(1, t.c3)].collect().to_pandas()['f2']
        r1 = t[self.f2(None, t.c3, 2.0)].collect().to_pandas()['f2']
        assert np.all(r0 == r1)
        r2 = t[self.f2(2, t.c3, None)].collect().to_pandas()['f2']
        assert np.all(r1 == r2)
        # kwarg with None
        r3 = t[self.f2(c=None, a=t.c2)].collect().to_pandas()['f2']
        # kwarg with Expr
        r4 = t[self.f2(c=t.c3, a=None)].collect().to_pandas()['f2']
        assert np.all(r3 == r4)

        with pytest.raises(TypeError) as exc_info:
            _ = t[self.f1(t.c2, c=0.0)].collect()
        assert "'b'" in str(exc_info.value)
        with pytest.raises(TypeError) as exc_info:
            _ = t[self.f1(t.c2)].collect()
        assert "'b'" in str(exc_info.value)
        with pytest.raises(TypeError) as exc_info:
            _ = t[self.f1(c=1.0, a=t.c2)].collect()
        assert "'b'" in str(exc_info.value)

        # bad default value
        with pytest.raises(excs.Error) as exc_info:
            @pxt.udf
            def f1(a: int, b: float, c: float = '') -> float:
                return a + b + c
        assert 'default value' in str(exc_info.value).lower()
        # missing param type
        with pytest.raises(excs.Error) as exc_info:
            @pxt.udf
            def f1(a: int, b: float, c = '') -> float:
                return a + b + c
        assert 'cannot infer pixeltable type for parameter c' in str(exc_info.value).lower()
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

    @pxt.udf(is_method=True)
    def increment(n: int) -> int:
        return n + 1

    @pxt.udf(is_property=True)
    def successor(n: int) -> int:
        return n + 1

    @pxt.udf(is_method=True)
    def append(s: str, suffix: str) -> str:
        return s + suffix

    def test_member_access_udf(self, reset_db) -> None:
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
        assert "`is_property=True` expects a UDF with exactly 1 parameter, but `udf8` has 2" in str(exc_info.value)

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

    def test_query(self, reset_db) -> None:
        t = pxt.create_table('test', {'c1': pxt.Int, 'c2': pxt.Float})
        name = t._name
        rows = [{'c1': i, 'c2': i + 0.5} for i in range(100)]
        validate_update_status(t.insert(rows))

        @t.query
        def lt_x(x: int) -> int:
            return t.where(t.c2 < x).select(t.c2, t.c1)

        res1 = t.select(out=t.queries.lt_x(t.c1)).order_by(t.c2).collect()
        validate_update_status(t.add_column(query1=t.queries.lt_x(t.c1)))
        _ = t.select(t.query1).collect()

        reload_catalog()
        t = pxt.get_table(name)
        _ = t.select(t.query1).collect()
        # insert more rows in order to verify that lt_x() is still executable after catalog reload
        validate_update_status(t.insert(rows))

    def test_query2(self, reset_db) -> None:
        schema = {
            'query_text': pxt.String,
            'i': pxt.Int,
        }
        queries = pxt.create_table('queries', schema)
        query_rows = [
            {'query_text': 'how much is the stock of AI companies up?', 'i': 1},
            {'query_text': 'what happened to the term machine learning?', 'i': 2},
        ]
        validate_update_status(queries.insert(query_rows), expected_rows=len(query_rows))

        chunks = pxt.create_table('test_doc_chunks', {'text': pxt.StringType()})
        chunks.insert([
            {'text': 'the stock of artificial intelligence companies is up 1000%'},
            {
                'text': (
                         'the term machine learning has fallen out of fashion now that AI has been '
                         'rehabilitated and is now the new hotness'
                )
            },
            {'text': 'machine learning is a subset of artificial intelligence'},
            {'text': 'gas car companies are in danger of being left behind by electric car companies'},
        ])

        # # TODO: make this work
        # @chunks.query
        # def retrieval(n: int):
        #     """ simply returns 2 passages from the table"""
        #     return chunks.select(chunks.text).limit(n)

        @chunks.query
        def retrieval(s: str, n: int):
            """ simply returns 2 passages from the table"""
            return chunks.select(chunks.text).limit(2)

        res = queries.select(queries.i, out=chunks.queries.retrieval(queries.query_text, queries.i)).collect()
        assert all(len(out) == 2 for out in res['out'])
        validate_update_status(queries.add_column(chunks=chunks.queries.retrieval(queries.query_text, queries.i)))
        res = queries.select(queries.i, queries.chunks).collect()
        assert all(len(c) == 2 for c in res['chunks'])

        reload_catalog()
        queries = pxt.get_table('queries')
        res = queries.select(queries.chunks).collect()
        assert all(len(c) == 2 for c in res['chunks'])
        validate_update_status(queries.insert(query_rows), expected_rows=len(query_rows))
        res = queries.select(queries.chunks).collect()
        assert all(len(c) == 2 for c in res['chunks'])

    def test_query_errors(self, reset_db) -> None:
        schema = {'a': pxt.Int, 'b': pxt.Int}
        t = pxt.create_table('test', schema)
        rows = [{'a': i, 'b': i + 1} for i in range(100)]
        validate_update_status(t.insert(rows), expected_rows=len(rows))

        # query name conflicts with column name
        with pytest.raises(excs.Error) as exc_info:
            @t.query
            def a(x: int, y: int) -> int:
                return t.order_by(t.a).where(t.a > x).select(c=t.a + y).limit(10)
        assert 'conflicts with existing column' in str(exc_info.value).lower()

        @t.query
        def c(x: int, y: int) -> int:
            return t.order_by(t.a).where(t.a > x).select(c=t.a + y).limit(10)

        # duplicate query name
        with pytest.raises(excs.Error) as exc_info:
            @t.query
            def c(x: int, y: int) -> int:
                return t.order_by(t.a).where(t.a > x).select(c=t.a + y).limit(10)
        assert 'duplicate query name' in str(exc_info.value).lower()

        # column name conflicts with query name
        with pytest.raises(excs.Error) as exc_info:
            t.add_column(c=pxt.Int)
        assert 'conflicts with a registered query' in str(exc_info.value).lower()

        # unknown query
        with pytest.raises(AttributeError) as exc_info:
            _ = t.queries.not_a_query
        assert "table 'test' has no query with that name: 'not_a_query'" in str(exc_info.value).lower()

    @pxt.udf
    def binding_test_udf(p1: str, p2: str, p3: str, p4: str = 'default') -> str:
        return f'{p1} {p2} {p3} {p4}'

    def test_partial_binding(self, reset_db) -> None:
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
            self.binding_test_udf.using(p1=5)
        assert "Expected type `String` for parameter `p1`; got `Int`" in str(exc_info.value)

        with pytest.raises(TypeError) as exc_info:
            _ = pb1(p1='a')
        assert 'missing a required argument' in str(exc_info.value).lower()

    @pxt.expr_udf
    def add1(x: int) -> int:
        return x + 1

    @pxt.expr_udf
    def add2(x: int, y: int):
        return x + y

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
            def add1(x, y) -> int:
                return x + y
        assert 'cannot infer pixeltable type' in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            # missing param types
            @pxt.expr_udf(param_types=[pxt.IntType()])
            def add1(x, y) -> int:
                return x + y
        assert 'missing type for parameter y' in str(exc_info.value).lower()

        with pytest.raises(TypeError) as exc_info:
            # signature has correct parameter kind
            @pxt.expr_udf
            def add1(*, x: int) -> int:
                return x
            _ = t.select(add1(t.c2)).collect()
        assert 'takes 0 positional arguments' in str(exc_info.value).lower()

        res1 = t.select(out=self.add2_with_default(t.c2)).order_by(t.c2).collect()
        res2 = t.select(out=self.add2(t.c2, 1)).order_by(t.c2).collect()
        assert_resultset_eq(res1, res2)

    # Test that various invalid udf definitions generate
    # correct error messages.
    def test_invalid_udfs(self):
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
        assert 'cannot infer pixeltable type for parameter array' in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            @pxt.udf
            def udf5(name: str, untyped) -> str:
                return ''
        assert 'cannot infer pixeltable type for parameter untyped' in str(exc_info.value).lower()

        with pytest.raises(ValueError) as exc_info:
            @udf6.conditional_return_type
            def _(wrong_param: str) -> pxt.ColumnType:
                return pxt.StringType()
        assert '`wrong_param` that is not in the signature' in str(exc_info.value).lower()

        with pytest.raises(excs.Error) as exc_info:
            from .module_with_duplicate_udf import duplicate_udf
        assert 'A UDF with that name already exists: tests.module_with_duplicate_udf.duplicate_udf' in str(exc_info.value)

    def test_udf_docstring(self) -> None:
        assert self.func.__doc__ == "A UDF."
        assert self.agg.__doc__ == "An aggregator."

    @pxt.udf
    def overloaded_udf(x: str, y: str, z: str = 'a') -> str:
        return x + y

    @overloaded_udf.overload
    def _(x: int, y: int, z: str = 'a') -> int:
        return x + y + 1

    @overloaded_udf.overload
    def _(x: float, y: float) -> float:
        return x + y + 2.0

    @pxt.udf(type_substitutions=({T: str}, {T: int}, {T: float}))
    def typevar_udf(x: T, y: T, z: str = 'a') -> T:
        return x + y

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

        with pytest.raises(excs.Error) as exc_info:
            fn(t.c3, t.c3)
        assert 'has no matching signature' in str(exc_info.value)

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
    class _(pxt.Aggregator):
        def __init__(self) -> None:
            self.sum = 0

        def update(self, x: int) -> None:
            self.sum += x + 1

        def value(self) -> int:
            return self.sum

    @overloaded_uda.overload
    class _(pxt.Aggregator):
        def __init__(self) -> None:
            self.sum = 0.0

        def update(self, x: float) -> None:
            self.sum += x + 2.0

        def value(self) -> float:
            return self.sum

    @pxt.uda(type_substitutions=({T: str}, {T: int}, {T: float}))
    class typevar_uda(pxt.Aggregator, typing.Generic[T]):
        def __init__(self) -> None:
            self.max = None

        def update(self, x: T) -> None:
            if self.max is None or x > self.max:
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
        assert res[0] == {
            'c1': max(res_direct['c1']),
            'c2': max(res_direct['c2']),
            'c3': max(res_direct['c3']),
        }

    def test_tool_errors(self):
        with pytest.raises(excs.Error) as exc_info:
            pxt.tools(pxt.functions.sum)
        assert 'Aggregator UDFs cannot be used as tools' in str(exc_info.value)


@pxt.udf
def udf6(name: str) -> str:
    return ''
