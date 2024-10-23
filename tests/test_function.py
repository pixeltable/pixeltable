from typing import Optional

import numpy as np
import pytest

import pixeltable as pxt
import pixeltable.exceptions as excs
import pixeltable.func as func
from pixeltable import catalog
from pixeltable.func import Batch, Function, FunctionRegistry

from .utils import assert_resultset_eq, reload_catalog, validate_update_status


def dummy_fn(i: int) -> int:
    return i

class TestFunction:
    @pxt.udf
    def func(x: int) -> int:
        """A UDF."""
        return x + 1

    @pxt.uda(value_type=pxt.IntType(), update_types=[pxt.IntType()])
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

        res1 = t.select(out=self.add1(t.c2)).order_by(t.c2).collect()
        res2 = t.select(t.c2 + 1).order_by(t.c2).collect()
        assert_resultset_eq(res1, res2)

        # return type inferred from expression
        res1 = t.select(out=self.add2(t.c2, t.c2)).order_by(t.c2).collect()
        res2 = t.select(t.c2 * 2).order_by(t.c2).collect()
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
def udf6(name: str) -> str:
    return ''
