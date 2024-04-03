from typing import Optional

import numpy as np
import pytest

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable import catalog
from pixeltable.func import Function, FunctionRegistry, Batch
from pixeltable.type_system import IntType, FloatType
from pixeltable.tests.utils import assert_resultset_eq
import pixeltable.func as func


def dummy_fn(i: int) -> int:
    return i

class TestFunction:
    @pxt.udf(return_type=IntType(), param_types=[IntType()])
    def func(x: int) -> int:
        return x + 1

    @pxt.uda(value_type=IntType(), update_types=[IntType()])
    class agg:
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

    @pytest.mark.skip(reason='deprecated')
    def test_create(self, test_client: pxt.Client) -> None:
        cl = test_client
        cl.create_function('test_fn', self.func)
        assert self.func.md.fqn == 'test_fn'
        FunctionRegistry.get().clear_cache()
        cl = pxt.Client(reload=True)
        _ = cl.list_functions()
        fn2 = cl.get_function('test_fn')
        assert fn2.md.fqn == 'test_fn'
        assert fn2.py_fn(1) == 2

        with pytest.raises(excs.Error):
            cl.create_function('test_fn', self.func)
        with pytest.raises(excs.Error):
            cl.create_function('dir1.test_fn', self.func)
        with pytest.raises(excs.Error):
            library_fn = make_library_function(IntType(), [IntType()], __name__, 'dummy_fn')
            cl.create_function('library_fn', library_fn)

    @pytest.mark.skip(reason='deprecated')
    def test_update(self, test_client: pxt.Client, test_tbl: catalog.Table) -> None:
        cl = test_client
        t = test_tbl
        cl.create_function('test_fn', self.func)
        res1 = t[self.func(t.c2)].show(0).to_pandas()

        # load function from db and make sure it computes the same thing as before
        FunctionRegistry.get().clear_cache()
        cl = pxt.Client(reload=True)
        fn = cl.get_function('test_fn')
        res2 = t[fn(t.c2)].show(0).to_pandas()
        assert res1.col_0.equals(res2.col_0)
        fn.py_fn = lambda x: x + 2
        cl.update_function('test_fn', fn)
        assert self.func.md.fqn == fn.md.fqn  # fqn doesn't change

        FunctionRegistry.get().clear_cache()
        cl = pxt.Client(reload=True)
        fn = cl.get_function('test_fn')
        assert self.func.md.fqn == fn.md.fqn  # fqn doesn't change
        res3 = t[fn(t.c2)].show(0).to_pandas()
        assert (res2.col_0 + 1).equals(res3.col_0)

        # signature changes
        with pytest.raises(excs.Error):
            cl.update_function('test_fn', make_function(FloatType(), [IntType()], fn.py_fn))
        with pytest.raises(excs.Error):
            cl.update_function('test_fn', make_function(IntType(), [FloatType()], fn.py_fn))
        with pytest.raises(excs.Error):
            cl.update_function('test_fn', self.agg)

    @pytest.mark.skip(reason='deprecated')
    def test_move(self, test_client: pxt.Client) -> None:
        cl = test_client
        cl.create_function('test_fn', self.func)

        FunctionRegistry.get().clear_cache()
        cl = pxt.Client(reload=True)
        with pytest.raises(excs.Error):
            cl.move('test_fn2', 'test_fn')
        cl.move('test_fn', 'test_fn2')
        func = cl.get_function('test_fn2')
        assert func.py_fn(1) == 2
        assert func.md.fqn == 'test_fn2'

        with pytest.raises(excs.Error):
            _ = cl.get_function('test_fn')

        # move function between directories
        cl.create_dir('functions')
        cl.create_dir('functions2')
        cl.create_function('functions.func1', self.func)
        with pytest.raises(excs.Error):
            cl.move('functions2.func1', 'functions.func1')
        cl.move('functions.func1', 'functions2.func1')
        func = cl.get_function('functions2.func1')
        assert func.md.fqn == 'functions2.func1'


        FunctionRegistry.get().clear_cache()
        cl = pxt.Client(reload=True)
        func = cl.get_function('functions2.func1')
        assert func.py_fn(1) == 2
        assert func.md.fqn == 'functions2.func1'
        with pytest.raises(excs.Error):
            _ = cl.get_function('functions.func1')

    @pytest.mark.skip(reason='deprecated')
    def test_drop(self, test_client: pxt.Client) -> None:
        cl = test_client
        cl.create_function('test_fn', self.func)
        FunctionRegistry.get().clear_cache()
        cl = pxt.Client(reload=True)
        cl.drop_function('test_fn')

        with pytest.raises(excs.Error):
            _ = cl.get_function('test_fn')

    def test_list(self, test_client: pxt.Client) -> None:
        _ = FunctionRegistry.get().list_functions()
        print(_)

    def test_stored_udf(self, test_client: pxt.Client) -> None:
        cl = test_client
        t = cl.create_table('test', {'c1': pxt.IntType(), 'c2': pxt.FloatType()})
        rows = [{'c1': i, 'c2': i + 0.5} for i in range(100)]
        status = t.insert(rows)
        assert status.num_rows == len(rows)
        assert status.num_excs == 0

        @pxt.udf(_force_stored=True)
        def f1(a: int, b: float) -> float:
            return a + b
        t['f1'] = f1(t.c1, t.c2)

        func.FunctionRegistry.get().clear_cache()
        cl = pxt.Client(reload=True)
        t = cl.get_table('test')
        status = t.insert(rows)
        assert status.num_rows == len(rows)
        assert status.num_excs == 0

    @pxt.udf(return_type=IntType(), param_types=[IntType(), FloatType(), FloatType(), FloatType()])
    def f1(a: int, b: float, c: float = 0.0, d: float = 1.0) -> float:
        return a + b + c + d

    @pxt.udf(
        return_type=IntType(),
        param_types=[IntType(nullable=True), FloatType(nullable=False), FloatType(nullable=True)])
    def f2(a: int, b: float = 0.0, c: float = 1.0) -> float:
        return (0.0 if a is None else a) + b + (0.0 if c is None else c)

    def test_call(self, test_tbl: catalog.Table) -> None:
        t = test_tbl

        r0 = t[t.c2, t.c3].show(0).to_pandas()
        # positional params with default args
        r1 = t[self.f1(t.c2, t.c3)].show(0).to_pandas()['col_0']
        assert np.all(r1 == r0.c2 + r0.c3 + 1.0)
        # kw args only
        r2 = t[self.f1(c=0.0, b=t.c3, a=t.c2)].show(0).to_pandas()['col_0']
        assert np.all(r1 == r2)
        # overriding default args
        r3 = t[self.f1(d=0.0, c=1.0, b=t.c3, a=t.c2)].show(0).to_pandas()['col_0']
        assert np.all(r2 == r3)
        # overriding default with positional arg
        r4 = t[self.f1(t.c2, t.c3, 0.0)].show(0).to_pandas()['col_0']
        assert np.all(r3 == r4)
        # overriding default with positional arg and kw arg
        r5 = t[self.f1(t.c2, t.c3, 1.0, d=0.0)].show(0).to_pandas()['col_0']
        assert np.all(r4 == r5)
        # d is kwarg
        r6 = t[self.f1(t.c2, d=1.0, b=t.c3)].show(0).to_pandas()['col_0']
        assert np.all(r5 == r6)
        # d is Expr kwarg
        r6 = t[self.f1(1, d=t.c3, b=t.c3)].show(0).to_pandas()['col_0']
        assert np.all(r5 == r6)

        # test handling of Nones
        r0 = t[self.f2(1, t.c3)].show(0).to_pandas()['col_0']
        r1 = t[self.f2(None, t.c3, 2.0)].show(0).to_pandas()['col_0']
        assert np.all(r0 == r1)
        r2 = t[self.f2(2, t.c3, None)].show(0).to_pandas()['col_0']
        assert np.all(r1 == r2)
        # kwarg with None
        r3 = t[self.f2(c=None, a=t.c2)].show(0).to_pandas()['col_0']
        # kwarg with Expr
        r4 = t[self.f2(c=t.c3, a=None)].show(0).to_pandas()['col_0']
        assert np.all(r3 == r4)

        with pytest.raises(TypeError) as exc_info:
            _ = t[self.f1(t.c2, c=0.0)].show(0)
        assert "'b'" in str(exc_info.value)
        with pytest.raises(TypeError) as exc_info:
            _ = t[self.f1(t.c2)].show(0)
        assert "'b'" in str(exc_info.value)
        with pytest.raises(TypeError) as exc_info:
            _ = t[self.f1(c=1.0, a=t.c2)].show(0)
        assert "'b'" in str(exc_info.value)

        # bad default value
        with pytest.raises(excs.Error) as exc_info:
            @pxt.udf(return_type=IntType(), param_types=[IntType(), FloatType(), FloatType()])
            def f1(a: int, b: float, c: str = '') -> float:
                return a + b + c
        assert 'default value' in str(exc_info.value).lower()
        # missing param type
        with pytest.raises(excs.Error) as exc_info:
            @pxt.udf(return_type=IntType(), param_types=[IntType(), FloatType()])
            def f1(a: int, b: float, c: str = '') -> float:
                return a + b + c
        assert 'missing type for parameter c' in str(exc_info.value).lower()
        # bad parameter name
        with pytest.raises(excs.Error) as exc_info:
            @pxt.udf(return_type=IntType(), param_types=[IntType()])
            def f1(group_by: int) -> int:
                return group_by
        assert 'reserved' in str(exc_info.value)
        # bad parameter name
        with pytest.raises(excs.Error) as exc_info:
            @pxt.udf(return_type=IntType(), param_types=[IntType()])
            def f1(order_by: int) -> int:
                return order_by
        assert 'reserved' in str(exc_info.value)

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
            @pxt.expr_udf(param_types=[IntType()])
            def add1(x, y) -> int:
                return x + y
        assert 'missing type for parameter y' in str(exc_info.value).lower()

        with pytest.raises(TypeError) as exc_info:
            # signature has correct parameter kind
            @pxt.expr_udf
            def add1(*, x: int) -> int:
                return x + y
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

