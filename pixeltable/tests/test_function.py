import numpy as np
import pytest

from pixeltable.function import Function, FunctionRegistry
from pixeltable.type_system import IntType, FloatType
from pixeltable import catalog
import pixeltable as pt
from pixeltable import exceptions as exc


def dummy_fn(i: int) -> int:
    return i

class TestFunction:
    @pt.udf(return_type=IntType(), param_types=[IntType()])
    def func(x: int) -> int:
        return x + 1

    class Aggregator:
        def __init__(self):
            self.sum = 0
        @classmethod
        def make_aggregator(cls) -> 'Aggregator':
            return cls()
        def update(self, val) -> None:
            if val is not None:
                self.sum += val
        def value(self):
            return self.sum
    agg = Function.make_aggregate_function(
        IntType(), [IntType()], Aggregator.make_aggregator, Aggregator.update, Aggregator.value)

    def test_serialize_anonymous(self, init_env) -> None:
        d = self.func.as_dict()
        FunctionRegistry.get().clear_cache()
        deserialized = Function.from_dict(d)
        assert deserialized.eval_fn(1) == 2

    def test_create(self, test_client: pt.Client) -> None:
        cl = test_client
        cl.create_function('test_fn', self.func)
        assert self.func.md.fqn == 'test_fn'
        FunctionRegistry.get().clear_cache()
        cl = pt.Client()
        _ = cl.list_functions()
        fn2 = cl.get_function('test_fn')
        assert fn2.md.fqn == 'test_fn'
        assert fn2.eval_fn(1) == 2

        with pytest.raises(exc.Error):
            cl.create_function('test_fn', self.func)
        with pytest.raises(exc.Error):
            cl.create_function('dir1.test_fn', self.func)
        with pytest.raises(exc.Error):
            library_fn = Function.make_library_function(IntType(), [IntType()], __name__, 'dummy_fn')
            cl.create_function('library_fn', library_fn)

    def test_update(self, test_client: pt.Client, test_tbl: catalog.MutableTable) -> None:
        cl = test_client
        t = test_tbl
        cl.create_function('test_fn', self.func)
        res1 = t[self.func(t.c2)].show(0).to_pandas()

        # load function from db and make sure it computes the same thing as before
        FunctionRegistry.get().clear_cache()
        cl = pt.Client()
        fn = cl.get_function('test_fn')
        res2 = t[fn(t.c2)].show(0).to_pandas()
        assert res1.col_0.equals(res2.col_0)
        fn.eval_fn = lambda x: x + 2
        cl.update_function('test_fn', fn)
        assert self.func.md.fqn == fn.md.fqn  # fqn doesn't change

        FunctionRegistry.get().clear_cache()
        cl = pt.Client()
        fn = cl.get_function('test_fn')
        assert self.func.md.fqn == fn.md.fqn  # fqn doesn't change
        res3 = t[fn(t.c2)].show(0).to_pandas()
        assert (res2.col_0 + 1).equals(res3.col_0)

        # signature changes
        with pytest.raises(exc.Error):
            cl.update_function('test_fn', Function.make_function(FloatType(), [IntType()], fn.eval_fn))
        with pytest.raises(exc.Error):
            cl.update_function('test_fn', Function.make_function(IntType(), [FloatType()], fn.eval_fn))
        with pytest.raises(exc.Error):
            cl.update_function('test_fn', self.agg)

    def test_move(self, test_client: pt.Client) -> None:
        cl = test_client
        cl.create_function('test_fn', self.func)

        FunctionRegistry.get().clear_cache()
        cl = pt.Client()
        with pytest.raises(exc.Error):
            cl.move('test_fn2', 'test_fn')
        cl.move('test_fn', 'test_fn2')
        func = cl.get_function('test_fn2')
        assert func.eval_fn(1) == 2
        assert func.md.fqn == 'test_fn2'

        with pytest.raises(exc.Error):
            _ = cl.get_function('test_fn')

        # move function between directories
        cl.create_dir('functions')
        cl.create_dir('functions2')
        cl.create_function('functions.func1', self.func)
        with pytest.raises(exc.Error):
            cl.move('functions2.func1', 'functions.func1')
        cl.move('functions.func1', 'functions2.func1')
        func = cl.get_function('functions2.func1')
        assert func.md.fqn == 'functions2.func1'


        FunctionRegistry.get().clear_cache()
        cl = pt.Client()
        func = cl.get_function('functions2.func1')
        assert func.eval_fn(1) == 2
        assert func.md.fqn == 'functions2.func1'
        with pytest.raises(exc.Error):
            _ = cl.get_function('functions.func1')

    def test_drop(self, test_client: pt.Client) -> None:
        cl = test_client
        cl.create_function('test_fn', self.func)
        FunctionRegistry.get().clear_cache()
        cl = pt.Client()
        cl.drop_function('test_fn')

        with pytest.raises(exc.Error):
            _ = cl.get_function('test_fn')

    def test_list(self, test_client: pt.Client) -> None:
        _ = FunctionRegistry.get().list_functions()
        print(_)

    def test_call(self, test_tbl: catalog.MutableTable) -> None:
        t = test_tbl

        @pt.udf(return_type=IntType(), param_types=[IntType(), FloatType(), FloatType(), FloatType()])
        def f1(a: int, b: float, c: float = 0.0, d: float = 1.0) -> float:
            return a + b + c + d

        r0 = t[t.c2, t.c3].show(0).to_pandas()
        # positional params with default args
        r1 = t[f1(t.c2, t.c3)].show(0).to_pandas()['col_0']
        assert np.all(r1 == r0.c2 + r0.c3 + 1.0)
        # kw args only
        r2 = t[f1(c=0.0, b=t.c3, a=t.c2)].show(0).to_pandas()['col_0']
        assert np.all(r1 == r2)
        # overriding default args
        r3 = t[f1(d=0.0, c=1.0, b=t.c3, a=t.c2)].show(0).to_pandas()['col_0']
        assert np.all(r2 == r3)
        # overriding default with positional arg
        r4 = t[f1(t.c2, t.c3, 0.0)].show(0).to_pandas()['col_0']
        assert np.all(r3 == r4)
        # overriding default with positional arg and kw arg
        r5 = t[f1(t.c2, t.c3, 1.0, d=0.0)].show(0).to_pandas()['col_0']
        assert np.all(r4 == r5)
        # d is kwarg
        r6 = t[f1(t.c2, d=1.0, b=t.c3)].show(0).to_pandas()['col_0']
        assert np.all(r5 == r6)
        # d is Expr kwarg
        r6 = t[f1(1, d=t.c3, b=t.c3)].show(0).to_pandas()['col_0']
        assert np.all(r5 == r6)

        # test handling of Nones
        @pt.udf(
            return_type=IntType(),
            param_types=[IntType(nullable=True), FloatType(nullable=False), FloatType(nullable=True)])
        def f2(a: int, b: float = 0.0, c: float = 1.0) -> float:
            return (0.0 if a is None else a) + b + (0.0 if c is None else c)
        r0 = t[f2(1, t.c3)].show(0).to_pandas()['col_0']
        r1 = t[f2(None, t.c3, 2.0)].show(0).to_pandas()['col_0']
        assert np.all(r0 == r1)
        r2 = t[f2(2, t.c3, None)].show(0).to_pandas()['col_0']
        assert np.all(r1 == r2)
        # kwarg with None
        r3 = t[f2(c=None, a=t.c2)].show(0).to_pandas()['col_0']
        # kwarg with Expr
        r4 = t[f2(c=t.c3, a=None)].show(0).to_pandas()['col_0']
        assert np.all(r3 == r4)

        with pytest.raises(TypeError) as exc_info:
            _ = t[f1(t.c2, c=0.0)].show(0)
        assert "'b'" in str(exc_info.value)
        with pytest.raises(TypeError) as exc_info:
            _ = t[f1(t.c2)].show(0)
        assert "'b'" in str(exc_info.value)
        with pytest.raises(TypeError) as exc_info:
            _ = t[f1(c=1.0, a=t.c2)].show(0)
        assert "'b'" in str(exc_info.value)

        # bad default value
        with pytest.raises(exc.Error) as exc_info:
            @pt.udf(return_type=IntType(), param_types=[IntType(), FloatType(), FloatType()])
            def f1(a: int, b: float, c: str = '') -> float:
                return a + b + c
        assert 'default value' in str(exc_info.value).lower()
        # missing param type
        with pytest.raises(exc.Error) as exc_info:
            @pt.udf(return_type=IntType(), param_types=[IntType(), FloatType()])
            def f1(a: int, b: float, c: str = '') -> float:
                return a + b + c
        assert 'number of parameters' in str(exc_info.value)
        # bad parameter name
        with pytest.raises(exc.Error) as exc_info:
            @pt.udf(return_type=IntType(), param_types=[IntType()])
            def f1(group_by: int) -> int:
                return group_by
        assert 'reserved' in str(exc_info.value)
        # bad parameter name
        with pytest.raises(exc.Error) as exc_info:
            @pt.udf(return_type=IntType(), param_types=[IntType()])
            def f1(order_by: int) -> int:
                return order_by
        assert 'reserved' in str(exc_info.value)
