import builtins
import math
from typing import Callable

import numpy as np
import pytest

import pixeltable as pxt
from pixeltable import env, exprs, functions as pxtf

pytestmark = pytest.mark.local('UDF/integration test')


class TestMath:
    TEST_FLOATS = (0.0, 1.6, -19.27469, 1.32e57, math.inf, -math.inf, math.nan)
    TEST_INTS = (0, 1, -19, 4171780)

    @pytest.mark.parametrize('method_type', [pxt.Float, pxt.Int], ids=['Float', 'Int'])
    def test_methods(self, method_type: type, uses_db: None) -> None:
        t = pxt.create_table('test_tbl', {'x': method_type})
        values = sorted(self.TEST_FLOATS if method_type is pxt.Float else self.TEST_INTS)
        t.insert({'x': x} for x in values)
        r = t.order_by(t.x).collect()
        assert np.array_equal(r['x'], np.array(values), equal_nan=True), r

        test_params: list[tuple[pxt.Function, Callable, list, dict]]

        if method_type == pxt.Float:
            test_params = [
                (pxtf.math.abs, builtins.abs, [], {}),
                (pxtf.math.ceil, lambda x: float(math.ceil(x)) if math.isfinite(x) else x, [], {}),
                (pxtf.math.floor, lambda x: float(math.floor(x)) if math.isfinite(x) else x, [], {}),
                (pxtf.math.round, builtins.round, [0], {}),
                (pxtf.math.round, builtins.round, [2], {}),
                (pxtf.math.round, builtins.round, [4], {}),
                # round(x) without an explicit digits argument: behaves like round(x, 0)
                (pxtf.math.round, lambda x: builtins.round(x, 0), [], {}),
            ]
        else:
            test_params = [
                (pxtf.math.pow, builtins.pow, [2], {}),
                (pxtf.math.bitwise_and, lambda x, y: x & y, [2], {}),
                (pxtf.math.bitwise_or, lambda x, y: x | y, [2], {}),
                (pxtf.math.bitwise_xor, lambda x, y: x ^ y, [2], {}),
            ]

        for pxt_fn, py_fn, args, kwargs in test_params:
            print(f'Testing {pxt_fn.name} ...')
            actualdb = t.select(out=pxt_fn(t.x, *args, **kwargs)).order_by(t.x).collect()['out']
            expected = [py_fn(x, *args, **kwargs) for x in values]
            if (
                env.Env.get().is_using_cockroachdb
                and method_type == pxt.Float
                and pxt_fn == pxtf.math.round
                and (len(args) == 1)
            ):
                # cockroachdb does not support values of +-Infinity for NUMERIC / DECIMAL types
                # This means that our implementation of round(x, d) returns NaN if x is +-inf
                expecteddb = [math.nan if (math.isinf(x)) else y for x, y in zip(values, expected)]
            else:
                expecteddb = expected
            print(f'  values:   {values}')
            print(f'  actualdb:   {actualdb}')
            print(f'  expected: {expected}')
            print(f'  expecteddb: {expecteddb}')
            assert np.array_equal(actualdb, expecteddb, equal_nan=True), f'{actualdb} != {expecteddb}'
            # Run the same query, forcing the calculations to be done in Python (not SQL)
            # by interposing a non-SQLizable identity function
            actual_py = (
                t.select(out=pxt_fn(t.x.apply(lambda x: x, col_type=method_type), *args, **kwargs))
                .order_by(t.x)
                .collect()['out']
            )
            print(f'  actualpy: {actual_py}')
            assert np.array_equal(actual_py, expected, equal_nan=True), f'{actual_py} != {expected}'

        # Check that they can all be called with method syntax too
        for pxt_fn, _, _, _ in test_params:
            mref = getattr(t.x, pxt_fn.name)
            if isinstance(mref, exprs.MethodRef):
                # method
                assert mref.method_name == pxt_fn.name, pxt_fn
            elif isinstance(mref, exprs.FunctionCall):
                # property
                assert mref.fn.name == pxt_fn.name, pxt_fn
            else:
                raise AssertionError()

    def test_log_fns(self, uses_db: None) -> None:
        # tested separately from test_methods: these are only defined for positive (finite) inputs
        t = pxt.create_table('test_tbl', {'x': pxt.Float})
        values = [0.25, 1.0, 2.5, 100.0]
        t.insert({'x': x} for x in values)

        test_params: list[tuple[pxt.Function, Callable]] = [
            (pxtf.math.sqrt, math.sqrt),
            (pxtf.math.exp, math.exp),
            (pxtf.math.log, math.log),
            (pxtf.math.log10, math.log10),
        ]
        for pxt_fn, py_fn in test_params:
            expected = [py_fn(x) for x in values]
            actual = t.select(out=pxt_fn(t.x)).order_by(t.x).collect()['out']
            assert np.allclose(actual, expected), pxt_fn
            # Run the same query, forcing the calculations to be done in Python (not SQL)
            # by interposing a non-SQLizable identity function
            actual_py = t.select(out=pxt_fn(t.x.apply(lambda x: x, col_type=pxt.Float))).order_by(t.x).collect()['out']
            assert np.allclose(actual_py, expected), pxt_fn
            # method syntax
            mref = getattr(t.x, pxt_fn.name)
            assert isinstance(mref, exprs.MethodRef)
            assert mref.method_name == pxt_fn.name, pxt_fn

    def test_pow(self, uses_db: None) -> None:
        t = pxt.create_table('test_tbl', {'i': pxt.Int, 'f': pxt.Float})
        t.insert([{'i': 2, 'f': 1.5}, {'i': 5, 'f': 0.25}])

        # pow takes float parameters; int arguments are accepted in both call forms, and method syntax on an
        # int column resolves via the Int-to-Float method lookup
        res = (
            t.select(ff=pxtf.math.pow(t.f, 2), fi=t.f.pow(t.i), ii=t.i.pow(2), int_sqrt=t.i.sqrt())
            .order_by(t.i)
            .collect()
        )
        assert np.allclose(res['ff'], [1.5**2, 0.25**2])
        assert np.allclose(res['fi'], [1.5**2, 0.25**5])
        assert np.allclose(res['ii'], [4.0, 25.0])
        assert np.allclose(res['int_sqrt'], [math.sqrt(2), math.sqrt(5)])
