import builtins
import functools
import math
from typing import Callable

import numpy as np
import pytest

import pixeltable as pxt
from pixeltable import exprs, functions as pxtf

pytestmark = pytest.mark.local('UDF/integration test')


class TestMath:
    TEST_FLOATS = (0.0, 0.25, 1.0, 1.6, -19.27469, 100.0, 1.32e57, math.inf, -math.inf, math.nan)
    TEST_INTS = (0, 1, 3, -19, 100, 4171780)

    @pytest.mark.parametrize('method_type', [pxt.Float, pxt.Int], ids=['Float', 'Int'])
    def test_methods(self, method_type: type, uses_db: None) -> None:
        t = pxt.create_table('test_tbl', {'x': method_type})
        values = sorted(self.TEST_FLOATS if method_type is pxt.Float else self.TEST_INTS)
        t.insert({'x': x} for x in values)
        r = t.order_by(t.x).collect()
        assert np.array_equal(r['x'], np.array(values), equal_nan=True), r

        # the bool field indicates functions that are only defined for positive inputs (sqrt/exp/log/log10)
        test_params: list[tuple[pxt.Function, Callable, bool, list, dict]]

        if method_type == pxt.Float:
            test_params = [
                (pxtf.math.abs, builtins.abs, False, [], {}),
                (pxtf.math.ceil, lambda x: float(math.ceil(x)) if math.isfinite(x) else x, False, [], {}),
                (pxtf.math.floor, lambda x: float(math.floor(x)) if math.isfinite(x) else x, False, [], {}),
                (pxtf.math.round, builtins.round, False, [0], {}),
                (pxtf.math.round, builtins.round, False, [2], {}),
                (pxtf.math.round, builtins.round, False, [4], {}),
                # round(x) without an explicit digits argument: behaves like round(x, 0)
                (pxtf.math.round, lambda x: builtins.round(x, 0), False, [], {}),
                (pxtf.math.sqrt, math.sqrt, True, [], {}),
                (pxtf.math.exp, math.exp, True, [], {}),
                (pxtf.math.log, math.log, True, [], {}),
                (pxtf.math.log10, math.log10, True, [], {}),
            ]
        else:
            test_params = [
                (pxtf.math.pow, builtins.pow, False, [2], {}),
                (pxtf.math.bitwise_and, lambda x, y: x & y, False, [2], {}),
                (pxtf.math.bitwise_or, lambda x, y: x | y, False, [2], {}),
                (pxtf.math.bitwise_xor, lambda x, y: x ^ y, False, [2], {}),
                (pxtf.math.sqrt, math.sqrt, True, [], {}),
                (pxtf.math.exp, math.exp, True, [], {}),
                (pxtf.math.log, math.log, True, [], {}),
                (pxtf.math.log10, math.log10, True, [], {}),
            ]

        for pxt_fn, py_fn, requires_positive, args, kwargs in test_params:
            print(f'Testing {pxt_fn.name} ...')
            matches: Callable[[list, list], bool]
            base: pxt.Table | pxt.Query
            if requires_positive:
                # strictly positive and bounded: some functions overflow on large inputs
                base = t.where((t.x > 0) & (t.x <= 100))
                fn_values = [x for x in values if 0 < x <= 100]
                # exp/log/log10 results aren't guaranteed bit-identical between SQL and Python
                matches = np.allclose
            else:
                base = t
                fn_values = values
                matches = functools.partial(np.array_equal, equal_nan=True)
            actualdb = base.select(out=pxt_fn(t.x, *args, **kwargs)).order_by(t.x).collect()['out']
            expected = [py_fn(x, *args, **kwargs) for x in fn_values]
            expecteddb = expected
            print(f'  values:   {fn_values}')
            print(f'  actualdb:   {actualdb}')
            print(f'  expected: {expected}')
            print(f'  expecteddb: {expecteddb}')
            assert matches(actualdb, expecteddb), f'{actualdb} != {expecteddb}'
            # Run the same query, forcing the calculations to be done in Python (not SQL)
            # by interposing a non-SQLizable identity function
            actual_py = (
                base.select(out=pxt_fn(t.x.apply(lambda x: x, col_type=method_type), *args, **kwargs))
                .order_by(t.x)
                .collect()['out']
            )
            print(f'  actualpy: {actual_py}')
            assert matches(actual_py, expected), f'{actual_py} != {expected}'

        # Check that they can all be called with method syntax too
        for pxt_fn, _, _, _, _ in test_params:
            mref = getattr(t.x, pxt_fn.name)
            if isinstance(mref, exprs.MethodRef):
                # method
                assert mref.method_name == pxt_fn.name, pxt_fn
            elif isinstance(mref, exprs.FunctionCall):
                # property
                assert mref.fn.name == pxt_fn.name, pxt_fn
            else:
                raise AssertionError()

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
