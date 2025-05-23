import builtins
import math
from typing import Callable

import numpy as np
import pytest

import pixeltable as pxt
from pixeltable import exprs, functions as pxtf


class TestTimestamp:
    TEST_FLOATS = (0.0, 1.6, -19.274, 1.32e57, math.inf, -math.inf, math.nan)
    TEST_INTS = (0, 1, -19, 4171780)

    @pytest.mark.parametrize('method_type', [pxt.Float, pxt.Int])
    def test_methods(self, method_type: type, reset_db: None) -> None:
        t = pxt.create_table('test_tbl', {'x': method_type})
        values = self.TEST_FLOATS if method_type is pxt.Float else self.TEST_INTS
        t.insert({'x': x} for x in values)

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
            actual = t.select(out=pxt_fn(t.x, *args, **kwargs)).collect()['out']
            expected = [py_fn(x, *args, **kwargs) for x in values]
            assert np.array_equal(actual, expected, equal_nan=True), f'{actual} != {expected}'
            # Run the same query, forcing the calculations to be done in Python (not SQL)
            # by interposing a non-SQLizable identity function
            actual_py = t.select(out=pxt_fn(t.x.apply(lambda x: x, col_type=method_type), *args, **kwargs)).collect()[
                'out'
            ]
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
