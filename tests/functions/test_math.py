import builtins
import math
from typing import Callable

import numpy as np

import pixeltable as pxt
import pixeltable.functions as pxtf


class TestTimestamp:
    TEST_FLOATS = (0.0, 1.6, -19.274, 1.32e57, math.inf, -math.inf, math.nan)

    def test_methods(self, reset_db) -> None:
        t = pxt.create_table('test_tbl', {'x': pxt.Float})
        t.insert({'x': x} for x in self.TEST_FLOATS)

        test_params: list[tuple[pxt.Function, Callable, list, dict]] = [
            (pxtf.math.abs, builtins.abs, [], {}),
            (pxtf.math.ceil, lambda x: float(math.ceil(x)) if math.isfinite(x) else x, [], {}),
            (pxtf.math.floor, lambda x: float(math.floor(x)) if math.isfinite(x) else x, [], {}),
            (pxtf.math.round, builtins.round, [0], {}),
            (pxtf.math.round, builtins.round, [2], {}),
            (pxtf.math.round, builtins.round, [4], {}),
            # round(x) without an explicit digits argument: behaves like round(x, 0)
            (pxtf.math.round, lambda x: builtins.round(x, 0), [], {}),
        ]

        for pxt_fn, py_fn, args, kwargs in test_params:
            print(f'Testing {pxt_fn.name} ...')
            actual = t.select(out=pxt_fn(t.x, *args, **kwargs)).collect()['out']
            expected = [py_fn(x, *args, **kwargs) for x in self.TEST_FLOATS]
            assert np.array_equal(actual, expected, equal_nan=True), f'{actual} != {expected}'
            # Run the same query, forcing the calculations to be done in Python (not SQL)
            # by interposing a non-SQLizable identity function
            actual_py = t.select(out=pxt_fn(t.x.apply(lambda x: x, col_type=pxt.Float), *args, **kwargs)).collect()[
                'out'
            ]
            assert np.array_equal(actual_py, expected, equal_nan=True), f'{actual_py} != {expected}'

        # Check that they can all be called with method syntax too
        for pxt_fn, _, _, _ in test_params:
            mref = t.x.__getattr__(pxt_fn.name)
            if isinstance(mref, pxt.exprs.MethodRef):
                # method
                assert mref.method_name == pxt_fn.name, pxt_fn
            elif isinstance(mref, pxt.exprs.FunctionCall):
                # property
                assert mref.fn.name == pxt_fn.name, pxt_fn
            else:
                assert False
