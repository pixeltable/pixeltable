import builtins
import functools
import math
from typing import Any, Callable

import numpy as np
import pytest
import sqlalchemy.exc as sql_exc

import pixeltable as pxt
from pixeltable import exprs, functions as pxtf

pytestmark = pytest.mark.db_roots('local', reason='UDF/integration test')


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

        # each entry: function, Python reference, domain predicate (None: defined everywhere), whether SQL and Python
        # results are only approximately equal, args, kwargs; the predicate is applied to both the column expression
        # and the Python values, which works because & is defined on both Expr and bool
        test_params: list[tuple[pxt.Function, Callable, Callable | None, bool, list, dict]]

        def positive(x: Any) -> Any:
            # bounded: some functions overflow on large inputs
            return (x > 0) & (x <= 100)

        def unit(x: Any) -> Any:
            return (x >= -1) & (x <= 1)

        def finite(x: Any) -> Any:
            # sin/cos/tan raise an error for infinity; Postgres can't be given an infinite bound here
            return (x > -1e100) & (x < 1e100)

        if method_type == pxt.Float:
            test_params = [
                (pxtf.math.abs, builtins.abs, None, False, [], {}),
                (pxtf.math.ceil, lambda x: float(math.ceil(x)) if math.isfinite(x) else x, None, False, [], {}),
                (pxtf.math.floor, lambda x: float(math.floor(x)) if math.isfinite(x) else x, None, False, [], {}),
                (pxtf.math.round, builtins.round, None, False, [0], {}),
                (pxtf.math.round, builtins.round, None, False, [2], {}),
                (pxtf.math.round, builtins.round, None, False, [4], {}),
                # round(x) without an explicit digits argument: behaves like round(x, 0)
                (pxtf.math.round, lambda x: builtins.round(x, 0), None, False, [], {}),
                (pxtf.math.sqrt, math.sqrt, positive, True, [], {}),
                (pxtf.math.exp, math.exp, positive, True, [], {}),
                (pxtf.math.log, math.log, positive, True, [], {}),
                (pxtf.math.log10, math.log10, positive, True, [], {}),
                (pxtf.math.sin, math.sin, finite, True, [], {}),
                (pxtf.math.cos, math.cos, finite, True, [], {}),
                (pxtf.math.tan, math.tan, finite, True, [], {}),
                (pxtf.math.asin, math.asin, unit, True, [], {}),
                (pxtf.math.acos, math.acos, unit, True, [], {}),
                (pxtf.math.atan, math.atan, None, True, [], {}),
                (pxtf.math.degrees, math.degrees, None, True, [], {}),
                (pxtf.math.radians, math.radians, None, True, [], {}),
                (pxtf.math.sign, np.sign, None, True, [], {}),
            ]
        else:
            test_params = [
                (pxtf.math.pow, builtins.pow, None, False, [2], {}),
                (pxtf.math.bitwise_and, lambda x, y: x & y, None, False, [2], {}),
                (pxtf.math.bitwise_or, lambda x, y: x | y, None, False, [2], {}),
                (pxtf.math.bitwise_xor, lambda x, y: x ^ y, None, False, [2], {}),
                (pxtf.math.sqrt, math.sqrt, positive, True, [], {}),
                (pxtf.math.exp, math.exp, positive, True, [], {}),
                (pxtf.math.log, math.log, positive, True, [], {}),
                (pxtf.math.log10, math.log10, positive, True, [], {}),
                (pxtf.math.sin, math.sin, finite, True, [], {}),
                (pxtf.math.cos, math.cos, finite, True, [], {}),
                (pxtf.math.tan, math.tan, finite, True, [], {}),
                (pxtf.math.asin, math.asin, unit, True, [], {}),
                (pxtf.math.acos, math.acos, unit, True, [], {}),
                (pxtf.math.atan, math.atan, None, True, [], {}),
                (pxtf.math.degrees, math.degrees, None, True, [], {}),
                (pxtf.math.radians, math.radians, None, True, [], {}),
                (pxtf.math.sign, np.sign, None, True, [], {}),
            ]

        for pxt_fn, py_fn, domain, approx, args, kwargs in test_params:
            print(f'Testing {pxt_fn.name} ...')
            matches: Callable[[list, list], bool]
            base: pxt.Table | pxt.Query
            if domain is not None:
                base = t.where(domain(t.x))
                fn_values = [x for x in values if domain(x)]
            else:
                base = t
                fn_values = values
            if approx:
                # results aren't guaranteed to be bit-identical between SQL and Python
                matches = functools.partial(np.allclose, equal_nan=True)
            else:
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
        for pxt_fn, _, _, _, _, _ in test_params:
            mref = getattr(t.x, pxt_fn.name)
            if isinstance(mref, exprs.MethodRef):
                # method
                assert mref.method_name == pxt_fn.name, pxt_fn
            elif isinstance(mref, exprs.FunctionCall):
                # property
                assert mref.fn.name == pxt_fn.name, pxt_fn
            else:
                raise AssertionError()

    def test_trig_and_sign(self, uses_db: None) -> None:
        """Domain errors, and sign(), trig and conversion functions in where(), update() and delete() filters."""
        t = pxt.create_table('test_tbl', {'id': pxt.Int, 'x': pxt.Float})
        t.insert({'id': i, 'x': x} for i, x in enumerate([-2.0, -0.5, 0.0, 0.5, 2.0]))

        # sign(), trigonometric and conversion functions run in Postgres in where(), update() and delete()
        assert t.where(t.x.sign() < 0).select(t.id).order_by(t.id).collect()['id'] == [0, 1]
        assert t.where(t.x.sign() == 0).select(t.id).collect()['id'] == [2]
        assert t.where(t.x.radians().cos() > 0.99999).select(t.id).collect()['id'] == [2]
        status = t.update({'x': 10.0}, where=t.x.radians().cos() > 0.99999)
        assert status.num_rows == 1
        assert t.where(t.id == 2).select(t.x).collect()['x'] == [10.0]
        assert t.delete(where=t.x.sign() < 0).num_rows == 2
        assert t.delete(where=t.x.degrees().atan() > 0).num_rows == 3
        assert t.count() == 0

        # out-of-domain inputs are errors, not silent NaNs, in both SQL and Python
        t = pxt.create_table('errors_tbl', {'id': pxt.Int, 'x': pxt.Float})
        t.insert([{'id': 0, 'x': 2.0}, {'id': 1, 'x': math.inf}])
        # in Postgres, the error surfaces as a database error
        with pytest.raises(sql_exc.DataError, match='out of range'):
            _ = t.where(t.id == 0).select(out=t.x.asin()).collect()
        with pytest.raises(sql_exc.DataError, match='out of range'):
            _ = t.where(t.id == 1).select(out=t.x.sin()).collect()
        # a non-SQLizable identity function forces evaluation in Python
        ident = t.x.apply(lambda x: x, col_type=pxt.Float)
        # the wording of the ValueError message differs across Python versions
        with pytest.raises(pxt.Error, match='ValueError'):
            _ = t.where(t.id == 0).select(out=ident.asin()).collect()
        with pytest.raises(pxt.Error, match='ValueError'):
            _ = t.where(t.id == 1).select(out=ident.sin()).collect()

        # computed columns are evaluated in Python on insert
        t = pxt.create_table('computed_tbl', {'id': pxt.Int, 'x': pxt.Float})
        t.add_computed_column(asin=t.x.asin())
        t.add_computed_column(sin=t.x.sin())
        status = t.insert([{'id': 0, 'x': 2.0}, {'id': 1, 'x': math.inf}], on_error='ignore')
        assert status.num_excs == 3
        res = t.select(t.asin, t.sin, asin_err=t.asin.errortype, sin_err=t.sin.errortype).order_by(t.id).collect()
        assert res['asin'] == [None, None]
        assert res['sin'] == [math.sin(2.0), None]
        assert res['asin_err'][0] == 'ValueError'
        assert res['sin_err'][1] == 'ValueError'

    def test_to_int(self, uses_db: None) -> None:
        """to_int() truncates toward zero and returns the same values from Python and from Postgres."""
        floats = [0.0, 2.5, 3.5, 2.7, -2.7, 3e9, 9.2e18]
        ints = [0, 5, -5, 2**31, 2**53 + 1, 2**62 + 1, -(2**62) - 1]
        bools = [i == 0 for i in ints]
        t = pxt.create_table('test_tbl', {'id': pxt.Int, 'x': pxt.Float, 'i': pxt.Int, 'b': pxt.Bool})
        t.add_computed_column(x_before=t.x.to_int())
        t.add_computed_column(i_before=pxtf.math.to_int(t.i))
        t.insert({'id': row_id, 'x': x, 'i': i, 'b': b} for row_id, (x, i, b) in enumerate(zip(floats, ints, bools)))
        t.add_computed_column(x_after=t.x.to_int())
        t.add_computed_column(i_after=pxtf.math.to_int(t.i))
        assert t.x.to_int().col_type.is_int_type()

        res = (
            t.select(
                t.x_before,
                t.x_after,
                t.i_before,
                t.i_after,
                x_sql=t.x.to_int(),
                i_sql=pxtf.math.to_int(t.i),
                # a non-SQLizable identity function forces evaluation in Python
                x_py=t.x.apply(lambda x: x, col_type=pxt.Float).to_int(),
                i_py=pxtf.math.to_int(t.i.apply(lambda i: i, col_type=pxt.Int)),
                i_abs=pxtf.math.to_int(t.i.abs()),
                b_col=pxtf.math.to_int(t.b),
                b_pred=pxtf.math.to_int(t.i == 0),
            )
            .order_by(t.id)
            .collect()
        )
        for col in ('x_before', 'x_after', 'x_sql', 'x_py'):
            assert res[col] == [int(x) for x in floats], col
            assert all(type(val) is int for val in res[col]), col
        for col in ('i_before', 'i_after', 'i_sql', 'i_py'):
            assert res[col] == ints, col
        assert res['i_abs'] == [abs(i) for i in ints]
        assert res['b_col'] == res['b_pred'] == [int(b) for b in bools]

        assert t.where(t.x.to_int() == 2).select(t.x).order_by(t.id).collect()['x'] == [2.5, 2.7]

        # the other rounding modes
        res = (
            t.where((t.x == 2.7) | (t.x == -2.7))
            .select(floor=t.x.floor().to_int(), ceil=t.x.ceil().to_int(), round=t.x.round().to_int())
            .order_by(t.id)
            .collect()
        )
        assert res['floor'] == [2, -3]
        assert res['ceil'] == [3, -2]
        assert res['round'] == [3, -3]

        # update() and delete() accept only a filter that runs in Postgres
        status = t.update({'x': 1.5}, where=pxtf.math.to_int(t.b) == 1)
        assert status.num_rows == 1
        assert t.where(t.id == 0).select(t.x, t.x_before, t.x_after).collect()[0] == {
            'x': 1.5,
            'x_before': 1,
            'x_after': 1,
        }
        filters = [
            (pxtf.math.to_int(t.i.abs()) == 2**62 + 1, 2),
            ((t.x // 2).to_int() == -2, 1),
            (t.x.pow(2).to_int() == 7, 1),
            (t.x.abs().to_int() == 2, 1),
            (pxtf.math.to_int(t.i // 2) == -3, 1),
            ((t.x**2).to_int() == 2, 1),
        ]
        for where, num_rows in filters:
            assert t.delete(where=where).num_rows == num_rows, where
        assert t.count() == 0

        t = pxt.create_table('json_tbl', {'id': pxt.Int, 'j': pxt.Json})
        t.insert({'id': row_id, 'j': j} for row_id, j in enumerate([[1.0, 2.0], [0.5, 0.25], [2.5, 5.0], [7.5, 8.5]]))
        filters = [
            (pxtf.math.to_int(pxtf.json.sum(t.j)) == 3, 0),
            (pxtf.math.to_int(pxtf.json.max(t.j)) == 5, 2),
            (pxtf.math.to_int(pxtf.json.min(t.j)) == 7, 3),
            (pxtf.math.to_int(pxtf.json.mean(t.j)) == 0, 1),
        ]
        for where, row_id in filters:
            assert t.where(where).select(t.id).collect()['id'] == [row_id], where
            assert t.delete(where=where).num_rows == 1, where
        assert t.count() == 0

        # values without an Int equivalent
        t = pxt.create_table('errors_tbl', {'id': pxt.Int, 'x': pxt.Float})
        t.add_computed_column(out=t.x.to_int())
        status = t.insert(
            ({'id': row_id, 'x': x} for row_id, x in enumerate([math.inf, math.nan, 1e19, -1e19])), on_error='ignore'
        )
        assert status.num_excs == 4
        res = t.select(t.out, errormsg=t.out.errormsg).order_by(t.id).collect()
        assert res['out'] == [None] * 4
        expected_msgs = [
            'cannot convert float infinity to integer',
            'cannot convert float NaN to integer',
            '1e+19 is outside the range of Int',
            '-1e+19 is outside the range of Int',
        ]
        assert all(expected in msg for expected, msg in zip(expected_msgs, res['errormsg'])), res['errormsg']

    def test_pow(self, uses_db: None) -> None:
        t = pxt.create_table('test_tbl', {'i': pxt.Int | None, 'f': pxt.Float | None})
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
