"""End-to-end tests of calls of user Python code: UDFs, templates, query templates, iterators (see ARCHITECTURE.md,
"Python functions")."""

import glob
import os
import warnings
from typing import Any, Iterator, TypedDict

import numpy as np
import pytest

import pixeltable as pxt
from pixeltable import _core, exceptions as excs, func
from pixeltable._testing import reload_catalog

VIDEOS = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'data', 'videos')


def raises(code: pxt.ErrorCode, match: str | None = None) -> pytest.RaisesExc:
    return pytest.raises(excs._error_class(code), match=match)


@pxt.udf
def add(a: int, b: float, c: float = 0.0) -> float:
    return a + b + c


@pxt.udf
def nullable_add(a: int | None, b: float = 0.0) -> float:
    return (0 if a is None else a) + b


@pxt.udf
def inverse(x: int) -> float:
    return 1 / x


@pxt.udf
def overloaded(x: str) -> str:
    return x + x


@overloaded.overload
def _(x: int) -> int:
    return x * 2


@pxt.udf(batch_size=4)
def batched(xs: func.Batch[int]) -> func.Batch[int]:
    return [x + 100 for x in xs]


@pxt.udf
async def async_double(x: int) -> int:
    return x * 2


@pxt.udf(is_method=True)
def triple(x: int) -> int:
    return x * 3


@pxt.expr_udf
def plus_one(x: int) -> int:
    return x + 1


@pxt.uda
class total(pxt.Aggregator):
    def __init__(self, start: int = 0) -> None:
        self.sum = start

    def update(self, x: int) -> None:
        self.sum += x

    def value(self) -> int:
        return self.sum


evolving: Any = None


class Row(TypedDict):
    n: int
    s: str


@pxt.iterator
def count_to(x: int, prefix: str = 'p') -> Iterator[Row]:
    for i in range(x):
        yield Row(n=i, s=f'{prefix}{i}')


@pytest.fixture
def t() -> pxt.Table:
    t = pxt.create_table('t', {'i': pxt.Int | None, 'f': pxt.Float, 's': pxt.String | None})
    t.insert([{'i': i, 'f': i + 0.5, 's': str(i)} for i in range(4)] + [{'i': None, 'f': 1.0, 's': None}])
    return t


class TestUdfCalls:
    def test_binding_and_nulls(self, t: pxt.Table) -> None:
        res = t.select(r=add(t.i, t.f), k=add(c=1.0, b=t.f, a=t.i), n=nullable_add(t.i, b=t.f)).collect()
        assert res['r'] == [0.5, 2.5, 4.5, 6.5, None]  # NULL for a non-nullable parameter short-circuits
        assert res['k'] == [1.5, 3.5, 5.5, 7.5, None]
        assert res['n'] == [0.5, 2.5, 4.5, 6.5, 1.0]
        assert repr(add(t.i, t.f).col_type) == 'Float | None'
        with raises(pxt.ErrorCode.INVALID_ARGUMENT, match=r"missing a required argument: 'b'; expected \(a: pxt\.Int"):
            add(t.i)
        with raises(pxt.ErrorCode.TYPE_MISMATCH, match="argument type String | None does not match"):
            add(t.s, t.f)

    def test_overloads_batches_async_methods_templates(self, t: pxt.Table) -> None:
        res = t.where(t.i != None).select(o1=overloaded(t.s), o2=overloaded(t.i), b=batched(t.i), d=async_double(t.i)).collect()
        assert res['o1'] == ['00', '11', '22', '33'] and res['o2'] == [0, 2, 4, 6]
        assert res['b'] == [100, 101, 102, 103] and res['d'] == [0, 2, 4, 6]
        assert t.where(t.i == 2).select(m=t.i.triple(), p=plus_one(t.i)).collect()[0] == {'m': 6, 'p': 3}
        pb = add.using(c=10.0)
        assert pb.arity == 2
        assert t.where(t.i == 1).select(r=pb(t.i, t.f)).collect()[0]['r'] == 12.5

    def test_cell_errors_and_on_error(self, t: pxt.Table) -> None:
        with raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='division by zero'):
            t.add_computed_column(inv=inverse(t.i))
        status = t.add_computed_column(inv=inverse(t.i), on_error='ignore')
        assert status.num_excs == 1
        errs = t.where(t.inv.errortype != None).select(t.inv.errortype, t.inv.errormsg).collect()
        assert list(errs) == [{'inv_errortype': 'ZeroDivisionError', 'inv_errormsg': 'division by zero'}]

    def test_storable_and_reload(self, t: pxt.Table) -> None:
        @pxt.udf(_force_stored=True)
        def local(x: int) -> int:
            return x

        with raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='defined as a local function'):
            t.add_computed_column(bad=local(t.i))
        t.add_computed_column(r=add(t.i, t.f))
        reload_catalog()
        t = pxt.get_table('t')
        t.insert(i=10, f=1.0)
        assert t.where(t.i == 10).select(t.r).collect()[0]['r'] == 11.0
        q = t.select(x=local(t.i))  # local functions round-trip by value in serialized queries
        assert pxt.Query.from_dict(q.as_dict()).collect()['x'] == q.collect()['x']

    def test_evolution(self, t: pxt.Table) -> None:
        import test_functions

        @pxt.udf(_force_stored=True)
        def v1(a: int) -> int:
            return a

        @pxt.udf(_force_stored=True)
        def v2(a: str) -> int:
            return 0

        def mimic(fn: func.CallableFunction) -> None:
            test_functions.evolving = func.CallableFunction(fn.signatures, fn.py_fns, 'test_functions.evolving')

        mimic(v1)
        t.add_computed_column(e=test_functions.evolving(t.i))
        mimic(v2)
        reload_catalog()
        with pytest.warns(pxt.PixeltableWarning, match="The computed column 'e' in table 't' is no longer valid"):
            t = pxt.get_table('t')
        with warnings.catch_warnings():
            warnings.simplefilter('error', pxt.PixeltableWarning)
            t = pxt.get_table('t')  # warned once per catalog load
        assert len(t.collect()) == 5  # stored values stay readable
        with raises(pxt.ErrorCode.INVALID_STATE, match="Data cannot be inserted into the Table 't',\nbecause the column 'e'"):
            t.insert(i=1, f=1.0)
        test_functions.evolving = None

    def test_query_template(self, t: pxt.Table) -> None:
        @pxt.query
        def below(x: int) -> pxt.Query:
            return t.where(t.i < x).select(t.i).order_by(t.i)

        res = t.where(t.i != None).order_by(t.i).select(r=below(t.i)).collect()['r']
        assert res == [[], [{'i': 0}], [{'i': 0}, {'i': 1}], [{'i': 0}, {'i': 1}, {'i': 2}]]

        @pxt.query(return_scalar=True)
        def first(n: int, skip: int = 0) -> pxt.Query:
            return t.where(t.i != None).select(t.i).order_by(t.i).limit(n, offset=skip)

        assert t.where(t.i == 0).select(r=first(2, skip=1)).collect()[0]['r'] == [1, 2]
        assert t.where(t.i == 0).select(r=first(0)).collect()[0]['r'] == []
        with raises(pxt.ErrorCode.INVALID_ARGUMENT, match="'limit' parameter must be >= 0"):
            t.select(r=first(-1)).collect()
        # adding a computed column wraps a row's failure (upstream `write_column()`)
        with raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match="computed column 'c':\n'offset' parameter must be >= 0"):
            t.add_computed_column(c=first(1, skip=-1))
        with raises(pxt.ErrorCode.INVALID_ARGUMENT, match='limit\\(\\): parameter must be an int constant'):

            @pxt.query
            def bad(n: int) -> pxt.Query:
                return t.select(t.i).limit(n + 1)

    def test_uda(self, t: pxt.Table) -> None:
        assert t.select(s=total(t.i)).collect()[0]['s'] == 6  # NULL for a non-nullable parameter is skipped
        res = t.where(t.i != None).group_by(t.i % 2).select(k=t.i % 2, s=total(t.i, start=10)).order_by(t.i % 2)
        assert list(res.collect()['s']) == [12, 14]
        reload_catalog()
        t = pxt.get_table('t')
        q = t.select(s=total(t.i))  # stored by reference to the class, resolved again after a reload
        assert pxt.Query.from_dict(q.as_dict()).collect()[0]['s'] == 6


class TestIterators:
    def test_call_rows_and_seek(self, t: pxt.Table) -> None:
        call = count_to(t.i, prefix='x')
        assert {k: repr(v) for k, v in call.outputs.items()} == {'pos': 'Int', 'n': 'Int', 's': 'String'}
        assert _core._iterator_rows(call, {'i': 2}) == [{'pos': 0, 'n': 0, 's': 'x0'}, {'pos': 1, 'n': 1, 's': 'x1'}]
        assert _core._iterator_rows(call, {'i': None}) == []  # NULL for a non-nullable parameter: no rows
        with raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match="Invalid iterator arguments: got an unexpected keyword"):
            count_to(t.i, y=1)
        d = call._core.as_dict()
        assert d['fn'] == {'fqn': 'test_functions.count_to'} and list(d['outputs']) == ['pos', 'n', 's']

    def test_frame_iterator(self) -> None:
        videos = sorted(glob.glob(os.path.join(VIDEOS, 'bangkok*.mp4')))
        if len(videos) == 0:
            pytest.skip('no test videos')
        from pixeltable.functions.video import frame_iterator

        v = pxt.create_table('v', {'video': pxt.Video})
        call = frame_iterator(v.video, fps=1)
        rows = _core._iterator_rows(call, {'video': videos[0]})
        assert len(rows) > 0 and rows[0]['frame_attrs']['index'] == 0
        pos = len(rows) // 2
        again = _core._iterator_seek(call, {'video': videos[0]}, pos, {'frame_attrs': rows[pos]['frame_attrs']})
        assert np.array_equal(np.asarray(again['frame']), np.asarray(rows[pos]['frame']))
