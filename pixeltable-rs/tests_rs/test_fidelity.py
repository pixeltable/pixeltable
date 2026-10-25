"""Observable behavior that must match upstream exactly: error messages, reprs, names."""

import pytest

import pixeltable as pxt


@pxt.udf
def div0(a: pxt.Json, b: int) -> float:
    return 1 / b


@pxt.uda
class init_exc(pxt.Aggregator):
    def __init__(self) -> None:
        self.sum = 1 / 0

    def update(self, val: int) -> None:
        pass

    def value(self) -> int:
        return 0


@pxt.uda
class update_exc(pxt.Aggregator):
    def __init__(self) -> None:
        self.sum = 0

    def update(self, val: int) -> None:
        self.sum += 1 // val

    def value(self) -> int:
        return self.sum


@pxt.uda
class value_exc(pxt.Aggregator):
    def __init__(self) -> None:
        self.sum = 0

    def update(self, val: int) -> None:
        self.sum += val

    def value(self) -> float:
        return 1 / (self.sum - self.sum)


def _tbl() -> pxt.Table:
    t = pxt.create_table('t', {'a': pxt.Int, 'j': pxt.Json})
    t.insert([{'a': i, 'j': {'f': i, 'g': [1, 2]}} for i in range(3)])
    return t


def _msg(f: object) -> str:
    with pytest.raises(pxt.Error) as exc_info:
        f()  # type: ignore[operator]
    return str(exc_info.value)


def test_eval_error_names_failing_node_and_inputs() -> None:
    """The message names the innermost failing expression and its inputs' values (upstream `ExprEvalError`)."""
    t = _tbl()
    expected = (
        'In row 0 the expression div0(a + 1, a) encountered exception ZeroDivisionError:\n'
        "division by zero\nwith 'a + 1' = 1, 'a' = 0"
    )
    assert _msg(lambda: t.select(t.a + div0(t.a + 1, t.a)).collect()) == expected
    assert _msg(lambda: t.where(div0(t.a + 1, t.a) > 0).collect()) == expected
    assert _msg(lambda: t.select(t.j.f / (t.a - 1)).collect()) == (
        "In row 0 the expression j.f / (a - 1) encountered exception ZeroDivisionError:\n"
        "division by zero\nwith 'j.f' = 1, 'a - 1' = 0"
    )
    assert _msg(lambda: t.select(div0(t.j, t.a)).collect()).endswith("with 'j' = {'f': 0, 'g': [1, 2]}, 'a' = 0")


def test_aggregate_errors() -> None:
    """Failures of a UDA's methods name the method, show the inputs and the user's stack."""
    t = _tbl()
    m = __name__
    msg = _msg(lambda: t.select(init_exc(t.a)).collect())
    assert msg.startswith(
        f'In row 0 the init() function of the aggregate {m}.init_exc(a) encountered exception ZeroDivisionError:\n'
        'division by zero\nStack:\n'
    )
    assert 'self.sum = 1 / 0' in msg
    msg = _msg(lambda: t.select(update_exc(t.a - 1)).collect())
    assert msg.startswith(
        f'In row 0 the update() function of the aggregate {m}.update_exc(a - 1) encountered exception '
        "ZeroDivisionError:\ninteger division or modulo by zero\nwith 'a - 1' = 0\nStack:\n"
    )
    # value() is evaluated in the group's last row
    msg = _msg(lambda: t.select(value_exc(t.a)).collect())
    assert msg.startswith(
        f"In row 0 the expression {m}.value_exc(a) encountered exception ZeroDivisionError:\ndivision by zero\n"
        "with 'a' = 2\nStack:\n"
    )


def test_aggregate_display_names() -> None:
    """Aggregates display with their path below `pixeltable.functions` (upstream `Function.display_name`)."""
    t = _tbl()
    assert str(pxt.functions.sum(t.a)) == 'globals.sum(a)'
    assert str(pxt.functions.json.make_list(t.j)) == 'json.make_list(j)'
    with pytest.raises(pxt.Error, match=r'Grouping expression contains aggregate function: globals\.count\(a\)$'):
        t.group_by(pxt.functions.count(t.a)).select(t.a).collect()
    assert t.select(pxt.functions.sum(t.a)).collect().schema == {'sum': 'Int | None'}
