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


@pxt.udf(resource_pool='request-rate:test')
def sync_with_pool(s: str) -> str:
    return s


@pxt.udf
def overloaded(a: int, z: str) -> int:
    return a


@overloaded.overload
def _(a: str, z: str) -> str:
    return a


def test_setup_errors_are_not_row_errors() -> None:
    """A resource pool on a sync function fails the operation as it is, also with on_error='ignore' (upstream raises
    it when it sets up the call's evaluation)."""
    t = pxt.create_table('t', {'s': pxt.String | None})
    t.add_computed_column(r=sync_with_pool(t.s))
    for on_error in ('abort', 'ignore'):
        with pytest.raises(pxt.Error) as exc_info:
            t.insert([{'s': 'a'}], on_error=on_error)  # type: ignore[arg-type]
        assert str(exc_info.value) == 'sync_with_pool: resource_pool requires an async function'
        assert exc_info.value.error_code == pxt.ErrorCode.INVALID_CONFIGURATION


def test_misc_messages() -> None:
    t = pxt.create_table('t', {'s': pxt.String | None, 'i': pxt.Int | None})
    # a function made by using() has no name
    assert _msg(lambda: overloaded.using(z='b')(1.5)) == 'Function None has no matching signature for arguments'
    # the base table is named
    v = pxt.create_view('v', t)
    assert _msg(lambda: pxt.create_view('v2', v, additional_columns={'s': pxt.Int})) == (
        "Column 's' already exists in the base table 'v'."
    )
    # names are validated as given
    assert _msg(lambda: t.add_columns({'Count': pxt.Int})) == (
        "'Count' is a reserved name in Pixeltable; please choose a different column name."
    )
    assert _msg(lambda: t.add_computed_column(w=pxt.functions.sum(t.i, group_by=t.s), stored=False)) == (
        "Column 'w': `stored=False` is not valid for image columns computed with a streaming function"
    )
    t.add_column(d=pxt.Int | None)
    q = t.select(t.d)
    t.drop_column('d')
    assert _msg(lambda: q.collect()) == (
        "Column was dropped (no record for column ID 2 in table 't'; it was dropped in table version 2)"
    )
    # a string key column of a data-versioned table is indexed by its first 256 characters
    pk = pxt.create_table('pk', {'a': pxt.Int, 'b': pxt.String}, primary_key=['a', 'b'])
    pk.insert([{'a': 1, 'b': 'x' * 300}])
    assert _msg(lambda: pk.insert([{'a': 1, 'b': 'x' * 300}])) == (
        f'Duplicate primary key value: Key (a, "left"(b::text, 256))=(1, {"x" * 256}) already exists.'
    )


def test_media_validation_error_is_raised_as_is(tmp_path: object) -> None:
    import pathlib

    bad = pathlib.Path(str(tmp_path)) / 'bad.jpg'
    bad.write_bytes(b'not an image')
    t = pxt.create_table('t', {'img': pxt.Image | None})
    assert _msg(lambda: t.insert(img=str(bad))) == f'Not a valid image: {bad}'


def test_query_udf_invalid_after_drop() -> None:
    """A stored query function whose query reads a dropped column is invalid once the catalog reloads."""
    import warnings

    import pixeltable._testing

    src = pxt.create_table('src', {'id': pxt.Int, 'extra': pxt.Int | None})
    src.insert([{'id': 1, 'extra': 1}])

    @pxt.query
    def q_col(lower: int) -> pxt.Query:
        return src.where(src.id >= lower).select(src.extra)

    @pxt.query
    def q_star(n: int) -> pxt.Query:
        return src.select().limit(n)

    host = pxt.create_table('host', {'threshold': pxt.Int})
    host.add_computed_column(rows_col=q_col(host.threshold))
    host.add_computed_column(rows_star=q_star(5))
    src.drop_column('extra')
    pixeltable._testing.reload_catalog()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        host = pxt.get_table('host')
    msgs = [str(x.message) for x in w]
    assert any(
        "The UDF `q_col` cannot be located, because\nthe @pxt.query UDF 'q_col' cannot be loaded: Column was dropped "
        "(no record for column ID 1 in table 'src')" in m
        for m in msgs
    ), msgs
    assert any('The return type stored in the database for a UDF call to `None` no longer' in m for m in msgs), msgs
    with pytest.raises(pxt.Error, match="because the column 'rows_col' is currently invalid"):
        host.insert([{'threshold': 30}])
