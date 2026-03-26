"""Tests for the expression evaluation engine (pixeltable/exec/expr_eval/)."""

import pixeltable as pxt


@pxt.udf
def add_one(x: int) -> int:
    return x + 1


@pxt.udf
def add_two(x: int, y: int) -> int:
    return x + y


def test_gc_bug_leaked_slot(uses_db: None) -> None:
    """Reproduce the GC bug where has_val doesn't distinguish 'not computed' from 'already GC'd'.

    Graph:
        x (from DB, slot 0)
        S = add_one(x)           -- gc target, depends on x
        T = add_one(S)           -- gc target, depends on S (fast branch)
        V = add_one(T)           -- output (fast branch finishes first)
        U = add_two(S, V)        -- gc target, depends on S AND V
        W = add_one(U)           -- output

    The DAG structure alone determines execution order (all scalar UDFs run
    synchronously on the event loop, so no timing tricks are needed):
    1. S computed -> x GC'd
    2. T computed -> S should be GC-able but has 2 deps (T, U). U not done -> S stays.
    3. V computed (output) -> T's only dep (V) done -> T GC'd. has_val[T]=False
    4. U can now start (needs S and V). S still has val.
       Bug: new_missing_dep[S] counts T (GC'd, has_val=False) as needing S -> S not GC'd!
    5. U computed -> W scheduled
    6. W computed -> row complete. S still has has_val=True -> ASSERTION FIRES
    """
    t = pxt.create_table('test_gc_leak', {'x': pxt.Int})
    t.insert({'x': i} for i in range(3))

    s = add_one(t.x)
    fast = add_one(s)  # T
    v_out = add_one(fast)  # V - output
    joined = add_two(s, v_out)  # U - depends on S and V
    w_out = add_one(joined)  # W - output

    result = t.select(v_out, w_out).collect()
    assert len(result) == 3
