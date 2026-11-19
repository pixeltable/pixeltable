// The /rpc encoding of expressions. Each expected dict is what Python's Expr.as_dict() produces for the same
// expression, minus the FunctionCall fields the daemon recomputes (signatures, return_type).

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { count } from '../src/functions/globals.ts';
import { string } from '../src/functions/index.ts';
import * as pxt from '../src/index.ts';
import { layout } from '../src/model.ts';
import { encode, type PyDict, type Scope } from '../src/pyexpr.ts';
import { decodeValue, encodeValue } from '../src/rpc.ts';

const TableModel = pxt.modelBase();

class Films extends TableModel('films') {
  title = pxt.string().orNull();
  year = pxt.int();
  budget = pxt.float();
  released = pxt.timestamp();
  meta = pxt.json();
  title_len = this.title.len();
}

const COL_IDS: Record<string, number> = { title: 0, year: 1, budget: 2, released: 3, meta: 4, title_len: 5 };

function ref(name: string): PyDict {
  return {
    _classname: 'ColumnRef',
    tbl_id: 't1',
    effective_version: null,
    col_tbl_id: 't1',
    col_tbl_effective_version: null,
    col_id: COL_IDS[name],
    perform_validation: false,
  };
}

const l = layout(Films);
const f = l.instance as Films;
const scope: Scope = { columnName: (e) => l.names.get(e), columnRef: ref };
const int = (val: number) => ({ _classname: 'Literal', val, col_type: { _classname: 'IntType', nullable: false } });
const str = (val: string) => ({ _classname: 'Literal', val, col_type: { _classname: 'StringType', nullable: false } });

function call(path: string, components: unknown[], options: { method?: boolean; aggregate?: boolean } = {}) {
  return {
    _classname: 'FunctionCall',
    fn: {
      _classpath: options.aggregate
        ? 'pixeltable.func.aggregate_function.AggregateFunction'
        : 'pixeltable.func.callable_function.CallableFunction',
      path,
    },
    return_type: null,
    arg_idxs: components.map((_, i) => i),
    kwarg_idxs: {},
    group_by_start_idx: 0,
    group_by_stop_idx: 0,
    order_by_start_idx: components.length,
    is_method_call: options.method ?? false,
    components,
  };
}

test('a computed column is referenced by its column id, not by its expression', () => {
  assert.deepEqual(encode(f.title_len.gte(5), scope), {
    _classname: 'Comparison',
    operator: 5,
    components: [ref('title_len'), int(5)],
  });
});

test('operators', () => {
  assert.deepEqual(encode(f.year.gt(1990).and(f.title_len.gte(5)), scope), {
    _classname: 'CompoundPredicate',
    operator: 0,
    components: [
      { _classname: 'Comparison', operator: 4, components: [ref('year'), int(1990)] },
      { _classname: 'Comparison', operator: 5, components: [ref('title_len'), int(5)] },
    ],
  });
  assert.deepEqual(encode(f.year.add(1), scope), {
    _classname: 'ArithmeticExpr',
    operator: 0,
    components: [ref('year'), int(1)],
  });
  assert.deepEqual(encode(f.year.neg(), scope), {
    _classname: 'ArithmeticExpr',
    operator: 2,
    components: [ref('year'), int(-1)],
  });
  assert.deepEqual(encode(f.title.concat('x'), scope), {
    _classname: 'StringOp',
    operator: 0,
    components: [ref('title'), str('x')],
  });
  assert.deepEqual(encode(f.title.isNotNull(), scope), {
    _classname: 'CompoundPredicate',
    operator: 2,
    components: [{ _classname: 'IsNull', components: [ref('title')] }],
  });
  assert.deepEqual(encode(f.title.eq(null), scope), { _classname: 'IsNull', components: [ref('title')] });
  assert.deepEqual(encode(f.year.isIn([1, 2]), scope), {
    _classname: 'InPredicate',
    value_list: [1, 2],
    components: [ref('year')],
  });
  assert.deepEqual(encode(f.year.cast(pxt.string().orNull()), scope), {
    _classname: 'TypeCast',
    new_type: { _classname: 'StringType', nullable: true },
    components: [ref('year')],
  });
});

test('functions are called by their Python path', () => {
  assert.deepEqual(encode(f.title.upper(), scope), call('pixeltable.functions.string.upper', [ref('title')], { method: true }));
  assert.deepEqual(encode(f.released.year, scope), call('pixeltable.functions.timestamp.year', [ref('released')]));
  assert.deepEqual(encode(f.budget.round(2), scope), call('pixeltable.functions.math.round', [ref('budget'), int(2)]));
  assert.deepEqual(
    encode(string.format('{0}!', f.title), scope),
    call('pixeltable.functions.string.format', [str('{0}!'), ref('title')]),
  );
  assert.deepEqual(encode(count(f.title), scope), call('pixeltable.functions.globals.count', [ref('title')], { aggregate: true }));
  assert.deepEqual(encode(f.title.contains('x', { case: false }), scope), {
    ...call('pixeltable.functions.string.contains', [ref('title'), str('x')], { method: true }),
    kwarg_idxs: { case: 2 },
    order_by_start_idx: 3,
    components: [ref('title'), str('x'), { _classname: 'Literal', val: false, col_type: { _classname: 'BoolType', nullable: false } }],
  });
});

test('a json path accumulates its elements, as e["a"]["b"] does in Python', () => {
  assert.deepEqual(encode(f.meta.get('a').at(0).slice(1, 3), scope), {
    _classname: 'JsonPath',
    path_elements: ['a', 0, [1, 3, null]],
    root_type: null,
    components: [ref('meta')],
  });
});

test('a timestamp literal is UTC ISO', () => {
  assert.deepEqual(encode(f.released.gt(new Date('2020-01-01T00:00:00Z')), scope), {
    _classname: 'Comparison',
    operator: 4,
    components: [
      ref('released'),
      { _classname: 'Literal', val: '2020-01-01T00:00:00.000Z', col_type: { _classname: 'TimestampType', nullable: false } },
    ],
  });
});

test("a column of another model is refused", () => {
  class Other extends TableModel('other') {
    year = pxt.int();
  }
  const other = layout(Other).instance as Other;
  assert.throws(() => encode(other.year.gt(1), scope), /'year' is not a column of this table/);
});

test('values carry the protocol tags for what JSON cannot represent', () => {
  const when = new Date('2026-11-19T12:00:00Z');
  assert.deepEqual(encodeValue({ t: when, x: Infinity, j: { a: [1] } }), {
    t: { $pxt: 'datetime', v: '2026-11-19T12:00:00.000Z' },
    x: { $pxt: 'float', v: 'inf' },
    j: { a: [1] },
  });
  assert.deepEqual(decodeValue([{ $pxt: 'datetime', v: '2026-11-19T12:00:00+00:00' }, { $pxt: 'UUID', v: 'u' }]), [when, 'u']);
});

test('an unbound model has no table', () => {
  assert.throws(() => Films.table(), /Films is not bound to a table; call bindAll\(\) on its model base/);
});
