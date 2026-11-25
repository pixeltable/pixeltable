import assert from 'node:assert/strict';
import { test } from 'node:test';

import { modelSpec } from '../src/codec.ts';
import { string } from '../src/functions/index.ts';
import * as pxt from '../src/index.ts';

const TableModel = pxt.modelBase();

class Films extends TableModel('films', { comment: 'films' }) {
  title = pxt.string();
  budget = pxt.float().orNull().comment('USD');
  title_upper = this.title.upper();
  title_len = this.title_upper.len();
  half = this.budget.div(2);
  label = string.format('{0}!', this.title);
  recent = pxt.column(this.title_len.gt(3).and(this.budget.isNotNull()), { stored: false });
  pinned = this.title.eq(pxt.lit(1, 'float').cast(pxt.string()));
}

test('a model encodes its columns in order, referring to earlier ones by name', () => {
  assert.deepEqual(modelSpec(Films), {
    cls: 'Films',
    name: 'films',
    options: { comment: 'films' },
    columns: [
      ['title', { type: { type: 'string', nullable: false } }],
      ['budget', { type: { type: 'float', nullable: true }, comment: 'USD' }],
      ['title_upper', { value: { method: 'upper', on: { col: 'title' }, args: [], kwargs: {} } }],
      ['title_len', { value: { method: 'len', on: { col: 'title_upper' }, args: [], kwargs: {} } }],
      ['half', { value: { op: 'div', args: [{ col: 'budget' }, 2] } }],
      ['label', { value: { call: 'pixeltable.functions.string.format', args: ['{0}!', { col: 'title' }], kwargs: {} } }],
      [
        'recent',
        {
          value: {
            op: 'and',
            args: [
              { op: 'gt', args: [{ col: 'title_len' }, 3] },
              { op: 'isNotNull', args: [{ col: 'budget' }] },
            ],
          },
          stored: false,
        },
      ],
      [
        'pinned',
        {
          value: {
            op: 'eq',
            args: [{ col: 'title' }, { cast: { type: 'string', nullable: false }, on: { lit: 1, type: 'float' } }],
          },
        },
      ],
    ],
    indexes: [],
  });
});

test('literals that JSON cannot carry bare are typed', () => {
  const when = new Date('2026-11-19T12:00:00Z');
  class Literals extends TableModel('literals') {
    t = pxt.timestamp();
    j = pxt.json();
    after = this.t.gt(when);
    tagged = this.j.eq({ kind: 'film' });
  }
  const columns = modelSpec(Literals).columns;
  assert.deepEqual(columns[2], [
    'after',
    { value: { op: 'gt', args: [{ col: 't' }, { lit: { $pxt: 'timestamp', v: '2026-11-19T12:00:00.000Z' }, type: 'timestamp' }] } },
  ]);
  assert.deepEqual(columns[3], [
    'tagged',
    { value: { op: 'eq', args: [{ col: 'j' }, { lit: { kind: 'film' }, type: 'json' }] } },
  ]);
});

test('one column object in two fields is refused', () => {
  class Twice extends TableModel('twice') {
    a = pxt.string();
    b = this.a;
  }
  assert.throws(() => modelSpec(Twice), /Twice\.b and Twice\.a hold the same column object/);
});

test("a reference to another model's column is refused", () => {
  const shared = pxt.string();
  class Owner extends TableModel('owner') {
    a = shared;
  }
  class Borrower extends TableModel('borrower') {
    b = pxt.string();
    c = shared.upper();
  }
  modelSpec(Owner);
  assert.throws(() => modelSpec(Borrower), /Borrower\.c refers to a column that is not a field of Borrower/);
});

test('a model base refuses a second model with the same table name', () => {
  const Base = pxt.modelBase();
  Base('films');
  assert.throws(() => Base('Films'), /two models on one model base have the table name 'Films'/);
  pxt.modelBase()('films'); // another base may reuse it
});
