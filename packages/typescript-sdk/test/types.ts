// Compile-time checks only: `tsc -p .` must report no errors, and each @ts-expect-error must catch one. Never run.

import * as pxt from '../src/index.ts';
import { Films, type Film, type NewFilm } from '../examples/films/app.tsx';

type Equals<A, B> = (<X>() => X extends A ? 1 : 2) extends <X>() => X extends B ? 1 : 2 ? true : false;
function assertType<T extends true>(): T | void {}

// row types derive from the class
assertType<Equals<Film['title'], string>>();
assertType<Equals<Film['budget'], number | null>>();
assertType<Equals<Film['title_len'], number>>();
assertType<Equals<Film['profit'], number | null>>();
assertType<Equals<Film['is_hit'], boolean>>();
assertType<Equals<Film['label'], string>>();

// expression types follow Python's result types
assertType<Equals<Films['title_len'], pxt.IntExpr<false>>>();
assertType<Equals<Films['profit'], pxt.FloatExpr<true>>>();
assertType<Equals<Films['decade'], pxt.IntExpr<false>>>();

const newFilm: NewFilm = { title: 'Arrival', year: 2016 };
const withBudget: NewFilm = { title: 'Arrival', year: 2016, budget: 47, revenue: null };

// @ts-expect-error: a required column is missing
const missingYear: NewFilm = { title: 'Arrival' };

// @ts-expect-error: a computed column cannot be inserted
const computed: NewFilm = { title: 'Arrival', year: 2016, title_upper: 'ARRIVAL' };

declare const film: Film;
// @ts-expect-error: a nullable column can be null
const budget: number = film.budget;

const CheckModel = pxt.modelBase();

export class Checks extends CheckModel('checks') {
  n = pxt.int().orNull();
  s = pxt.string();

  plus_int = this.n.add(1);
  plus_float = this.n.add(0.5);
  ratio = this.s.len().div(2);
  squared = this.s.len().pow(2);
  root = this.s.len().pow(0.5);
  as_text = this.n.cast(pxt.string().orNull());
  either = this.n.isNull().or(this.s.eq('x'));

  // @ts-expect-error: a typo in a column name
  typo = this.ss.upper();

  // @ts-expect-error: a reference to a column declared further down
  forward = this.later.upper();

  // @ts-expect-error: a string operand to an arithmetic operator
  bad_operand = this.n.add('1');

  // @ts-expect-error: upper() is a string method
  bad_method = this.n.upper();

  // @ts-expect-error: comparing a string with a number
  bad_comparison = this.s.gt(3);

  later = pxt.string();
}

assertType<Equals<Checks['plus_int'], pxt.IntExpr<true>>>();
assertType<Equals<Checks['plus_float'], pxt.FloatExpr<true>>>();
assertType<Equals<Checks['ratio'], pxt.FloatExpr<false>>>();
assertType<Equals<Checks['squared'], pxt.IntExpr<false>>>();
assertType<Equals<Checks['root'], pxt.FloatExpr<false>>>();
assertType<Equals<Checks['as_text'], pxt.StringExpr<true>>>();
assertType<Equals<Checks['either'], pxt.BoolExpr<false>>>();

// a bound model's table is typed by the class
async function boundModel(): Promise<void> {
  const t = Films.table();
  await t.insert([{ title: 'Arrival', year: 2016 }]);
  // @ts-expect-error: a required column is missing
  await t.insert([{ title: 'Arrival' }]);

  const all = await t.collect();
  assertType<Equals<typeof all, Film[]>>();

  const selected = await t.where(t.is_hit).select({ title: t.title, roi: t.margin.mul(100) }).collect();
  assertType<Equals<typeof selected, { title: string; roi: number | null }[]>>();

  // @ts-expect-error: a filter must be a boolean expression
  t.where(t.year);

  // @ts-expect-error: a typo in a column name
  t.where(t.is_hitt);
}

// a table opened by path has its columns as properties, typed only at run time, as in Python
async function byPath(): Promise<void> {
  const t = await pxt.getTable('pxt://local:tsdemo/films');
  const rows = await t.where(t.year.gt(2000)).select(t.title, t.margin).collect();
  assertType<Equals<typeof rows, Record<string, unknown>[]>>();
  const n: number = await t.count();
}

export { newFilm, withBudget, missingYear, computed, budget, boundModel, byPath };
