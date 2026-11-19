import assert from 'node:assert/strict';
import * as pxt from '@pixeltable/sdk';

import { Films, TableModel } from './app.ts';

const target = process.argv[2];
if (target === undefined) throw new Error('Pass the pxt:// target of a disposable Films catalog');

await TableModel.bindAll(target);
const films = Films.table();
const title = `Heat (SDK smoke ${Date.now()})`;
const missingBudget = `${title} without budget`;
const insertedFilms = films.title.isIn([title, missingBudget]);
await films.insert([
  { title, year: 1995, budget: 60, revenue: 187.5 },
  { title: missingBudget, year: 1996, budget: null, revenue: 100 },
]);

try {
  const rows = await films
    .where(insertedFilms)
    .select({ title: films.title, margin: films.margin, isHit: films.is_hit })
    .orderBy(films.year)
    .collect() satisfies { title: string; margin: number | null; isHit: boolean }[];

  assert.deepEqual(rows, [
    { title, margin: 2.125, isHit: true },
    { title: missingBudget, margin: null, isHit: false },
  ]);

  const DriftedModel = pxt.modelBase();
  class DriftedFilms extends DriftedModel('films') {
    year = pxt.string();
  }
  await DriftedModel.bindAll(target);
  assert.throws(() => DriftedFilms.table(), /films\.year.*string.*int/);
  console.log('Films smoke passed: typed projection, null budget, and mismatched schema.');
} finally {
  await films.delete(insertedFilms);
}
