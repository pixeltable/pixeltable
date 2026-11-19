// Reads and writes the films table from TypeScript, over the same /rpc protocol as the Python client.
// Run after `pxt schema update app.tsx <target>`:
//   node --import @pixeltable/sdk/register demo.ts pxt://org:db

import * as pxt from '@pixeltable/sdk';

import { Films, TableModel } from './app.tsx';

const target = process.argv[2] ?? 'pxt://local:tsdemo';

// a table opened by path, as pxt.get_table() opens one in Python; its columns are its properties
const t = await pxt.getTable(`${target}/films`);
console.log(`${t.uri}: ${await t.count()} rows; columns ${t.columns().join(', ')}`);
console.log('insert:', await t.insert([{ title: 'Heat', year: 1995, budget: 60, revenue: 187.4 }]));
const recent = await t
  .where(t.year.gt(1990).and(t.title_len.gte(5)))
  .select(t.title, t.margin)
  .orderBy(t.margin, { asc: false })
  .limit(2)
  .collect();
console.log('top margins since 1990:', recent);

// the models, bound to their tables as TableModel.bind_all() binds them in Python; a model's rows are typed
await TableModel.bindAll(target);
const f = Films.table();
const hits: { label: string; roi_pct: number | null }[] = await f
  .where(f.is_hit)
  .select({ label: f.label, roi_pct: f.margin.mul(100).round(1) })
  .orderBy(f.margin, { asc: false })
  .collect();
console.log('hits:');
console.table(hits);
const byDecade = await f
  .groupBy(f.decade)
  .select({ decade: f.decade, films: pxt.count(f.title), revenue: pxt.sum(f.revenue) })
  .orderBy(f.decade)
  .collect();
console.log('by decade:');
console.table(byDecade);
console.log('delete:', await f.delete(f.title.eq('Heat')));
console.log(`${await f.count()} rows`);
