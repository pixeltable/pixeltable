# @pixeltable/sdk (prototype)

Define Pixeltable tables in TypeScript. As in Python, models extend a model base; a table is a class, each field is a
column, and a computed column is an expression over the fields above it:

```ts
import * as pxt from '@pixeltable/sdk';

export const TableModel = pxt.modelBase();

export class Films extends TableModel('films') {
  title = pxt.string();
  budget = pxt.float().orNull();
  revenue = pxt.float().orNull();
  title_upper = this.title.upper();
  profit = this.revenue.sub(this.budget);
}

type Film = pxt.Row<Films>; // { title: string; budget: number | null; ...; profit: number | null }
```

The `pxt` CLI applies a TypeScript schema file the same way it applies a Python one:

```bash
pxt schema diff app.tsx pxt://org:db
pxt schema update app.tsx pxt://org:db
```

The CLI runs Node.js once to evaluate the file. `src/export.ts` prints each exported model as data: its columns in
order, with each computed value as a tree of names (`{"method": "upper", "on": {"col": "title"}}`). The CLI replays
that tree through the Python expression DSL, so types, function signatures and error messages are Python's own, and
then reconciles the tables as it does for a Python schema.

## Reading and writing rows

Programs read and write tables directly over the daemon's `/rpc` protocol, the one the Python client uses, so they
need no Python. As in Python, a table opened by path has its columns as properties:

```ts
const t = await pxt.getTable('pxt://org:db/films');
await t.insert([{ title: 'Heat', year: 1995, budget: 60, revenue: 187.4 }]);
const rows = await t.where(t.year.gt(1990)).select(t.title, t.margin).orderBy(t.margin, { asc: false }).collect();
```

A model base binds its models to their tables, as `TableModel.bind_all()` does in Python. `Films.table()`, the
counterpart of Python's `Films.table`, is then the same kind of table, with its columns and rows typed by the class:

```ts
await TableModel.bindAll('pxt://org:db');
const f = Films.table();
await f.insert([{ title: 'Heat', year: 1995 }]); // typed by pxt.Insert<Films>
const hits = await f.where(f.is_hit).select({ title: f.title, roi: f.margin.mul(100) }).collect();
// hits: { title: string; roi: number | null }[]
```

It is a method because TypeScript types a static method, but not a static property, by the class it is called on.
A base learns which class extends `TableModel('films')` only when that class first asks for its table, so a field
missing from the table is reported then; a missing table is reported by `bindAll()`.

The client keeps each table's metadata, as the Python client does: the column ids its queries refer to, and the
table version a write is checked against. The daemon refuses a write against an older version and returns the current
metadata, and the client retries. A local proxy is `pxt://local:<db>`; a hosted database is reached through the
same TLS tunnel as from Python, with `PIXELTABLE_API_KEY`. `examples/films/demo.ts` runs both forms:

```bash
node --import @pixeltable/sdk/register demo.ts pxt://org:db
```

`@pixeltable/sdk/register` lets Node import `app.tsx`.

## Requirements

- Node.js 22.18 or later, on `PATH` where `pxt` runs. The package runs from its `.ts` sources through Node's type
  stripping, so it has no dependencies.
- The schema file imports `@pixeltable/sdk`, so the project needs it installed, for example with a
  `"@pixeltable/sdk": "file:..."` dependency and `npm install`.

## Development

```bash
tsc -p .                      # type-check src/, test/types.ts and examples/
node --test test/*.test.ts    # encoding tests
```

`test/types.ts` is never run: each `@ts-expect-error` in it must catch a schema mistake at compile time.
`examples/films/` is a project to apply with `pxt schema update app.tsx <target>`.

## Not in the prototype

Views and iterators, indexes, media and array columns, wrappers generated from the Python function registry,
`update()`, cursors, and connection reuse. The client speaks /rpc protocol version 10 and must match the daemon's
exactly.
