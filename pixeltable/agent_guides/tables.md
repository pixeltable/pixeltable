# Tables

Declaring tables, writing rows, and changing a table's schema.

## Declaring a table

An application declares its tables as `TableModel` classes in `app.py`; `pxt schema update app.py TARGET` creates
them under the catalog directory `TARGET`.

```python
import pixeltable as pxt
import pixeltable.functions as pxtf

TableModel = pxt.model_base()


class Docs(TableModel, name='docs'):
    id = pxt.Column(value=pxtf.uuid.uuid7(), primary_key=True)
    title: pxt.String
    body: pxt.String | None
    tags: pxt.Json[list[str]]
    cover: pxt.Image | None
    title_upper = pxtf.string.upper(title)
    thumbnail = pxt.Column(value=cover.resize((64, 64)), stored=False)
```

- An annotation declares an input column; an assignment declares a computed column.
- `pxt.String` declares a non-nullable column: an insert that omits it fails with `Missing required column(s)`, and
  an explicit null fails with `expected non-None value`.
- `pxt.String | None` declares a nullable column.
- `pxt.Column()` takes what an annotation cannot express: `primary_key=True`, `stored=False` (computed when a query
  reads it, never stored), `media_validation='on_read'` (the default is `'on_write'`), `destination=` (where media
  files go), `comment=` and `custom_metadata=`.
- An annotation does not set the type of a `pxt.Column()`: `id: pxt.Int = pxt.Column(primary_key=True)` raises
  `` defines neither `type` nor `value` ``; write `id = pxt.Column(type=pxt.Int, primary_key=True)`.
- Types: `String`, `Int`, `Float`, `Bool`, `Timestamp`, `Date`, `UUID`, `Binary`, `Json`, `Array[(3, 4), pxt.Float]`,
  `Image`, `Video`, `Audio`, `Document`.
- `pxt.Json[...]` takes a TypedDict, a Pydantic model or `list[...]`; paths into the column, such as `tags[0]`, are
  typed (`pxt guide json`).
- In a script, `TableModel.bind_all('my_app')` binds each class to its table in `my_app`; then `Docs.insert(...)` and
  `Docs.select(...)` work.
- `pxt.get_table('my_app.docs')` returns the same table as a `Table`.

## Primary keys

- A primary key is declared when the table is created.
- `pxt schema update` refuses a model that adds a primary key to an existing table:
  `Cannot add primary key column ... after table creation`.
- `batch_update()`, `pxt get`, `pxt errors`, `add_update_route()` and `add_delete_route()` require a primary key.
- Inserting a row whose key already exists fails with `CONSTRAINT_VIOLATION` (`Duplicate primary key value`).

## Writing rows

```python
pxt.create_dir('my_project', if_exists='ignore')
docs = pxt.create_table(
    'my_project.docs', {'id': pxt.Int, 'title': pxt.String, 'body': pxt.String | None}, primary_key='id'
)
docs.add_computed_column(title_upper=pxtf.string.upper(docs.title))

status = docs.insert([{'id': 1, 'title': 'a'}, {'id': 2, 'title': 'b', 'body': 'text'}], return_rows=True)
docs.insert(id=3, title='c')
docs.insert([{'id': 3, 'title': 'c2'}, {'id': 5, 'title': 'e'}], if_exists='ignore')
docs.update({'body': 'reviewed'}, where=docs.id == 1)
docs.batch_update([{'id': 2, 'title': 'b2'}, {'id': 4, 'title': 'd'}], if_not_exists='insert')
docs.delete(where=docs.id == 3)
```

- `insert()` takes a list of dicts, one row as keyword arguments, a file path or URL, or a query.
- For a file path or URL, `source_format=` is `'csv'`, `'excel'`, `'parquet'` or `'json'`.
- For a file path or URL, `schema_overrides=` sets column types.
- Known issue (PXT-1749): an insert from a query leaves the computed columns of the new rows null.
- `if_exists='ignore'` skips each row whose primary key is already in the table and computes nothing for it.
- `status.row_count_stats.skipped_rows` counts the skipped rows.
- With `if_exists='ignore'`, a batch that repeats a new key inserts only the first of those rows.
- With `if_exists='ignore'`, concurrent inserts of the same key store one row.
- The default, `if_exists='error'`, fails the insert.
- `return_rows=True` returns the inserted rows, computed columns included, in `status.rows`.
- If the expression of a computed column raises, the whole insert fails and stores nothing.
- With `on_error='ignore'` the row is stored, the failed cell is null, and `col.errortype` and `col.errormsg` record
  the error.
- `update()` recomputes the computed columns that depend on the updated columns (`cascade=False` keeps their old
  values).
- `batch_update()` matches rows by primary key.
- `batch_update(if_not_exists='insert')` inserts the rows with unknown keys; `'ignore'` skips them.
- Each row inserted by `batch_update()` needs every non-nullable input column; a missing one raises
  `Missing required column(s)`, and `batch_update()` applies none of the rows.
- `batch_update()` recomputes the dependent computed columns of every matched row, even when the new values equal the
  old ones; a column that calls a paid model runs again.
- `delete()` without `where` deletes every row.
- Known issue (PXT-1760): an insert or an update that changes 0 rows still creates a table version; a delete that
  matches no row does not.
- Each `insert()`, `update()`, `batch_update()` and `delete()` is atomic.
- Writes to one table run one at a time.
- `status.num_rows` counts the changed rows.
- For a compare-and-set, put the expected current value in the `where` of `update()` and check `status.num_rows`.

## Computed columns

- An insert computes the computed columns of the new rows in the same operation.
- An update recomputes, in the same operation, the computed columns that depend on the changed columns.
- Adding a computed column computes it for every existing row.
- A computed column that references no column, such as `uuid7()`, gets its value at insert; updates never recompute
  it.
- `recompute_columns('summary', errors_only=True)` recomputes the failed cells of `summary`.
- The `where=` argument of `recompute_columns()` restricts the rows.
- `recompute_columns()` also recomputes the columns that depend on the recomputed column, unless `cascade=False`.
- The CLI form is `pxt recompute PATH COLUMN`, with `--errors-only` and `--no-cascade`.

## Changing the schema

`pxt schema diff app.py TARGET` shows what `pxt schema update app.py TARGET` would change in each table:

| Change in `app.py` | Classified as | Effect of `pxt schema update` |
|---|---|---|
| A new column | additive | Added; a new computed column is computed for every existing row |
| A new expression for a computed column | additive | Existing rows keep their values; `pxt schema update` prints the `pxt recompute` command for them |
| An input column becoming nullable (`pxt.String` to `pxt.String \| None`) | additive; unsupported if a non-nullable computed column in the table or in one of its views references the column | Altered in place; the refusal lists those computed columns |
| A removed column | destructive | Applied only with `--allow-destructive`; the column's data is dropped |
| Any other change to a column: its type, `primary_key`, `stored`, `media_validation`, `destination`, `comment` or `custom_metadata` | unsupported | Refused |

If `pxt schema update` refuses one change, it applies nothing, not even the additive changes.

A computed column that references a non-nullable column is non-nullable unless its expression can return null, so it
blocks making that column nullable.

`pxt schema diff` and `pxt schema update` ignore a UDF return type that becomes narrower, such as `str | None` to `str`.

In a notebook, `add_column()`, `add_computed_column()`, `drop_column()`, `rename_column()`, `alter_column()` and
`alter_computed_column()` change the schema directly:

- `alter_column(col, type_=pxt.String | None)` makes a column nullable; a non-nullable computed column that references
  the column blocks the change.
- `alter_computed_column()` recomputes the column by default; `recompute=False` keeps the old values.

## Versions

- Each change creates a new version of the table.
- `t.history()` and `pxt history PATH` list the versions.
- `t.revert()` and `pxt revert PATH --steps N` undo the last operation, or the last `N`, for every user of the table.
- A revert fails when the current version is 0 (`Cannot revert version 0`) or when a snapshot needs the current
  version (`Current version is needed for 1 snapshot: (docs_v3)`).
- `pxt.get_table('my_app.docs:3')` opens version 3, read-only.
- `pxt.create_snapshot('my_app.docs_v3', docs)` keeps the current rows as a read-only snapshot.

```bash
pxt schema diff app.py my_app
pxt schema update app.py my_app
pxt errors my_app/docs
pxt recompute my_app/docs title_upper --errors-only -f
pxt get my_app/docs 2
pxt history my_app/docs
pxt revert my_app/docs --steps 2 -n
```
