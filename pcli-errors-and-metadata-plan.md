# pcli errors + metadata commands — concrete plan

## `pcli errors <table> [--col NAME] [--full-rows] [--json]`

### What it does

For a given table, find every (row, column) where the column is computed
**and** has a non-null `errortype`/`errormsg`. Flat output, one entry per
(row, erroring column).

### Server-side query construction

```python
@router.post('/pcli/v0/errors', response_model=ErrorsResponse)
def errors(req: ErrorsRequest) -> ErrorsResponse:
    t = pxt.get_table(req.path)
    md = t.get_metadata()
    pk_names = md.get('primary_key')
    if not pk_names:
        raise HTTPException(400, f"{req.path} has no primary key; pcli errors requires one")
    computed = [
        name for name, c in md['columns'].items()
        if c['is_computed'] and (req.col is None or name == req.col)
    ]
    if not computed:
        return ErrorsResponse(entries=[])

    where = None
    for col in computed:
        cond = (t[col].errortype != None)  # noqa: E711 — pxt's `!=` returns an Expr
        where = cond if where is None else where | cond

    pk_cols = [t[c] for c in pk_names]
    select_args = list(pk_cols)
    for col in computed:
        select_args += [t[col].errortype, t[col].errormsg]
    rows = t.where(where).select(*select_args).collect()

    entries = []
    for r in rows:
        pk = {name: r[i] for i, name in enumerate(pk_names)}
        for j, col in enumerate(computed):
            errortype = r[len(pk_names) + 2*j]
            errormsg = r[len(pk_names) + 2*j + 1]
            if errortype is not None:
                entries.append(ErrorEntry(pk=pk, column=col, errortype=errortype, errormsg=errormsg))
    return ErrorsResponse(entries=entries)
```

Key points:
- One DB query per `pcli errors` invocation (OR'd predicates filter once).
- Requires a declared primary key. Refuses if the table has none.
- `pk` field is a dict (column name → value), not an opaque rowid.
- `t[col].errortype != None` returns a pxt `Expr` (matches existing usage
  in `tests/test_table.py`).

### Models

```python
class ErrorsRequest(BaseModel):
    path: str
    col: str | None = None

class ErrorEntry(BaseModel):
    pk: dict          # column name -> value
    column: str
    errortype: str
    errormsg: str | None

class ErrorsResponse(BaseModel):
    entries: list[ErrorEntry]
```

### Output

**Default (TSV)** — column-named pk for grep-friendliness:
```
pk                              column              errortype      errormsg
{id: 42}                        relevant_memories   ValueError     Failed to embed: bad input
{id: 42}                        response            RequestError   Rate limit exceeded
{user: 'alice', day: '2026-01'} response            TimeoutError   Connection timed out
```

The `pk` rendering is a compact `{name: val, ...}`; client can parse trivially.

**`--json`** — one entry per object:
```json
[
  {"pk": {"id": 42}, "column": "relevant_memories",
   "errortype": "ValueError", "errormsg": "Failed to embed..."}
]
```

**`--full-rows`** — same entries plus a `row` field with the full row data
(media rendered as paths). Requires a second pass: re-query with all visible
columns selected. Skip in v0; add later if needed.

### Deferred

- View-traversal (`--walk-views`) — not in v0; user can name the view directly.
- `--full-rows` — dropped from v0. Use `pcli rows <table> --where <pk>=...`
  once `rows` exists; the errormsg already carries the exception detail.
  (We avoid `pcli head` — head implies an ordering that won't be defined for
  unversioned tables that are coming.)
- Cross-catalog error summary (`pcli errors` with no arg → all tables) — not
  in v0; lean on `tree.error_count` (already in TableNode after the extension)
  to find tables with errors, then drill in.

---

## Metadata commands

### `pcli history <table> [-n N] [--json]`

Server returns raw `Table.get_versions(n)` (list of `VersionMetadata` dicts,
most recent first). Client renders TSV by default, raw JSON with `--json`.

```python
class HistoryRequest(BaseModel):
    path: str
    n: int | None = None

class HistoryResponse(BaseModel):
    versions: list[dict]   # raw VersionMetadata; client formats
```

Client TSV columns: `version | created_at | change_type | inserts | updates
| deletes | errors | schema_change`. ~10 lines of formatting.

### `pcli columns [<table>] [--computed] [--json]`

Cross-catalog column listing. With a table arg: that table only (similar to
`describe` but column-focused). Without: every column across every table.

Use case: "where do I use `openai.chat_completions`" — grep the
`computed_with` field across all computed columns.

Server walks `get_dir_tree()` → for each table, pull `metadata['columns']`
→ filter to `is_computed` if `--computed`. Returns
`list[{table_path, column}]`.

This is one DB walk and shouldn't be expensive; the metadata is read from the
catalog Tables records pxt already touches.

```python
class ColumnsRequest(BaseModel):
    path: str | None = None
    computed_only: bool = False

class ColumnEntry(BaseModel):
    table: str
    column: str
    is_computed: bool
    type_: str
    computed_with: str | None
    depends_on: list[tuple[str, str]] | None  # passthrough from ColumnMetadata

class ColumnsResponse(BaseModel):
    entries: list[ColumnEntry]
```

### `pcli indices [<table>] [--json]`

Cross-catalog embedding-index listing. Same shape as columns.

Use case: "what indexes exist", "which tables have a text-embedding index".

```python
class IndicesRequest(BaseModel):
    path: str | None = None

class IndexEntry(BaseModel):
    table: str
    name: str
    columns: list[str]
    index_type: str         # 'embedding' / 'btree'
    metric: str | None      # for embedding
    embedding: str | None   # fn ref, for embedding

class IndicesResponse(BaseModel):
    entries: list[IndexEntry]
```

### Not adding

- `pcli base <view>` — already covered by `describe` (`metadata.base`) and by
  TableNode's `base` field in `ls`. No separate command needed.
- `pcli stats` — could be useful but lower-priority; defer.

---

## Implementation order

1. **`pcli errors`** — concrete plan above; one route + client command.
2. **`pcli history`** — passthrough of `get_versions()`.
3. **`pcli columns`** — cross-catalog walk.
4. **`pcli indices`** — same shape as columns.

Each is ~30-50 LOC server + ~20 LOC client. The walking pattern is shared
across columns/indices and could be factored if it gets repetitive, but at
two commands' worth it's fine inline.

## Open questions

1. **`pcli columns` default**: list everything (with `--computed` to filter),
   or default to computed-only? Defaulting to everything is consistent with
   "list everything you've got"; computed-only is what you actually use it
   for. Leaning everything-by-default.

## Decisions taken (this round)

- `--full-rows` dropped from v0.
- History/columns/indices use **server-returns-raw, client-formats-TSV**
  pattern. `describe` keeps its server-side text rendering only because
  `repr(table)` already exists there.
