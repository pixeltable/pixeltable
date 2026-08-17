# Case-insensitive Pixeltable identifiers

**Status:** proposed · **Date:** 2026-08-17

> Code references are given as `file.py::symbol` rather than line numbers, so they survive unrelated
> edits.

## Context

Pixeltable identifiers — directory, table, column and index names — are matched **exactly** today.
`pxt.get_table('MyDir.MyTable')` will not find a table created as `mydir.mytable`, and nothing
prevents a single table from holding two columns `foo` and `Foo`. Users coming from SQL expect
identifiers to behave case-insensitively.

**Approach: case-fold at the front door.** Every user-supplied identifier is lowercased at the API
boundary, and that folded form is what gets stored. Original casing is *not* preserved. This mirrors
Postgres, which folds unquoted identifiers to lower case; Pixeltable has no quoting syntax, so there
is no case-preserving escape hatch to reconcile with.

The consequence that makes this design cheap: **once stored names are already folded, every internal
comparison, lookup, dict key and SQL predicate keeps working unchanged.** There is no second "folded
key" form to thread through the system, and no place where a folded value can be compared against an
unfolded one.

### Decisions

| Question | Decision |
|---|---|
| Stored form | **Case-folded (lowercase).** Original casing is not retained anywhere. |
| Display | Folded — `describe()`, `get_metadata()`, `columns()`, `collect()` keys and `to_pandas()` labels all show lowercase. |
| Migration | **Required.** `VERSION` 55 → 56, with a converter that folds all existing names. |
| Migration conflicts | **Fail loudly.** The converter validates before writing and aborts with a message naming the offenders; it never silently merges or renames around a collision. |
| Non-ASCII names | **Rejected.** `is_valid_identifier` gains an `isascii()` check; the converter validates existing names and aborts on any non-ASCII name. |
| Case-only rename | **No-op.** Both spellings fold to the same value, so renaming or moving to a case variant succeeds and changes nothing — no new schema version, no metadata write. |
| Remote `pxt://` paths | Folded in `Path.parse`, same as local. |

### Out of scope

JSON path components (`t.json_col.someField`) are **data**, not identifiers — they stay
case-sensitive. This is the one boundary the front door must not cross; see section 1.

---

## 1. Front-door folding

Fold as early as possible, so that every downstream raw-string use already receives a folded value.
Placement matters more than volume: a fold placed at the true entry point means the ~60 internal
identifier comparisons inventoried during design need no changes at all.

**The primitive** — `catalog/globals.py`, next to `is_valid_identifier`:

```python
def fold_identifier(name: str) -> str:
    """Fold an identifier to its stored form. Identifiers are ASCII, so this matches SQL lower()."""
    return name.lower()
```

`fold_identifier` is **total and non-raising** — it never validates. Validation stays in
`is_valid_identifier`, which has real uses with no folding involved: `globals.py` uses it as a filter
to hide system paths during listings, `json_path.py` asserts a column name *derived* from a JSON path
is legal, and `column.py::Column.__init__` validates on load from stored metadata. More importantly,
lookup paths need fold-without-validate: `t['not-an-identifier']` and `row['weird key']` must stay
`COLUMN_NOT_FOUND` rather than becoming `INVALID_COLUMN_NAME`.

Where both apply, **fold first, then validate**, so the message quotes the stored form. For ASCII,
lowercasing cannot change whether a string is a legal identifier, so the order affects only the error
text, not the outcome.

**Restrict to ASCII** in `is_valid_identifier`: add `if not name.isascii(): return False`.

This is *not* required for correctness. Only Python ever folds — stored names are already folded, so
the catalog predicates stay plain `==` and Postgres collation never enters the picture. The same
`fold_identifier` runs at write and at read time, so the two sides cannot disagree whatever the
alphabet. The restriction is taken for predictability: on ASCII, `str.lower()`, `str.casefold()` and
SQL `lower()` all coincide, so no one has to reason about which folding an identifier went through;
and it sidesteps Unicode normalization, where `'café'` (NFC) and `'cafe\u0301'` (NFD) stay distinct
names after case folding because folding does not normalize.

It does have a hard consequence for the migration. `is_valid_identifier` is called from
`column.py::Column.__init__`, which runs when columns are **loaded** from stored metadata, not only
when they are created. Adding the `isascii()` check therefore makes any pre-existing non-ASCII column
unloadable, so the converter must reject non-ASCII names outright — there is no option to grandfather
them.

**Entry points.** Two of these are chokepoints that cover many callers at once:

| Entry point | Covers |
|---|---|
| `catalog/path.py::Path.__post_init__` | **All** path-based APIs — `create_table`, `create_view`, `create_snapshot`, `create_dir`, `get_table`, `drop_table`, `drop_dir`, `move`, `ls`, `list_tables`, `list_dirs`, plus `org`/`db` in `pxt://` URIs. Fold `org`, `db` and every component there and the entire path surface is covered in one place. |
| `catalog/globals.py::normalize_schema` | Schema-dict keys for `create_table` / `create_view` (and the `table_proxy.py` equivalents). Must reject folded-duplicate keys *before* building the result dict — see the note below |
| `local_table.py::LocalTable.add_columns` | `add_columns` and `add_column` (which forwards to it). **Not** covered by `normalize_schema` today: it calls `_validate_column_schema` and then `Column.create` per key, and `_ignore_or_drop_existing_columns` does its duplicate check on the **raw** dict keys, and only against *existing* columns. Fold the schema keys at the top, rejecting folded duplicates within the input first — ideally by routing through `normalize_schema`, which also removes the asymmetry with `TableProxy.add_columns`, which already normalizes |
| `LocalTable.add_computed_column` **and** `TableProxy.add_computed_column` | **A separate path that must be folded on its own — see the note below.** |
| `local_table.py::LocalTable.__getattr__`/`__getitem__`, `table_proxy.py::TableProxy.__getattr__` | `t.MyCol`, `t['MyCol']` |
| `local_table.py::LocalTable._resolve_column_parameter` | `drop_column`, `alter_column`, `recompute_columns` taking a name |
| `TableVersion.rename_column`, `LocalTable.rename_column` | both arguments |
| `add_btree_index`, `add_embedding_index`, `_drop_index`, `TableVersion.get_idx`, `ColumnRef.similarity(idx=)`, `model.py::EmbeddingIndex.name` | index names |
| `_query.py::Query.select` | `select(**kwargs)` output names |
| `TableVersion._validate_update_spec`, `LocalTable.batch_update` | `update()` / `batch_update()` dict keys |
| `io/table_data_conduit.py::check_source_columns_are_insertable` and `RowDataTableDataConduit._translate_row` | `insert()` dict keys; `_translate_row` must write the folded key, because `exec/in_memory_data_node.py::InMemoryDataNode._open` **asserts** rather than raises on a miss |
| `insertable_table.py::InsertableTable._create` | `primary_key=[...]` |
| `row.py::Row.__getitem__`/`get`/`__contains__`, `Row.errors`, `Row.index_values`, `_query.py::ResultSet.__getitem__` | `row['MyCol']` keeps working even though the stored key is `mycol`. `errors` and `index_values` are public mappings keyed by column/index name and must fold too — see the note below |
| `func/iterator.py::GeneratingFunction.__call__` and its `outputs is None` reconstruction; `unstored_cols` | iterator output names, which become view column names. Fold the **keys** of the `outputs` dict — and **not** `IteratorOutput.orig_name`, which is a runtime data key — see the note below |
| `catalog/model.py::TableModelMeta.__prepare__` | the model's **table** name — the `name=` kwarg of `class M(TableModel, name='foo')`. A separate identifier surface from the columns; fold before the `is_valid_identifier` check and the `__registered_models__` probe — see the note below |
| `catalog/model.py::_ModelNamespace.set_col_value` **and** `set_col_type` | Python class attribute names → column names. Fold in **both**, not in `__setitem__`: bare annotations bypass it entirely — `_AnnotationRecorder.__setitem__` calls `set_col_type` directly. And fold only the *column* name, not the Python binding — see the note below |
| `serving/_fastapi.py::_validate_dml_args`; `pixeltable_cli/server/routes.py` | HTTP request field names, `cols`/`order_by` query params |
| `pixeltable_cli/client/commands/localproxy.py`; the config `db` key (`config.py`) | The local-database name. It must fold in **all** the places it enters, not just in `Path`: `pxt localproxy {create,start,stop,delete} <db>` takes it straight to `service/proxy_daemon.py::proxy_home`, and the config `db` key becomes the Postgres database name in `env.py`. Fold only in `Path` and `pxt localproxy start MyDb` creates `proxy_MyDb` while `pxt://local:MyDb` looks for `proxy_mydb` |
| `io/utils.py::normalize_schema_names` | external source column names (pandas/parquet/HF/SQL). Fold **before** the `_1` disambiguation loop, or a source carrying both `A` and `a` passes disambiguation untouched and then collides |

**Folding inside `Path` needs `object.__setattr__`, and leaves validation on the raw input.** `Path`
is `@dataclasses.dataclass(frozen=True, slots=True, order=True, kw_only=True)`, so `__post_init__`
cannot assign to `self`: fold with
`object.__setattr__(self, 'components', tuple(fold_identifier(c) for c in self.components))`, and
likewise for `org` and `db`. `order=True` means the generated `__lt__` then compares folded values,
which is what the `sorted(dir_paths)` deadlock-avoidance ordering in `catalog.py::_prepare_dir_op`
relies on. `is_valid_identifier(..., allow_hyphens=True)` guards components, `org` and `db` alike, so
the ASCII restriction reaches all three.

Paths are the one exception to *fold first, then validate*. `Path.parse` deliberately validates the
components **before** constructing the instance — "so that if validation fails, the error message
exactly matches the input" — and `__post_init__` validates again after folding. Keep both, so
`pxt.get_table('Café/x')` reports `Invalid path: Café/x` rather than a folded form the user never
typed. That does not conflict with section 3's note about messages echoing the folded path: that one
concerns *not-found* messages, which are rendered by `Path.__str__` from the stored components.

**`add_computed_column` needs its own fold; it shares nothing with `add_columns`.** It reads
`col_name, spec = next(iter(kwargs.items()))` from the raw kwarg, validates *that* with
`is_valid_identifier`, runs `_ignore_or_drop_existing_columns([col_name], ...)` on it, builds
`Column.create(col_name, ...)` from it, and then calls `TableVersion.add_columns` — a different
method from `LocalTable.add_columns`. There is no shared chokepoint. The proxy has the same hole from
the other direction: `TableProxy.add_columns` and `add_column` both call `normalize_schema`, but
`TableProxy.add_computed_column` just does `bound_args['columns'] = bound_args.pop('kwargs')`.

Left unfolded, `add_computed_column(MyCol=...)` stores a mixed-case name, which breaks the invariant
every other part of this design rests on — that stored names are *always* folded — and makes
duplicate and reserved-name handling inconsistent with `add_column`. Fold at the **top** of the
method, before both the `is_valid_identifier` call and the duplicate check: the existing
`assert cols_to_ignore[0] == col_name` after `_ignore_or_drop_existing_columns` fires if the fold
happens inside that helper instead.

**Fold before deduplicating, and reject folded duplicates in the input.** Every place that folds a
mapping's keys must detect folded collisions *before* building the normalized mapping, because
assignment silently discards the loser. `catalog/globals.py::normalize_schema` builds
`result[name] = ...`; fold `name` at that assignment and `{'a': ..., 'A': ...}` quietly keeps whichever
came last, in direct contradiction of the rejection that section 5 requires. The same applies to
`LocalTable.add_columns`, which materializes `schema_copy = dict(schema)` — and note that
`_ignore_or_drop_existing_columns` will not save it, since that checks the input against columns that
*already exist*, not against itself.

Both need a pre-pass over the incoming mapping that groups keys by folded name and raises
`INVALID_SCHEMA` on any group larger than one, naming both spellings. Use `INVALID_SCHEMA` rather than
`COLUMN_ALREADY_EXISTS` here: nothing pre-existing has been collided with — the supplied mapping is
self-contradictory. (Python collapses a literal `{'a': 1, 'a': 2}` before we ever see it, so only
case-distinct keys are detectable, which is exactly the case that matters.)

`_query.py::Query.select` already has the equivalent check — its `seen` set raises
`Repeated column name` — so it needs no new logic, provided the kwarg names are folded *upstream* of
`_normalize_select_list`.

`io/utils.py::normalize_schema_names` is the deliberate exception: it **suffixes rather than rejects**.
External source names are raw strings, not identifiers the user chose — `normalize_pxt_col_name`
already rewrites non-alphanumeric characters to `_` and the existing loop already de-collides `my col`
against `my-col` as `my_col`/`my_col_1`. Folding makes case one more way two source names normalize to
the same identifier, so `A` and `a` import as `a` and `a_1`, with `col_mapping` recording both. That is
why the fold goes *before* the disambiguation loop, per its entry above.

**Iterator outputs fold their column name, not their runtime key.** `func/iterator.py` builds
`outputs = {name: IteratorOutput(orig_name=name, ...)}` from `call_output_schema`, in two places — the
normal call path and the `outputs is None` reconstruction for legacy iterators. The dict *key* becomes
a view column name: `view.py::View._create` creates one `Column` per key, bypassing both
`normalize_schema` and `_verify_schema`, so nothing folds it today and a `ComponentIterator` whose
`output_schema` declares `MyOutput` would store a mixed-case column name — the one remaining hole in
"stored names are always folded".

`orig_name` is a different thing despite starting out equal to the key: it is the field name in the
dict the iterator actually *yields*, read at `exec/component_iteration_node.py` as
`component_dict[output_info.orig_name]`. Folding it would break every iterator whose `__next__`
returns a mixed-case field. Fold the key, leave `orig_name` alone — the same split as
`_ModelNamespace` below.

`unstored_cols` folds too. It is a user-supplied list of output names, tested as
`name not in self.unstored_cols`, so `@pxt.iterator(unstored_cols=['Frame'])` over an output `Frame`
would silently start storing a column meant to be unstored. Two checks then become case-insensitive
for free: `view.py`'s iterator-output-vs-declared-column rejection ("produced by the iterator and also
declared by the view"), and `_ModelNamespace._check_reserved` for the names
`add_reserved_column_ref` seeds from an iterator.

**`_ModelNamespace` is two namespaces at once, and only one of them folds.** It is both the Pixeltable
column registry and the **Python class namespace**. `set_col_value`/`set_col_type` write
`known_cols[name]` — which becomes `__columns__`, the actual column names — *and*
`super().__setitem__(name, ColumnRefByName(name, ...))`, which is what makes the name resolvable in the
rest of the class body and, after `dict(namespace)` in `TableModelMeta.__new__`, an attribute on the
class. Folding the dict key breaks both:

```python
class M(TableModel, name='m'):
    MyCol: pxt.Int
    doubled = MyCol * 2      # NameError: only 'mycol' is in the namespace
```

and `M.MyCol` afterwards raises `AttributeError`. Python name and attribute resolution is
case-sensitive; that is not something this design can fold away.

So fold only the Pixeltable-facing side — `known_cols` keys, `reserved_cols` keys, the `_check_reserved`
probe, and the `name` inside the `ColumnRefByName` — and keep the **as-written** spelling as the dict
key and class attribute. Record the raw→folded mapping (say `__declared_names__`, alongside
`__columns__`) so later stages can get from one to the other. `__setitem__`'s existing
`is_valid_identifier` guard stays on the raw key, which is correct: it is a Python identifier as well as
a column name.

Three consequences that need explicit work:

- **`set_col_type` does not currently reject duplicates.** With folded `known_cols` keys,
  `set_col_value`'s `if name in self.known_cols` catches `Foo = ...` after `foo = ...`. Its counterpart
  in `set_col_type` does *not* raise — an existing entry is read as an annotation confirming an
  already-assigned value (`col = expr` stores before `__annotations__` records), so it only errors on a
  type *mismatch*. `Foo: pxt.Int` followed by `foo: pxt.Int` would therefore be silently merged into one
  column. Track which names arrived as bare annotations and raise `INVALID_SCHEMA` on a second one.
- **`_bind` must rebind the as-written spellings.** It does
  `setattr(cls, col_name, col_ref) for col_name in tbl.columns()`, and `tbl.columns()` returns folded
  names — so `M.mycol` becomes the real `ColumnRef` while `M.MyCol` still holds the stale
  `ColumnRefByName` placeholder left by the class body, silently diverging. Walk
  `__declared_names__` and `setattr` both spellings to the same ref.
- **`TableModelMeta.__getattr__` should fold as a fallback**, so a casing that was never declared
  (`M.MYCOL`) resolves against the bound table. Without it the model class is the one handle where
  arbitrary casing does not work, since `LocalTable.__getattr__` folds. Note `__getattr__` only runs
  when normal lookup fails, so it cannot fix the declared-spelling case above — that is why `_bind`
  has to rebind.

`model.py::EmbeddingIndex.name` folds too, as an index name; `_validate_indexes` then catches `Idx`
against `idx` with its existing `len(all_index_names) != len(set(...))` check. Index *columns* need
nothing — they must already be `ColumnRefByName`s, whose `name` is folded by the above.

**A model's table name is a second identifier surface, and it is the one that silently duplicates.**
`TableModelMeta.__prepare__` validates the `name=` kwarg and probes
`bases[0].__registered_models__` with the **raw** string; `__new__` then registers under the raw
`namespace.table_spec['name']`. But `_create` builds `bound_path = f'{catalog_dir}{table_spec["name"]}'`
and hands it to `Path.parse`, which folds. So models declared `name='Foo'` and `name='foo'` both
register as distinct entries that resolve to the *same* table, and every collection keyed by that name
inherits the duplicate:

- `validate_models` iterates `registered_models` and returns `results` keyed by it — two diffs computed
  against one table.
- `create_all` sees both as `exists=False`, so `expect_created` holds both spellings. `_create_models`
  creates the table on the first and finds it existing on the second, whose name *is* in
  `expect_created` — so it raises `ConcurrencyError`, "was created concurrently; re-run the operation",
  for a condition that is not concurrency and that re-running cannot clear.
- `update_all` applies two diffs to one table, the second computed against state the first already
  changed.

Fold in `__prepare__` — before the `is_valid_identifier(tbl_name, allow_hyphens=True)` call and before
the `if tbl_name in base_models` probe — and store the folded value in `table_spec['name']`. That needs
no new rejection rule: the existing probe then catches the second declaration with the message it
already has — *has name 'foo', but that name was previously used by `Foo`* — and `registered_models`,
`expect_created` and the diff keys are folded by construction. `table_spec['display_name']` is a
different thing: it is built from the Python class name for use in messages, and must **not** fold.

**View models reject base-column shadowing, and keep rejecting it.** This is the one place where the
model API is deliberately stricter than the imperative one. `prepare_model` refuses an additional
column whose name is already an inherited base column —
`if exprs.ColumnRefByName(name) in subst_dict: raise COLUMN_ALREADY_EXISTS`, under the comment "we
don't allow an explicitly named column to shadow one". `ColumnRefByName` identity is name-only
(`_equals()` compares only the name, and `_id_includes_col_type = False`), so once names fold, a model
declaring `Foo` over a base column `foo` starts hitting that probe where today it slips past.

Keep it. Declaring `foo` over base `foo` already raises here, and the whole point of folding is that
`Foo` behaves exactly as `foo` does — carving out an exception so that the case variant is *more*
permissive than the exact name would be the surprising choice. The asymmetry with `create_view`
(section 5, where shadowing is allowed) also stands on its own: shadowing in an imperative call is a
local, visible act, whereas a declarative model is re-diffed against the catalog on every
`update_all()`, and a column that shadows an inherited one makes that diff ambiguous. So this is a
documented TableModel-only restriction rather than an inconsistency to remove.

Worth knowing before writing the test: the check is only reachable when the base query is `select(*)`.
`subst_dict` is populated solely in the `base.select_list is None` branch; with an explicit select list
it stays empty and the probe never fires. That is pre-existing and unrelated to folding.

**`Row.errors` and `Row.index_values` fold as well.** They are public mappings keyed by column and
index name respectively, and both currently return their backing `dict` directly. Folding only
`__getitem__`/`get`/`__contains__` would leave `row.errors['MyCol']` and
`row.index_values['MyIdx']` case-sensitive while `row['MyCol']` works — precisely the "works here,
not there" seam this design exists to remove.

Give `Row` one internal folding mapping and use it in all three places, so there is a single
implementation of *fold the key, then look up*. Their keys are already folded (they come from stored
names), so the wrapper only has to fold the incoming key; iteration, `keys()` and `len()` pass through
unchanged. This widens the declared return type from `dict[str, ...]` to `Mapping[str, ...]` — a
public signature change, though mutating those dicts was never meaningful.

`RowBatch.schema`, `RowBatch.column_names` and `ResultSet.schema` need nothing: they enumerate names
rather than look them up.

**Do not fold** `exprs/column_ref.py::ColumnRef.__getattr__` or
`exprs/json_path.py::JsonPath.__getattr__`. Attribute access on a `ColumnRef` falls through to JSON
path element access, and those elements are data keys — a JSON field `someField` must stay
`someField`. Folding in `Table.__getattr__` is correct and safe; folding one level down is not.

**Both name bans become case-insensitive, and neither check changes.** `Column.validate_name` rejects
a name that is either a Pixeltable reserved symbol (`is_system_column_name`, i.e. a member of
`dir(InsertableTable) | dir(View)` — `count`, `select`, `insert`, `where`, `head`, `columns`,
`update`, ...) or a Python keyword (`is_python_keyword`, which is `keyword.iskeyword`). Both receive
an already-folded name, so `Count` is rejected exactly as `count` is, and `Class` exactly as `class`
is. That follows entirely from where the fold sits. No data changes either: `_PREDEF_SYMBOLS` is
`set(itertools.chain(dir(InsertableTable), dir(View)))`, and every member of it is already lowercase —
they are snake_case methods and dunders. Build it folded anyway, as a one-line guard: it costs nothing
and it stops the ban from silently depending on a naming convention holding forever, since a
mixed-case public attribute added later would otherwise stop being rejected in its folded form.

Rejecting them is correct rather than incidental — after folding, the column really would be named
`count` or `class`, and `t.class` does not parse. But it *is* a behavior change: `Count` and `Class`
are legal column names today. See section 5 for the tests, and section 2 for the migration note.

**`rename_column` must start enforcing them.** It validates the new name with `is_valid_identifier`
only and never calls `Column.validate_name`, so `rename_column('x','select')` installs a reserved name
today and `rename_column('x','Count')` would install `count` after folding — leaving the ban uniform
for `add_column` but bypassable through rename. Replace the `is_valid_identifier` call in
`TableVersion.rename_column` with `Column.validate_name`, which checks both bans *and* validity. This
is a pre-existing hole rather than one folding creates, and closing it makes
`rename_column('x','select')` an error where it succeeds today.

**Case-only renames and moves are no-ops.** Both spellings fold to the same value, so there is
nothing to change and nothing to report — the operation succeeds and does nothing. Neither of the two
paths does that today:

- **`move`.** Drop the raw `if path == new_path` string comparison in `globals.py::move`; the identity
  test belongs on the parsed `Path`s. Put it in `catalog.py::Catalog._move`, **before**
  `_prepare_dir_op` and resolving the source first: when `path == new_path`, call `_prepare_dir_op`
  with only its *drop* side (`drop_dir_path`/`drop_name`, `raise_if_not_exists` from `if_not_exists`)
  and return without moving. Two reasons it cannot be an early return in `globals.py`: a missing source
  must still raise `PATH_NOT_FOUND` (or return quietly under `if_not_exists='ignore'`) exactly as any
  other move does; and calling the full `_prepare_dir_op` with equal paths would look up the
  destination, *find the source itself*, and raise `PATH_ALREADY_EXISTS` under
  `raise_if_exists=True` — a wrong answer with a confusing message. `is_ancestor` needs no change: it
  returns `False` for equal-length paths, so the "cannot move into its own subdirectory" check does not
  fire first. This also makes the exact-identical `pxt.move('d/t','d/t')` a no-op where it raises
  today; after folding the two cases are the same comparison.
- **`rename_column`.** `TableVersion.rename_column` has no identity check at all — the existing
  `if new_name in self.cols_by_name` guard fires instead and reports `Column 'a' already exists`,
  which is baffling for someone who typed `'A'`. Add the check **after** the existing
  `self.path.get_column(old_name)` lookup and the `col.get_tbl().id != self.id` base-column check, so
  `rename_column('nope','NOPE')` still raises `COLUMN_NOT_FOUND` and a base column is still refused,
  and **before** `bump_version`/`_write_md`, so the no-op creates no new schema version. Leave it
  *after* the `is_mutable` guard too: renaming on a snapshot stays an error whatever the casing.

## 2. Migration — `convert_55.py`

Bump `metadata/__init__.py::VERSION` to 56 and add `@register_converter(version=55)`, following
`convert_54.py`.

### What holds a name

| Location | Field |
|---|---|
| `dirs.md` | `DirMd.name` |
| `tables.md` | `TableMd.name` |
| `tables.md` | `IndexMd.name`, under `index_md[*]` |
| `tableschemaversions.md` | `SchemaColumn.name`, under `columns[*]` — **one row per schema version per table** |
| `tables.md` | the keys of `outputs` in `ViewMd.iterator_call`, for component views |

That last one is the bulk of the work: column names are versioned, so a table with 20 schema versions
has 20 rows each carrying the full column list, and all of them must be folded to keep historical
versions and snapshots loadable.

One thing that looks like it does *not* need rewriting and does: **`ViewMd.iterator_call`**.
`GeneratingFunctionCall.as_dict` persists `'outputs': {name: ...}`, and those keys must match the
view's column names exactly — `exec/component_iteration_node.py` does
`cols_by_name[name] for name in iterator_call.outputs`, so a folded column name against an unfolded
key raises `KeyError` on every query of the view. Fold the keys; leave each entry's `orig_name`
unfolded, since that one names a field the iterator yields at runtime, not a column.

Two things that *look* like they need rewriting and do not:

- **Serialized expressions.** `exprs/column_ref.py::ColumnRef._as_dict` persists `tbl_id`/`col_id`,
  not a name, so computed-column value exprs, view predicates and index `init_args` need no rewriting.
  `ColumnRefByName` is a creation-time placeholder that is substituted before storage
  (`table_version.py`, `catalog.py::create_from_model`) — but the converter should still fold any
  `name` it finds under a persisted `ColumnRefByName`, as cheap insurance, via the `substitution_fn`
  hook in `converters/util.py::convert_table_md`.
- **External-store metadata.** Removed from the schema by `convert_54`, so there is no column-name
  mapping left there.

`convert_table_md` covers `tables.md` (and the legacy `functions` table). It does **not** touch
`dirs` or `tableschemaversions`, so the converter needs direct updates for those — see
`convert_53.py` / `convert_54.py` for the `TableSchemaVersion` pattern.

### Validate first, then write

The converter runs in two passes. The validation pass reads everything and collects **all** problems
before aborting, so a user sees the full list rather than fixing them one at a time.

**ASCII violations** — any `DirMd.name`, `TableMd.name`, `IndexMd.name` or `SchemaColumn.name`
containing a non-ASCII character. Report the containing object and the offending name. This scope is
**everything, with no exclusions** — in particular the dropped-table exclusion below does *not* apply
here. A dropped table's columns are still loaded through `Column.__init__`: the drop defers its work
to pending ops, the first of which (`DeleteTableMediaFilesOp`) has `needs_tv = True` and is loaded with
`check_pending_ops=False`, deliberately bypassing the dropped-table guard, because
`TableVersion.delete_media` derives the storage destinations from the reconstructed `Column` objects.
Skipping such a name would leave a `DROP_TABLE` roll-forward — which `TableStatement.can_abort()`
excludes, so it cannot be rolled back — failing on every attempt.

**Fold conflicts** — three scopes. All of them **exclude dropped tables**: `schema.py` documents
`dir_id: NULL for dropped tables`, and a dropped table keeps its `md.name`, so a long-dropped `Foo`
would otherwise abort the upgrade by colliding with a live `foo` the user cannot even see. Skip rows
with `dir_id IS NULL` or `pending_stmt = DROP_TABLE` when scanning — but still **fold** their names.
That is not defensive: as described above, an interrupted drop is resumed by reloading exactly those
rows, so leaving them unfolded leaves real inconsistency behind, not hypothetical inconsistency.

1. **Directory entries.** Within one parent directory, two entries whose folded names collide. Note
   this is a *single* namespace: `catalog.py::_get_dir_entry` checks subdirectories and then tables, so
   a directory `Foo` and a table `foo` in the same parent are a conflict.
2. **Columns, per schema version.** Within `tableschemaversions.md.columns` for **each** schema
   version — not only the current one, since snapshots and time-travel queries resolve against
   historical versions. Report table id, schema version, and both names. Be explicit in the message
   that a conflict in a historical version can only be resolved by dropping the table or the
   snapshots that reach it — the user cannot rename a column in a past version.
3. **Indexes, per table.** Within `TableMd.index_md`. (Index and column names are separate
   namespaces — `idxs_by_name` vs `cols_by_name` — so a column/index collision is not a conflict.)

**Deliberately not checked: view/base shadowing.** A view column whose folded name collides with an
inherited base column's would, after folding, shadow the base column instead of coexisting with it.
This is *not* a new condition — shadowing is already reachable today (a base table can add a column
whose name a view already uses; nothing rejects it), and the resolution code already accommodates it:
`table_path.py::TableVersionPath.columns` filters base columns "that don't conflict with one of our
column names". Folding only widens which pairs collide. The worst case is bounded — no data is lost,
the base column remains reachable directly, and a view query that referenced it silently resolves to
the view's own column instead. Detecting it would require reconstructing view/base relationships from
raw `ViewMd.base_versions` metadata without the catalog layer, which is disproportionate to the risk.
(Note `func/udf.py` *asserts* column names are unique across a view and its bases — that assert is
already reachable today, independently of this change.)

A shadowing view that came from a **view model** is left in a legal but non-reproducible state: it
migrates and keeps working, and `update_all()` diffs it clean, but `prepare_model` now refuses that
declaration (section 1), so dropping and re-creating it raises `COLUMN_ALREADY_EXISTS`. Accepted — the
alternative is aborting the upgrade over a condition that costs nothing until then. The fix is to
rename the view's own column.

On any violation, raise and leave the database untouched — the converter must not have written
anything at that point.

### Non-blocking, but worth logging

Folding can move a name into the reserved set: a column named `Count` becomes `count`, which
`is_system_column_name` rejects. Existing tables keep loading (the reserved check runs at creation,
not at load), but such a column can no longer be re-created or replaced. Log a warning listing them
rather than failing the migration.

## 3. User-visible changes

- Identifiers are case-insensitive, and **the casing you supply is not retained**. A column created
  as `MyCol` is thereafter `mycol` everywhere it is displayed: `describe()`, `get_metadata()`,
  `columns()`, `collect()` result keys, `to_pandas()` column labels.
- Lookups by any casing keep working — `t.MyCol`, `t['MYCOL']`, `row['MyCol']` all resolve.
- Identifiers must be ASCII.
- Upgrading runs a migration that rewrites existing names and **aborts** if it finds case collisions
  or non-ASCII names, with a list of what to fix.
- `ResultSet.to_pydantic()` keeps working with any field casing. It matches result columns against
  pydantic model **field** names, which are Python attributes and case-sensitive, and result columns
  are now always lowercase — so a model field `MyCol` would stop matching. Fold the comparison
  (`required_fields - col_names`, `col_names - model_fields`) **and** remap each row's keys to the
  model's spelling before the `model(**row)` splat. Folding only the validation is the one outcome to
  avoid: the check would pass and the splat would then fail.
- Client/server version skew: a folded client against an un-migrated remote catalog will fail to
  resolve mixed-case names. Both sides must be on the same metadata version.
- **Error messages echo the folded path, not what was typed.** `pxt.get_table('MyDir/MyTable')`
  reports `Path 'mydir/mytable' does not exist`, because `Path.__str__` renders the stored form. This
  is expected behavior, and matches Postgres, which reports unquoted identifiers folded.
- **The `db` in `pxt://local:<db>` is folded like everything else.** That name is not only a lookup
  key: it maps to the daemon's home directory (`~/.pixeltable/proxy_<db>`, via
  `service/proxy_daemon.py::proxy_home`) and to the Postgres database name (`env.py`). It is folded
  everywhere it enters — in `Path`, in `pxt localproxy <db>`, and in the config `db` key — since
  folding only some of those would make the same database resolve two different ways. A local
  database created with mixed case therefore resolves to a different directory after this change and
  must be recreated. Accepted; consistency with the rest of the path is worth more than preserving
  those installs.

## 4. Commit sequence

Each independently green under `make slimtest`.

| # | Commit |
|---|---|
| 1 | `fold_identifier`; ASCII restriction in `is_valid_identifier`; folded `_PREDEF_SYMBOLS` |
| 2 | Fold in `Path.__post_init__` (covers every path-based API) |
| 3 | Fold column/index entry points (schema dicts, `__getattr__`, rename/drop/recompute, index names, `primary_key`); `rename_column` through `Column.validate_name`; the `move`/`rename_column` no-ops |
| 4 | Fold query/DML entry points (`select` kwargs, `update`, `batch_update`, insert conduit, `Row`/`ResultSet`, `normalize_schema_names`, the `to_pydantic` remapping) |
| 5 | Fold the model API — `TableModelMeta.__prepare__` (the table name and registry), `_ModelNamespace`, `_bind`, `TableModelMeta.__getattr__` — and iterator outputs |
| 6 | Fold serving and CLI entry points, incl. the `db` name in `pxt localproxy` and the config `db` key |
| 7 | `convert_55.py` + `VERSION` 56: validation pass, then fold |
| 8 | Docs |

The migration is the largest and riskiest single piece. Commits 1–6 are independent of 7 and can land
first.

## 5. Verification

`make format && make check && make slimtest` after every commit; `make test` before the PR. The
`make_catalog_path` fixture in `tests/conftest.py` auto-forks across local and proxy, so dir and table
tests written against it cover remote `pxt://` paths for free.

- `tests/test_path.py` — non-ASCII rejected; `Path.parse('A/B').components == ('a','b')`;
  `Path.parse('A/B') == Path.parse('a/b')` with equal hashes; `str()` returns the folded form;
  `pxt://Org:DB/x` folds org and db.
- `tests/test_dirs.py`, `test_catalog.py` — `create_dir('D')` then `get_dir_contents('d')`; creating
  `d` after `D` raises `PATH_ALREADY_EXISTS`; `ls()` shows `d`; `pxt.move('d/t','d/T')` is a no-op
  that raises nothing and leaves `d/t` in place, as is `pxt.move('d/t','d/t')` — and specifically
  **not** `PATH_ALREADY_EXISTS`, which is what the full `_prepare_dir_op` would produce. A missing
  source still behaves like any other: `pxt.move('d/nope','d/NOPE')` raises `PATH_NOT_FOUND`, and
  returns quietly under `if_not_exists='ignore'`.
- `tests/test_table.py`, `test_alter_column.py` — `create_table('T', {'MyCol': ...})` then `t.mycol`,
  `t['MYCOL']`, `insert([{'MYCOL': ...}])`, `update({'MyCol': ...})`, `drop_column('MYCOL')`,
  `recompute_columns(['MYCOL'])`; `{'a':..., 'A':...}` rejected with `INVALID_SCHEMA` naming both
  spellings — assert on **both** `create_table` and `add_columns`, since each builds its own mapping
  and a silent overwrite there would look like success; `get_metadata()['columns']` keyed
  `mycol`; `add_computed_column(MyCol=...)` stores `mycol` and collides with an existing `mycol`
  under each `if_exists` mode, on both `LocalTable` and `TableProxy`;
  `rename_column('a','A')` is a no-op — it raises nothing (**not** `COLUMN_ALREADY_EXISTS`, which is
  what the pre-existing guard produces without the new check) and, critically, leaves
  `get_metadata()['version']` unchanged, while `rename_column('nope','NOPE')` still raises
  `COLUMN_NOT_FOUND`.
- **Reserved names and Python keywords are rejected in every casing** — extend the existing
  `add_column`/`insert` cases at `tests/test_table.py` (around the `'is a reserved name in Pixeltable'`
  assertions, which today cover only the lowercase spellings). Add `Count`, `INSERT`, `Select` for the
  reserved-symbol ban and `Class`, `FOR` for the keyword ban, each expecting `INVALID_COLUMN_NAME`.
  These are the regression tests for a behavior change: all of those are legal column names today.
  Assert the same bans on `rename_column`'s *new* name — `rename_column('x','Count')` and
  `rename_column('x','select')` both raise `INVALID_COLUMN_NAME` — which is the regression test for
  routing it through `Column.validate_name`; `rename_column('x','select')` succeeds today.
- `tests/test_view.py` — a view column `Foo` over a base column `foo` is **allowed** and shadows the
  base column, matching what section 2 permits for the migration. Nothing rejects shadowing today —
  `view.py::View._create` checks only iterator outputs against declared view columns, never declared
  columns against inherited ones — and this design adds no such check. Assert that `v.foo` resolves to
  the view's own column and that `table_path.py::TableVersionPath.columns` no longer lists the base's
  (`c.name not in tv.cols_by_name`), while the base column stays reachable through the base table
  directly. Also `t.join(u, on=u.Foo)` against `t.foo`. View **models** are deliberately stricter —
  see the view-model bullet below.
- `tests/test_component_view.py` — a custom iterator declaring an output `MyOutput` creates the view
  column `myoutput`, **while its `__next__` keeps yielding `'MyOutput'`** and the view still populates
  correctly: the regression test for folding the `outputs` key but not `orig_name`.
  `@pxt.iterator(unstored_cols=['Frame'])` over an output `Frame` still yields an unstored column. An
  iterator output `frame` against a declared view column `Frame` is rejected with `INVALID_SCHEMA`.
- `tests/test_index.py` — `add_embedding_index(idx_name='Idx')` then `drop_index(idx_name='IDX')`;
  `similarity(idx='IDX')`; generated `idx0` does not collide with a user's `IDX0`.
- `tests/test_query.py` — `select(A=..., a=...)` repeated-name error; `select(t.MyCol)` yields key
  `mycol`; `row['MyCol']`, `'MYCOL' in row`, `rs['MyCol']`, `rs[0,'MyCol']` all resolve. On a table
  with a failing computed column and a named embedding index: `row.errors['MyCol']` and
  `row.index_values['MyIdx']` resolve, while iterating either mapping still yields folded keys.
- `tests/test_table_model.py` — the stored column of `MyCol: pxt.Int` is `mycol`, while **the Python
  spellings keep working**: a class-body reference (`doubled = MyCol * 2`) resolves, `M.MyCol` resolves
  both before and after `_bind` and is the *same* ref as `M.mycol` afterwards (the regression test for
  the stale-placeholder divergence), and `M.MYCOL` — a casing never declared — resolves on a bound
  model. A class declaring both `Foo` and `foo` is rejected in **each** declaration form and
  combination: two bare annotations *of the same type* (the case `set_col_type` would otherwise merge
  silently), two assignments, and one of each. A view model redeclaring a base-query or iterator name in
  a different casing is rejected by `_check_reserved`. Two `EmbeddingIndex(name=...)` differing only in
  case are rejected. `update_all()` after re-casing an attribute emits no ops.
- `tests/test_table_model.py`, view models — a view model over a `select(*)` base that declares `Foo`
  against the base's own `foo` raises `COLUMN_ALREADY_EXISTS` naming the base table, **unlike** the
  ordinary `create_view` path above. This is the regression test for the documented TableModel-only
  restriction; write it against a `select(*)` base, since the check is unreachable with an explicit
  select list.
- `tests/test_table_model.py`, table names — a model declared `name='Foo'` creates the table `foo`;
  declaring a second model `name='foo'` raises `INVALID_SCHEMA` naming the first class. On a
  single-model base, `create_all()` returns exactly one diff, keyed `foo`, and does **not** raise
  `ConcurrencyError` — the regression test for the raw-keyed `registered_models`/`expect_created`
  pair.
- **JSON paths stay case-sensitive** — `t.json_col.someField` and `t.json_col.somefield` are distinct;
  this is the regression test for not folding one level too deep.
