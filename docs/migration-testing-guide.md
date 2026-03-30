# Metadata Migration Testing Guide

This guide explains how Pixeltable's metadata migration system works and how to validate new migrations.

## Overview

Pixeltable stores table and column metadata as JSONB in PostgreSQL. When the metadata schema changes, a **converter** upgrades old metadata to the new format. Each converter is registered against a version number and runs sequentially during startup.

## Key Files

| File | Purpose |
|------|---------|
| `pixeltable/metadata/__init__.py` | `VERSION` constant and `upgrade_md()` loop |
| `pixeltable/metadata/notes.py` | Human-readable description per version |
| `pixeltable/metadata/converters/convert_NN.py` | Converter from version NN to NN+1 |
| `pixeltable/metadata/converters/util.py` | `convert_table_md()` helper and patterns |
| `pixeltable/metadata/schema.py` | Dataclasses (`TableMd`, `ColumnMd`, etc.) and SQLAlchemy models |
| `tool/create_test_db_dump.py` | Creates a fresh database and exports a `pg_dump` |
| `tests/test_migration.py` | Loads every dump, migrates it, and verifies correctness |
| `tests/data/dbdumps/` | Gzipped `pg_dump` files and `.toml` info files |

## How a Migration Runs

1. On startup, `upgrade_md()` acquires an exclusive lock on `SystemInfo`.
2. It reads the stored `schema_version` and compares it to the installed `VERSION`.
3. For each gap, it calls the registered converter: `converter_cbs[old_version](engine)`.
4. After all converters run, `SystemInfo.schema_version` is updated to `VERSION`.

## Converter Patterns

`convert_table_md()` in `util.py` iterates over every row in the `tables` metadata table and offers several hooks:

| Hook | Signature | Use Case |
|------|-----------|----------|
| `substitution_fn` | `(key, value) -> (key, value) \| None` | Recursive find-and-replace in nested metadata |
| `table_md_updater` | `(table_md: dict, table_id: UUID) -> None` | Modify the `TableMd` dict in place |
| `column_md_updater` | `(column_md: dict) -> None` | Modify each `ColumnMd` dict in place |
| `table_modifier` | `(conn, tbl_id, orig_md, updated_md) -> None` | Execute SQL against storage tables (called **after** metadata is persisted) |

### Execution Order Inside `convert_table_md()`

For each table row:
1. Deep-copy the metadata dict.
2. Apply `table_md_updater`, `column_md_updater`, `external_store_md_updater`, `substitution_fn`.
3. If the dict changed, write it back to the database.
4. Call `table_modifier` (with a live `Connection` inside the same transaction).

**Important**: If `table_modifier` needs to change metadata (e.g., to undo a failed operation), it must issue its own `UPDATE` statement since the metadata was already persisted in step 3.

### Using Savepoints

When a `table_modifier` performs an operation that may fail (e.g., creating a unique index on data that might have duplicates), use PostgreSQL savepoints to prevent the failure from aborting the entire migration transaction:

```python
conn.execute(sql.text('SAVEPOINT my_savepoint'))
try:
    conn.execute(sql.text('CREATE UNIQUE INDEX ...'))
    conn.execute(sql.text('RELEASE SAVEPOINT my_savepoint'))
except Exception:
    conn.execute(sql.text('ROLLBACK TO SAVEPOINT my_savepoint'))
    # Handle failure: update metadata, log warning, etc.
```

## Adding a New Migration

### Step 1: Write the Converter

Create `pixeltable/metadata/converters/convert_NN.py` (where NN is the **current** `VERSION`):

```python
import sqlalchemy as sql
from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md

@register_converter(version=NN)
def _(engine: sql.engine.Engine) -> None:
    convert_table_md(engine, table_md_updater=_update, table_modifier=_modify)

def _update(table_md: dict, table_id) -> None:
    # Modify metadata dict in place
    ...

def _modify(conn, tbl_id, orig_md, updated_md) -> None:
    # Execute SQL against storage tables
    ...
```

### Step 2: Bump the Version

In `pixeltable/metadata/__init__.py`, increment `VERSION` from NN to NN+1.

### Step 3: Add Version Notes

In `pixeltable/metadata/notes.py`, add an entry for version NN+1.

### Step 4: Update the Dump Script

In `tool/create_test_db_dump.py`, add any new tables, columns, or data that exercise the new migration path. For migrations that need specific database states (e.g., tables with metadata attributes but no corresponding PostgreSQL objects), create tables normally and then patch the metadata via direct SQL updates.

### Step 5: Generate the Dump

```bash
# Must be run on Python 3.10 (due to UDF pickling compatibility)
rm target/*.dump.gz target/*.toml
python tool/create_test_db_dump.py
mv target/*.dump.gz target/*.toml tests/data/dbdumps/
```

This creates `pixeltable-vNNN-test.dump.gz` and `pixeltable-vNNN-test-info.toml`.

### Step 6: Add Verification Tests

In `tests/test_migration.py`, add a `_verify_vNN()` method and hook it into `test_db_migration()`:

```python
if old_version >= NN:
    self._verify_vNN()
```

The verification method should check:
- Metadata correctness (query `Table.md` directly).
- Physical schema state (query `pg_indexes`, `information_schema.columns`, etc.).
- Functional correctness (use the Pixeltable SDK to query the migrated tables).

### Step 7: Run the Tests

```bash
make format   # auto-format code
make check    # mypy + ruff static checks
make test     # run the test suite (migration test requires Python 3.10)
```

## Dump File Format

Each dump consists of two files in `tests/data/dbdumps/`:

- `pixeltable-vNNN-test.dump.gz` — gzipped `pg_dump -Fc` output
- `pixeltable-vNNN-test-info.toml` — metadata:

```toml
[pixeltable-dump]
metadata-version = NNN
git-sha = "abc123..."
datetime = 2026-03-30T12:00:00+00:00
user = "developer"
```

## Common Pitfalls

1. **`table_modifier` vs `table_md_updater`**: Use `table_md_updater` for pure metadata changes. Use `table_modifier` when you need SQL access to storage tables. Don't confuse them — `table_md_updater` cannot execute SQL.

2. **String column keys in metadata dicts**: JSON only allows string keys. Column IDs in `column_md` are string keys (`"5"`), but `indexed_col_ids` in `PrimaryIndexMd` are integers. Always use `str(col_id)` when looking up columns.

3. **Python version for dumps**: Dumps must be created on Python 3.10 due to UDF pickling incompatibilities across versions.

4. **Idempotent converters**: Converters may run on dumps from many different versions. Check for pre-existing state (e.g., `IF NOT EXISTS`, checking `pg_indexes`) before making changes.

5. **Savepoints for fallible operations**: If a converter operation might fail on some tables but not others, use savepoints instead of letting the exception abort the entire migration.
