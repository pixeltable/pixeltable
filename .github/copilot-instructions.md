When reviewing Pixeltable PRs, apply these project-specific rules. Pixeltable is a Python library (3.10–3.13) for declarative multimodal AI data infrastructure. UDFs (`@pxt.udf`) and query functions (`@pxt.query`) extend the system.

## Expression DSL — Most Common Bug Source

Pixeltable overloads Python operators. These are CORRECT (E711/E712 suppressed in ruff):
```python
t.where(t.status == None)   # correct
t.where(t.active == False)  # correct
```
These are WRONG and silently produce incorrect results:
```python
t.where(t.status is None)   # WRONG
t.where(not t.active)       # WRONG
```
Flag any `is None` or `not` in expression contexts.

## Type Safety

- All functions need complete type annotations (mypy strict).
- UDF annotations are critical: Pixeltable derives column schema from them. Wrong types → silent data corruption.
- Pixeltable auto-lifts nullable params: `fn(x: int)` becomes `fn(x: int | None) -> int | None`. Don't add unnecessary None-guard boilerplate. Use `X | None` (not `Optional[X]`) when custom None-handling is needed.

## Security

- No hardcoded API keys in code or notebooks. Keys come from env vars.
- `~/.pixeltable/` paths must not appear in user-facing error messages.

## Code Style (Don't Contradict)

120-char lines, single quotes, `ruff` formatting. `make format` is authoritative. Don't suggest style changes conflicting with ruff config. `PascalCase` for classes, `snake_case` for everything else (aggregate classes use lowercase — N801 suppressed).

## Testing

- New features need tests in `tests/` (mirrors `pixeltable/` structure).
- AI provider tests → `tests/functions/test_<provider>.py`, marked `@pytest.mark.remote_api`.
- `pytest.raises` must always use `match=` to verify error text.
- Tests must assert on user-visible behavior via public API — not `col.stored`, `ColumnRef`, or `TableVersion` internals. Use `Table.get_metadata()`, `t.describe()`, or queries.
- Test names must be specific for `pytest -k` filtering. Prefer `pytest.parametrize` over duplication.
- Use shared utilities (`validate_update_status()`, `skip_test_if_not_installed()`, `ReloadTester`). Extend shared fixtures, don't duplicate setup.

## Error Handling

- `assert` is for internal invariants only. User-reachable paths must raise `excs.Error`.
- Error messages should be friendly and specific. Don't stringify large objects.

## Schema & Migrations

- Schema ops in examples/notebooks must use `if_exists='ignore'` / `if_not_exists=True`.
- Computed columns form a DAG — changes to `catalog/` must propagate correctly.
- Migrations (`metadata/`) must be backward-compatible. Prefer batching over one-off migrations.
- Cache invalidation: any code writing table metadata must clear cache via `try/finally`.

## Performance

- No full-table scans where indexes/filters should be used.
- Pixeltable is incremental — only new/changed rows should be processed.
- In `exec/`, verify resource cleanup. Large media must stream, not load into memory.

## Documentation

Docstrings deploy as Mintlify MDX. Check: code fences on own lines, paired backticks, self-closing HTML, all code in fenced blocks. Notebooks need Raw cell with YAML frontmatter, no H1 headers.

## Co-Changes (Flag if Missing)

| Changed | Should also change |
|---|---|
| `pixeltable/functions/<provider>.py` | `tests/functions/test_<provider>.py` + `docs/public_api.opml` |
| `pixeltable/catalog/` | `tests/test_table.py` or `tests/test_view.py` |
| `pixeltable/metadata/` | Migration tests + `tests/data/` + `tool/create_test_db_dump.py` |
| `pyproject.toml` (deps) | `uv.lock` |

## Review Priority

1. Expression DSL correctness (`is None`/`not` in expressions)
2. Type annotations (especially UDFs)
3. Security (no leaked keys/paths)
4. Test quality (public API, match=, user-visible behavior)
5. Incremental computation correctness
6. Co-change completeness
