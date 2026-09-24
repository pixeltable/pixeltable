When reviewing Pixeltable PRs, apply these project-specific rules. Pixeltable is a Python library (3.11–3.14) for declarative multimodal AI data infrastructure. UDFs (`@pxt.udf`) and query functions (`@pxt.query`) extend the system.

Path-specific rules live in `.github/instructions/`: `tests.instructions.md`, `docs.instructions.md`, `serving.instructions.md`. They apply on top of this file.

## Expression DSL: Most Common Bug Source

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

## Comments (Most Frequent Human-Review Complaint)

Roughly 18 of the last 100 PRs drew a maintainer comment about this. It outranks every other review theme.

- Delete a comment that restates the identifier, signature, or the line below it. Do not reword it.
- A comment states the function's own contract. It must not narrate the caller, the callee's internals, the change history, or which test exercises the code.
- When behavior changes, every comment, docstring, `help=` string, and console message stating the old behavior changes with it. Flag any survivor.
- Do not accept a comment asserting behavior the adjacent code does not have, especially atomicity and ordering claims. Check the transaction boundary before believing "all-or-nothing".
- Match the file's existing comment style rather than introducing backticks or prose it does not use.

## Protected Configuration

- `integrations.telemetry.enabled` in `docs/release/docs.json` must stay `true`. Flag any PR that changes or removes it.
- Nothing may write to `~/.pixeltable/` directly. Go through the SDK.

## Type Safety

- All functions need complete type annotations (mypy strict).
- UDF annotations are critical: Pixeltable derives column schema from them. Wrong types → silent data corruption.
- Pixeltable auto-lifts nullable params: `fn(x: int)` becomes `fn(x: int | None) -> int | None`. Don't add unnecessary None-guard boilerplate.
- Skip all comments regarding potential mypy errors or type hints. CI validates this via mypy.

## Security

- No hardcoded API keys in code or notebooks. Keys come from env vars.
- `~/.pixeltable/` paths must not appear in user-facing error messages, including in `dashboard/src/`.

## Code Style (Don't Contradict)

120-char lines, single quotes, `ruff` formatting. `make format` is authoritative. Don't suggest style changes conflicting with ruff config. `PascalCase` for classes, `snake_case` for everything else (aggregate classes use lowercase; N801 suppressed).

## Error Handling

- `assert` is for internal invariants only. User-reachable paths must raise a subclass of `excs.Error`; `Error` itself asserts on construction.
- The `pxt.Error` subclass and error code must match: the code determines the class, and a mismatch asserts. `UserError` carries `GENERIC_USER_ERROR` when nothing more specific fits.
- Error messages should be friendly and specific. Don't stringify large objects.

## Schema & Migrations

- Computed columns form a DAG: changes to `catalog/` must propagate correctly.
- Migrations (`metadata/`) must be backward-compatible. Prefer batching over one-off migrations.
- Cache invalidation: any code writing table metadata must clear cache via `try/finally`.

## Performance

- No full-table scans where indexes/filters should be used.
- Pixeltable is incremental: only new/changed rows should be processed.
- Release resources explicitly; never rely on GC. A handle from a C-backed library (pdfium, PIL, sockets, async clients) belongs in `with`/`closing()`, and its owner needs a `close()`. Four separate leaks were fixed this way.

## Co-Changes (Flag if Missing)

| Changed | Should also change |
|---|---|
| `pixeltable/functions/<provider>.py` | `tests/functions/test_<provider>.py` + `docs/public_api.opml` + `docs/release/howto/providers/working-with-<provider>.ipynb` |
| A provider model ID string | The same ID in the docstring example, the test, and the provider notebook |
| New public SDK surface | `docs/public_api.opml` + `docs/release/docs.json` navigation |
| `pixeltable/catalog/` | `tests/test_table.py` or `tests/test_view.py` |
| `pixeltable/catalog/model/` | `tests/test_table_model.py` (and `test_concurrent_model.py` for locking) |
| `pixeltable_cli/client/commands/` | `tests/pixeltable_cli/` + `docs/release/platform/cli.mdx` |
| `pixeltable/serving/_fastapi.py` | `tests/serving/test_fastapi.py` |
| `pixeltable/config.py` (new key) | `tests/conftest.py` + `docs/release/platform/configuration.mdx` |
| A key in a dict returned to the dashboard | `dashboard/src/types/` + its consumers. Grep the old key across `dashboard/src/` |
| `pixeltable/metadata/` | Migration tests + `tests/data/` + `tool/create_test_db_dump.py` |
| `tests/data/dbdumps/*-info.toml` | Regenerate the matching `.dump.gz` in the same commit |
| `pyproject.toml` (deps) | `uv.lock` |
| The `app.py` example | Its copies in `README.md`, `AGENTS.md`, `quick-start.mdx`, `cloud.mdx` |

## Current API (Citing a Stale One Wastes Review Time)

- Non-nullable is the default. `col: pxt.String` is NOT NULL; nullable is `pxt.String | None`. `pxt.Required` is deprecated.
- Indexes are declared in `__indexes__` as `pxt.EmbeddingIndex(Col, ...)` / `pxt.BtreeIndex(Col)`. `TableMetadata.indices` is now `.indexes`.
- The nomenclature is "operational" and "data versioned". `is_versioned` and `unversioned` are gone.
- `pxt serve` does not exist. `pxt service run` serves locally and cannot target Cloud; `pxt service update` starts HTTP. `pxt service` serves `FastAPIRouter` objects only.
- Minimum Python is 3.11. Use `X | None`, never `Optional[X]`.

## Before Raising a Comment

Maintainers have said Copilot review "cost me a lot of time". False positives are expensive here.

- Verify a runtime-introspection claim before asserting it. A confident wrong claim about `typing.get_origin` burned a maintainer.
- Do not flag a `TableMd` field rename as breaking without first checking `pixeltable/metadata/converters/` and `metadata.VERSION`. This field is pre-launch and renames have shipped deliberately without converters.
- Do not restate a rule the diff already follows, and do not raise style that `ruff` or `mypy` already gates.
- Prefer one specific, checkable comment over several speculative ones.

## Review Priority

1. Expression DSL correctness (`is None`/`not` in expressions)
2. Stale or vacuous comments, and prose that no longer matches the code
3. Protected configuration (telemetry flag flipped)
4. Type annotations (especially UDFs)
5. Security (no leaked keys/paths)
6. Test quality (db_root, markers, public API, match=)
7. Hosted-execution and incremental-computation correctness
8. Co-change completeness
