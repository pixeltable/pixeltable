When reviewing Pixeltable PRs, apply these project-specific rules. Pixeltable is a Python library (3.11–3.14) for declarative multimodal AI data infrastructure. UDFs (`@pxt.udf`) and query functions (`@pxt.query`) extend the system.

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

## Type Safety

- All functions need complete type annotations (mypy strict).
- UDF annotations are critical: Pixeltable derives column schema from them. Wrong types → silent data corruption.
- Pixeltable auto-lifts nullable params: `fn(x: int)` becomes `fn(x: int | None) -> int | None`. Don't add unnecessary None-guard boilerplate. Use `X | None` (not `Optional[X]`) when custom None-handling is needed.
- Skip all comments regarding potential mypy errors, type hints, or type-checking issues.
- Assume that type safety is already validated by our CI pipeline via mypy.

## Protected Configuration

- `integrations.telemetry.enabled` in `docs/release/docs.json` must stay `true`. Flag any PR that changes or removes it.
- Nothing may write to `~/.pixeltable/` directly. Go through the SDK.

## Security

- No hardcoded API keys in code or notebooks. Keys come from env vars.
- `~/.pixeltable/` paths must not appear in user-facing error messages.
- Pixeltable reads the process environment only. It never loads `.env`, and `python-dotenv` is not a dependency. Flag any doc or example telling a reader to put a key in `.env` without also saying to source or export it.

## Comments (Most Frequent Human-Review Complaint)

Roughly 18 of the last 100 PRs drew a maintainer comment about this. It outranks every other review theme.

- Delete a comment that restates the identifier, signature, or the line below it. Do not reword it.
- A comment states the function's own contract. It must not narrate the caller, the callee's internals, the change history, or which test exercises the code.
- When behavior changes, every comment, docstring, `help=` string, console message, and `docs/release/**` page stating the old behavior changes with it. Flag any survivor.
- Do not accept a comment asserting behavior the adjacent code does not have, especially atomicity and ordering claims. Check the transaction boundary before believing "all-or-nothing".
- Match the file's existing comment style rather than introducing backticks or prose it does not use.

## Code Style (Don't Contradict)

120-char lines, single quotes, `ruff` formatting. `make format` is authoritative. Don't suggest style changes conflicting with ruff config. `PascalCase` for classes, `snake_case` for everything else (aggregate classes use lowercase; N801 suppressed).

## Testing

- New features need tests in `tests/` (mirrors `pixeltable/` structure).
- AI provider tests → `tests/functions/test_<provider>.py`, marked `@pytest.mark.remote_api`.
- any test for a pxt.Error or one of its subclasses needs to use pxt_raises() instead of pytest.raises()
- pxt_raises() (and pytest.raises(), if justified)  must always use `match=` to verify error text.
- Tests must assert on user-visible behavior via public API, not `col.stored`, `ColumnRef`, or `TableVersion` internals. Use `Table.get_metadata()`, `t.describe()`, or queries.
- Test names must be specific for `pytest -k` filtering. Prefer `pytest.parametrize` over duplication.
- Use shared utilities (`validate_update_status()`, `skip_test_if_not_installed()`, `ReloadTester`). Extend shared fixtures, don't duplicate setup.
- A test takes `db_root: DatabaseRoot` and builds paths with `db_root.make_catalog_path(...)`. Never hardcode a catalog path. 586 tests already do this.
- A test using `uses_db` needs `@pytest.mark.db_roots('local', reason='...')`; `tests/conftest.py` raises a `UsageError` at collection without it. The reason names a PXT ticket, not a bare TODO. Delete the exclusion when the ticket lands.
- Never dodge one backend with a bare `@pytest.mark.skip`. Scope it with `db_roots`. Register any new marker in `pyproject.toml`.
- Tests hitting a third-party model or service need `very_expensive`, not just `remote_api` or `expensive`.
- No `http://` or `https://` literals in tests. Use the `sample_file_server` fixture, which serves the repo tree over localhost and still exercises the download path.
- A `skipif`/`xfail` on a test parametrized over catalog mode disables every variant. Gate inside the test body instead when the reason names one parametrization.
- Prefer extending an existing parametrized test over adding a bespoke one beside it.

## Error Handling

- `assert` is for internal invariants only. User-reachable paths must raise `excs.Error` or one of its subclasses.
- make sure that the pxt.Error subclass and error code matches the actual error being signalled.
- Error messages should be friendly and specific. Don't stringify large objects.

## Schema & Migrations

- Schema ops in examples/notebooks must use `if_exists='ignore'` / `if_not_exists=True`.
- Computed columns form a DAG: changes to `catalog/` must propagate correctly.
- Migrations (`metadata/`) must be backward-compatible. Prefer batching over one-off migrations.
- Cache invalidation: any code writing table metadata must clear cache via `try/finally`.

## Performance

- No full-table scans where indexes/filters should be used.
- Pixeltable is incremental: only new/changed rows should be processed.
- In `exec/`, verify resource cleanup. Large media must stream, not load into memory.
- Release resources explicitly; never rely on GC. A handle from a C-backed library (pdfium, PIL, sockets, async clients) belongs in `with`/`closing()`, and its owner needs a `close()`. Four separate leaks were fixed this way.
- No unbounded `join()`, `result()`, or `wait()` in a shutdown path. Give it a timeout.

## Hosted Execution

Code under `pixeltable/serving/`, `pixeltable/service/`, or `pixeltable_cli/server/` runs on a hosted pod as well as locally. Four distinct bugs came from forgetting this.

- Do not assume a local catalog. `col.col` resolves through it and is empty for a hosted table; use `col.col_md.name`.
- Do not assume a local filesystem. Pod storage is ephemeral; media belongs in the home bucket, and results return presigned HTTP URLs, never a raw `pxtfs://` or `s3://` URI.
- Do not assume the request path. The gateway strips the service prefix, so URLs need `root_path`.
- Any traversal rooted at `Config.get().project_root` must exclude the Python environment. `.gitignore` is not enough; an in-project `.venv` caused two regressions.
- No `assert` in a request handler. It vanishes under `-O`.

## Documentation

- Docstrings deploy as Mintlify MDX.
- Always use >>> prompts for code blocks in docstrings. Never use python fences. (Other fences such as bash or json are fine.)
- Check: fenced blocks on own lines, paired backticks, self-closing HTML.
- Never use double backticks in docstrings. Use single backticks with inline code or triple backticks for fenced blocks.

## Prose (Applies to MDX, Notebooks, READMEs, Docstrings)

No CI job checks prose, so these are only caught in review.

- No em dashes (U+2014). Use a period, a colon, or a comma. ASCII `-` for empty placeholders.
- Name the command and say what it does: `pxt schema update` creates tables and does not start HTTP; `pxt service update` starts HTTP and does not create tables. Do not label the loop Declare / Experiment / Serve / Pack on a user-facing page.
- One name per idea. "Application file", "schema file", and "the file" are not three objects.
- No emojis unless asked for.

## Notebooks

- Exactly one title source: either a raw cell with YAML frontmatter, or a leading H1 that Quarto converts. Flag a notebook carrying both, which renders a double title. Do not flag a leading H1 on its own; 93 of 100 notebooks use one.
- Code cells format at line length **74**, not 120 (`scripts/check-notebooks.sh`). The 120-char rule is for `.py` files.
- At least 50% of code cells must have outputs (`tool/check_notebooks.py`). Never advise clearing all outputs.
- Markdown cells must be `nbqa mdformat` clean. Use `raw.githubusercontent.com`, never `raw.github.com`.
- No badge images in markdown cells. Kaggle/Colab/download links belong in the frontmatter `description`.

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
| The `app.py` example | Its copies in `README.md`, `AGENTS.md`, `docs/release/skill.md`, `quick-start.mdx`, `cloud.mdx` |

## Current API (Citing a Stale One Wastes Review Time)

- Non-nullable is the default. `col: pxt.String` is NOT NULL; nullable is `pxt.String | None`. `pxt.Required` is deprecated.
- Indexes are declared in `__indexes__` as `pxt.EmbeddingIndex(Col, ...)` / `pxt.BtreeIndex(Col)`. `TableMetadata.indices` is now `.indexes`.
- The nomenclature is "operational" and "data versioned". `is_versioned` and `unversioned` are gone.
- `pxt serve` does not exist. `pxt service run` serves locally and cannot target Cloud; `pxt service update` starts HTTP. `pxt service` serves `FastAPIRouter` objects only and rejects a bare `fastapi.FastAPI`.
- Minimum Python is 3.11. Use `X | None`, never `Optional[X]`.
- Send `e.message` to a client, not `str(e)`, which appends the detail. `Error.detail` does not cross to a remote client.

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
