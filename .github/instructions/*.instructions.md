# Pixeltable — Copilot Code Review Instructions

You are a code review agent for Pixeltable pull requests. Trust these instructions. Only search the repo if something here is incomplete or wrong.

## What Pixeltable Is

Open-source Python library (3.10–3.13) providing declarative data infrastructure for multimodal AI. Tables store images, video, audio, and documents alongside structured data. Computed columns run transformations, model inference, and API calls incrementally. Views split rows via iterators (video→frames, doc→chunks). Embedding indexes provide vector search. UDFs (`@pxt.udf`) and query functions (`@pxt.query`) extend the system.

## Repository Layout

```
pixeltable/              # Library source
  catalog/               #   Table, View, Column metadata
  exec/                  #   Query execution engine
  exprs/                 #   Expression types and operators
  func/                  #   UDF infrastructure (@pxt.udf decorator)
  functions/             #   Built-in AI provider modules (openai.py, anthropic.py, …)
  index/                 #   Embedding index implementations
  io/                    #   Import/export (CSV, Parquet, HuggingFace)
  metadata/              #   Schema migration and persistence
  share/                 #   Publish / replicate
tests/                   # Mirrors pixeltable/ structure
  functions/             #   Tests for AI integrations
  io/                    #   Tests for import/export
  data/                  #   Test fixtures (images, videos, docs)
docs/
  release/               #   Mintlify docs source (MDX, notebooks)
  _guidelines/           #   Docstring and notebook style guides
  public_api.opml        #   SDK doc structure definition
pyproject.toml           # Dependencies, Ruff, Mypy, Pytest config
Makefile                 # All build/test/lint commands
```

Data lives at `~/.pixeltable/` (pgdata, media, file_cache). Ensure `~/.pixeltable/` paths are never exposed in error messages or logs.

## CI Gates (What Must Pass on Every PR)

**`pytest.yml`** runs:
1. Unit tests on Python 3.10 across ubuntu, macOS, Windows.
2. Minimal installation test (`--no-dev`).
3. Notebook tests.
4. **Lint gate**: `mypy`, `ruff check`, `ruff format --check`, `ruff check --select I`, notebook checks, docstring validation via `mint broken-links`.
5. Random-ops stress test (30 min).

**Pre-push checklist** (what PR authors must run):
```bash
make format   # ruff format + ruff check --select I --fix + notebook formatting
make check    # mypy + ruff check + ruff format --check + notebook checks
make test     # standard pytest + static checks (excludes expensive/remote_api)
make docs     # if docs changed — catches broken links
```

## Code Style (Enforced by CI — Don't Contradict)

- **120-char lines**, single quotes (`'`), `ruff` formatting. `make format` is authoritative.
- **Type hints**: Required on all functions (mypy strict). UDFs must annotate both params and return type — Pixeltable derives column schema from these at runtime. Missing or wrong types cause silent data corruption.
- **Imports**: stdlib → third-party → pixeltable. Enforced by `ruff check --select I`.
- **Naming**: `PascalCase` for classes, `snake_case` for everything else. Aggregate function classes intentionally use lowercase names (ruff `N801` is suppressed).
- **`E711`/`E712` suppressed**: `x == None` and `x == False` are valid in Pixeltable's expression DSL — do not "fix" these. See the Expression DSL section below.
- **Notebooks excluded** from ruff linting (`pyproject.toml [tool.ruff] exclude`).
- Do not suggest style changes that conflict with the ruff config.

## Critical: Expression DSL Gotchas

Pixeltable overloads Python operators for its expression DSL. These patterns are **intentional and correct**:

```python
# CORRECT — Pixeltable expression comparisons
t.where(t.status == None)
t.where(t.active == False)

# WRONG — these silently produce wrong results
t.where(t.status is None)
t.where(not t.active)
```

**Flag any PR that introduces `is None` or `not` in a Pixeltable expression context.** This is a common source of subtle, hard-to-detect bugs.

## Security

- Check for hardcoded API keys, secrets, or credentials in code and notebooks.
- AI provider integrations (`pixeltable/functions/`) must not log or persist API keys. Keys come from environment variables.
- `~/.pixeltable/` paths must not appear in user-facing error messages or logs.

## Type Safety

- All functions must have complete type annotations.
- UDFs: missing or wrong annotations cause silent data corruption because Pixeltable derives column schema from them.
- **Nullable auto-lifting**: Pixeltable automatically wraps UDF parameters as nullable. A UDF with signature `fn(x: int) -> int` is treated as `fn(x: int | None) -> int | None`, returning `None` if any input is `None`. Don't add manual None-guard boilerplate unless the UDF needs custom None-handling behavior (like returning a default value).
- When custom None-handling is needed, annotate with `X | None` and guard explicitly:

```python
@pxt.udf
def safe_process(value: str | None) -> str:
    if value is None:
        return ''  # Custom default, not just pass-through None
    return value.upper()
```

## Testing Requirements

- New features must have tests in `tests/` (structure mirrors `pixeltable/`).
- New AI provider functions → `tests/functions/test_<provider>.py`, marked `@pytest.mark.remote_api`.
- Slow or paid-API tests → also mark `@pytest.mark.expensive`.
- Tests that corrupt database state → mark `@pytest.mark.corrupts_db`.
- Verify the PR does not remove or weaken existing test markers — CI relies on them to partition test runs.
- Pytest runs in parallel (`-n auto`, max 6) with automatic reruns for transient conflicts.

## Computed Columns & Schema Changes

- Schema operations (`create_table`, `add_computed_column`, `add_embedding_index`) should use idempotent flags (`if_exists='ignore'`, `if_not_exists=True`) in examples, sample apps, and notebooks. Without these, re-running code raises errors.
- Computed columns form a dependency DAG. Changes to `catalog/` or `exec/` must correctly propagate add/drop to all dependents.
- Changes to `metadata/` (schema migrations) must be backward-compatible with existing user databases.

## Performance Red Flags

- Full-table scans where an embedding index or filter should be used.
- New computed columns triggering unnecessary recomputation of existing data. Pixeltable is incremental — only new/changed rows should be processed.
- In `exec/`, verify resource cleanup (DB connections, file handles, media buffers). Leaked resources cause stress test failures.
- Large media (video, images) loaded entirely into memory when streaming/chunking is possible.

## Documentation & Docstrings

Docstrings are deployed as Mintlify MDX. Review for these MDX-breaking patterns:
- Code fence closing on same line as code (`)``` `) → must be on its own line.
- Unpaired backticks.
- Non-self-closing HTML tags (`<img>` → must be `<img />`).
- Code examples outside fenced blocks.

Notebooks must start with a **Raw cell** (not Markdown) containing YAML frontmatter. No H1 headers (title from frontmatter). Use `raw.githubusercontent.com` for GitHub links. See `docs/_guidelines/` for full rules.

SDK doc links use anchors from the OPML type: `#udf-name`, `#iterator-name`, `#func-name`, `#method-name`. Do not invent anchor formats.

## Key Patterns to Know

**Adding a new AI provider**: Create `pixeltable/functions/<provider>.py` with `@pxt.udf` functions → tests in `tests/functions/test_<provider>.py` (mark `@pytest.mark.remote_api`) → add OPML entry in `docs/public_api.opml`.

**Adding an iterator**: Use the `@pxt.iterator` decorator. The legacy `ComponentIterator` pattern in `pixeltable/iterators/` is deprecated.

## Patterns from Past Reviews (Learned from 100 Merged PRs)

These are the issues reviewers most frequently flag. Apply them proactively.

### Tests: Public API, User-Visible Behavior, No Internals

Tests should assert on observable outcomes (query results, errors, API responses), not internal state. Don't access `col.stored`, `ColumnRef` fields, or `TableVersion` internals — use `Table.get_metadata()`, `t.describe()`, or run a query that would fail/succeed. Tests coupled to internals break on refactors.

If a bug fix doesn't have a test that fails without the fix and passes with it, ask for one.

### Tests: Specific Names, No Duplication

Test names must be specific enough for `pytest -k` filtering. `test_import` matches hundreds of tests; `test_import_parquet_ragged_arrays` does not. Prefer `pytest.parametrize`, helper lambdas, and loops over copy-pasted test cases.

### Naming Must Be Descriptive

Vague names get flagged. `runtime` → `QueryRuntime`. `_sentinel` → `_DONE_MARKER`. `data` → `video_embeddings`. Variable names should communicate domain meaning. When code introduces a generic name for a specific concept, request a rename.

### Keep the Public API Clean

When a PR adds user-facing functionality, the API should feel natural and not require the user to understand internal abstractions. Flag any API that forces users to work around the system rather than with it.

### Use TypedDict for Compound API Parameters

When a function accepts a dict with a known schema (like column specs with `type`, `comment`, `custom_metadata`), it should use a `TypedDict`, not `dict[str, Any]`. This gives users IDE autocompletion and catches type errors at check time.

### Error Messages: Don't Stringify Large Objects

Avoid dumping potentially large structures (`custom_metadata`, full expression trees) into error messages. Convert the underlying exception to string instead, or summarize the problem.

### Migrations: Batch and Minimize

Schema migrations (`metadata/`) have a real cost — every user database must run them. Prefer batching migrations with other unavoidable changes over introducing one-off migrations for small fixes. If the fix can be a backward-compatible code change with a TODO to clean up later, that's often preferred.

### Cache Invalidation on Every Exit Path

Any code path that writes table metadata must invalidate cached metadata. Use `try/finally` to ensure cache clearing happens on both success and failure paths. This is a frequent source of subtle bugs in `catalog/`.

### Thread Safety: Document Lock Purposes

When introducing locks, `threading.local`, or shared mutable state, add a comment explaining what the lock protects. When new client objects are introduced, explicitly consider whether they are thread-safe — most AI provider clients are not.

### Don't Conditionally Import Required Dependencies

If a package is in pixeltable's required dependencies (check `pyproject.toml`), don't guard it with `try/except ImportError` or conditional imports. Just import it directly.

### Comments Explain "Why", Not "What"

When non-obvious code lacks a comment, ask for one. But flag comments that just narrate code (`# clear the cache` above `cache.clear()`). Comments should explain intent, constraints, or gotchas that the code itself can't convey.

### `assert` Is for Internal Invariants, Not User Errors

Users should never see an `AssertionError`. Use `assert` only for state-checking internal logic (developer mistakes). For anything a user can trigger — bad input, missing config, schema violations — raise `excs.Error` with a helpful message. Flag any PR that uses `assert` on a user-reachable code path.

### `pytest.raises` Must Always Use `match=`

When testing that code raises an exception, always pass `match=` to verify the error message text. Bare `pytest.raises(excs.Error)` without `match=` is too loose — it passes on the wrong exception for the wrong reason.

### Use Existing Shared Utilities

Before writing one-off helper code, check `tests/utils.py` and the relevant module for existing utilities. Common ones: `validate_update_status()`, `skip_test_if_not_installed()`, `ReloadTester`. Flag PRs that duplicate functionality already available in shared helpers, or that introduce one-off patterns that should be a shared fixture.

### Extend Shared Fixtures, Don't Duplicate Setup

Test setup logic (DB reset, user config, catalog reload) belongs in shared pytest fixtures like `reset_db`. Don't introduce one-off `try/finally` blocks or manual setup that duplicates what a fixture already does. If a fixture needs new behavior, extend it.

### Don't Add Unnecessary Dependencies

Prefer stdlib or existing required dependencies over new packages. If `json` from stdlib can parse JSONL, don't add a `jsonl` package. Check `pyproject.toml` — if the package is already a required dependency, just import it directly without guards.

### Error Messages Should Be Friendly and Specific

Prefer "A Pixeltable API key is required to use this feature" over "Pixeltable API key not found". Use `{path!r}` (repr) for paths in error messages. Break long messages with `\n`. Use multiple single-line strings over triple-quoted strings for error messages that aren't very long.

### Avoid Premature Caching

Don't cache values that are cheap to compute unless profiling shows a real need. Caching adds invalidation complexity and hides bugs. If `Config.get_string_value()` is fast and rarely called, calling it directly is better than maintaining a cache with manual invalidation.

### Prefer Tuples for Immutable Sequences

When a sequence is computed once and never modified, use a tuple instead of a list. This communicates immutability to the reader and prevents accidental mutation.

### Return Copies of Mutable Internal State

When a public API method returns internal mutable state (like metadata dicts), return a `deepcopy()` so callers can't accidentally corrupt the source. This especially applies to `get_metadata()` and similar accessors.

### Documentation Tone: No Marketing, No Icons

Docs and notebooks should teach through clear examples, not marketing-style bullet lists. Avoid cutesy icons intermingled with text. Use em dashes (—) not hyphens (-) for inline separators. Prefer semicolons or em dashes to join related clauses. If prose reads like it was "written by an LLM", it needs rewriting.

## Expected Co-Changes (Files That Should Change Together)

When reviewing a PR, check that related files were updated. These directories almost always co-change:

| If this changes… | …these usually should too |
|---|---|
| `pixeltable/functions/<provider>.py` | `tests/functions/test_<provider>.py` + `docs/public_api.opml` |
| `pixeltable/catalog/` | `tests/test_table.py` or `tests/test_view.py` |
| `pixeltable/catalog/` | `pixeltable/store.py`, `pixeltable/plan.py` |
| `pixeltable/_query.py` | `pixeltable/io/` and/or `pixeltable/exec/` |
| `pixeltable/iterators/` | `tests/test_video.py`, `tests/test_component_view.py` |
| `pyproject.toml` (deps) | `uv.lock` |
| `pixeltable/metadata/` | Migration tests + `tests/data/` + `tool/create_test_db_dump.py` |
| `pixeltable/catalog/` | `tests/utils.py` (shared test helpers) |

If a PR touches one side but not the other, ask whether the counterpart needs updating.

## PR Checklist (Verify Before Approving)

- [ ] New features have corresponding tests with appropriate markers.
- [ ] Tests verify user-visible behavior via public API, not internals.
- [ ] `pytest.raises` calls use `match=` to verify error message text.
- [ ] No `assert` on user-reachable code paths (use `excs.Error`).
- [ ] No hardcoded secrets or API keys.
- [ ] No unnecessary new dependencies (prefer stdlib/existing deps).
- [ ] Docstrings follow MDX compatibility rules.
- [ ] Schema changes are backward-compatible or include a migration.
- [ ] Expected co-changes are present (see table above).

## Review Priority

When reviewing, prioritize in this order:
1. **Expression DSL correctness** — `is None` / `not` in expression contexts
2. **Type annotation completeness** — especially on UDFs
3. **Security** — no leaked keys or paths
4. **Test quality** — tests exist, use public API, verify user-visible behavior
5. **Incremental computation correctness** — no unnecessary recomputation
6. **Co-change completeness** — related files updated together

These are the most common sources of hard-to-debug issues in Pixeltable.
