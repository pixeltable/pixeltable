# pxt CLI — AI-Agent-First Design

## Goal

A CLI that makes inspection, debugging, and lifecycle ops cheap for an AI agent
writing a Pixeltable + FastAPIRouter app. Code stays the artifact for schema,
UDFs, queries; CLI handles state-of-the-world questions and one-shot ops.

## Non-goals

- No subcommands that create tables, add columns, define UDFs, or otherwise
  duplicate the Python API. Code is the program; the CLI is for inspecting it.
- No interactive REPL. Agents already have Bash; one-shot composable commands
  beat a shell.
- No raw SQL passthrough. Pixeltable's API is the abstraction.

## Design principles

1. **Structured output, always available.** Default tabular for humans, `--json`
   for agents.
2. **Fast cold-start.** Target <50ms feels-instant; <200ms acceptable for v1.
   Achieved via long-running daemon.
3. **Read-only by default.** Mutating commands require `-f` to skip
   confirmation.
4. **Composable.** Output flows into `jq`, `grep`, pipes.
5. **Rooted, no session state.** All paths absolute from catalog root.

## Architecture

### Daemon

Python + FastAPI process. Loads pixeltable once; holds the runtime + catalog
connection; one HTTP endpoint per CLI subcommand.

- Bind localhost only. No auth in v1.
- Port + PID recorded in `~/.pixeltable/cli-daemon.{port,pid}`.
- Auto-shutdown after N minutes idle (default 30).
- Health endpoint reports pixeltable version; client kills+respawns on
  mismatch.

**Daemon scope:** single global. Per-request cwd is supplied by the client
via `X-Cwd` header; the daemon resolves project context from that on each call.
Parallel-safe by construction since no daemon-side cwd state.

### Client

**v1: Python with import discipline.** Lazy-import everything (no top-level
`httpx`/`click`); manual argv dispatch for the first level; `httpx` imported
only after the subcommand demands a network call. `python -S` to skip
site-packages init.

Target: <50ms cold start. Measurement-driven — if real numbers exceed ~100ms
after import audit, switch to Go.

Per-request: client reads `os.getcwd()` and sends it as `X-Cwd` header. Daemon
holds no cwd state; parallel calls are independent by construction.

**v2 (if needed): Go binary** shipped via platform-specific wheels (same model
as `ruff`/`uv`/`black-mypyc`; tooling: `cibuildwheel`). `pip install pixeltable`
picks the right wheel by platform tag. Unsupported platforms fall back to a
Python-only wheel.

Cost of v2: CI multi-arch builds, slightly larger wheels, separate binary-build
step in release. Benefit: ~10ms cold start.

### Daemon lifecycle

- `pxt daemon start` — explicit start (also auto-spawned on first command).
- `pxt daemon stop` — kill daemon.
- `pxt daemon status` — running, PID, port, idle timer, pxt version.
- `pxt daemon restart` — stop + start, after pixeltable upgrades.

## Command surface

### Catalog inspection (read-only)

#### `pxt ls [path] [--tree] [-l] [--json] [--no-counts]`

Default: flat listing rooted at the given path (root if omitted).

Per entry: `name` | `kind` (table/view/dir) | `num_rows`.

- `-l`: long form. Adds `num_cols`, last-modified-version, flag letters
  (`c` = has computed columns, `i` = has embedding indexes).
- `--tree`: hierarchical view, equivalent to `get_dir_tree()`.
- `--no-counts`: skip num_rows (faster on large catalogs).
- `--json`: structured.

#### `pxt describe <table> [--json]`

Default: text output equivalent to `Table.describe()` (reuses existing impl —
captures the print, doesn't reimplement).

`--json`: `table.get_metadata()` verbatim. Canonical structured form. Slice
with `jq`; no in-CLI filters.

#### `pxt head <table> [-n N] [--cols c1,c2,...] [--json]`

First N rows (default 10). Media as paths/URLs.

`--cols`: project a subset.

`--json`: rows as JSON array.

#### `pxt errors <table> [--col NAME] [--full-rows] [--json]`

Rows where any computed column has non-null `errortype`/`errormsg`. Flat
output:

```
rowid | column | errortype | errormsg
```

Columns with zero errors not surfaced. Per-table only (no view-walk in v1).

- `--col NAME`: filter to one column.
- `--full-rows`: bundle the full row alongside each entry.
- `--json`: array of objects.

#### `pxt plan <table> [--json]`

Computed-column DAG + which columns are SQL-pushdown vs Python.

#### `pxt get <table> <pk> [--json]`

Lookup a single row by primary key. `<pk>` is a positional list matching the
PK column order, or `--pk col1=val1,col2=val2` for clarity.

#### `pxt count <table> [--where EXPR] [--json]`

Row count. `--where` supports a limited expression DSL: column-literal
comparisons (`c1 == 42`, `c1 > 'foo'`), `and`/`or`. No UDFs (use Python for
that).

#### `pxt history <table> [-n N] [--json]`

Table version timeline. Default last 20 ops.

### Cross-catalog discovery

These are "find X across the project / catalog" commands that compose with
each other. Honestly the most context-saving commands for an AI.

#### `pxt udfs [--module PATH] [--kind=udf|uda|query|expr_udf] [--json]`

Discovered UDFs in a module or the whole project. Replaces today's only option
(`grep -rn "@pxt\.\(udf\|uda\|query\|expr_udf\)"`).

Per entry: name, kind, signature (typed), defined-at (file:line), docstring
first line.

Project resolution: same as `pxt endpoints`.

#### `pxt indexes [<table>] [--json]`

Embedding indexes. Without a table arg, all of them across the catalog. Per
entry: table, column, embedding fn, similarity metric.

#### `pxt computed [<table>] [--json]`

Computed columns. Without a table arg, all of them. Per entry: table, column,
expression text, dependency column names, SQL-or-Python eval. Useful for
"find all places using `openai.chat_completions`".

#### `pxt endpoints [--app PATH] [--json]`

FastAPI routes a given app config would expose. Per-endpoint: path, method,
query UDF, parameter types, return type.

Project resolution:
1. `--app path/to/app.py` explicit
2. `PXT_APP` env var
3. `pyproject.toml` `[tool.pxt] app = "..."`
4. Default `app.py` in cwd

#### `pxt serve --print-routes [other serve flags]`

Print routes that would be served, without starting the server. Same shape
as `pxt endpoints` but from the actual server-config code path.

#### `pxt logs [-f] [-n N] [--since TIME] [--errors] [--slow MS] [--endpoint PATH] [--recent] [--request ID] [--verbose] [--json]`

Two-source log viewer:
- Per-request entries (high-signal, one JSON line per HTTP request) from
  `~/.pixeltable/serve.log`.
- Pixeltable runtime log (already exists at `~/.pixeltable/logs/...`,
  verbose), filtered by request_id when joining.

Flags:
- `-f`: tail follow.
- `-n N`: last N entries.
- `--since TIME`: relative (`30s ago`, `5m ago`) or absolute ISO.
- `--errors`: status ≥ 400.
- `--slow MS`: latency > MS ms.
- `--endpoint PATH`: filter to a route.
- `--recent`: read from in-memory ring buffer (fast, no disk).
- `--request ID`: show the per-request entry + all matching pxt log lines
  tagged with that request_id. Joins the two log streams.
- `--verbose`: with `--request`, include DEBUG-level pxt log lines (default
  INFO+).
- `--json`: one JSON object per line.

Per-request entry: `{ts, method, path, status, latency_ms, request_id,
route_udf, error_type?, error_msg?}`. Bodies not logged by default
(`PXT_SERVE_LOG_BODIES=1` to opt in).

### Server logging (FastAPIRouter + pxt logger)

For `pxt logs --request <id>` to work, the existing pxt log stream needs to
carry the request_id, and the FastAPIRouter middleware needs to set it.

**Pixeltable logger change** (`pixeltable/env.py` logging setup):
- Define `request_id: ContextVar[str] = ContextVar('request_id', default='')`.
- Add a `logging.Filter` that copies the current value to every `LogRecord`
  as `record.request_id`.
- Extend the format string with `%(request_id)s` (blank for non-request
  contexts).

**FastAPIRouter middleware** (default-on, opt out via `PXT_SERVE_LOG=0`):
- Generate a request_id at request start (ULID or short UUID).
- Set the ContextVar.
- After response, write a JSON line to `$PXT_SERVE_LOG` (default
  `~/.pixeltable/serve.log`), rotated at ~10 MB × 5 files.
- Maintain in-memory ring buffer (last 1000 entries) for `--recent`.

**Executor dispatch:** wrap `executor.submit(...)` calls with
`copy_context().run(...)` so the request_id ContextVar survives the thread
hop into worker threads. ~3 lines.

Net new code: ~70 lines.

### Static analysis

#### `pxt lint <paths>... [--json] [--rules R1,R2,...]`

Catches AI-failure-mode bugs that pixeltable can't see at runtime (or sees too
late). Type/schema checks intentionally omitted — pixeltable already infers
column types from exprs and validates schemas internally; lint shouldn't
duplicate that.

**Idempotency — `PXT0xx`:**
- `PXT001`: `create_table`/`create_view`/`create_dir` with neither
  `if_exists=` nor `if_not_exists=`. The actual AI failure mode is forgetting
  *any* idempotency knob; setup scripts then break on re-run.
- `PXT002`: `if_exists='replace'` in a path that could silently nuke
  production data. Heuristic: flag unless an adjacent marker comment opts in.

**Reproducibility & hygiene — `PXT1xx`:**
- `PXT101`: top-level `import` inside a UDF body (suggest module level
  unless the import is genuinely lazy).
- `PXT102`: `@pxt.udf` likely returning unpicklable type (lambdas, closures
  over non-pickled state).
- `PXT103`: `sample()` in a test file without explicit `seed=`.

**Performance smells — `PXT2xx`:**
- `PXT201`: computed column calls slow external API (openai, anthropic, …)
  without batching enabled.
- `PXT202`: SQL-pushdownable comparison wrapped in a Python UDF.
- `PXT203`: similarity query on a column with no embedding index defined.

**Output:** ruff-style — `path:line:col: PXTxxx description`. `--json` gives
the same data as objects. `--rules`: subset rules to apply (default all).

### State

#### `pxt status [--json]`

One-shot environment view:
- pgdata path, db name, redacted connection string
- file cache size + path
- media dir size + path
- plan cache size (in-daemon)
- recent op log tail
- runtime: pid, started-at

Cache sizes here replace what was originally a `pxt cache` command.

#### `pxt env [--json]`

Pixeltable env vars + active config file. For "why is this picking the wrong
DB" debugging.

### Mutation

All mutating commands require confirmation unless `-f`.

All mutating commands support `-n` / `--dry-run` — prints what would happen
without executing. Combine with `-f` (skip prompts) to script safely.

#### `pxt drop <path> [-f] [--cascade] [-n|--dry-run]`

Drop a table or view.
- `--cascade`: also drop dependent views.
- `-n`: print the set of tables/views that would be dropped.
- Refuses dirs — prints "use `pxt rm` for directories".

#### `pxt rm <dir> [-f] [-r] [-n|--dry-run]`

Remove a directory.
- `-r`: recursive (kills contained tables/views).
- `-n`: print the contents that would be removed.
- Refuses tables/views — prints "use `pxt drop`".

#### `pxt rename <path> <new_name> [-n|--dry-run]`

Rename in place. Keeps parent dir.

#### `pxt mv <path> <new_dir> [-n|--dry-run]`

Move to a different directory.

#### `pxt revert <table> [--to VERSION] [-f] [-n|--dry-run]`

Default: undo the last op on `<table>`. `--to VERSION` jumps to that version.
Always reverts both data and schema together — pixeltable's versioning model
doesn't separate them.

- `-n` / `--dry-run`: print the resulting state delta (rows added/removed,
  schema changes, view-cascade effects) without executing.
- `-f`: skip confirmation prompt.

If downstream views would break, fails loudly rather than attempting to repair.

## Dropped from earlier drafts

- `pxt sql` — no concrete use case that isn't better served by other
  commands. Exposing pg invites SQL that bypasses pxt's invariants in ways
  that are very hard to reason about, with no upside today.

- `pxt cache` as a top-level command — sizes surfaced in `pxt status` instead;
  no clear use case for CLI-driven cache wipes.

## Output format

Default: plain TSV-like text (composable, pipes cleanly, easy to grep). No
boxes, no colors. `--json` for structured.

## Open questions

1. After Python import audit, does cold start land under ~100ms? If not,
   trigger Go v2.
2. **pre_flight(intent)** — term floated externally; we don't know the
   intended meaning. Plausible reads: validate planned op before running,
   cost-estimate an expensive op, CORS-style pre-flight. Punted: don't design
   for it until the concrete problem it solves is identified.

## History (running notes)

- 2026-05-11 r1: AI-agent-first framing. Sketched ls / describe / head /
  errors / plan / endpoints / status / drop / clear-cache.
- 2026-05-11 r2: Confirmed `pxt ls` rooted (no session state). `pxt describe`
  reuses `Table.describe()`; `--json` is `get_metadata()`. `pxt errors`
  per-table only. Split mutation: `pxt drop` for table/view, `pxt rm` for dir.
  Added history, sql (later dropped), env, rename, mv, revert, cache (later
  collapsed into status). Architecture: FastAPI daemon + Python/Go client.
- 2026-05-11 r3: Dropped `pxt sql` (no real use case) and `pxt cache` (collapsed
  into status). Added cross-catalog discovery commands (udfs, indexes,
  computed). Revised on 150ms-vs-10ms: v1 Python client is acceptable
  short-term but v2 Go binary is the target — bursts of 20 commands at 150ms
  add up to 3s, noticeable. Defined revert semantics defaulting to last-op,
  schema+data together.
- 2026-05-11 r4: Revert always covers data+schema (no separable knobs).
  Daemon stateless on cwd: client sends `X-Cwd` per request, daemon resolves
  project context from that. `--dry-run` (alias `-n`) is the standard form.
  Client v1 strategy: Python with aggressive lazy imports targeting <50ms;
  Go via cibuildwheel-style wheels as fallback if Python can't make it.
- 2026-05-11 r5: `pxt revert` confirmed exposed. `-n` extended to every
  mutating command (drop, rm, rename, mv, revert). Added `pxt lint`: highest-
  value single command for AI-generated code; rule namespace `PXT0xx`-`PXT3xx`
  static + `PXT4xx` catalog-aware. Plain TSV is default output format.
- 2026-05-11 r6: Removed type/schema rules from `pxt lint` (pxt already
  infers/validates these; column types aren't even declared). Removed
  `--catalog` mode and `PXT4xx` rules — leftover from the type-check premise.
  Added data ops: `pxt get <table> <pk>` and `pxt count <table> [--where]`.
  Skipped insert/update/delete — Python API stays clearer. Pre-flight
  concept seeded as open question; likely a Python API rather than a
  CLI-first feature.
- 2026-05-11 r7: Confirmed `pxt delete` unnecessary — `pxt revert` covers
  data undo. Pre-flight downgraded: term floated but meaning unknown, deferred
  until concrete need surfaces. Added `pxt logs` plus FastAPIRouter
  default-on logging middleware: JSON-line file at `~/.pixeltable/serve.log`,
  in-memory ring buffer for `--recent`, off-switch via `PXT_SERVE_LOG=0`.
  Filters: `--errors`, `--slow MS`, `--endpoint`, `--since`. Bodies not
  logged by default.
- 2026-05-11 r8: Pixeltable already has a verbose log stream at
  `~/.pixeltable/logs/...` but lacks request correlation. Added
  request_id-via-ContextVar plumbing: pxt logger gains a `%(request_id)s`
  field, FastAPIRouter middleware sets the ContextVar per request, executor
  dispatch uses `copy_context().run` to propagate across threads. `pxt logs`
  gains `--request <id>` to join the per-request entry with all pxt log
  lines tagged with that id — killer query for "what happened in this
  request".
