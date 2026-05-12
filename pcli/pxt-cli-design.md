# pxt CLI — AI-Agent-First Design

A CLI that makes inspection, debugging, and lifecycle ops cheap for an AI agent
writing a Pixeltable + FastAPIRouter app. Code stays the artifact for schema,
UDFs, queries; CLI handles state-of-the-world questions and one-shot ops.

## Goals

- **Sub-150ms per command.** Agents fire bursts of commands; 10 calls at 150ms
  vs 1500ms is the difference between "instant catalog view" and a
  context-blowing wait. Measured at ~110ms warm.
- **Errors are self-correcting.** Every misuse (bad arg, missing PK, unknown
  column, dependent views) returns enough info for the agent to retry
  correctly in one shot — argparse failures re-print example invocations,
  HTTP errors surface pxt's diagnostic verbatim.
- **No daemon literacy required.** Auto-spawns on first call. No PID files,
  no `start`/`stop` ceremony. Help text never mentions "daemon".
- **Catalog state is always current.** Changes made from the user's Python
  REPL or notebook are visible to the CLI without any sync step.
- **Bounded blast radius.** No mutation runs without `-f` or an interactive
  y/N. Non-TTY callers without `-f` fail closed. `--cascade`/`-r` are never
  inferred.
- **Mirrors the Python API.** Each command corresponds to a Python call the
  user could write: `describe` ≡ `t.describe()`, `rows` ≡ `t.head()`,
  `rename` ≡ `pxt.move()`. No DSLs. Learning the CLI teaches the API.
- **One transport.** All catalog ops go through HTTP to the daemon. No
  "fast path" that direct-imports pxt in the client. Single state model.

## Non-goals

- No subcommands that create tables, add columns, define UDFs, or otherwise
  duplicate the Python API. Code is the program; the CLI is for inspecting it.
- No interactive REPL. Agents already have Bash; one-shot composable commands
  beat a shell.
- No raw SQL passthrough. Pixeltable's API is the abstraction.

## Design principles

1. **Structured output, always available.** Default tabular for humans, `--json`
   for agents.
2. **Composable.** Output flows into `jq`, `grep`, pipes.
3. **Rooted, no session state.** All paths absolute from catalog root.

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

### Version coherence

Client compares its `pixeltable.__version__` with `health.pxt_version` on each
call; if they differ, it SIGTERMs the daemon and respawns. Keeps post-`pip
install -U pixeltable` behavior correct with zero user action.

## Commands

All commands accept `--json`. Mutations accept `-f` (skip confirmation) and
`-n` / `--dry-run`. Non-TTY callers must pass `-f` to mutate.

### Inspection

| Command    | Purpose |
|------------|---------|
| `ls`       | List catalog entries (table/view/dir), flat or `--tree`. |
| `describe` | Schema + metadata for one table. |
| `rows`     | First N rows; optional `--cols` subset. |
| `get`      | Single-row lookup by primary key. |
| `count`    | Row count for a table. |
| `errors`   | Rows where any computed column failed. PK required. |
| `history`  | Table version timeline. |
| `plan`     | Computed-column DAG + SQL-pushdown vs Python. |

### Discovery

| Command     | Purpose |
|-------------|---------|
| `columns`   | All columns across the catalog (or one table). |
| `computed`  | Only computed columns; alias for `columns --computed`. |
| `idxs`      | Embedding/btree indexes across the catalog. |
| `udfs`      | UDFs in a module or project. |
| `endpoints` | FastAPIRouter routes a serve config would expose. |

### State

| Command  | Purpose |
|----------|---------|
| `status` | Daemon pid, pxt version, pgdata path, cache sizes. |
| `env`    | `PIXELTABLE_*` env vars + active config file. |
| `logs`   | Per-request log + pxt log, joinable by request_id. |
| `health` | Liveness probe; also used internally to detect version drift. |

### Static analysis

| Command | Purpose |
|---------|---------|
| `lint`  | Catch AI-failure-mode bugs pxt can't see at runtime. |

### Mutation

| Command  | Purpose |
|----------|---------|
| `drop`   | Drop a table/view. Refuses directories. |
| `rm`     | Remove a directory. Refuses tables/views. |
| `rename` | Rename in place (parent dir preserved). |
| `mv`     | Move to a different directory (leaf preserved). |
| `revert` | Undo last op(s) on a table. Irreversible. |

### Inspection (read-only)

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

### Discovery

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

## Output format

Default: plain TSV-like text (composable, pipes cleanly, easy to grep). No
boxes, no colors. `--json` for structured.
