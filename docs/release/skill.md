---
name: pixeltable
description: >
  Build multimodal AI apps with Pixeltable. One application file (app.py)
  declares TableModel tables and FastAPIRouter routes. Create tables with
  pxt schema update. Start HTTP with pxt service update. Insert a row or
  POST to try the app. Use computed columns instead of LangChain,
  pandas-as-store, or a separate vector DB. Use when building RAG,
  processing images/video/audio/documents, or serving an API. Do NOT use for
  general Python or direct PostgreSQL administration.
license: Apache-2.0
allowed-tools: []
metadata:
  author: Pixeltable
  version: 2.11.1
  type: documentation
  executes-code: false
  category: data-infrastructure
  tags: [multimodal, ai, data, tables, embeddings, rag, udf, video, audio, images, documents, agents, tools, fastapi, declarative, computed-columns, vector-search]
  documentation: https://docs.pixeltable.com/
  support: https://github.com/pixeltable/pixeltable/discussions
  priority: 6
  pathPatterns: ["**/*.py"]
  importPatterns: ["pixeltable", "import pixeltable as pxt", "from pixeltable"]
  bashPatterns: ['^\s*pxt(?:\s|$)']
  promptSignals:
    phrases: ["pixeltable", "computed column", "embedding index", "add_embedding_index", "TableModel", "create_view", "document_splitter", "invoke_tools", "pxt schema", "pxt service", "pxt shell", "pxt errors", "pxt dashboard"]
    minScore: 6
  source: https://github.com/pixeltable/pixeltable-skill/blob/main/skills/pixeltable-skill/SKILL.md
---


## STOP

If you find yourself importing any of these, you are off-path:

1. **Do not use LangChain / LlamaIndex / Haystack / LangGraph.** Chunking is `document_splitter`. Search is `.similarity()`. Tools are `pxt.tools()` + `invoke_tools()`.
2. **Do not use pandas as a working store.** Tables are the store. `.collect().to_pandas()` is export only.
3. **Do not write `for row in ...:` loops calling models.** Wrap the call in a computed column.
4. **Do not install a separate vector database.** In an app, `__indexes__ = [pxt.EmbeddingIndex(...)]` on the model. In a notebook, `t.add_embedding_index(col, embedding=fn)`. Search with `.similarity(string=query)`.
5. **Do not write `while not done:` agent loops.** Insert a row. The computed-column chain runs.

See [anti-patterns.md](#reference-anti-patterns) (6 macros).

## What is Pixeltable?

One application file (`app.py`) is the backend.

- `pxt schema update`: creates tables from `TableModel` classes. Does not start HTTP.
- Insert a sample, `.select()`, `pxt dashboard`, or `pxt schema diff`. Compute runs on insert. After `pxt service update`, curl POST.
- `pxt service update`: starts HTTP (local or `pxt://`). `pxt service list` prints the URL. This is the serving command; do not reach for `pxt service run`.

`pxt db update` sets hosted image, secrets, and workers. It does not insert rows and does not start app HTTP.

First run: [Quickstart](https://docs.pixeltable.com/overview/quick-start). Why: [Why Pixeltable](https://docs.pixeltable.com/overview/pixeltable).

## Starting a new project

```bash
pip install 'pixeltable[serve]'   # Python 3.11+
pxt init
pxt service example --out app.py
pxt schema check app.py           # validates the file; warns if 'app' is shadowed
pxt schema update app.py my_app
pxt service update app.py my_app
pxt service list                  # assigned port; do not hard-code :8000
```

`pxt service example` writes models plus a `FastAPIRouter`. Schema only (no HTTP): `pxt schema example --brief --out app.py`. Then edit `app.py` and run `pxt schema update` again. After a schema change, run `pxt service update` again if routes exist. Do not `python app.py`. Full flags: [cli.md](#reference-pixeltable-cli-reference-pxt).

The last argument (`my_app`, or `pxt://org:db` on Cloud) is a catalog directory, not a folder on disk. `pxt init` marks the project root. Schema does not start HTTP. Service does not create tables. Non-interactive: `pxt service update ... -f`. Local handle: `pxt.get_table('my_app.docs')`, or bind the models: `import app; app.TableModel.bind_all('my_app')`, then `app.Docs.insert(...)` / `app.Docs.select(...).collect()`.

Same file on Cloud: set `PIXELTABLE_API_KEY`, add `[[pixeltable.database]]` with `name = 'pxt://org:db'`, then `pxt db update pxt://org:db -f`, then `pxt schema update app.py pxt://org:db -f`, then `pxt service update app.py pxt://org:db -f`. Cloud handle: `pxt.get_table('pxt://org:db/docs')`. Cloud databases store media in their managed home bucket by default; set a column `destination=` only to override it. On Cloud, try the app with dashboard insert plus `pxt schema diff`, and inspect failures with `pxt service logs` / `pxt db logs`. [Cloud](https://docs.pixeltable.com/howto/deployment/cloud).

## The application file

`pxt service example --out app.py` writes this shape. Edit it. Then `pxt schema update app.py my_app`.

```python
import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.serving import FastAPIRouter

TableModel = pxt.model_base()


@pxt.udf
def excerpt(text: str, n: int = 12) -> str:
    return text if len(text) <= n else f'{text[:n]}...'


class Docs(TableModel, name='docs'):
    id = pxt.Column(value=pxtf.uuid.uuid7(), primary_key=True)
    title: pxt.String
    body: pxt.String | None
    title_upper = pxtf.string.upper(title)
    summary = excerpt(title)


ingest = FastAPIRouter(name='ingest')
ingest.add_insert_route(
    Docs, path='/docs', inputs=[Docs.title, Docs.body],
    outputs=[Docs.id, Docs.title_upper, Docs.summary],
)
ingest.add_update_route(
    Docs, path='/docs/update', inputs=[Docs.title],
    outputs=[Docs.id, Docs.title_upper],
)
ingest.add_compute_route(Docs, path='/titles', inputs=[Docs.title], outputs=[Docs.title_upper])
```

Annotation is a stored column. Assignment is a computed column. Optional is `T | None`. Primary key is `pxt.Column(..., primary_key=True)`; `add_update_route` matches rows by it, so the request body carries `id` even though `inputs` does not list it. Indexes on the model: `__indexes__ = [pxt.EmbeddingIndex(...)]`. `from pixeltable.serving import FastAPIRouter`.

Already have FastAPI: after schema update, `ingest.bind('my_app')` then `app.include_router(ingest)`. Or define the `fastapi.FastAPI` object in `app.py` and `include_router()` each router there; `pxt service update` then serves that one application. Call `pxt.get_table()` inside custom handlers. [workflows.md](#reference-fastapirouter).

RAG, views, and search: [workflows.md](#reference-fastapirouter). Do not add Hugging Face or spaCy unless the user asked.

## Apps vs notebooks

- **Apps:** `app.py` + `pxt schema update` + `pxt service update`. Indexes on the model.
- **Notebooks / REPL:** `pxt.create_table()`, `add_computed_column()`, `add_embedding_index()`. The appendix below uses that form.

## Where to look

| Need | Open |
|------|------|
| `pxt schema`, `pxt service`, inspect | [cli.md](#reference-pixeltable-cli-reference-pxt) |
| Types, views, UDFs, UDAs | [core-api.md](#reference-pixeltable-core-api-reference) |
| Provider import and output shape | [providers.md](#reference-pixeltable-ai-provider-reference) |
| Serving, FastAPIRouter, routes | [workflows.md](#reference-fastapirouter) |
| Wrong stack | [anti-patterns.md](#reference-anti-patterns) |

Add video, audio, agents, or a UI by editing `app.py`. A view is either a filter (`base=Docs.where(...)`) or an iterator (`frame_iterator`, `audio_splitter`, `document_splitter`, `video_splitter`, `string_splitter`, `list_iterator`, `tile_iterator`). Check `pixeltable.functions` before writing a UDF. Start from `pxt service example` or `pxt schema example`. Do not invent a second `pxt schema update` path.

## API traps

| Wrong | Correct |
|-------|---------|
| `openai.vision(...)` | Deprecated (the only deprecated function in `pixeltable.functions`). Use `chat_completions` with `image_url`, or `responses` |
| `from pixeltable.iterators import ...` | The whole `pixeltable.iterators` package is a deprecated shim (`FrameIterator`, `VideoSplitter`, `DocumentSplitter`, `StringSplitter`, `AudioSplitter`, `TileIterator`). Import the function from `pixeltable.functions.*` -- e.g. `from pixeltable.functions.video import frame_iterator` |
| `similarity(query)` | `similarity(string=query)`. Also `image=` / `audio=` / `video=` / `document=` / `vector=`; `idx=` picks among several indexes on one column |
| Re-run with `if_exists='ignore'` to fix logic | Notebook: `add_computed_column(..., if_exists='replace')`. App: **rename** the column, then `pxt schema update --allow-destructive` |
| Edit a computed column's expression in place, then `--allow-destructive` | Editing an existing column's expression is `UNSUPPORTED`; the flag does not help and the whole update applies nothing. Rename the column |
| `t.summary_errortype` | `t.summary.errortype` / `t.summary.errormsg`, on stored computed or media columns. `t.<col>.fileurl` / `.localpath` for media |
| `pxt.Required[pxt.String]` | Non-nullable by default. Optional: `T \| None` |
| `@pxt.udf def f(x: str)` fed a nullable column | A non-nullable parameter that receives `None` **skips the call**: the cell is `None` and `errormsg` is empty. Annotate `x: str \| None` and handle `None` in the body |
| `whisper.load_model(...)` inside a UDF body | Weights reload on every row. Use the shipped wrapper (`pxtf.whisper.transcribe`, `clip.using(...)`), or a module-scope cached loader |
| `recompute_columns(columns=['summary'])` | `t.recompute_columns('summary', errors_only=True)` |
| TOML routes or a retired serve CLI | `FastAPIRouter` + `pxt schema update` + `pxt service update` |
| `add_embedding_index()` in `app.py` | `__indexes__` on the TableModel. Note the DSL names an index `name=`, the SDK `idx_name=` |
| `make_video(order_by=...)` / `stitch_tiles(order_by=...)` | Both are `requires_order_by` UDAs: the ordering expression is the **first positional** argument -- `make_video(t.pos, t.frame, fps=25)`. `order_by=` raises |
| `pxt.create_table()` / `get_table()` at import in `app.py` | `TableModel` + `pxt schema update`. Import must not mutate the catalog |
| `EmbeddingIndex(frame, image_embed=clip)` | `embedding=clip` (covers text and image). Or both `string_embed=` and `image_embed=`. `image_embed=` alone cannot answer `similarity(string=...)` |
| `uuid.astype(pxt.String)` | `uuid.to_string()` (`from pixeltable.functions.uuid import to_string`). `astype` does not cast UUID to String |

Extract the field (`.text`, `.choices[0].message.content`). Cast Json with `.astype(pxt.String)` only before embedding or concatenating.

## Notebook / REPL appendix

```python
import pixeltable as pxt

pxt.create_dir('my_project', if_exists='ignore')
t = pxt.create_table('my_project.documents', {
    'title': pxt.String,
    'content': pxt.String,
    'image': pxt.Image,
    'video': pxt.Video,
    'audio': pxt.Audio,
    'doc': pxt.Document,
}, if_exists='ignore')
```

Types are non-nullable by default. Optional is `T | None`. Do not use `pxt.Required`.

```python
from pixeltable.functions.uuid import uuid7

t = pxt.create_table('my_project.items', {
    'content': pxt.String,
    'uuid': uuid7(),
}, primary_key=['uuid'], if_exists='ignore')
```

Insert: `t.insert([{...}])`. Computed column:

```python
from pixeltable.functions.openai import chat_completions

t.add_computed_column(
    summary=chat_completions(
        messages=[{'role': 'user', 'content': t.content}],
        model='gpt-4o-mini',
    ).choices[0].message.content,
    if_exists='ignore',
)
```

Views: `document_splitter`, `frame_iterator` (from `pixeltable.functions.video`), `string_splitter`, `audio_splitter`. Notebook indexes: `t.add_embedding_index('content', embedding=embed_fn, if_exists='ignore')`.

Query: `t.where(...).select(...).collect()`. Similarity: `t.content.similarity(string=query)`. In `@pxt.query`, alias as `score=sim`.

UDFs are recorded as a module path relative to the project root (`app.excerpt`).

Always `if_exists='ignore'` on notebook `create_*` / `add_*`. Failed cells: `t.recompute_columns('summary', errors_only=True)`. `string_splitter` / `document_splitter(..., separators='sentence')` need spaCy. Embedding indexes need `.using(...)`.

## pxt CLI

```bash
pxt init
pxt service example --out app.py
pxt schema check app.py
pxt schema update app.py my_app
pxt service update app.py my_app
pxt service list
pxt ls -l
pxt errors my_app/docs
pxt recompute my_app/docs summary --errors-only -f
pxt dashboard
```

[cli.md](#reference-pixeltable-cli-reference-pxt).

## Resources

- [Quickstart](https://docs.pixeltable.com/overview/quick-start)
- [CLI](https://docs.pixeltable.com/platform/cli)
- [MCP Server](https://github.com/pixeltable/mcp-server-pixeltable-developer)
- [Docs](https://docs.pixeltable.com/llms-full.txt)

## Reference: Pixeltable CLI Reference (`pxt`)


Agent-focused map of the `pxt` CLI. Official source: [platform/cli.md](https://docs.pixeltable.com/platform/cli.md). Always run `pxt <command> --help` for version-specific flags -- never guess.

Python 3.11+. There is no `pxt serve`, no `pxt deploy`, no `pxt app`, no `pxt db create` (`pxt db update` creates), and no `[tool.pixeltable.service]` TOML.

### Two surfaces

| Surface | Purpose | Requires |
|---------|---------|----------|
| **Catalog** | Inspect, query, mutate tables/views/dirs | `pip install pixeltable` |
| **Schema / service** | Apply a `TableModel` file; run `FastAPIRouter` services | `pip install 'pixeltable[serve]'` for `pxt service` |

Verify: `pxt --help` and `pxt health`.

### Project root

`pxt init` marks this directory as a project root. Schema and service refuse an application file with no project root. Fresh dir: writes `pixeltable.toml`. Already configured: no-op. Existing `pyproject.toml`: appends `[[tool.pixeltable.database]]` (does not write `pixeltable.toml`). Nested under another root: refused (exit 3). Start from `pxt service example --out app.py` (models plus routes) or `pxt schema example --brief --out app.py` (models only).

```bash
pxt init                          # project root; see cases above
pxt schema update app.py my_app   # creates catalog dir + tables; does NOT start HTTP
pxt service update app.py my_app  # starts local HTTP; does NOT create tables
pxt service update app.py my_app -f   # CI / no TTY when changes are pending
```

`my_app` is a catalog directory, not a folder on disk. After apply: `t = pxt.get_table('my_app.docs')`. Cloud: `t = pxt.get_table('pxt://org:db/docs')`. Tables live under `~/.pixeltable`, not in the repo. Directory names from the project root to `app.py` must be Python identifiers. Do not `python app.py` if the file only declares models and routers. After a schema change, run `pxt service update` again if routes exist.

### Daemon

On the first catalog command, `pxt` auto-spawns a daemon at `127.0.0.1:22089` (~40 ms per command after warm-up). Override with `PXT_PORT`. Lifecycle: `pxt daemon status`, `pxt daemon stop`, `pxt daemon start`.

### Command categories

| Category | Commands |
|----------|----------|
| **Project** | `init` |
| **Inspection** | `ls`, `describe`, `columns`, `computed`, `idxs`, `history`, `status`, `config` |
| **Query** | `rows`, `get`, `count`, `errors` |
| **Mutation** | `drop`, `drop-dir`, `rename`, `mv`, `recompute`, `revert` |
| **Schema** | `schema diff`, `schema update`, `schema prune`, `schema check`, `schema example` |
| **Serving** | `service diff`, `service update`, `service run`, `service prune`, `service stop`, `service restart`, `service list`, `service logs`, `service check`, `service example` |
| **Cloud** | `db`, `org`, `secret` |
| **Interactive** | `shell`, `cd`, `pwd` |
| **Lifecycle** | `daemon`, `dashboard`, `localproxy`, `health` |

`cd` / `pwd` set and print a working directory prepended to relative paths. It is scoped to the invoking shell's process, so it does **not** survive between separate tool calls: always pass full catalog paths instead. `localproxy` manages the daemons behind `pxt://local:<db>` URIs and is not part of the normal app loop.

### Universal flags

| Flag | Description |
|------|-------------|
| `-h`, `--help` | Every command |
| `--json` | Machine-readable output on catalog commands, `schema` / `service` verbs, `db`, `org`, `secret`, `daemon status`. Not on `shell`, `dashboard`, or `daemon start`/`stop`. `health` is always JSON. |
| `-n`, `--dry-run` | Catalog mutations (`drop`, `drop-dir`, `rename`, `mv`, `recompute`, `revert`) plus `schema update`, `schema prune`, `service update`, `service prune`, and `db update` |
| `-f`, `--force` | Skip `[y/N]` on `drop`, `drop-dir`, `recompute`, `revert`, schema/service update and prune, and `db update`. Use it in non-interactive runs when a pending plan can prompt. Additive or no-op schema updates do not prompt. Not on `rename`/`mv`. |

### Agent workflows

| Task | Prefer CLI | Example |
|------|-----------|---------|
| Mark a project root | `pxt init` | no-op if already configured; exit 3 if nested |
| Write a starting file | `pxt service example` | `pxt service example --out app.py`. Models only: `pxt schema example --brief --out app.py` |
| Validate a file | `pxt schema check`, `pxt service check` | no `TARGET`; reads no catalog |
| Apply tables | `pxt schema update` | `pxt schema update app.py my_app` |
| Review schema drift | `pxt schema diff` | exit `0` in sync, `2` pending |
| Start HTTP | `pxt service update` | `pxt service update app.py my_app -f` (force pending changes in CI) |
| Inspect catalog | `pxt ls -l`, `pxt describe`, `pxt columns --computed` | `pxt ls --json \| jq '.entries[] \| select(.kind == "table")'` |
| Debug failed columns | `pxt errors`, `pxt rows --cols` | `pxt errors my_app/docs --col embedding` |
| Retry failed cells | `pxt recompute` | `pxt recompute my_app/docs embedding --errors-only -f` |
| Debug a hosted service | `pxt service logs` | `pxt service logs pxt://org:db/ingest --since 10m --tail 50`. A local service is not readable this way: the command exits 1 and prints the log file's path; `tail` that file |
| Check runtime/config | `pxt status`, `pxt config` | `pxt config --section openai` |
| Many commands in sequence | `pxt shell` | amortizes startup; errors don't kill session |
| Visual inspection | `pxt dashboard` | read-only UI at daemon port |
| Hosted database | `pxt db update` | `pxt db update pxt://myorg:mydb -f`, then schema, then service |

**SDK vs CLI:** Notebooks and one-off REPL use the Python SDK (`create_table`, `add_computed_column`). Apps use a `TableModel` file plus `pxt schema` / `pxt service`. Use CLI for inspect, debug, and CI drift checks.

### Quick reference

```bash
## project, then schema, then service
pxt init
pxt service example --out app.py
pxt schema update app.py my_app
pxt service update app.py my_app

## inspect
pxt ls -l
pxt describe my_app/docs
pxt rows my_app/docs -n 5

## query / debug
pxt get my_app/docs 42
pxt count my_app/docs
pxt errors my_app/docs
pxt recompute my_app/docs summary --errors-only -f

## mutations (use -f in CI)
pxt drop my_app/docs -f
pxt revert my_app/docs --steps 3 -f

## interactive
pxt shell
pxt dashboard
```

### Inspection highlights

- **`pxt ls`**: `-l` (metadata), `--counts` (row counts), `--tree`
- **`pxt describe`**: schema; `--json` returns full `get_metadata()` dict
- **`pxt computed`**: shorthand for `pxt columns --computed`
- **`pxt idxs`**: `--embedding` for embedding indexes only
- **`pxt history`**: `-n N` for last N versions (run before `revert`)
- **`pxt status`**: daemon PID, version, total errors; `--sizes` for disk usage

### Query highlights

- **`pxt rows`**: `-n N` (default 10), `--cols a,b,c`. Unstored computed columns skipped unless listed in `--cols` (forces eval).
- **`pxt get`**: PK lookup; composite PKs in declared order. Table must have a primary key.
- **`pxt errors`**: rows where stored computed columns failed; `--col NAME` to filter. Table must have a primary key.

### Mutation highlights

- **`pxt drop`**: tables/views; `--cascade` drops dependent views; use `pxt drop-dir` for directories
- **`pxt schema prune`**: never force-drops, and drops a view before its base; a table something outside the pruned set depends on is left in place
- **`pxt drop-dir`**: `-r` for recursive directory removal
- **`pxt revert`**: irreversible -- run `pxt history` first
- **`pxt recompute`**: `pxt recompute PATH COLUMN...`; `--errors-only` narrows it to the rows that failed and takes one column; `--no-cascade` leaves dependent columns alone; `-n` reports the row count without running. The CLI form of `t.recompute_columns()`

Table paths accept `my_app/docs` or `my_app.docs`.

### Schema (`pxt schema`)

Reconcile a catalog directory with the `TableModel` classes in a Python file. Provisioning an empty target and evolving an existing one are the same command.

| Command | Description |
|---------|-------------|
| `pxt schema diff APP TARGET` | What `update` would change. Read-only. Exit `2` if pending |
| `pxt schema update APP TARGET` | Create the catalog dir + tables; migrate existing ones. Does **not** start HTTP |
| `pxt schema prune APP TARGET` | Drop tables under `TARGET` that the file does not declare |
| `pxt schema check APP` | Validate the file only. No `TARGET`. Reads no catalog |
| `pxt schema example` | Write a working file (`--brief` for the minimal one) |

```bash
pxt schema example --out app.py
pxt schema check  app.py
pxt schema diff   app.py my_app
pxt schema update app.py my_app
pxt schema update app.py my_app -n                       # plan only; exit 2 if pending
pxt schema update app.py my_app --allow-destructive -f   # including column/index drops
pxt schema prune  app.py my_app -n
```

`TARGET` is a catalog directory or a `pxt://org:db/...` URI.

Reading a diff: `+` created / added, `-` dropped, `~` migrated, `=` already matches, `!` cannot be migrated in place. Each op is marked safe, DESTRUCTIVE, or UNSUPPORTED.

- **DESTRUCTIVE** (dropping a column or index) needs `--allow-destructive`; exit `3` without it. Applying is **all-or-nothing** -- without the flag a destructive plan applies *nothing at all*, not the safe parts.
- **UNSUPPORTED** cannot be applied by any flag: a kind or iterator mismatch, or a column whose type or **value expression** changed. One unsupported table aborts the whole update, including other models' pending additive changes. Rename the column, or drop it and re-add it in a second pass. See [core-api.md](#tables).

The daemon imports the application file, so it must be readable there; the file's own directory joins `sys.path`, so it can import modules sitting next to it.

**Run `pxt schema check APP` before the first update.** It validates the file with no catalog access, confirms every udf a column calls resolves to a module path another process can import, and warns when a top-level name in the project is shadowed:

```
app.py: an import of 'app' reads /.../site-packages/app/__init__.py, so this project
cannot record a udf under 'app'; rename it
```

The project root goes on `sys.path` *after* installed packages, so an installed distribution of the same name wins. `check` warns and still exits `0`; `schema update`, `service update` and `service run` do **not** warn -- they import the wrong module silently. Generic single-file names collide most often, so heed the warning and rename.

A CI drift check:

```bash
pxt schema diff app.py pxt://acme:main/prod    # 0 = in sync, 2 = drift, 1 = error
```

### Serving (`pxt service`)

Runs the `FastAPIRouter` instances an application file declares. Requires `pip install 'pixeltable[serve]'`. Same file as the models: apply tables first, then start HTTP.

| Command | Description |
|---------|-------------|
| `pxt service diff APP TARGET` | What `update` would change. Exit `2` if pending |
| `pxt service update APP TARGET [SERVICE]` | Start declared services in the background; restart those that changed. Does **not** create tables. `--port` pins one named service's port; a restarted service keeps its port |
| `pxt service run APP TARGET [SERVICE]` | Serve one service in the foreground until interrupted (`--host`, `--port`; default `127.0.0.1:8000`). For a container entrypoint, which must not return; **not** the command to recommend otherwise -- use `update` |
| `pxt service prune APP TARGET` | Stop and forget services at `TARGET` that the file does not declare |
| `pxt service stop NAME...` | Stop named services (`ingest` or `my_app/ingest`) |
| `pxt service restart NAME...` | Restart named services onto the current project and secrets |
| `pxt service logs NAME` | Read a hosted service's log; a local service gets its log file's path instead |
| `pxt service list [TARGET]` | What is running, and where |
| `pxt service check APP` | Validate the file only. No `TARGET`. Reads no catalog |
| `pxt service example` | Write a working application file |

```bash
pxt service example --out app.py
pxt service check app.py
pxt schema update app.py my_app
pxt service update app.py my_app -f
pxt service list
pxt service logs ingest                # local: exits 1 and prints the log file's path
tail -50 "$(pxt service logs ingest 2>&1 | sed 's/.*the log is at //')"
pxt service stop ingest
```

Hosted: `pxt service logs pxt://org:db/ingest --since 10m --tail 50`.

`update` starts one background process per service, each on its own port, and is the serving command to use. A no-op or dry run exits without prompting; pass `-f` when a pending update runs without a TTY. Adding a route is additive; changing or removing one needs `--allow-destructive`. OpenAPI docs are at `/docs`. `pxt service run` refuses a `pxt://` TARGET and does not record anything, so `list` and `stop` cannot find it.

`pxt service logs NAME` reads a hosted service's log: `pxt://org:db/ingest` or `pxt://org:db/path/ingest`. `--since` accepts `30s`, `10m`, `1h`, `2d` (default `1h`), `--tail` is capped at 10,000 lines (default 200), and `--include-health` keeps health-probe requests. Hosted logs merge request records with console output, including startup tracebacks. A service on this machine is not readable through it: the command exits 1 and prints the log file's path (`$PIXELTABLE_HOME/logs/services/<target>/<name>.log`); `tail` that file.

**Tracing.** `service diff`, `service update` and `service run` take `--otel`, which emits OpenTelemetry traces and needs `pip install 'pixeltable[otel]'` (`serve` and `otel` are the only two extras). The setting belongs to the running service, not to the file: a service already running without it restarts when `update` is given the flag, dropping the flag restarts it again, and `diff --otel` reports tracing that is off but was asked for as a pending change.

Do **not** write `[tool.pixeltable.service]` TOML or call `pxt serve`.

### Cloud (`pxt db`, `pxt org`, `pxt secret`)

Require `PIXELTABLE_API_KEY`. URIs are `pxt://org` or `pxt://org:db`.

```bash
pxt db update pxt://myorg:mydb -f  # also: diff, list, status, logs, start, stop, restart, build-image, delete
pxt db logs pxt://myorg:mydb --since 10m --tail 50
pxt org status pxt://myorg         # also: list
```

`pxt db update pxt://org:db` selects `[[pixeltable.database]]` by `name = 'pxt://org:db'`. A URI with no matching entry is an error. First `update` creates the hosted database.

Hosted order: `pxt db update pxt://org:db -f` sets secrets, image, and workers, then `pxt schema update app.py pxt://org:db -f`, then `pxt service update app.py pxt://org:db -f`. Database capacity or secret changes can also require `--allow-destructive`. If `pxt db diff` says the database project is behind the working copy, run `pxt db update` first.

Every Cloud database stores inserted and computed media in its managed home bucket by default. Set a column `destination=` to send that output elsewhere, or configure `input_media_dest` / `output_media_dest` to change the database defaults.

A UDF is recorded as a module path relative to the project root (`app.excerpt`), not a raw file path. `pxt db update` packs the project so Cloud can import it.

#### Secrets

```bash
pxt secret set pxt://myorg OPENAI_API_KEY=<your-key>    # also: list, delete
```

An org secret applies to every database in the org; a database secret wins on a key collision. A project declares database secrets under the `secrets` mapping, for example `secrets.openai_api_key = '<env:OPENAI_API_KEY>'`; `pxt db update` sets them. A process reads its secrets once, at startup, so a running one keeps the values it began with. After `pxt secret set` or `pxt secret delete`, run `pxt db restart pxt://org:db` for the database's tables and `pxt service restart pxt://org:db/NAME` for its services.

### Scripting with `--json`

```bash
pxt ls --json | jq '.entries[] | select(.kind == "table")'
pxt get my_app/docs 42 --json | jq '.row'
pxt count my_app/docs --json | jq '.count'
pxt schema diff app.py my_app --json
pxt service diff app.py my_app --json
```

### Related references

- [core-api.md, Serving](#serving) -- `FastAPIRouter` Python API
- [workflows.md](#reference-fastapirouter) -- application-file example
- [Configuration](https://docs.pixeltable.com/platform/configuration) -- API keys, paths, env vars

## Reference: Pixeltable Core API Reference


Apps use `app.py` plus `pxt schema update`. This file is notebook form (`pxt.create_table()`) unless noted. CLI: [cli.md](#reference-pixeltable-cli-reference-pxt). Routes: [workflows.md](#reference-fastapirouter).

Types are non-nullable by default. Optional is `T | None`. Do not use `pxt.Required`.

### Contents

- [Tables](#tables)
- [Querying](#querying)
- [Computed columns](#computed-columns)
- [Views](#views)
- [Indexes](#indexes)
- [UDFs](#udfs)
- [UDAs](#udas)
- [Built-in functions](#built-in-functions)
- [Import and export](#import-and-export)
- [Serving](#serving)
- [Tools](#tools)

### Tables

App:

```python
class Docs(TableModel, name='docs'):
    title: pxt.String
    body: pxt.String | None
    title_upper = pxtf.string.upper(title)
    uuid = pxt.Column(value=pxtf.uuid.uuid7(), primary_key=True)
    # astype(pxt.String) fails on UUID; use uuid.to_string()
```

Notebook:

```python
import pixeltable as pxt
from pixeltable.functions.uuid import uuid7

t = pxt.create_table('dir.docs', {
    'title': pxt.String,
    'body': pxt.String | None,
    'image': pxt.Image,
    'video': pxt.Video,
    'audio': pxt.Audio,
    'doc': pxt.Document,
    'tags': pxt.Json[list[str]],
    'records': pxt.Json[list[dict[str, str]]],
    'uuid': uuid7(),
}, primary_key=['uuid'], if_exists='ignore')
```

Types: `String`, `Int`, `Float`, `Bool`, `Image`, `Video`, `Audio`, `Document`, `Json`, `Timestamp`, `Date`, `UUID`, `Binary`, `Array[(3, 4), pxt.Float]`.

`pxt.Column(...)` carries what an annotation cannot: `stored=False` (computed on read, never materialized), `media_validation='on_read'` (defer validation to first read; default `'on_write'`), and `destination=` (object store for computed media -- `s3`/`gs`/`az`/`r2`/`b2`/`tigris`/`http`/a local path/`pxtfs`).

```python
thumbnail = pxt.Column(value=cover.rotate(90), stored=False)
scan = pxt.Column(type=pxt.Image, media_validation='on_read', comment='validated lazily')
```

From a file: `pxt.create_table('dir.data', source='data.csv', if_exists='ignore')`.

Insert: `t.insert([{...}])`. `return_rows=True` returns computed columns on the status object. `source=` also takes a path or URL (`source_format='csv'|'excel'|'parquet'|'json'`, `schema_overrides=`), a DataFrame, a list of dicts or Pydantic models, another table or query, or a HF dataset.

**Do not let one bad row abort a batch.** `insert(..., on_error='ignore')` and `add_computed_column(..., on_error='ignore')` keep the row, leave the failed cell `None`, and record the reason in `t.<col>.errortype` / `t.<col>.errormsg` (stored computed or media columns only). `t.<col>.fileurl` / `.localpath` locate a media cell.

```python
t.update({'score': 1.0}, where=t.category == 'important')
t.delete(where=t.is_active == False)
t.recompute_columns('summary', errors_only=True)
```

Changing a computed column's logic:

- **App.** Editing an existing column's expression in place is reported `UNSUPPORTED`. `--allow-destructive` does **not** help, and one unsupported table makes the whole `pxt schema update` apply nothing. **Rename** the column instead: the old name is a destructive drop, the new one an additive add, so it lands in one pass with `--allow-destructive`. Or drop it in one pass and re-add it in a second. Either way the data is destroyed and recomputed. `Table.rename_column()` preserves an existing column and its existing expression; it does not install new logic. `pxt.move()` moves a table or directory.
- **Notebook.** `t.add_computed_column(summary=..., if_exists='replace')` replaces it in one call when the column is directly replaceable and has no dependents. Otherwise drop dependents first, or `drop_column` and recreate; do not depend on one exception class for every rejected replacement.
- `if_exists='ignore'` never fixes logic -- it skips the call.

### Querying

```python
t.select(t.title, doubled=t.n * 2).where(t.n > 10).order_by(t.n, asc=False).limit(10).collect()
t.where(t.title.like('%pattern%')).count()
t.sample(n=100, seed=42).collect()
df = t.select(t.title).collect().to_pandas()  # export only
```

One `.where()` per query. Compose extra predicates with `&` / `|` (`t.where((t.n > 10) & t.active)`); do not chain `.where()`.

Aggregates run in queries, not computed columns:

```python
t.select(t.amount.sum()).collect()
t.group_by(t.region).select(t.region, total=t.amount.sum()).collect()
```

`@pxt.query` compiles at decoration time: do not `.collect()` or `get_table()` a table that does not exist yet inside one.

Local handle: `pxt.get_table('my_app.docs')`. Cloud: `pxt.get_table('pxt://org:db/docs')`. Or bind the models themselves: `import app; app.TableModel.bind_all('my_app')`, after which `app.Docs.insert(...)`, `app.Docs.count()` and `app.Docs.select(...).collect()` work.

### Computed columns

Notebook: `t.add_computed_column(...)`. App: assignment on the model.

```python
from pixeltable.functions.openai import chat_completions

t.add_computed_column(
    summary=chat_completions(
        messages=[{'role': 'user', 'content': t.body}],
        model='gpt-4o-mini',
    ).choices[0].message.content,
    if_exists='ignore',
)
```

Extract the field (`.text`, `.choices[0].message.content`). Cast Json with `.astype(pxt.String)` only before embedding or concatenating.

### Views

A view is either a **filter** over a base or an **iterator** that expands each base row into many. Rows follow the base; you never insert into a view.

#### Filtered views

`base=` takes a query over another model, not just the model:

```python
class Titled(TableModel, name='titled', base=Docs.where(Docs.title != '')):
    headline = Docs.title_upper + '!'      # may reference the base's computed columns
```

Notebook: `pxt.create_view('dir.titled', t.where(t.title != ''), if_exists='ignore')`. Add `is_snapshot=True` to freeze it.

#### Iterator views

Iterator output columns are reserved. Do not redeclare them. `string_splitter` / `document_splitter(..., separators='sentence')` need spaCy. `token_limit` needs `tiktoken`.

```python
from pixeltable.functions.document import document_splitter
from pixeltable.functions.video import frame_iterator
from pixeltable.functions.string import string_splitter
from pixeltable.functions.audio import audio_splitter
from pixeltable.functions.json import list_iterator

chunks = pxt.create_view(
    'dir.chunks', t,
    iterator=document_splitter(t.doc, separators='token_limit', limit=300),
    if_exists='ignore',
)
## Image elements only with separators='page' on PDFs
pages = pxt.create_view(
    'dir.pages', t,
    iterator=document_splitter(t.doc, separators='page', elements=['text', 'image']),
    if_exists='ignore',
)

## Output columns: pos, frame, frame_attrs  (timestamp is frame_attrs.time)
## Not frame_idx / pos_msec / pos_frame (legacy_frame_iterator only)
frames = pxt.create_view('dir.frames', t, iterator=frame_iterator(t.video, fps=1.0), if_exists='ignore')
frames.select(frames.pos, frames.frame, time=frames.frame_attrs.time).collect()

## text
sentences = pxt.create_view(
    'dir.sentences', t, iterator=string_splitter(text=t.body, separators='sentence'), if_exists='ignore',
)

audio = pxt.create_view(
    'dir.audio', t, iterator=audio_splitter(audio=t.audio, duration=30.0), if_exists='ignore',
)

tags = pxt.create_view('dir.tags', t, iterator=list_iterator(tag=t.tags), if_exists='ignore')
records = pxt.create_view('dir.records', t, iterator=list_iterator(t.records), if_exists='ignore')
```

`list_iterator` requires typed Json. Use keyword arguments for typed scalar lists, such as `tag=t.tags`; the keyword becomes the output column. Its single positional form requires a typed list of dictionaries with compatible keys. Untyped `pxt.Json` is rejected because Pixeltable cannot infer the view schema.

App: `base=` plus `iterator=` on the model. See [workflows.md](#reference-fastapirouter).

An unbound model class builds queries (`.where()`, `.select()`, `.order_by()`, `.group_by()`, `.limit()`, `.sample()`, `.join()`) -- that is what makes `base=Docs.where(...)` work. Anything that touches rows (`.collect()`, `.insert()`, `.count()`, `.update()`) needs the table to exist: use `pxt.get_table()`, or let a bound router reach it.

Every iterator view also gets `pos`. The nine that ship:

| Iterator | Module | Output columns |
|----------|--------|----------------|
| `document_splitter` | `functions.document` | subset of `text, image, title, heading, sourceline, page, bounding_box` per `elements=` / `metadata=` |
| `frame_iterator` | `functions.video` | `frame` (unstored), `frame_attrs` (`.time`, `.index`, `.key_frame`, ...) |
| `video_splitter` | `functions.video` | `segment_start`, `segment_start_pts`, `segment_end`, `segment_end_pts`, `video_segment` |
| `audio_splitter` | `functions.audio` | `segment_start`, `segment_end`, `audio_segment` |
| `string_splitter` | `functions.string` | `text` |
| `list_iterator` | `functions.json` | keys of `elements=`, or the kwarg names |
| `tile_iterator` | `functions.image` | `tile` (unstored), `tile_coord`, `tile_box` |

`legacy_frame_iterator` (`frame_idx` / `pos_msec` / `pos_frame`) and `sam3_for_video_segmentation` also exist; prefer `frame_iterator`.

### Indexes

App: `__indexes__ = [pxt.EmbeddingIndex(col, embedding=fn, name='...'), pxt.BtreeIndex(col)]`. Do not call `add_embedding_index()` in `app.py`. Use `.using(...)` when provider or model arguments must be bound. A custom embedding UDF must return a fixed-length, one-dimensional `pxt.Array[(N,), pxt.Float]`; Json or an array with an unknown length is not a valid embedding index function.

`embedding=` is tried against every modality and registers each one whose signature it matches, so a bidirectional function covers them all at once. That is why a CLIP index on an **image** column answers `similarity(string=...)`:

```python
clip_embed = pxtf.huggingface.clip.using(model_id='openai/clip-vit-base-patch32')
__indexes__ = [pxt.EmbeddingIndex(frame, embedding=clip_embed, name='frames_clip')]
## frame is an Image column, yet this resolves:
Frames.frame.similarity(string='a red bicycle')
```

Do not pass only `image_embed=` if you will query with `similarity(string=...)`.

Reach for the per-modality parameters -- `string_embed=`, `image_embed=`, `audio_embed=`, `video_embed=`, `document_embed=` -- when you want a *different* function per modality, or when you want a hard error: a per-modality argument that does not resolve raises, while `embedding=` quietly skips the modalities it cannot serve. Also `metric=` (`'cosine'` default), `precision=` (`'fp16'` default, `'fp32'` available). **The DSL names an index `name=`; `add_embedding_index()` names it `idx_name=`.**

```python
from pixeltable.functions.openai import embeddings
from pixeltable.functions.huggingface import sentence_transformer

embed_fn = embeddings.using(model='text-embedding-3-small')
## or: sentence_transformer.using(model_id='sentence-transformers/all-MiniLM-L6-v2')

class Docs(TableModel, name='docs'):
    body: pxt.String
    __indexes__ = [pxt.EmbeddingIndex(body, embedding=embed_fn, name='body_idx')]

t.add_embedding_index('body', embedding=embed_fn, if_exists='ignore')

sim = t.body.similarity(string=query)
t.where(sim > 0.3).order_by(sim, asc=False).limit(10).select(t.body, score=sim).collect()
```

`similarity()` takes exactly one of `string=`, `image=`, `audio=`, `video=`, `document=`, `vector=` (a raw array of the index's dimensionality). A positional argument is deprecated. When a column carries more than one embedding index, `idx='name'` picks one.

CLIP: `clip.using(model_id='openai/clip-vit-base-patch32')` then `similarity(string=...)` or `similarity(image=...)`. Metrics: `cosine` (default), `ip`, `l2`.

### UDFs

A UDF is recorded as a module path from the project root (`app.excerpt`). Type hints are required.

```python
@pxt.udf
def excerpt(text: str, n: int = 12) -> str:
    return text if len(text) <= n else f'{text[:n]}...'
```

#### A non-nullable parameter that receives `None` skips the call

The cell is set to `None`, nothing raises, and `errormsg` stays empty. This is by design, and it applies to the shipped UDFs as much as to yours. Annotate `T | None` when the argument can be null, and handle `None` in the body:

```python
@pxt.udf
def label(severity: str | None) -> str:
    return severity or 'unknown'
```

Binding a nullable argument to a non-nullable parameter also widens the column's declared type, so `excerpt(Docs.body)` over `body: pxt.String | None` is a `String | None` column.

Load models at module scope, never in the UDF body: [anti-patterns.md](#reference-anti-patterns).

```python
from pixeltable.func import Batch

@pxt.udf(batch_size=32)
def batch_process(texts: Batch[str]) -> Batch[list[float]]:
    return model.encode(texts).tolist()

lookup_fn = pxt.retrieval_udf(t, name='lookup_items', description='Look up items by name',
    parameters=['name'], limit=5)
```

### UDAs

`@pxt.uda` is many rows to one value. Use in `select()` / `group_by()`, not `add_computed_column`. Subclass `pxt.Aggregator`: `__init__`, `update`, `value`. `__init__` args must be constants.

```python
@pxt.uda
class avg_int(pxt.Aggregator):
    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, val: int) -> None:
        if val is not None:
            self.sum += val
            self.count += 1

    def value(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0

t.select(avg_int(t.value)).collect()
t.group_by(t.category).select(t.category, avg_val=avg_int(t.value)).collect()
```

Built-ins: `make_video`, `concat_videos_agg` (`pixeltable.functions.video`), `make_list` (`json`), `stitch_tiles` (`image`), `mean_ap` (`vision`). Scalar `concat_videos` takes a **list** of videos.

`requires_order_by` UDAs take the ordering expression as their **first positional argument**; passing `order_by=` raises. Built-ins include:

```python
t.select(pxtf.video.make_video(t.pos, t.frame, fps=30))          # t.pos orders; order_by= raises
t.group_by(base).select(pxtf.image.stitch_tiles(t.pos, t.tile, t.tile_box, width, height))
t.select(pxtf.video.concat_videos_agg(t.pos, t.video))
```

### Built-in functions

Before writing a UDF, check whether the operation already ships. `pixeltable.functions` (`pxtf`) covers strings, json, math, dates, arrays, images, audio, documents, and video (`video.editing`, `video.filters`, `video.scene_detect`), plus `vision` and `net`. Import the module and read its docs rather than guessing a name.

The one path worth spelling out, because nothing else documents it -- video to transcript:

```python
class Clips(TableModel, name='clips'):
    video: pxt.Video
    audio = pxtf.video.extract_audio(video, format='mp3')
    transcript = pxtf.openai.transcriptions(audio=audio, model='whisper-1').text
```

`extract_audio` returns `pxt.Audio | None`, and `transcriptions` takes a non-nullable `audio`. A silent video therefore leaves `transcript` as `None` with no `errormsg`. See the skip rule under [UDFs](#udfs).

### Import and export

Do not hand-roll a reader or writer -- check `pxt.io.import_*` / `export_*` first (csv, json, parquet, excel, pandas, SQL, Iceberg, LanceDB, HuggingFace).

### Serving

`from pixeltable.serving import FastAPIRouter`. Start from `pxt service example --out app.py`. `add_update_route` requires the target's primary key and matches rows by it: the request body carries the key even though `inputs` does not list it. `add_delete_route` uses the primary key by default and can instead take a nonempty `match_columns=` list.

Routes: `add_insert_route` (stores the row), `add_compute_route` (same request shape, computes without storing), `add_update_route`, `add_delete_route`, `add_query_route` (wraps a `@pxt.query`).

```python
ingest.add_update_route(Docs, path='/update', inputs=[Docs.title], outputs=[Docs.title])
```

Call `pxt.get_table()` inside custom FastAPI handlers. Do not `python app.py` if the file only declares models and routers. After a schema change, run `pxt service update` again. Worked example, upload and media URLs, and `background=True` job polling: [workflows.md](#reference-fastapirouter).

### Tools

```python
from pixeltable.functions.openai import chat_completions, invoke_tools

tools = pxt.tools(search_docs, lookup_fn)
## invoke_tools is per provider: openai.invoke_tools vs anthropic.invoke_tools
```

MCP: `pxt.mcp_udfs(url)` returns one UDF per remote tool over streamable HTTP; tools returning images or audio are not supported. Keys: env or [Configuration](https://docs.pixeltable.com/platform/configuration), not `api_key=` in calls.

## Reference: Pixeltable AI Provider Reference


Every provider is a module under `pixeltable.functions.`. Call it in a computed column; never in a `for` loop. In an app, embeddings go on `__indexes__`; in a notebook, `add_embedding_index()`. Embedding and index functions are bound with `.using(...)`.

Keys come from the environment or [config](https://docs.pixeltable.com/platform/configuration), never `api_key=` in the call.

### Quick reference

| Provider | Module | Functions | Extract |
|----------|--------|-----------|---------|
| OpenAI | `openai` | `chat_completions`, `responses`, `embeddings`, `speech`, `transcriptions`, `translations`, `image_generations`, `image_edits`, `image_variations`, `moderations`, `invoke_tools` | chat: `.choices[0].message.content`; Responses: `.output_text` |
| Anthropic | `anthropic` | `messages`, `invoke_tools` | `.content[0].text` |
| Gemini | `gemini` | `generate_content`, `embed_content`, `generate_images`, `generate_videos`, `generate_speech`, `transcribe`, `invoke_tools` | content: Json; generation/transcription returns typed Image, Video, Audio, or String |
| Bedrock | `bedrock` | `converse`, `invoke_model`, `embed`, `invoke_tools` | `.output.message.content[0].text` |
| Groq | `groq` | `chat_completions`, `invoke_tools` | `.choices[0].message.content` |
| Together | `together` | `chat_completions`, `completions`, `embeddings`, `image_generations` | `.choices[0].message.content` |
| Mistral | `mistralai` | `chat_completions`, `fim_completions`, `embeddings` | `.choices[0].message.content` |
| Nebius | `nebius` | `chat_completions`, `embeddings` | `.choices[0].message.content` |
| Fireworks | `fireworks` | `chat_completions` | `.choices[0].message.content` |
| DeepSeek | `deepseek` | `chat_completions` | `.choices[0].message.content` |
| OpenRouter | `openrouter` | `chat_completions` | `.choices[0].message.content` |
| Fabric | `fabric` | `chat_completions`, `embeddings` | `.choices[0].message.content` |
| Ollama (local) | `ollama` | `chat`, `generate`, `embed` | chat: `.message.content`; generate: `.response` |
| llama.cpp (local) | `llama_cpp` | `create_chat_completion` | `.choices[0].message.content` |
| vLLM (local) | `vllm` | `chat_completions`, `generate` | `.choices[0].message.content` |
| Hugging Face | `huggingface` | `sentence_transformer`, `clip`, `cross_encoder`, `detr_for_object_detection`, `sam3_for_segmentation`, `image_captioning`, `summarization`, `text_to_image`, ~15 more | index fn, or Json |
| Whisper (local) | `whisper` | `transcribe` | `.text` |
| WhisperX (local) | `whisperx` | `transcribe` | Json with `segments` |
| Voyage AI | `voyageai` | `embeddings`, `rerank`, `multimodal_embed` | index fn / Json |
| Jina AI | `jina` | `embeddings`, `rerank` | index fn / Json |
| Twelve Labs | `twelvelabs` | `embed` | video index fn |
| BFL FLUX | `bfl` | `generate`, `edit`, `fill`, `expand` | `pxt.Image` |
| RunwayML | `runwayml` | `text_to_image`, `text_to_video`, `image_to_video`, `video_to_video` | image: `response['output'][0].astype(pxt.Image)`; video: `response['output'].astype(pxt.Video)` |
| fal.ai | `fal` | `run` | Json |
| Replicate | `replicate` | `run` | Json |
| YOLOX | `yolox` | `yolox`, `yolo_to_coco` | Json detections |

Not model providers, but in the same namespace: `net.presigned_url` turns a blob-storage URI into a time-limited HTTP URL for serving media, and `vision` carries `eval_detections`, the `mean_ap` UDA, `bboxes_draw` / `overlay_segmentation` and the `bboxes_*` conversion family.

### Shapes

```python
from pixeltable.functions.openai import chat_completions, embeddings
from pixeltable.functions.huggingface import sentence_transformer

summary = chat_completions(messages=[{'role': 'user', 'content': body}], model='gpt-4o-mini') \
    .choices[0].message.content

embed_fn = embeddings.using(model='text-embedding-3-small')
## or local: sentence_transformer.using(model_id='sentence-transformers/all-MiniLM-L6-v2')
```

OpenAI-compatible chat-completion providers return `.choices[0].message.content`; OpenAI `responses` exposes simple text as `.output_text`. Anthropic returns `.content[0].text`. An image goes in a message as `{'type': 'image_url', 'image_url': {'url': t.image}}` -- `openai.vision` is deprecated. Tool calling is per provider: pair `pxt.tools(...)` with that module's own `invoke_tools`.

Rerankers (`voyageai.rerank`, `jina.rerank`, `huggingface.cross_encoder`) score query/document pairs; run one over the rows `.similarity()` returned rather than reaching for a framework.

**Model ids go stale.** The ids here are examples, not recommendations -- check the provider's current list. Pixeltable's own docstrings lag further behind than this file does.

## Reference: FastAPIRouter


`from pixeltable.serving import FastAPIRouter`. One application file declares `TableModel` classes and routers. Start from `pxt service example --out app.py`. Apply tables with `pxt schema update`. Start HTTP with `pxt service update`.

```python
## app.py
import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.serving import FastAPIRouter

TableModel = pxt.model_base()
embed_fn = sentence_transformer.using(model_id='intfloat/multilingual-e5-large-instruct')


class Docs(TableModel, name='docs'):
    document: pxt.Document
    timestamp: pxt.Timestamp
    uuid = pxt.Column(value=pxtf.uuid.uuid7(), primary_key=True)


class Chunks(
    TableModel,
    name='chunks',
    base=Docs,
    iterator=pxtf.document.document_splitter(
        Docs.document, separators='page, sentence', metadata='title,heading,page'
    ),
):
    __indexes__ = [pxt.EmbeddingIndex(text, embedding=embed_fn, name='chunks_embed')]  # type: ignore[name-defined]
    # Needs sentence-transformers + torch, and spaCy if separators include 'sentence'.


ingest = FastAPIRouter(name='ingest', prefix='/api', tags=['data'])
ingest.add_insert_route(
    Docs, path='/upload', uploadfile_inputs=[Docs.document], inputs=[Docs.timestamp],
    outputs=[Docs.document], background=True,
)
ingest.add_delete_route(Docs, path='/delete')

@pxt.query
def list_docs():
    return Docs.select(Docs.document, Docs.timestamp).order_by(Docs.timestamp, asc=False)

@pxt.query
def search_docs(query_text: str):
    sim = Chunks.text.similarity(string=query_text)
    return Chunks.where(sim > 0.3).order_by(sim, asc=False).select(
        text=Chunks.text, score=sim).limit(20)

ingest.add_query_route(path='/list', query=list_docs, method='get')
ingest.add_query_route(path='/search', query=search_docs, method='post')
```

```bash
pxt init
pxt schema update app.py my_app
pxt service update app.py my_app
```

After apply: `t = pxt.get_table('my_app.docs')`.

Already have FastAPI: after schema update, bind the catalog, then include the router. Call `pxt.get_table()` inside custom handlers.

```python
ingest.bind('my_app')
app.include_router(ingest)
```

The other way round also works: define the `fastapi.FastAPI` object in `app.py` and `include_router()` every router the file declares. `pxt service update` then serves that one application, named after the module, with the models bound at `TARGET` before it starts. Without a `FastAPI` object, each router is its own service on its own port.

- `add_insert_route`: POST from model columns. `uploadfile_inputs` for files. Persists the row. A file column is `uploadfile_inputs` or `inputs`, not both.
- `add_compute_route`: same request shape as insert, but `Table.compute()`: no row stored
- `add_update_route`: POST matches the row by primary key, so the request body carries the key (`id`) even though `inputs` does not list it. No `match_columns`
- `add_query_route`: wraps `@pxt.query`. Default `{ "rows": [...] }`. `one_row=True` returns the object (0 rows is a 404, more than one is a 409). `return_fileresponse=True` returns the one media column as a file (implies one-row)
- `add_delete_route`: POST delete by primary key, or by a nonempty `match_columns=` list
- Indexes on the model (`__indexes__`)

Media columns in JSON are URLs under `{prefix}/_pxt/media/...` (this file: `/api/_pxt/media/...`). Use that URL in a browser or `<img>` / `<video>`. Do not base64 the bytes. `return_fileresponse=True` streams the file instead of a URL.

`background=True` returns `{ "id", "job_url" }`. Poll `job_url` (`{prefix}/_pxt/jobs/{id}`). Status is `pending` | `done` | `error`, not `succeeded`. Mutually exclusive with `return_fileresponse`.

No HTTP: apply, then insert from Python. [Self-hosting](https://docs.pixeltable.com/howto/deployment/overview).

[cli.md](#reference-pixeltable-cli-reference-pxt) | [core-api.md](#serving)

## Reference: Anti-Patterns


Apps use `app.py` plus `pxt schema update`. This file is notebook form unless noted. These priors are wrong for Pixeltable.

### 1. Framework addiction (LangChain / LlamaIndex / Haystack / LangGraph)

**Wrong:** RecursiveCharacterTextSplitter + Chroma + RetrievalQA.

**Right:**

```python
class Chunks(
    TableModel,
    name='chunks',
    base=Docs,
    iterator=pxtf.document.document_splitter(Docs.document, separators='token_limit', limit=512),
):
    __indexes__ = [
        pxt.EmbeddingIndex(text, embedding=embeddings.using(model='text-embedding-3-small'), name='chunks_embed')
    ]  # type: ignore[name-defined]
```

Full pattern: [workflows.md](#reference-fastapirouter).

Chunking is `document_splitter`. Search is `.similarity()`. Tools are `pxt.tools()` + `invoke_tools()`.

### 2. pandas as a working store

**Wrong:** `df['summary'] = df['text'].apply(call_openai)` then parquet as the store.

**Right:** columns on the model (or `add_computed_column` in a notebook). `.collect().to_pandas()` is export only.

### 3. For-loops calling models

**Wrong:** `for row in df.iterrows(): openai.chat.completions.create(...)`.

**Right:** assignment on the model / computed column. Retry: `t.recompute_columns('summary', errors_only=True)`.

### 4. Separate vector database

**Wrong:** Pinecone, Chroma, FAISS, Qdrant, Weaviate, pgvector.

**Right:** `__indexes__ = [pxt.EmbeddingIndex(col, embedding=fn.using(...), name='...')]`. Notebook: `add_embedding_index`. Query: `col.similarity(string=query)`.

### 5. While-loop agents

**Wrong:** `while True:` tool loop that loses state on failure.

**Right:** insert a row. The computed-column chain runs (`chat_completions`, then `invoke_tools`, then the final answer). `invoke_tools` is per provider.

### 6. Loading a model inside the UDF body

**Wrong:** `whisper.load_model('tiny.en')` or `Model.from_pretrained(...)` called inside `@pxt.udf`. The weights reload on every row: 100ms of work becomes 1.8s.

**Right:** the shipped wrapper. It keeps a process-level model cache keyed on (model, device), so the weights load once.

```python
transcript = pxtf.whisper.transcribe(audio, model='tiny.en').text
```

Embeddings the same way: `clip.using(model_id=...)`, `sentence_transformer.using(model_id=...)`. [providers.md](#reference-pixeltable-ai-provider-reference) lists the wrappers.

If nothing ships for your model, load it once at module scope and cache the handle:

```python
import functools


@functools.cache
def _scorer():
    from my_lib import Scorer

    return Scorer.load('checkpoint.pt')


@pxt.udf
def score(text: str) -> float:
    return _scorer().score(text)
```

### Also wrong

| Prior | Do this |
|-------|---------|
| `python app.py` for models + router | `pxt schema update` then `pxt service update` |
| Drop + recreate tables as "init" | Edit `app.py`, then `pxt schema update` |
| Hard-coded `api_key=` | Env or config.toml |
| `psycopg2` against `~/.pixeltable/pgdata` | SDK / CLI only |
| Chat history in Redis | A table |
| `def f(x: str)` to "handle" a nullable column | A non-nullable parameter that receives `None` skips the call and leaves the cell `None`. Annotate `x: str \| None` |
