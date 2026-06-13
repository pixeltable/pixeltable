# OTEL Integration Plan (PXT-923)

## Context

Pixeltable has no metrics or tracing today; the only visibility is the `pixeltable` logger and ad-hoc `time.perf_counter()` timings. The product design (Notion: "Pixeltable OTEL: Product Design") calls for standard OpenTelemetry packaging where telemetry flows into whatever backend the user already runs, with visibility down to per-computed-column and per-UDF level, token/cost accounting, error counters, and throughput metrics (Notion: "OTEL", PXT-923).

Post-meeting revisions to the original docs:
- Not a purely hook-based design: `pixeltable[otel]` is a real extra that pulls in Phoenix (Arize) plus the OTEL SDK; the instrumentation bridge ships inside the pixeltable wheel.
- When the otel deps are installed, instrumentation is ON by default (traces + metrics + log correlation) for ANY running pixeltable application at Env init, not just `pxt serve`. Opt-out via config/env var.
- Core `pixeltable` carries zero OTEL dependency, not even `opentelemetry-api`. Core ships only a dependency-free hook system anyone can subscribe to.
- Spans have logging-style levels: low-value/high-volume spans are only emitted at higher verbosity.
- Per-UDF-call spans are NOT the default. The parent ExecNode span carries aggregated per-UDF stats as attributes (count, avg/total runtime, errors, tokens), since UDF calls are interwoven and aggregate tracking is what is actually useful. Individual call spans appear only at higher verbosity.

Decisions confirmed with user:
1. Instrumentation lives as a subpackage in the pixeltable wheel (`pixeltable/otel/`), import-gated. A standalone `opentelemetry-instrumentation-pixeltable` distribution can be extracted later.
2. Phoenix is the default backend: `phoenix.otel.register()` semantics (local Phoenix collector at `localhost:6006`) when nothing else is configured; standard `OTEL_*` / `PHOENIX_*` env vars and pixeltable config override.
3. Default-on scope: traces + metrics + log correlation, single opt-out.
4. `pxt serve` wiring (`[service.otel]` TOML, `--enable-otel` flag) is OUT of scope; this design must not preclude it (it doesn't: everything is Env-scoped, serve just inherits it).

## Learnings from branch `o` (partial reference implementation)

Branch `o` (commits a7c37017..a6d98bff, `pixeltable/_otel.py` + call sites) is reference only; implementation starts fresh from main.

Keep:
- **Work-span philosophy** (its `_otel.py` docstring, and the commit progression "nested self-time spans" -> "non-overlapping work spans"): spans cover contiguous units of real computation (a UDF call, an iterator step, a DB insert batch, model loading), not node lifetimes. The exec pipeline's Python work is single-threaded, so work spans don't overlap on the timeline; gaps between them are event-loop wait. Rule: no `yield`/`await` inside a CPU work span (genuinely-async external calls are the exception; in-flight overlap is semantically correct there).
- `model.load` / `processor.load` spans in `functions/huggingface.py::_lookup_model/_lookup_processor`: rare, chunky, answers "why was my first insert slow".
- `store.build_rows` (per row-batch) and `store.sql_insert` (per insert batch) granularity in `store.py::insert_rows`.
- `agg.<name>` finalize span in `aggregation_node.py` (e.g. make_video's encode, once per group).
- API ergonomics: kwargs attrs with `pxt.` prefix, None values skipped.

Avoid:
- Direct `from opentelemetry import trace` in core modules (store.py, insertable_table.py, evaluators.py, aggregation_node.py, component_iteration_node.py, huggingface.py). Replaced by the hooks hub.
- Always-on per-step spans: one span per video frame (`iter.*`), per agg-update row, per UDF call -> span explosion on any real workload. Replaced by aggregates-by-default + span levels.
- Hot-loop overhead even with no SDK configured: `get_tracer().start_as_current_span()` creates non-recording spans and does contextvar attach/detach per row/frame. The hub's `if not _SUBSCRIBERS` / level check costs ~nothing.
- `start_span()` + explicit `.end()` with no try/finally (`store.build_rows`): span leaks on the `abort_on_exc` raise path. All call sites use CM or try/finally.
- Invasive restructuring for per-step spans: `ComponentIterationNode` rewrote `for pos, component_dict in enumerate(iterator)` into a while/next loop just to wrap `next()`. With aggregate-by-default, per-step timing uses a timing-generator wrapper around `iterator` instead (no loop rewrite); per-step spans exist only at TRACE level where the wrapper emits them.
- Missing entirely (gaps this plan fills): end-attrs (insert status), metrics, error/token accounting, config/opt-out, SDK setup, tests.

## Architecture

```
core pixeltable (zero otel deps)              pixeltable[otel] installed
┌──────────────────────────────┐    ┌──────────────────────────────────────┐
│ pixeltable/hooks.py (new)    │    │ pixeltable/otel/ (new subpackage)    │
│  - subscriber registry       │◄───│  _bridge.py: _OtelSubscriber         │
│  - span levels (INFO/DEBUG/  │    │     hooks → tracer/meter calls       │
│    TRACE threshold)          │    │  _sdk.py: provider setup, Phoenix    │
│  - span_start/span_end/emit  │    │  __init__.py: init(),                │
│  - capture/restore_context   │    │     PixeltableInstrumentor           │
│                              │    └──────────────────────────────────────┘
│ call sites:                  │       activated by Env._set_up_runtime()
│  catalog ops, begin_xact,    │       when deps installed & not opted out
│  exec nodes (+udf aggregates)│
│  store, model loading        │
└──────────────────────────────┘
```

Span handle pattern (chosen over context-manager factories and pure ambient propagation): `hooks.span_start()` returns an opaque `SpanHandle` that core threads to `span_end()` and passes as `parent=` where deterministic parentage matters. A core-owned `ContextVar[SpanHandle]` covers ambient parenting (op span → everything under it); `capture_context()/restore_context()` covers the two thread-handoff points (`ExecNode._thread_iter`, `Runtime.run_coro`). Rationale: node spans start in `_open_aux` and end in `_close_aux` (not one lexical block), and iteration can run on a daemon thread with an empty contextvars context, so OTEL's own `start_as_current_span` semantics can't be relied on from core.

### Span levels (logging analogue)

Levels reuse logging-style ints: `INFO=20` (default), `DEBUG=10`, `TRACE=5`. Module-level threshold in the hub (`hooks.set_span_level()`, default INFO), set by the bridge from `otel.span_level` config at init; global across subscribers, like a root logger level. Behavior:
- The level check is the FIRST thing in `span_start`, before any attrs construction.
- A suppressed span returns its would-be parent handle (pass-through), so descendants parent to the nearest emitted ancestor and end calls on the pass-through handle are no-ops for it.

Level assignments:
- **INFO** (default): op spans (`pixeltable.insert|update|delete|create_table`), `catalog.begin_xact`, `exec.<NodeClass>` node spans, `store.sql_insert`, `model.load`/`processor.load`. Node spans are structural containers (they overlap by nature in a streaming pipeline); the meaningful numbers live in their aggregate attributes, per the work-span learning above.
- **DEBUG**: per-batch work spans (`store.build_rows`, `expr.eval` per row-batch, batched `udf.<name>` exec, `agg.<name>` finalize), per-call provider UDF spans (`udf.openai.vision` per request, matching the tech doc's sample trace).
- **TRACE**: per-row/per-step spans (scalar UDF calls, `iter.<name>` per iterator step, `agg.<name>` per-row update).

## Phase 1: Core hook hub

**New file `pixeltable/hooks.py`** (stdlib-only: `threading`, `contextvars`, `logging`; imports nothing from pixeltable, so usable from exec/exprs/catalog/store without cycles).

```python
INFO, DEBUG, TRACE = 20, 10, 5

class Subscriber:                      # all methods optional no-ops
    def on_span_start(self, name, parent_token, attrs, set_current) -> Any: ...
    def on_span_end(self, token, exc, attrs) -> None: ...
    def on_event(self, name, attrs) -> None: ...
    def capture_context(self) -> Any: ...      # called on spawning thread
    def restore_context(self, ctx) -> Any: ... # called on worker thread
    def exit_context(self, token) -> None: ...

def subscribe(s) / unsubscribe(s) / active() -> bool
def set_span_level(level: int) -> None
def span_start(name, *, level=INFO, parent=None, set_current=False, attrs=None) -> SpanHandle | None
def span_end(handle, *, exc=None, attrs=None) -> None
def emit(name, attrs=None) -> None             # discrete metric events
def span(name, *, level=INFO, **attrs)         # contextmanager sugar for lexical-block work spans
def capture_context() / restore_context(snap) / exit_context(token)
def current_span() -> SpanHandle | None
```

Properties:
- Registry is an immutable tuple swapped under a lock; every call site is guarded by `if hooks._SUBSCRIBERS:` (hot loops cache the bool per `_open()`) before building any attrs dict. `attrs` may be a callable for expensive cases. Level filtering before attrs construction.
- `SpanHandle.tokens` aligned with the subscriber-tuple snapshot taken at start (late subscribers never see an end without a start). Pass-through handles (level-suppressed) carry the parent's tokens but mark themselves no-end.
- Every subscriber call wrapped in `try/except Exception` logged to the `pixeltable.hooks` logger (rate-limited); subscriber bugs never break the host operation.
- `set_current=True` only for spans whose start/end happen in the same thread/context (op spans, begin_xact); the bridge may then `context.attach/detach` safely. Node/work spans never set ambient.
- Ambient ContextVar follows asyncio task-context copy semantics automatically.

Tests: `tests/test_hooks.py` with a `RecordingSubscriber` (no otel imports), including level-threshold and pass-through-parenting tests. Validate phase before moving on.

## Phase 2: Call sites (verified against source)

Spans: `pixeltable.insert|update|delete|create_table`, `catalog.begin_xact`, `exec.<NodeClass>`, `store.sql_insert`, `store.build_rows`, `model.load`, `processor.load`, and at DEBUG/TRACE the work spans listed under levels above.
Events: `rows.written`, `cells.computed`, `cell.error {pxt.table, pxt.column, error_type, count}`, `udf.call {pxt.udf, pxt.column, pxt.resource_pool, model, duration_s, retries, _result}` (scheduler/provider path only), `udf.stats` (per-slot aggregate at node close), `xact.retry`. Convention: `_`-prefixed keys carry raw Python objects (UDF result) consumed by bridges, never emitted as OTEL attributes.

1. **Operation spans**
   - `pixeltable/catalog/insertable_table.py::insert_table_data_source` (~line 173): the single insert funnel (`insert()` and create-table-with-source both land here). Span around the body starting at `start_ts = time.perf_counter()`, `set_current=True`, attrs `pxt.table`; end attrs from `UpdateStatus` (`num_rows`, `num_excs`, `cols_with_excs`); ends in try/finally (branch `o` had no end-attrs and combined the span CM with begin_xact in one `with`, losing status).
   - Same pattern: `insertable_table.py::delete`, `catalog/table.py::update` (~1501) / `batch_update`, `pixeltable/globals.py::create_table` (~232-256, covering DDL + optional initial insert).
2. **`catalog/catalog.py::Catalog.begin_xact`** (~332): no span for nested joins (reentrant short-circuit at ~369). Start span before the `while True` retry loop; end in a NEW outer try/finally wrapping the whole loop (the existing finally is per-attempt). One span covers all retries, end attr `pxt.retries`. Suppression rule: only start when `hooks.current_span() is not None` (begin_xact runs on every metadata access; unparented root spans would be overwhelming). Retries emit `xact.retry` events (catalog.py ~417-426 and `retry_loop()` ~138-141).
3. **`exec/exec_node.py::ExecNode`**: one wrap covers all node types in `_open_aux`/`_close_aux` (verified ~lines 177-200, run on the caller thread via `__enter__`/`__exit__`). New overridables `span_attributes()` / `span_end_attributes()` on the base. `self._span` must be reset in `_open()` (cached plans re-execute; same invariant as existing per-execution state).
   - Thread handoff: in `_thread_iter` capture next to `caller_runtime = get_runtime()`; restore inside `run()` after `copy_db_context`; also in `Runtime.run_coro` (persistent executor thread, must `exit_context` to avoid stack growth). Do NOT put in `copy_db_context` itself (capture must run on the spawning thread).
4. **Aggregated UDF stats on the ExprEvalNode span (the default story)**
   - Core-side per-slot accumulator (plain dataclass: `count`, `errors`, `total_s`, `min_s`, `max_s`, `retries`), owned by the dispatcher/node, updated by evaluators and schedulers with `time.perf_counter()` deltas: cheap float ops per call, no dict building, active only when hooks are active.
   - `ExprEvalNode._open()`: when hooks active, invert `row_builder.table_columns` into `col_names: dict[slot_idx, str]`; expose `span_handle`, `col_names`, and the accumulator on the `Dispatcher` protocol (`exec/expr_eval/globals.py`) so evaluators/schedulers reach them.
   - At node close, `span_end_attributes()` flattens aggregates onto the `exec.ExprEvalNode` span as `pxt.udf.<display_name>.count|total_s|avg_s|max_s|errors|retries` (OTEL attrs must be primitives, hence flat dotted keys), plus `num_input_rows`/`num_output_rows` (fields exist). One `udf.stats` event per slot is emitted alongside for the bridge to feed metrics. This answers "avg udf runtime etc. without logging 1000s of spans".
   - Scheduler/provider path (`schedulers.py::RateLimitsScheduler._exec` / `RequestRateScheduler._exec`): additionally emits a per-call `udf.call` event with `_result`, duration, retries; volume is network-call bound, and the raw result is required for token extraction in the bridge (providers do NOT push usage anywhere reusable today; openai.py records only rate-limit headers; usage lives in the returned JSON). Per-call spans here are DEBUG. The bridge may enrich the node span's aggregates with token totals via handle-keyed accumulation in `on_span_end`.
   - `evaluators.py::FnCallEvaluator`: batched exec gets a DEBUG span per batch; per-row scalar calls are TRACE spans, aggregate-only at lower levels. `DefaultExprEvaluator.eval`: DEBUG `expr.eval` span per row-batch.
   - Errors: `ExprEvalNode.dispatch_exc` (~336-358) is the single choke point for cell errors; in the `ignore_errors` branch emit `cell.error` (this is the "on_error='ignore' drops are visible" requirement) and bump the slot accumulator's error count. Abort path covered by op span ending with the exception.
   - `cells.computed` throughput: emit next to the existing `progress_reporter.update(...)` in `dispatch`, but NOT gated on the reporter (it is None whenever `show_progress` is off, i.e. always in production).
5. **Iterators and aggregates** (replacing branch `o`'s per-step spans)
   - `component_iteration_node.py`: wrap `iterator` in a timing-generator (`_timed_iter(iterator, acc)`) that times each `next()` into a per-node accumulator; keeps the original `for pos, component_dict in enumerate(iterator)` loop intact. Node span end-attrs: `pxt.iter.<name>.count|total_s|avg_s`. At TRACE, the wrapper emits per-step `iter.<name>` spans.
   - `aggregation_node.py`: per-row `_update_agg_state` timed into an accumulator (node span attrs); `agg.<name>` finalize span at DEBUG (once per group, chunky, e.g. make_video encode).
6. **`store.py`**: `rows.written` events next to each `sql_insert` call in `StoreBase.insert_rows` (~567-578) and `write_column` (~474-480), unconditionally (hook-guarded), since the progress reporter is None in non-interactive runs. `store.sql_insert` INFO span around `StoreBase.sql_insert` (~598), attrs `db.rows`. `store.build_rows` DEBUG span per row-batch, as a CM (branch `o` leaked it on the abort_on_exc raise path).
7. **`functions/huggingface.py::_lookup_model/_lookup_processor`**: `model.load`/`processor.load` INFO spans on the cache-miss path (adopted from branch `o`).
8. **Log correlation**: core needs NO new hook. The bridge installs a `logging.Filter` on `logging.getLogger('pixeltable')` stamping `otelTraceID/otelSpanID` from the current OTEL span; works because op spans use `set_current=True` and worker threads get context via the capture/restore handoff.

Tests: extend `tests/test_hooks.py`: insert with computed column yields correct span nesting and node-span aggregate attrs (count/avg present for the computed column's UDF); raising UDF with `on_error='ignore'` produces `cell.error` events and error counts in aggregates while insert succeeds; level threshold INFO hides batch/call spans and TRACE reveals them with correct parentage; a subscriber raising in every method doesn't break insert; start/end pairing across `_thread_iter`; unsubscribe restores inactive state.

## Phase 3: `pixeltable/otel/` bridge + SDK

```
pixeltable/otel/__init__.py   # public: init(), PixeltableInstrumentor
pixeltable/otel/_sdk.py       # provider detection/creation, Phoenix wiring, _State
pixeltable/otel/_bridge.py    # _OtelSubscriber(hooks.Subscriber)
pixeltable/otel/usage.py      # extract_usage(result) + cost table
pixeltable/otel/logs.py       # TraceContextFilter, optional OTLP LoggingHandler
```

- `PixeltableInstrumentor`: BaseInstrumentor-shaped (`instrument(tracer_provider=None, meter_provider=None)`, `uninstrument()`, idempotent) but does NOT inherit from `opentelemetry-instrumentation` (not a dep). Method names kept identical so a later `opentelemetry_instrumentor` entry point isn't precluded.
- `_OtelSubscriber`: `on_span_start` → `tracer.start_span(name, context=parent_ctx)`; if `set_current`, `context.attach`; `on_span_end` → end attrs, `record_exception` + ERROR status, `span.end()`, `detach`. `capture/restore/exit_context` map to `otel.context` get_current/attach/detach. Sets `hooks.set_span_level()` from config at instrument time.
- Metrics from events: counters `pixeltable.rows.written`, `pixeltable.cells.computed`, `pixeltable.cell.errors`, `pixeltable.udf.calls`, `pixeltable.udf.tokens` (model/column/type), `pixeltable.udf.cost`, `pixeltable.xact.retries`; histogram `pixeltable.udf.duration` (fed from `udf.stats` aggregates and per-call `udf.call` events). `usage.py::extract_usage` pops `_result` and recognizes OpenAI chat / Responses / Anthropic / embeddings usage shapes; model from attr or parsed from `pxt.resource_pool`.
- `_sdk.setup(config, ...)` init flow (must NEVER call `Env.get()`: it runs inside `Env._set_up` while `__initializing` is set and would trip the assert at env.py:104):
  1. Respect existing SDK: if `trace.get_tracer_provider()` is not a `ProxyTracerProvider`, instrument against it, never clobber (`owns_tracer_provider=False`). Manual `init(tracer_provider=...)` likewise.
  2. Phoenix default path: if `phoenix.otel` importable and no standard `OTEL_EXPORTER_OTLP_*` env vars: `phoenix.otel.register(endpoint=cfg_or_None, project_name=service_name, batch=True, set_global_tracer_provider=True, auto_instrument=True)`. None endpoint falls back to `PHOENIX_COLLECTOR_ENDPOINT` or `localhost:6006`. `auto_instrument=True` lights up any user-installed OpenInference provider instrumentors.
  3. Manual SDK path (standard OTLP env vars present or phoenix missing): `TracerProvider` + `BatchSpanProcessor(OTLPSpanExporter())`; withhold our config-derived kwargs whenever the corresponding standard env var is set so the exporter's native env resolution wins.
  4. Metrics/logs: Phoenix is traces-only; exporting metrics to `localhost:6006` would 404/retry-loop. DECISION (flag in PR): metrics are always RECORDED via the hook bridge, but the OTLP `MeterProvider`/`LoggerProvider` + exporters are only wired when an explicit OTLP endpoint exists (standard env vars or `otel.endpoint` config) or `otel.metrics`/`otel.logs` is explicitly true. This keeps "metrics on by default" without connection-error spam in the bare-Phoenix default.
  5. Process-exit flush comes free (SDK providers register their own atexit hooks); Env teardown only flushes + detaches the bridge, providers survive Env re-init (set-once per process; tests re-init Env repeatedly).
- `init()` public API: manual control, callable before any pixeltable op; sets `user_initialized` making the later Env auto-init a no-op; raises `excs.Error` if auto-init already ran (mirrors `Config.init` semantics).

## Phase 4: Packaging, config, Env wiring

1. **`pyproject.toml`**: add after the `serve` extra (~line 80):
   ```toml
   otel = [
       "opentelemetry-api>=1.39.0",
       "opentelemetry-sdk>=1.39.0",
       "opentelemetry-exporter-otlp-proto-http>=1.39.0",
       "arize-phoenix-otel>=0.16.0",
   ]
   ```
   `arize-phoenix-otel` is the lightweight `phoenix.otel` register helper (verified on PyPI: does NOT pull the full arize-phoenix server; does pull `opentelemetry-exporter-otlp` incl. grpcio, plus openinference-instrumentation/semantic-conventions and wrapt; flag grpcio weight in PR). OpenInference provider instrumentors (openai etc.) NOT in the extra; `auto_instrument=True` discovers user-installed ones; document instead. Dev group: keep the three explicit otel entries (mirrors the existing fastapi extra/dev duplication precedent), remove the redundant `opentelemetry-exporter-otlp>=1.38.0` (~line 189), add `arize-phoenix-otel>=0.16.0` so CI exercises the default path. Run `uv lock`. Wheel target already includes subpackages, so `pixeltable/otel/` ships automatically.
2. **`config.py::KNOWN_CONFIG_OPTIONS`** (~521), new section between `openrouter` and `pypi`:
   `'otel': {enabled, endpoint, service_name, headers, protocol, span_level, metrics, logs}` (scalars, no validator changes; `span_level` takes `info|debug|trace`, default `info`). Env-var mapping consequence of the `SECTION_KEY` pattern: `otel.service_name` → `OTEL_SERVICE_NAME` is a deliberate, semantics-identical overlap with the OTEL spec; `OTEL_ENABLED`/`OTEL_ENDPOINT` are incidental namespace squatting (document, don't advertise).
   Precedence: `PIXELTABLE_OTEL` > `OTEL_SDK_DISABLED=true` > `otel.enabled` config > default ON when deps installed. For endpoint/headers: standard `OTEL_EXPORTER_OTLP_*` / `PHOENIX_*` env vars > pixeltable config > Phoenix localhost default. `PIXELTABLE_OTEL` is an explicit `os.environ` check (NOT a config key: a top-level `otel = false` in TOML would error as "expected a table for section"); values `0/false/off/no` disable. This is THE documented kill switch.
3. **`env.py`**:
   - `_set_up_runtime()` (~612): append `self.__init_otel()` after `__register_packages()` (logging already configured earlier in `_set_up`; db already up, so bad otel config can never block bring-up).
   - `__init_otel()`: check `PIXELTABLE_OTEL`, `OTEL_SDK_DISABLED`, `Config.get().get_bool_value('enabled', section='otel') is False`, then probe `opentelemetry.sdk` AND `opentelemetry.exporter.otlp.proto.http` installed-ness (NOT just `opentelemetry-api`, a common transitive dep that proves nothing); only then `from pixeltable.otel import _sdk; _sdk.setup(Config.get())` inside `try/except Exception` logging a warning (failure isolation: pixeltable always comes up). Append bridge log handlers to `_managed_logging_handlers`.
   - `__register_packages()`: register `opentelemetry.sdk`, `opentelemetry.exporter.otlp.proto.http`, `phoenix.otel` with `library_name='pixeltable[otel]'` so `require_package` messages are correct.
   - `_clean_up()`: if `'pixeltable.otel' in sys.modules`, call `_sdk.on_env_teardown()` (uninstrument bridge, `force_flush` owned providers; do NOT shutdown providers, they survive Env re-init).
   - Timing: `import pixeltable` runs no Env code (init is lazy); instrumentation activates inside the first `Env.get()`, i.e. at `pxt.init()` or the first operation, before it executes.

## Phase 5: Tests, docs

1. **conftest kill switch**: `tests/conftest.py::init_env` (~93): set `os.environ['PIXELTABLE_OTEL'] = '0'` with the other env vars before `Env._init_env()` (otherwise default-on starts BatchSpanProcessor threads + connection-refused spam in every test process).
2. **`tests/test_otel.py`** (subprocess-based, pattern from `tests/test_config.py`; one fresh interpreter per scenario solves the set-once global-provider problem):
   - Import isolation (headline guarantee): `import pixeltable` then assert no `opentelemetry`/`phoenix` modules in `sys.modules`. Second variant: `pxt.init()` with `otel.enabled=false` also keeps them out.
   - Default-on: `PIXELTABLE_OTEL=1` + `OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:1` (forces manual path, no collector needed) → after `pxt.init()`, global provider is a real SDK `TracerProvider` with `service.name == 'pixeltable'`.
   - Opt-out via `PIXELTABLE_OTEL=0`, via `OTEL_ENABLED=false`, via config file.
   - Respect existing SDK: subprocess sets its own TracerProvider first → still the same object after `pxt.init()`, `owns_tracer_provider is False`.
   - No-otel-installed simulation: monkeypatch `importlib.util.find_spec` to return None for `opentelemetry*`/`phoenix*` before `import pixeltable` → `pxt.init()` succeeds, nothing imported.
   - Failure isolation: garbage `OTEL_ENDPOINT` → `pxt.init()` exits 0 with warning.
   - Env re-init: `Env._init_env()` twice → no provider-override warning, bridge re-attached.
   - Bridge correctness (in-process, otel deps are in dev group): `PixeltableInstrumentor().instrument(tracer_provider=TracerProvider(SimpleSpanProcessor(InMemorySpanExporter())), meter_provider=MeterProvider(metric_readers=[InMemoryMetricReader()]))` → run create_table/insert with a computed column; assert span names/parentage, ExprEvalNode span carries `pxt.udf.*` aggregate attrs, `cell.error` counter, `rows.written` sum, log record carries trace_id; repeat at `span_level=debug` and assert per-batch/per-call spans appear. `uninstrument()` in fixture teardown.
   - Lint-level guard: test grepping `pixeltable/` (excluding `pixeltable/otel/`) for `import opentelemetry`/`from opentelemetry`.
3. **`docs/public_api.opml`**: add `pixeltable.otel` module, `init`, `PixeltableInstrumentor` (+ methods), per repo convention (required in the same PR).
4. **Wiki**: new `wiki/subsystems/otel.md` (or extend runtime-env.md) + LOG.md entry after implementation.

## Execution order

Each phase lands as its own PR-sized unit, validated with its tests before the next (phases 1+2 could merge into one PR if review prefers):
1. `hooks.py` hub (levels included) + tests
2. Call sites (catalog ops + begin_xact, then exec/expr_eval aggregates/store/model-load) + recording-subscriber tests
3. `pixeltable/otel/` bridge + SDK (pyproject extra, config section, env wiring needed here to test end-to-end)
4. Tests/docs hardening (subprocess matrix, opml, wiki)

## Risks / open items (flag during implementation)

- **Metrics-default tension** (decision baked in above, needs PR sign-off): "metrics on by default" = always recorded; OTLP metric/log EXPORT only wires when an explicit endpoint exists, because Phoenix (the default backend) is traces-only and a default metrics exporter would retry-loop against it.
- Aggregate attrs vs the tech doc's sample trace (per-udf spans under ExprEvalNode): the doc's shape is reproduced at `span_level=debug`; the default shows the node span with aggregate attrs instead. Confirm this presentation is acceptable for the Phoenix default experience (Phoenix renders attrs fine, but per-call latency waterfalls need debug level).
- Plain `t.collect()` produces orphan `exec.*` root spans: phase-2 default is to suppress node spans when `hooks.current_span() is None` (consistent with begin_xact rule); a `pixeltable.query` op span in `_query.py` is a fast follow.
- Ordering hazard inherent to default-on: an app configuring its own OTEL after its first pixeltable op finds pixeltable's provider already global. Document: configure OTEL first, or `PIXELTABLE_OTEL=0`, or `pixeltable.otel.init(tracer_provider=...)`.
- `begin_xact` is a generator contextmanager: the new outer try/finally must end the span exactly once across retries, early return, and GeneratorExit.
- Accumulator reset: per-slot/per-node accumulators are per-execution state and must be reset in `_open()`/`reset()` (cached-plan reuse, same invariant as existing per-execution state).
- Verify at implementation time against pinned versions: exact `phoenix.otel.register()` signature (assumed from PyPI 0.16.1), metrics/logs proxy class names in otel 1.39.
- `_done_cb` swallows `KeyboardInterrupt`: interrupted background tasks may leave unended spans; OTEL tolerates this (dropped at processor shutdown).

## Verification

1. `make format && make check` after every phase.
2. Phase tests as listed; full `make test` before each PR.
3. End-to-end smoke: `pip install -e '.[otel]'` in a scratch venv, `phoenix serve` locally, run a script that creates a table with an openai computed column and inserts rows; verify in the Phoenix UI: `pixeltable.insert` → `begin_xact`/`exec.*` span tree, ExprEvalNode span showing per-udf aggregate attrs and token metrics; re-run with `span_level=debug` and verify per-call `udf.openai.*` spans appear.
4. Import-isolation check in a venv WITHOUT the extra: `python -c "import pixeltable, sys; assert not any(m.startswith('opentelemetry') for m in sys.modules)"` and confirm an insert works untraced.
