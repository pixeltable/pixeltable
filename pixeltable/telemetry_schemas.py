"""Span-attribute schemas for the telemetry reported through `pixeltable.telemetry`.

Each span family gets a TypedDict here describing its attributes; call sites construct these dicts, so
key names and value types are mypy-checked at the point of production, and the file doubles as the
written inventory of what every span carries. Keys are unprefixed; the hub adds the `pxt.` prefix on
export. The hub boundary itself (`span(**attrs)`, `add_attrs(**attrs)`) stays untyped: these schemas are
a construction-time contract, not a runtime enforcement.

Metric instruments are declared at the bottom of this module (the single inventory of pixeltable metrics)
and recorded from exactly one owning call site each.
"""

from __future__ import annotations

from typing import TypedDict as Attrs

from pixeltable import telemetry


class OpAttrs(Attrs, total=False):
    """Span attributes shared by all operation spans (`pixeltable.insert`, `pixeltable.update`, ...).

    `table`/`table_id` identify the operation's target at span start; the remaining fields are attached
    at span end (`version` is the post-operation version; the counts come from the operation's
    `UpdateStatus`, on the operations that produce one).
    """

    table: str
    table_id: str
    version: int
    num_rows: int
    num_computed_values: int
    num_excs: int
    updated_cols: list[str]
    cols_with_excs: list[str]


class XactAttrs(Attrs):
    """Span attributes for `pixeltable.catalog.begin_xact` (one span per acquisition attempt).

    A transaction targets zero or more tables (a bare metadata read targets none; `lock_mutable_tree`
    pulls in the whole view tree), so the targets are lists; None when there are none.
    """

    for_write: bool
    attempt: int
    read_table_ids: list[str] | None
    write_table_ids: list[str] | None


class CatalogAttrs(Attrs):
    """Span attributes for the `pixeltable.catalog.*` metadata spans and the `pixeltable.op.*` table ops.

    Each span sets the subset identifying its target: metadata reads/writes carry `table_id` (plus
    `version` where one applies); path resolution carries `path`.
    """

    table_id: str | None
    version: int | None
    path: str | None


class PlanAttrs(Attrs):
    """Span attributes for `pixeltable.plan.create`, attached at span end once the plan exists.

    `nodes` is the node class names in plan-chain order (consumer first, its input next, ...), matching
    the nesting of the `pixeltable.exec.<NodeClass>` spans the plan produces when executed.
    """

    nodes: list[str]


class ExecNodeAttrs(Attrs):
    """Span attributes for the `pixeltable.exec.<NodeClass>` pipeline spans.

    `node` (the class name) is set at span start; the row/batch counts are attached at span end.
    `ExprEvalNode` additionally flattens per-slot UDF stats onto its span (`pxt.udf.<column>.count`,
    ...); those keys are dynamic and therefore live outside this schema.
    """

    node: str
    rows: int
    batches: int


class UdfCallAttrs(Attrs):
    """Span attributes for the per-call `pixeltable.udf.<name>` spans (DEBUG/TRACE).

    `batch_size` is None for non-batched calls; `resource_pool` and `retries` are None outside the
    scheduler (resource-pool) execution paths.
    """

    column: str
    batch_size: int | None
    resource_pool: str | None
    retries: int | None


class MediaFetchAttrs(Attrs):
    """Span attributes for `pixeltable.media.fetch` (one span per downloaded file)."""

    # redacted at the call site (userinfo/query/fragment stripped): presigned URLs carry credentials
    url: str
    bytes: int


class MediaSaveAttrs(Attrs):
    """Span attributes for `pixeltable.media.save` (one span per persisted file)."""

    destination: str
    bytes: int


class MediaDeleteAttrs(Attrs):
    """Span attributes for `pixeltable.media.delete` (one span per bulk delete, covering all of a
    table version's column destinations).

    `num_files` is the sum of the per-destination delete counts; None when no store reports one.
    """

    destinations: list[str]
    num_files: int | None


class StoreAttrs(Attrs):
    """Span attributes for the `pixeltable.store.*` and `pixeltable.sa.*` spans.

    Each span sets the subset that applies: the row-oriented spans (`build_rows`, `insert_rows`,
    `soft_delete_rows`, `write_column`) carry `rows`; `write_column`/`add_column` carry `column`;
    `create_index` carries `index`.
    """

    rows: int | None
    column: str | None
    index: str | None


class ModelLoadAttrs(Attrs):
    """Span attributes for `pixeltable.model.load` and `pixeltable.processor.load` (cache-miss loads).

    The size fields are metadata-only sums over a torch module's parameters; None for non-torch models
    and processors.
    """

    model_id: str
    device: str | None
    param_count: int | None
    size_bytes: int | None


rows_written = telemetry.counter('pixeltable.rows.written', '{row}')
cells_computed = telemetry.counter('pixeltable.cells.computed', '{cell}')
cells_errors = telemetry.counter('pixeltable.cells.errors', '{error}')
udf_calls = telemetry.counter('pixeltable.udf.calls', '{call}')
udf_errors = telemetry.counter('pixeltable.udf.errors', '{error}')
udf_retries = telemetry.counter('pixeltable.udf.retries', '{retry}')
udf_latency = telemetry.histogram('pixeltable.udf.latency', 's')
udf_input_tokens = telemetry.counter('pixeltable.udf.input_tokens', '{token}')
udf_output_tokens = telemetry.counter('pixeltable.udf.output_tokens', '{token}')
media_fetched_bytes = telemetry.counter('pixeltable.media.fetched_bytes', 'By')
media_saved_bytes = telemetry.counter('pixeltable.media.saved_bytes', 'By')
