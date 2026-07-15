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

from typing import TYPE_CHECKING, TypedDict as Attrs

from pixeltable import telemetry

if TYPE_CHECKING:
    from pixeltable.catalog.update_status import UpdateStatus


class OpAttrs(Attrs, total=False):
    """Span attributes shared by all operation spans (`pixeltable.insert`, `pixeltable.create_dir`, ...).

    `table_id` (and `path` on the path-addressed entry points: `create_table`, `create_view`,
    `drop_table`, `create_dir`, `drop_dir`) identifies the operation's target at span start; the
    remaining fields are attached at span end (`table` is the table name, `version` the post-operation
    version; the counts come from the operation's `UpdateStatus`, on the operations that produce one).
    """

    table: str
    table_id: str
    path: str
    version: int
    num_rows: int
    num_computed_values: int
    num_excs: int
    updated_cols: list[str]
    cols_with_excs: list[str]


def op_status_attrs(status: UpdateStatus) -> OpAttrs:
    """The OpAttrs end attributes carried by an operation's UpdateStatus."""
    return OpAttrs(
        num_rows=status.num_rows,
        num_computed_values=status.num_computed_values,
        num_excs=status.num_excs,
        updated_cols=status.updated_cols,
        cols_with_excs=status.cols_with_excs,
    )


class XactAttrs(Attrs):
    """Span attributes for `pixeltable.catalog.begin_xact` (one span per acquisition attempt).

    A transaction targets zero or more tables (a bare metadata read targets none; `lock_mutable_tree`
    pulls in the whole view tree), so the targets are lists; None when there are none.
    """

    for_write: bool
    attempt: int
    read_table_ids: list[str] | None
    write_table_ids: list[str] | None


class CatalogAttrs(Attrs, total=False):
    """Span attributes for the `pixeltable.catalog.*` metadata spans and the `pixeltable.op.*` table ops.

    Each span sets the subset identifying its target: metadata reads/writes carry `table_id` (plus
    `version` where one applies); path resolution carries `path`.
    """

    table_id: str
    version: int | None
    path: str


class PlanAttrs(Attrs):
    """Span attributes for `pixeltable.plan.create`, attached at span end once the plan exists.

    `nodes` is the node class names in plan-chain order (consumer first, its input next, ...).
    """

    nodes: list[str]


class UdfCallAttrs(Attrs, total=False):
    """Span attributes for the per-call `pixeltable.udf.<name>` spans (DEBUG/TRACE).

    Each site sets the subset that applies: `column` when the call materializes a named table column,
    `batch_size` for batched calls, `resource_pool` and `retries` on the scheduler (resource-pool)
    execution paths.
    """

    column: str | None
    batch_size: int | None
    resource_pool: str
    retries: int


class MediaFetchAttrs(Attrs, total=False):
    """Span attributes for `pixeltable.media.fetch` (one span per downloaded file).

    `url` is set at span start; `bytes` is attached at span end on successful fetches only.
    """

    # redacted at the call site (userinfo/query/fragment stripped): presigned URLs carry credentials
    url: str
    bytes: int


class MediaSaveAttrs(Attrs):
    """Span attributes for `pixeltable.media.save` (one span per persisted file).

    `destination` is the column's configured store URI; None for the default object location.
    """

    destination: str | None
    bytes: int


class MediaDeleteAttrs(Attrs):
    """Span attributes for `pixeltable.media.delete` (one span per bulk delete, covering all of a
    table version's column destinations).

    `num_files` is the sum of the per-destination delete counts; None when no store reports one.
    """

    destinations: list[str]
    num_files: int | None


class StoreAttrs(Attrs, total=False):
    """Span attributes for the `pixeltable.store.*` and `pixeltable.sa.*` spans.

    Each span sets the subset that applies: the row-oriented spans (`build_rows`, `insert_rows`,
    `soft_delete_rows`, `write_column`) carry `rows`; `write_column`/`add_column` carry `column`;
    `create_index` carries `index`.
    """

    rows: int
    column: str
    index: str


class ModelLoadAttrs(Attrs, total=False):
    """Span attributes for `pixeltable.model.load` and `pixeltable.processor.load` (cache-miss loads).

    `model_id`/`device` are set at span start; the size fields are attached at span end and are
    metadata-only sums over a torch module's parameters (absent for non-torch models and processors).
    """

    model_id: str
    device: str | None
    param_count: int
    size_bytes: int


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
