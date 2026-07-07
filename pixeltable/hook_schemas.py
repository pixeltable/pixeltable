"""Shared schemas for the telemetry reported through `pixeltable.hooks`.

Each span family and each event gets a TypedDict here describing its payload; core call sites construct
these dicts and the OTEL bridge derives its metric instruments from them, so a field's name, type, and
metric binding are declared exactly once. Keys are unprefixed; the hub adds the `pxt.` prefix on export.

`Attrs` schemas describe span attributes (both are plain TypedDicts; the alias only marks the role). A
field annotated with a `CounterMetric` marker feeds that counter instrument; a field marked `DIM` is
exported as a metric attribute and must therefore be low-cardinality.
"""

from __future__ import annotations

import dataclasses
import typing
from typing import TypedDict as Attrs, TypedDict as Event


@dataclasses.dataclass(frozen=True)
class CounterMetric:
    """Marks a field whose value is added to the named counter instrument."""

    name: str
    unit: str = ''


class _Dim:
    pass


# Marks a field exported as a metric attribute on the datapoints its schema produces.
DIM: typing.Final = _Dim()


class OpAttrs(Attrs, total=False):
    """Span attributes shared by all operation spans (`pixeltable.insert`, `pixeltable.update`, ...).

    `table`/`table_id` identify the operation's target at span start; the remaining fields are attached
    at span end (`version` is the post-operation version; the counts come from the operation's
    `UpdateStatus`, on the operations that produce one).
    """

    table: typing.Annotated[str, DIM]
    table_id: typing.Annotated[str, DIM]
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

    node: typing.Annotated[str, DIM]
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
    bytes: typing.Annotated[int, CounterMetric('pixeltable.media.fetched_bytes', 'By')]


class MediaSaveAttrs(Attrs):
    """Span attributes for `pixeltable.media.save` (one span per persisted file)."""

    destination: str
    bytes: typing.Annotated[int, CounterMetric('pixeltable.media.saved_bytes', 'By')]


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


class RowsWritten(Event):
    """Payload for the `pixeltable.rows.written` event, emitted per store write batch."""

    table: typing.Annotated[str, DIM]
    table_id: typing.Annotated[str, DIM]
    rows: typing.Annotated[int, CounterMetric('pixeltable.rows.written', '{row}')]


class CellsComputed(Event):
    """Payload for the `pixeltable.cells.computed` event, emitted per dispatched batch in ExprEvalNode."""

    table: typing.Annotated[str, DIM]
    table_id: typing.Annotated[str, DIM]
    cells: typing.Annotated[int, CounterMetric('pixeltable.cells.computed', '{cell}')]
    errors: typing.Annotated[int, CounterMetric('pixeltable.cells.errors', '{error}')]


class UdfStats(Event):
    """Payload for the `pixeltable.udf.stats` event, emitted once per UDF slot when its node closes.

    `column` is None for slots that don't materialize a named column. `total_s` is exported as a counter
    rather than a histogram: sum plus `count` lets backends compute an average over any window without
    per-call instrument overhead. `min_s`/`max_s` appear on the ExprEvalNode span only.
    """

    udf: typing.Annotated[str, DIM]
    column: str | None
    count: typing.Annotated[int, CounterMetric('pixeltable.udf.calls', '{call}')]
    errors: typing.Annotated[int, CounterMetric('pixeltable.udf.errors', '{error}')]
    retries: typing.Annotated[int, CounterMetric('pixeltable.udf.retries', '{retry}')]
    total_s: typing.Annotated[float, CounterMetric('pixeltable.udf.duration_total', 's')]
    min_s: float | None
    max_s: float | None


class UdfUsage(Event):
    """Payload for the `pixeltable.udf.usage` event: token usage of one provider call.

    Emitted only when the provider response carries recognizable usage; extraction happens at the emit
    site (core owns the response shapes).
    """

    udf: typing.Annotated[str, DIM]
    input_tokens: typing.Annotated[int, CounterMetric('pixeltable.udf.input_tokens', '{token}')]
    output_tokens: typing.Annotated[int, CounterMetric('pixeltable.udf.output_tokens', '{token}')]
