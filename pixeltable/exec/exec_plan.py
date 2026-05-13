from __future__ import annotations

import asyncio
from types import TracebackType
from typing import Any, AsyncIterator, Iterator
from uuid import UUID

from typing_extensions import Self

import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable import exprs

from .data_row_batch import DataRowBatch
from .exec_context import ExecContext
from .exec_node import ExecNode


class ExecPlan:
    """
    Control structure for plan execution.

    All resources needed to execute a plan (sequence of ExecNodes) are included here.

    Not thread-safe.
    """

    exec_root: ExecNode
    ctx: ExecContext

    # the parameter signature, merged across nodes (one entry per distinct Variable name)
    param_types: dict[str, ts.ColumnType]
    # nodes that declared parameters; used to dispatch bind_params()
    param_nodes: list[ExecNode]

    # table id -> version map of tables referenced in this plan;
    # used to decide whether a plan is stale/needs to be re-generated
    compile_versions: dict[UUID, int]

    select_list_exprs: list[exprs.Expr]
    select_list_schema: dict[str, ts.ColumnType]

    # Serializes concurrent async iterations of the same cached plan. ExecNode state is reset in
    # _open() and mutated during iteration; two overlapping iterations would corrupt each other.
    # Lazy-init in aexec() so construction works in sync contexts (Python 3.10+ asyncio.Lock binds
    # to the running event loop on first await; constructing it without one is fragile).
    iter_lock: asyncio.Lock | None

    def __init__(
        self,
        exec_root: ExecNode,
        ctx: ExecContext,
        select_list_exprs: list[exprs.Expr],
        select_list_schema: dict[str, ts.ColumnType],
        compile_versions: dict[UUID, int] | None = None,
    ):
        self.exec_root = exec_root
        self.ctx = ctx
        self.select_list_exprs = select_list_exprs
        self.select_list_schema = select_list_schema
        self.compile_versions = compile_versions if compile_versions is not None else {}
        self.iter_lock = None

        self.param_types = {}
        self.param_nodes = []
        node: ExecNode | None = exec_root
        while node is not None:
            node.set_ctx(ctx)
            node.finalize()
            if node.bind_sources:
                self.param_nodes.append(node)
                for v in node.vars:
                    existing = self.param_types.get(v.name)
                    if existing is None:
                        self.param_types[v.name] = v.col_type
                    elif existing != v.col_type:
                        raise AssertionError(
                            f'Parameter {v.name!r} declared with conflicting types: {existing} vs {v.col_type}'
                        )
            node = node.input

    def _bind_params(self, args: dict[str, Any]) -> None:
        """Assign values to plan parameters"""
        # extra args are silently ignored (Variables may have been substituted away at compile time)
        missing = self.param_types.keys() - args.keys()
        if len(missing) > 0:
            raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f'bind_params: missing: {sorted(missing)}')

        # coerce each value to the canonical Pixeltable representation for its declared type
        coerced: dict[str, Any] = {}
        for name, expected_type in self.param_types.items():
            raw = args[name]
            # unwrap already-wrapped Literals so create_literal receives the underlying value
            if isinstance(raw, exprs.Literal):
                raw = raw.val
            try:
                coerced[name] = expected_type.create_literal(raw)
            except (TypeError, ValueError) as e:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_ARGUMENT,
                    f'bind_params: argument {name!r} is not a valid {expected_type}: {e}',
                ) from e

        for node in self.param_nodes:
            node.bind_params(coerced)

    def is_stale(self, current_versions: dict[UUID, int]) -> bool:
        """True if any captured table version differs from current_versions (or is missing)."""
        return any(current_versions.get(tbl_id) != version for tbl_id, version in self.compile_versions.items())

    def __enter__(self) -> Self:
        self.exec_root.__enter__()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        self.exec_root.__exit__(exc_type, exc_val, exc_tb)

    def __iter__(self) -> Iterator[DataRowBatch]:
        return iter(self.exec_root)

    def __aiter__(self) -> AsyncIterator[DataRowBatch]:
        return self.exec_root.__aiter__()

    def exec(self, args: dict[str, Any]) -> Iterator[exprs.DataRow]:
        """Bind args, open the plan, and yield rows. Synchronous path.

        The per-thread sync conn already serializes SQL execution; no asyncio lock here. Callers
        that drive the plan from an async context should use aexec() instead so concurrent
        iterations of the same cached plan don't interleave on shared ExecNode state.
        """
        self._bind_params(args)
        with self:
            for batch in self.exec_root:
                yield from batch

    async def aexec(self, args: dict[str, Any]) -> AsyncIterator[exprs.DataRow]:
        """Bind args, open the plan, and yield rows. Async path.

        Holds iter_lock for the lifetime of the iteration so concurrent async callers (e.g. a
        per-row inner-query template invoked from an outer ExprEvalNode) don't interleave their
        bind+iterate cycles on the same cached plan. Callers must drain the generator (or aclose
        it); standard async-generator finalization releases the lock when the generator is closed.
        """
        if self.iter_lock is None:
            self.iter_lock = asyncio.Lock()
        async with self.iter_lock:
            self._bind_params(args)
            with self:
                async for batch in self.exec_root:
                    for row in batch:
                        yield row
