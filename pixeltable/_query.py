from __future__ import annotations

import copy
import hashlib
import json
import urllib.parse
from pathlib import Path
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Generator,
    Hashable,
    Iterable,
    Iterator,
    Literal,
    NoReturn,
    Self,
    TypeVar,
)
from uuid import UUID

import pandas as pd
import PIL.Image
import pydantic
import sqlalchemy.exc as sql_exc

from pixeltable import catalog, exceptions as excs, exec, exprs
from pixeltable._query_base import QueryBase
from pixeltable.catalog import fold_identifier
from pixeltable.catalog.update_status import UpdateStatus
from pixeltable.env import Env
from pixeltable.plan import Planner
from pixeltable.query_clauses import FromClause, JoinClause, JoinType
from pixeltable.row import Row
from pixeltable.runtime import get_runtime
from pixeltable.service.proxy_client import ProxyClient
from pixeltable.type_system import ColumnType
from pixeltable.utils.formatter import Formatter

if TYPE_CHECKING:
    import torch.utils.data

__all__ = ['Query', 'ResultCursor', 'ResultSet']


class ResultSet:
    """
    A dataset obtained by executing a [`Query`][pixeltable.Query]. Returned by
    [`Query.collect()`][pixeltable.Query.collect], [`Query.head()`][pixeltable.Query.head],
    [`Query.tail()`][pixeltable.Query.tail], and the equivalent methods on class [`Table`][pixeltable.Table].

    A `ResultSet` is structured as a table with rows (indexed by integers) and columns (indexed by strings).
    The column names correspond to the expressions in the query's select list. The values in a `ResultSet` can
    be accessed in various ways:

    - `len(result)` returns the number of rows
    - `result[i]` returns the `i`th row as a `dict` mapping column names to values
    - `result['col']` returns a `list` of all values in the column named `'col'`
    - `result[i, 'col']` returns the specific value in the `i`th row and column `'col'`

    `ResultSet` implements the Sequence protocol, so it can be iterated over and converted to other sequence
    types in the usual fashion; for example:

    - `for row in result` (iterates over rows)
    - `list(result)` (converts to a list of rows)
    """

    _rows: list[Row]
    _col_names: list[str]
    _schema: dict[str, ColumnType]  # internal column types
    __formatter: Formatter

    def __init__(self, rows: list[Row], schema: dict[str, ColumnType]):
        self._rows = rows
        self._col_names = list(schema.keys())
        self._schema = schema
        self.__formatter = Formatter(len(self._rows), len(self._col_names), Env.get().http_address)

    @property
    def schema(self) -> dict[str, str]:
        """The result columns as a mapping from name to its type string."""
        # matches Table.get_metadata()
        return {name: repr(col_type) for name, col_type in self._schema.items()}

    def __len__(self) -> int:
        return len(self._rows)

    def __repr__(self) -> str:
        return self.to_pandas().__repr__()

    def _repr_html_(self) -> str:
        formatters: dict[Hashable, Callable[[object], str]] = {}
        for col_name, col_type in self._schema.items():
            formatter = self.__formatter.get_pandas_formatter(col_type)
            if formatter is not None:
                formatters[col_name] = formatter
        return self.to_pandas().to_html(formatters=formatters, escape=False, index=False)

    def __str__(self) -> str:
        return self.to_pandas().to_string()

    def _reverse(self) -> None:
        """Reverse order of rows"""
        self._rows.reverse()

    def to_pandas(self) -> pd.DataFrame:
        """Convert the `ResultSet` to a Pandas `DataFrame`.

        Returns:
            A `DataFrame` with one column per column in the `ResultSet`.
        """
        return pd.DataFrame.from_records([row._data for row in self._rows], columns=self._col_names)

    BaseModelT = TypeVar('BaseModelT', bound=pydantic.BaseModel)

    def to_pydantic(self, model: type[BaseModelT]) -> Iterator[BaseModelT]:
        """
        Convert the `ResultSet` to Pydantic model instances.

        Args:
            model: A Pydantic model class.

        Returns:
            An iterator over Pydantic model instances, one for each row in the result set.

        Raises:
            Error: If the row data doesn't match the model schema.
        """
        model_fields = model.model_fields
        model_config = getattr(model, 'model_config', {})
        forbid_extra_fields = model_config.get('extra') == 'forbid'

        # schema validation; model field names are Python attributes and case-sensitive, whereas result column names
        # are always folded, so match the two on their folded forms.
        folded_field_name_to_original: dict[str, str] = {}
        for name in model_fields:
            folded = fold_identifier(name)
            if folded in folded_field_name_to_original:
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'Result column names are case-insensitive, but model {model.__name__} has fields '
                    f'{folded_field_name_to_original[folded]!r} and {name!r} which both denote {folded!r}',
                )
            folded_field_name_to_original[folded] = name
        required_fields = {fold_identifier(name) for name, field in model_fields.items() if field.is_required()}
        col_names = set(self._col_names)
        missing_fields = {folded_field_name_to_original[name] for name in required_fields - col_names}
        if len(missing_fields) > 0:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                f'Required model fields ({missing_fields}) are missing from '
                f'result set columns ({", ".join(self._col_names)})',
            )
        if forbid_extra_fields:
            extra_fields = col_names - set(folded_field_name_to_original.keys())
            if len(extra_fields) > 0:
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f"Extra fields {extra_fields} are not allowed in model with extra='forbid'",
                )

        for row in self:
            # remap to the model's original spelling
            remapped_row = {folded_field_name_to_original.get(name, name): val for name, val in row.items()}
            try:
                yield model(**remapped_row)
            except pydantic.ValidationError as e:
                raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, str(e)) from e

    def _row_to_dict(self, row_idx: int) -> dict[str, Any]:
        return dict(self._rows[row_idx].items())

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, str):
            col_name = fold_identifier(index)
            if col_name not in self._col_names:
                raise excs.RequestError(excs.ErrorCode.INVALID_COLUMN_NAME, f'Invalid column name: {index}')
            return [row[col_name] for row in self._rows]
        if isinstance(index, int):
            return self._row_to_dict(index)
        if isinstance(index, tuple) and len(index) == 2:
            if not isinstance(index[0], int) or not isinstance(index[1], (str, int)):
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'Bad index, expected [<row idx>, <column name | column index>]: {index}',
                )
            if isinstance(index[1], str):
                col_name = fold_identifier(index[1])
                if col_name not in self._col_names:
                    raise excs.RequestError(excs.ErrorCode.INVALID_COLUMN_NAME, f'Invalid column name: {index[1]}')
            else:
                col_name = self._col_names[index[1]]
            return self._rows[index[0]][col_name]
        raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, f'Bad index: {index}')

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return (self._row_to_dict(i) for i in range(len(self)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResultSet):
            return False
        return self.to_pandas().equals(other.to_pandas())

    def __hash__(self) -> int:
        return hash(self.to_pandas())


class ResultCursor(Iterable[Row]):
    """Cursor that iterates over query results.

    Wraps a Query and yields Row objects one at a time,
    avoiding materializing all results into memory.

    A cursor transitions through three states: pending (created but not yet started), open (actively
    iterating), and closed (resources released). Iteration auto-opens and auto-closes the cursor, or you can
    use it as a context manager for explicit lifecycle control.

    Examples:
        Iterate over all rows in a table:

        >>> for row in t.cursor():
        ...     print(row['col_name'])

        Use as a context manager for early termination:

        >>> with t.select(t.col1, t.col2).cursor() as cur:
        ...     for row in cur:
        ...         if row['col1'] > threshold:
        ...             break  # resources are released on exit
    """

    _query: Query
    _row_iterator: Generator[list[Any], None, None] | None
    _schema: dict[str, ColumnType]
    _columns: dict[str, int]  # column name -> position
    _closed: bool

    def __init__(self, query: Query):
        self._query = query
        self._row_iterator = None
        # Known design issue: cursor construction is separated from transaction start. The
        # transaction that reads the rows is opened later, inside _output_row_iterator, so the
        # schema captured here has no causal link to the schema the iteration runs under. A
        # schema mutation between __init__ and the first next() can leave _schema/_columns
        # inconsistent with the yielded rows, especially for SELECT * queries where
        # Query.schema re-resolves against current catalog state on each access.
        self._schema = query.schema
        self._columns = {name: i for i, name in enumerate(self._schema)}
        self._closed = False

    @property
    def schema(self) -> dict[str, str]:
        """The result columns as a mapping from name to its type string."""
        # matches Table.get_metadata()
        return {name: repr(col_type) for name, col_type in self._schema.items()}

    def open(self) -> None:
        """Start the underlying query and prepare the cursor for iteration.

        Raises an error if the cursor is already open or has been closed.
        Called automatically when iterating if not already open.
        """
        if self._row_iterator is not None:
            raise excs.RequestError(excs.ErrorCode.INVALID_STATE, 'Cursor is already open.')
        if self._closed:
            raise excs.RequestError(excs.ErrorCode.INVALID_STATE, 'Cursor is closed and cannot be reopened.')
        self._row_iterator = self._query._output_row_iterator()

    def close(self) -> None:
        """Release the underlying database transaction and query resources.

        Safe to call multiple times. Once closed, the cursor cannot be reopened.
        Also called automatically via the context manager protocol and on garbage collection.
        """
        if self._closed:
            return
        if self._row_iterator is not None:
            # Sends GeneratorExit into _output_row_iterator, unwinding begin_xact()
            self._row_iterator.close()
        self._row_iterator = None
        self._closed = True

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def __iter__(self) -> Iterator[Row]:
        if self._closed:
            raise excs.RequestError(excs.ErrorCode.INVALID_STATE, 'Cursor is closed and cannot be iterated upon.')
        if self._row_iterator is None:
            self.open()
        assert self._row_iterator is not None
        try:
            for data in self._row_iterator:
                yield Row(data, self._columns, self._schema)
        finally:
            self.close()

    def __repr__(self) -> str:
        if self._closed:
            state = 'closed'
        elif self._row_iterator is not None:
            state = 'open'
        else:
            state = 'pending'
        cols = ', '.join(f'{name}: {col_type}' for name, col_type in self._schema.items())
        return f'ResultCursor({state}, columns=[{cols}])'


class ProxyResultCursor(ResultCursor):
    """A ResultCursor over a result fetched in full from a proxy catalog.

    The daemon's media URLs are fetched into the local store here, when the cursor is opened.

    TODO: implement a streaming protocol
    """

    _client: ProxyClient
    _rows: list[list[Any]]

    def __init__(self, query: Query, client: ProxyClient, schema: dict[str, ColumnType], rows: list[list[Any]]):
        super().__init__(query)
        self._client = client
        self._rows = rows
        # use the fetched result's schema/columns as authoritative (avoids any client-side schema staleness)
        self._schema = schema
        self._columns = {name: i for i, name in enumerate(schema)}

    def open(self) -> None:
        if self._row_iterator is not None:
            raise excs.RequestError(excs.ErrorCode.INVALID_STATE, 'Cursor is already open.')
        if self._closed:
            raise excs.RequestError(excs.ErrorCode.INVALID_STATE, 'Cursor is closed and cannot be reopened.')
        self._materialize_media()
        # a generator (not a plain iterator) so the inherited close() can call _row_iterator.close()
        self._row_iterator = (row for row in self._rows)

    def as_result_set(self) -> ResultSet:
        return ResultSet(list(self), self._schema)

    def _materialize_media(self) -> None:
        """Materialize daemon-resident media files into the local store/as PIL images."""
        media_items: list[tuple[int, bool]] = []  # (select list item index, whether PIL image)
        for i, (e, _) in enumerate(self._query._effective_select_list):
            is_localpath = (
                isinstance(e, exprs.ColumnPropertyRef) and e.prop == exprs.ColumnPropertyRef.Property.LOCALPATH
            )
            if e.col_type.is_image_type():
                media_items.append((i, True))
            elif e.col_type.is_media_type() or is_localpath:
                media_items.append((i, False))
        if len(media_items) == 0:
            return

        media_urls: set[str] = set()
        for row in self._rows:
            for i, _ in media_items:
                val = row[i]
                if isinstance(val, str):
                    scheme = urllib.parse.urlparse(val).scheme
                    if len(scheme) > 1 and scheme != 'file':
                        media_urls.add(val)
        local_paths = self._client.fetch_media(list(media_urls)) if len(media_urls) > 0 else {}

        for row in self._rows:
            for i, is_img in media_items:
                val = row[i]
                if isinstance(val, str):
                    val = local_paths.get(val, val)
                    if is_img:
                        with PIL.Image.open(val) as img:
                            val = img.copy()
                    row[i] = val


class Query(QueryBase):
    """A query over tables in a catalog: it can be executed, and its rows updated or deleted.

    Thread-safe.
    """

    def _mutation_target(self) -> catalog.Table:
        from_path = self._from_clause.tbls[0]
        tbl = get_runtime().get_table_by_id(from_path.tbl_id, version=from_path.effective_version())
        assert tbl is not None
        return tbl

    def _resolve_positive_int(self, e: exprs.Expr | None, role: str, args: dict[str, Any]) -> int | None:
        """Resolve Expr to a positive int, or None if not set."""
        if e is None:
            return None
        if isinstance(e, exprs.Literal):
            val = e.val
        else:
            assert isinstance(e, exprs.Variable)
            val = args[e.name]
        if val < 0:
            raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f"'{role}' parameter must be >= 0")
        return val

    def _resolved_limit(self, args: dict[str, Any]) -> int | None:
        return self._resolve_positive_int(self.limit_val, 'limit', args)

    def _resolved_offset(self, args: dict[str, Any]) -> int | None:
        return self._resolve_positive_int(self.offset_val, 'offset', args)

    def _validate_bound_args(self, args: dict[str, Any]) -> None:
        # Raised exceptions are caught and recorded per-cell when this Query is invoked
        # via a query UDF inside a computed column (see ExprEvalNode evaluators).
        # _resolved_limit/_resolved_offset perform the type and range checks.
        self._resolved_limit(args)
        self._resolved_offset(args)

    def _exec(self, args: dict[str, Any] | None = None) -> Iterator[exprs.DataRow]:
        """Run the query and yield rows.

        Slot indices live on the planned exprs returned by select_list_exprs(); callers that
        need them must read from there, not from this Query's _select_list_exprs (which are the
        pre-compile copies and don't carry slot_idx).
        """
        args = args or {}
        self._validate_bound_args(args)
        if self._resolved_limit(args) == 0:
            return
        plan = self._ensure_plan()
        for row in plan.exec(args):
            # stop progress output before we display anything, otherwise it'll mess up the output
            get_runtime().stop_progress()
            yield row

    async def _aexec(self, args: dict[str, Any] | None = None) -> AsyncIterator[exprs.DataRow]:
        """Run the query and yield rows."""
        args = args or {}
        self._validate_bound_args(args)
        if self._resolved_limit(args) == 0:
            return
        plan = self._ensure_plan()
        async for row in plan.aexec(args):
            yield row

    def _ensure_plan(self) -> exec.ExecPlan:
        assert get_runtime().in_xact
        cache = get_runtime().plan_cache
        plan = cache.get(self)
        if plan is not None and plan.matches_versions(self._from_clause_tbl_versions()):
            return plan
        plan = self._create_query_plan()
        cache[self] = plan
        return plan

    def _from_clause_tbl_versions(self) -> dict[UUID, int]:
        assert self._from_clause.is_local
        out: dict[UUID, int] = {}
        for tbl in self._from_clause.tvps:
            for tvh in tbl.get_tbl_versions():
                out[tvh.id] = tvh.get().version
        return out

    def _create_query_plan(self) -> exec.ExecPlan:
        assert self._from_clause.is_local
        tvps = self._from_clause.tvps
        has_operational_tbl = any(not tbl.tbl_version.get().is_data_versioned for tbl in tvps)
        if has_operational_tbl:
            # For now, we only support queries of the simplest form on operational tables
            assert len(self._from_clause.tbls) == 1, 'TODO: implement for operational tables [PXT-1101]'
            assert len(self._from_clause.join_clauses) == 0, 'TODO: implement for operational tables [PXT-1101]'
            assert self.grouping_tbl_key is None, 'TODO: implement for operational tables [PXT-1101]'
            assert self.group_by_clause is None, 'TODO: implement for operational tables [PXT-1101]'
            assert self.sample_clause is None, 'TODO: implement for operational tables [PXT-1101]'

        # construct a group-by clause if we're grouping by a table
        group_by_clause = self.group_by_clause
        if self.grouping_tbl_key is not None:
            assert group_by_clause is None
            grouping_tv = get_runtime().catalog.get_tbl_version(self.grouping_tbl_key)
            num_rowid_cols = len(grouping_tv.store_tbl.rowid_columns())
            # the grouping table must be a base of self.tbl
            first_tbl = tvps[0]
            assert num_rowid_cols <= len(first_tbl.tbl_version.get().store_tbl.rowid_columns())
            group_by_clause = Planner.rowid_columns(first_tbl.tbl_version, num_rowid_cols)

        select_list = self._effective_select_list
        select_list_exprs = [e for e, _ in select_list]
        select_list_schema = {n: e.col_type for e, n in select_list}
        root = Planner.create_query_plan(
            self._from_clause,
            select_list_exprs,
            where_clause=self.where_clause,
            group_by_clause=group_by_clause,
            order_by_clause=self.order_by_clause,
            limit=self.limit_val,
            offset=self.offset_val,
            sample_clause=self.sample_clause,
        )
        compile_versions = self._from_clause_tbl_versions()
        return exec.ExecPlan(
            root,
            root.ctx,
            select_list_exprs=select_list_exprs,
            select_list_schema=select_list_schema,
            compile_versions=compile_versions,
        )

    def __rowid_columns(self, num_rowid_cols: int | None = None) -> list[exprs.Expr]:
        """Return list of RowidRef for the given number of associated rowids"""
        return Planner.rowid_columns(self._first_tbl.tbl_version, num_rowid_cols)

    def show(self, n: int = 20) -> ResultSet:
        if self.sample_clause is not None:
            raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, 'show() cannot be used with sample()')
        assert n is not None
        return self.limit(n).collect()

    def head(self, n: int = 10) -> ResultSet:
        """Return the first n rows of the Query, in insertion order of the underlying Table.

        head() is not supported for joins.

        Args:
            n: Number of rows to select. Default is 10.

        Returns:
            A ResultSet with the first n rows of the Query.

        Raises:
            Error: If the Query is the result of a join or
                if the Query has an order_by clause.
        """
        return self._head(n)

    def _head(self, n: int = 10, *, media_as_urls: bool = False) -> ResultSet:
        if self.order_by_clause is not None:
            raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, 'head() cannot be used with order_by()')
        if self._has_joins():
            raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, 'head() not supported for joins')
        if self.sample_clause is not None:
            raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, 'head() cannot be used with sample()')
        if self.group_by_clause is not None:
            raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, 'head() cannot be used with group_by()')
        if not self._from_clause.is_local:
            # the rowid order_by needs the table's local store; run head() on the hosting catalog instead
            return self._exec_proxy('head', n=n).as_result_set()
        num_rowid_cols = len(self._first_tbl.tbl_version.get().store_tbl.rowid_columns())
        order_by_clause = [exprs.RowidRef(self._first_tbl.tbl_version, idx) for idx in range(num_rowid_cols)]
        return self.order_by(*order_by_clause, asc=True).limit(n)._collect(media_as_urls=media_as_urls)

    def tail(self, n: int = 10) -> ResultSet:
        """Return the last n rows of the Query, in insertion order of the underlying Table.

        tail() is not supported for joins.

        Args:
            n: Number of rows to select. Default is 10.

        Returns:
            A ResultSet with the last n rows of the Query.

        Raises:
            Error: If the Query is the result of a join or
                if the Query has an order_by clause.
        """
        return self._tail(n)

    def _tail(self, n: int = 10, *, media_as_urls: bool = False) -> ResultSet:
        if self.order_by_clause is not None:
            raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, 'tail() cannot be used with order_by()')
        if self._has_joins():
            raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, 'tail() not supported for joins')
        if self.sample_clause is not None:
            raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, 'tail() cannot be used with sample()')
        if self.group_by_clause is not None:
            raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, 'tail() cannot be used with group_by()')
        if not self._from_clause.is_local:
            # the rowid order_by needs the table's local store; run tail() on the hosting catalog instead
            return self._exec_proxy('tail', n=n).as_result_set()
        num_rowid_cols = len(self._first_tbl.tbl_version.get().store_tbl.rowid_columns())
        order_by_clause = [exprs.RowidRef(self._first_tbl.tbl_version, idx) for idx in range(num_rowid_cols)]
        result = self.order_by(*order_by_clause, asc=False).limit(n)._collect(media_as_urls=media_as_urls)
        result._reverse()
        return result

    def _raise_expr_eval_err(self, e: excs.ExprEvalError) -> NoReturn:
        excs.raise_from_expr_eval_err(e)

    def _compiled_select_list(self) -> list[exprs.Expr]:
        """Select list exprs that can be evaluated in the context of a plan (has slot_idxs assigned)."""
        return self._ensure_plan().select_list_exprs

    def _output_row_iterator(
        self, args: dict[str, Any] | None = None, *, media_as_urls: bool = False
    ) -> Generator[list, None, None]:
        assert self._from_clause.is_local
        tbl_ids = self.referenced_tbl_ids()
        tvps = self._from_clause.tvps
        with get_runtime().catalog.begin_xact(for_write=False, read_tvps=tvps, read_tbl_ids=tbl_ids):
            try:
                planned_exprs = self._compiled_select_list()
                if media_as_urls:
                    for data_row in self._exec(args=args):
                        # for a file-backed media cell, hand back its URL instead of materializing the value
                        yield [
                            data_row.file_urls[e.slot_idx]
                            if e.col_type.is_media_type() and data_row.file_urls[e.slot_idx] is not None
                            else data_row[e.slot_idx]
                            for e in planned_exprs
                        ]
                else:
                    for data_row in self._exec(args=args):
                        yield [data_row[e.slot_idx] for e in planned_exprs]
            except excs.ExprEvalError as e:
                self._raise_expr_eval_err(e)
            except (sql_exc.DBAPIError, sql_exc.OperationalError, sql_exc.InternalError) as e:
                single_tbl = next(iter(tbl_ids)) if len(tbl_ids) == 1 else None
                get_runtime().catalog.convert_sql_exc(e, tbl_id=single_tbl)
                raise  # just re-raise if not converted to a Pixeltable error

    def collect(self) -> ResultSet:
        return self._collect()

    _ProxyMethodNames = Literal['collect', 'head', 'tail']

    def _exec_proxy(self, method: _ProxyMethodNames, **extra: Any) -> ProxyResultCursor:
        from pixeltable.catalog.catalog_proxy import CatalogProxy

        cat = get_runtime().get_catalog(self._from_clause.catalog_uri)
        assert isinstance(cat, CatalogProxy)
        result = cat.client.run_query(method, self.as_dict(), **extra)
        return ProxyResultCursor(self, cat.client, result['schema'], result['rows'])

    def _collect(self, args: dict[str, Any] | None = None, *, media_as_urls: bool = False) -> ResultSet:
        if not self._from_clause.is_local:
            return self._exec_proxy('collect', args=args).as_result_set()
        tvps = self._from_clause.tvps
        with get_runtime().catalog.begin_xact(for_write=False, read_tvps=tvps, read_tbl_ids=self.referenced_tbl_ids()):
            schema = self.schema
            # url-mode takes the direct path; the cursor path (no args) stays as-is for normal execution
            if args is None and not media_as_urls:
                return ResultSet(list(self.cursor()), schema)
            columns = {name: i for i, name in enumerate(schema)}
            rows = [
                Row(tuple(data), columns, schema)
                for data in self._output_row_iterator(args=args, media_as_urls=media_as_urls)
            ]
            return ResultSet(rows, schema)

    def cursor(self) -> ResultCursor:
        """Return a [`ResultCursor`][pixeltable.ResultCursor] that iterates over the query results row by row.

        See [`ResultCursor`][pixeltable.ResultCursor] for usage examples and lifecycle details.
        """
        if not self._from_clause.is_local:
            return self._exec_proxy('collect')
        return ResultCursor(self)

    async def _acollect(self, args: dict[str, Any] | None = None) -> ResultSet:
        # this can only be called in the context of a running transaction
        assert get_runtime().in_xact
        single_tbl = self._first_tbl if len(self._from_clause.tbls) == 1 else None
        schema = self.schema
        columns = {name: i for i, name in enumerate(schema)}
        try:
            planned_exprs = self._compiled_select_list()
            result = [
                Row(tuple(row[e.slot_idx] for e in planned_exprs), columns, schema)
                async for row in self._aexec(args=args)
            ]
            return ResultSet(result, schema)
        except excs.ExprEvalError as e:
            self._raise_expr_eval_err(e)
        except (sql_exc.DBAPIError, sql_exc.OperationalError, sql_exc.InternalError) as e:
            get_runtime().catalog.convert_sql_exc(e, tbl=(single_tbl.tbl_version if single_tbl is not None else None))
            raise  # just re-raise if not converted to a Pixeltable error

    def count(self) -> int:
        """Return the number of rows in the Query.

        Returns:
            The number of rows in the Query.
        """
        from pixeltable.functions.globals import count as pxt_count

        if self.limit_val is not None or self.offset_val is not None:
            # supporting these would require wrapping the limited query in a subquery and counting
            # that, which the current SqlAggregationNode path doesn't do;
            # count() is meant for exploration, so no need to make every corner case work
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                'count() cannot be used with limit() or offset(). Use `select(pxtf.count())` instead.',
            )

        if not self._from_clause.is_local:
            from pixeltable.catalog.catalog_proxy import CatalogProxy

            cat = get_runtime().get_catalog(self._from_clause.catalog_uri)
            assert isinstance(cat, CatalogProxy)
            return cat.client.run_query('count', self.as_dict())

        count_query = Query(
            from_clause=self._from_clause,
            select_list=[(pxt_count(1), 'count')],
            where_clause=self.where_clause,
            group_by_clause=self.group_by_clause,
            grouping_tbl_key=self.grouping_tbl_key,
            sample_clause=copy.deepcopy(self.sample_clause),
        )
        is_grouped = self.group_by_clause is not None or self.grouping_tbl_key is not None

        assert self._from_clause.is_local
        with get_runtime().catalog.begin_xact(for_write=False, read_tvps=self._from_clause.tvps):
            plan_root = count_query._ensure_plan().exec_root
            if not isinstance(plan_root, exec.SqlNode):
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'count() cannot be used: query plan contains a non-SQL node ({type(plan_root).__name__})',
                )

        result = count_query.collect()
        if is_grouped:
            return len(result)
        assert len(result) == 1
        return int(result[0, 'count'])

    def update(self, value_spec: dict[str, Any], cascade: bool = True) -> UpdateStatus:
        """Update rows in the underlying table of the Query.

        Update rows in the table with the specified value_spec.

        Args:
            value_spec: a dict of column names to update and the new value to update it to.
            cascade: if True, also update all computed columns that transitively depend
                    on the updated columns, including within views. Default is True.

        Returns:
            UpdateStatus: the status of the update operation.

        Example:
            Given the Query person from a table t with all its columns and rows:

            >>> person = t.select()

            Via the above Query person, update the column 'city' to 'Oakland'
            and 'state' to 'CA' in the table t:

            >>> person.update({'city': 'Oakland', 'state': 'CA'})

            Via the above Query person, update the column 'age' to 30 for any
            rows where 'year' is 2014 in the table t:

            >>> person.where(t.year == 2014).update({'age': 30})
        """
        self._validate_mutable('update')
        return self._mutation_target().update(value_spec, where=self.where_clause, cascade=cascade)

    def recompute_columns(
        self, *columns: str | exprs.ColumnRef, errors_only: bool = False, cascade: bool = True
    ) -> UpdateStatus:
        """Recompute one or more computed columns of the underlying table of the Query.

        Args:
            columns: The names or references of the computed columns to recompute.
            errors_only: If True, only run the recomputation for rows that have errors in the column (ie, the column's
                `errortype` property indicates that an error occurred). Only allowed for recomputing a single column.
            cascade: if True, also update all computed columns that transitively depend on the recomputed columns.

        Returns:
            UpdateStatus: the status of the operation.

        Example:
            For table `person` with column `age` and computed column `height`, recompute the value of `height` for all
            rows where `age` is less than 18:

            >>> query = person.where(t.age < 18).recompute_columns(person.height)
        """
        self._validate_mutable('recompute_columns')
        return self._mutation_target().recompute_columns(
            *columns, where=self.where_clause, errors_only=errors_only, cascade=cascade
        )

    def delete(self) -> UpdateStatus:
        """Delete rows form the underlying table of the Query.

        The delete operation is only allowed for Queries on base tables.

        Returns:
            UpdateStatus: the status of the delete operation.

        Example:
            For a table `person` with column `age`, delete all rows where 'age' is less than 18:

            >>> person.where(t.age < 18).delete()
        """
        self._validate_mutable('delete')
        if self._from_clause.tbls[0].is_view():
            raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, 'Cannot use `delete` on a view.')
        return self._mutation_target().delete(where=self.where_clause)

    def _validate_mutable(self, op_name: str) -> None:
        """Tests whether this Query can be mutated (such as by an update operation).

        Args:
            op_name: The name of the operation for which the test is being performed.
        """
        if self.group_by_clause is not None or self.grouping_tbl_key is not None:
            raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, f'Cannot use `{op_name}` after `group_by`.')
        if self.order_by_clause is not None:
            raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, f'Cannot use `{op_name}` after `order_by`.')
        if self.select_list is not None:
            raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, f'Cannot use `{op_name}` after `select`.')
        if self.limit_val is not None:
            raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, f'Cannot use `{op_name}` after `limit`.')
        if self._has_joins():
            raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, f'Cannot use `{op_name}` after `join`.')

        # TODO: Reconcile these with Table.__check_mutable()
        assert len(self._from_clause.tbls) == 1
        # A pinned version is immutable. For a delegated catalog the from table of a pure-snapshot query is the base
        # pinned at the snapshot version, which reports is_snapshot() == False but a non-None effective_version().
        from_tbl = self._from_clause.tbls[0]
        if from_tbl.is_snapshot() or from_tbl.effective_version() is not None:
            raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, f'Cannot use `{op_name}` on a snapshot.')

    @classmethod
    def _from_clause_from_dict(cls, d: dict[str, Any]) -> FromClause:
        tbls: list[catalog.TablePath] = [
            catalog.TableVersionPath.from_key(catalog.TablePathKey.from_dict(tbl_dict)) for tbl_dict in d['tbls']
        ]
        join_clauses = [
            JoinClause(
                join_type=JoinType[clause_dict['join_type']],
                join_predicate=exprs.Expr.from_dict(clause_dict['join_predicate'])
                if clause_dict['join_predicate'] is not None
                else None,
            )
            for clause_dict in d['join_clauses']
        ]
        return FromClause(tbls=tbls, join_clauses=join_clauses)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Query:
        """The Query a serialized query names. Raises if it names a query of another class."""
        result = super().from_dict(d)
        if not isinstance(result, Query):
            raise excs.Error(excs.ErrorCode.INTERNAL_ERROR, f'Expected a serialized Query, got {type(result).__name__}')
        return result

    def _hash_result_set(self) -> str:
        """Return a hash that changes when the result set changes."""
        d = self.as_dict()
        # add list of referenced table versions (the actual versions, not the effective ones) in order to force cache
        # invalidation when any of the referenced tables changes
        d['tbl_versions'] = [
            tbl_version.get().version
            for tbl in self._from_clause.tbls
            if isinstance(tbl, catalog.TableVersionPath)
            for tbl_version in tbl.get_tbl_versions()
        ]
        summary_string = json.dumps(d)
        return hashlib.sha256(summary_string.encode()).hexdigest()

    def to_coco_dataset(self) -> Path:
        """Convert the Query to a COCO dataset.
        This Query must return a single json-typed output column in the following format:

        ```python
        {
            'image': PIL.Image.Image,
            'annotations': [
                {
                    'bbox': [x: int, y: int, w: int, h: int],
                    'category': str | int,
                },
                ...
            ],
        }
        ```

        Returns:
            Path to the COCO dataset file.
        """
        from pixeltable.utils.coco import write_coco_dataset

        cache_key = self._hash_result_set()
        dest_path = Env.get().dataset_cache_dir / f'coco_{cache_key}'
        if dest_path.exists():
            assert dest_path.is_dir()
            data_file_path = dest_path / 'data.json'
            assert data_file_path.exists()
            assert data_file_path.is_file()
            return data_file_path
        else:
            assert self._from_clause.is_local
            with get_runtime().catalog.begin_xact(
                for_write=False, read_tvps=self._from_clause.tvps, read_tbl_ids=self.referenced_tbl_ids()
            ):
                return write_coco_dataset(self, dest_path)

    def to_pytorch_dataset(self, image_format: str = 'pt') -> 'torch.utils.data.IterableDataset':
        """
        Convert the Query to a pytorch IterableDataset suitable for parallel loading
        with torch.utils.data.DataLoader.

        This method requires pyarrow >= 13, torch and torchvision to work.

        This method serializes data so it can be read from disk efficiently and repeatedly without
        re-executing the query. This data is cached to disk for future re-use.

        Args:
            image_format: format of the images. Can be 'pt' (pytorch tensor) or 'np' (numpy array).
                    'np' means image columns return as an RGB uint8 array of shape HxWxC.
                    'pt' means image columns return as a CxHxW tensor with values in [0,1] and type torch.float32.
                        (the format output by torchvision.transforms.ToTensor())

        Returns:
            A pytorch IterableDataset: Columns become fields of the dataset, where rows are returned as a dictionary
                compatible with torch.utils.data.DataLoader default collation.

        Constraints:
            The default collate_fn for torch.data.util.DataLoader cannot represent null values as part of a
            pytorch tensor when forming batches. These values will raise an exception while running the dataloader.

            If you have them, you can work around None values by providing your custom collate_fn to the DataLoader
            (and have your model handle it). Or, if these are not meaningful values within a minibtach, you can
            modify or remove any such values through selections and filters prior to calling to_pytorch_dataset().
        """
        # check dependencies
        Env.get().require_package('pyarrow', [13])
        Env.get().require_package('torch')
        Env.get().require_package('torchvision')

        from pixeltable.io import export_parquet
        from pixeltable.utils.pytorch import PixeltablePytorchDataset

        cache_key = self._hash_result_set()

        dest_path = (Env.get().dataset_cache_dir / f'df_{cache_key}').with_suffix('.parquet')
        if dest_path.exists():  # fast path: use cache
            assert dest_path.is_dir()
        else:
            assert self._from_clause.is_local
            with get_runtime().catalog.begin_xact(
                for_write=False, read_tvps=self._from_clause.tvps, read_tbl_ids=self.referenced_tbl_ids()
            ):
                # we need the metadata for PixeltablePytorchDataset
                export_parquet(self, dest_path, inline_images=True, _write_md=True)

        return PixeltablePytorchDataset(path=dest_path, image_format=image_format)
