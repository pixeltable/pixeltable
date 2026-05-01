from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal

import pyarrow as pa

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable.env import Env

if TYPE_CHECKING:
    from pyiceberg.catalog import Catalog
    from pyiceberg.catalog.sql import SqlCatalog


def sqlite_catalog(warehouse_path: str | Path, name: str = 'pixeltable') -> SqlCatalog:
    """
    Instantiate a sqlite Iceberg catalog at the specified path. If no catalog exists, one will be created.
    """
    from pyiceberg.catalog.sql import SqlCatalog

    if isinstance(warehouse_path, str):
        warehouse_path = Path(warehouse_path)
    warehouse_path.mkdir(exist_ok=True)
    return SqlCatalog(name, uri=f'sqlite:///{warehouse_path}/catalog.db', warehouse=f'file://{warehouse_path}')


def export_iceberg(
    table_or_query: pxt.Table | pxt.Query,
    catalog: Catalog,
    table_name: str,
    *,
    batch_size_bytes: int = 128 * 2**20,
    if_exists: Literal['error', 'replace', 'append'] = 'error',
) -> None:
    """
    Exports a query result or table to an Apache Iceberg table.

    Data is streamed into the Iceberg table via pyarrow
    [`RecordBatches`](https://arrow.apache.org/docs/python/generated/pyarrow.RecordBatch.html), the size of
    which can be controlled with the `batch_size_bytes` parameter.

    __Requirements:__

    - `pip install pyiceberg`

    Args:
        table_or_query: Pixeltable `Table` or `Query` to export.
        catalog: An Iceberg `Catalog` instance to write the table into.
        table_name: Fully-qualified Iceberg table identifier (e.g. `'pxt.my_table'`). If the namespace does not
            exist, it will be created.
        batch_size_bytes: Maximum size in bytes for each in-memory pyarrow batch.
        if_exists: Determines the behavior if the table already exists. Must be one of the following:

            - `'error'`: raise an error
            - `'replace'`: drop the existing table and create a new one
            - `'append'`: append to the existing table (source schema must be compatible)
    """
    Env.get().require_package('pyiceberg')

    from pyiceberg.exceptions import NoSuchTableError

    from pixeltable.utils.arrow import to_arrow_schema, to_record_batches

    if if_exists not in ('error', 'replace', 'append'):
        raise excs.RequestError(
            excs.ErrorCode.INVALID_ARGUMENT,
            "export_iceberg(): `if_exists` must be one of: ['error', 'replace', 'append']",
        )

    query: pxt.Query
    if isinstance(table_or_query, pxt.catalog.Table):
        query = table_or_query.select()
    else:
        query = table_or_query

    existing_tbl = None
    try:
        existing_tbl = catalog.load_table(table_name)
    except NoSuchTableError:
        pass

    # Iceberg's pyarrow implementation does not support arrays
    unsupported = [name for name in query.schema if query.schema[name].is_array_type()]
    if unsupported:
        raise excs.RequestError(
            excs.ErrorCode.UNSUPPORTED_OPERATION,
            f'export_iceberg(): cannot export fixed-shape tensor column(s) {unsupported}. '
            f'Iceberg has no fixed-shape tensor type; project the column to a list before exporting.',
        )

    if existing_tbl is not None and if_exists == 'error':
        raise excs.AlreadyExistsError(
            excs.ErrorCode.PATH_ALREADY_EXISTS, f'export_iceberg(): table {table_name!r} already exists'
        )

    batch_iter = to_record_batches(query, batch_size_bytes)
    first_batch = next(batch_iter, None)

    # Build a deterministic arrow schema up front so we can materialize the Iceberg table even
    # when the query yields no rows.
    arrow_schema = first_batch.schema if first_batch is not None else to_arrow_schema(query.schema)

    # `pa.infer_type` produces `pa.null()` for JSON keys whose value is None in every sampled row.
    # Iceberg format-version 2 cannot represent a null-only column, so reject it up front rather
    # than letting pyiceberg fail mid-write with a less actionable error.
    null_paths = _find_null_fields(arrow_schema)
    if null_paths:
        raise excs.RequestError(
            excs.ErrorCode.UNSUPPORTED_OPERATION,
            f'export_iceberg(): cannot infer a concrete type for JSON field(s) {null_paths} because every sampled '
            f'value is None. Iceberg has no null-only type; populate the field with at least one non-None value '
            f'or omit it from the data.',
        )

    if existing_tbl is not None and if_exists == 'append':
        target_schema = existing_tbl.schema().as_arrow()
        try:
            # Cast a zero-row slice to the target schema: pyarrow validates field names match
            # exactly and that source types can be promoted to target types (e.g. string -> large_string).
            sample = (
                first_batch.slice(0, 0)
                if first_batch is not None
                else pa.RecordBatch.from_arrays([pa.array([], type=f.type) for f in arrow_schema], schema=arrow_schema)
            )
            pa.Table.from_batches([sample]).cast(target_schema)
        except (pa.ArrowInvalid, ValueError) as e:
            raise excs.RequestError(
                excs.ErrorCode.TYPE_MISMATCH,
                f'export_iceberg(): source schema is not compatible with existing table {table_name!r}: {e}\n'
                f'Source schema:\n{arrow_schema}\n'
                f'Target schema:\n{target_schema}',
            ) from e
        iceberg_tbl = existing_tbl
    else:
        # All preflight checks have passed; safe to drop the existing table now.
        if existing_tbl is not None:
            assert if_exists == 'replace'
            catalog.drop_table(table_name)
        if '.' in table_name:
            catalog.create_namespace_if_not_exists(table_name.rsplit('.', 1)[0])
        iceberg_tbl = catalog.create_table(table_name, schema=arrow_schema)

    batches: Iterable[pa.RecordBatch] = chain([first_batch], batch_iter) if first_batch is not None else batch_iter
    with iceberg_tbl.transaction() as tx:
        for batch in batches:
            tx.append(pa.Table.from_batches([batch]))


def _find_null_fields(schema: pa.Schema) -> list[str]:
    """Return dotted paths of every `pa.null()`-typed field nested inside `schema`."""
    paths: list[str] = []

    def walk(arrow_type: pa.DataType, path: str) -> None:
        if pa.types.is_null(arrow_type):
            paths.append(path)
        elif pa.types.is_struct(arrow_type):
            for f in arrow_type:
                walk(f.type, f'{path}.{f.name}')
        elif (
            pa.types.is_list(arrow_type)
            or pa.types.is_large_list(arrow_type)
            or pa.types.is_fixed_size_list(arrow_type)
        ):
            walk(arrow_type.value_type, f'{path}[]')
        elif pa.types.is_map(arrow_type):
            walk(arrow_type.item_type, f'{path}{{}}')

    for field in schema:
        walk(field.type, field.name)
    return paths
