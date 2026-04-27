from __future__ import annotations

from typing import Literal

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable.env import Env


def export_iceberg(
    table_or_query: pxt.Table | pxt.Query,
    catalog: 'pyiceberg.catalog.Catalog',  # type: ignore[name-defined]  # noqa: F821
    table_name: str,
    *,
    batch_size_bytes: int = 128 * 2**20,
    if_exists: Literal['error', 'overwrite', 'append'] = 'error',
) -> None:
    """
    Exports a query result or table to an Apache Iceberg table.

    Data is streamed into the Iceberg table via pyarrow `RecordBatches`, the size of which can be controlled with
    the `batch_size_bytes` parameter. The destination Iceberg `Catalog` is supplied by the caller; for a local
    sqlite-backed catalog, see [`pixeltable.utils.iceberg.sqlite_catalog`][pixeltable.utils.iceberg.sqlite_catalog].

    __Requirements:__

    - `pip install pyiceberg`

    Args:
        table_or_query: Table or Query to export.
        catalog: An Iceberg `Catalog` instance to write the table into.
        table_name: Fully-qualified Iceberg table identifier (e.g. `'pxt.my_table'`). If the namespace does not
            exist, it will be created.
        batch_size_bytes: Maximum size in bytes for each in-memory pyarrow batch.
        if_exists: Determines the behavior if the table already exists. Must be one of the following:

            - `'error'`: raise an error
            - `'overwrite'`: drop the existing table and create a new one
            - `'append'`: append to the existing table (source schema must be compatible)
    """
    Env.get().require_package('pyiceberg')

    import pyarrow as pa
    from pyiceberg.exceptions import NoSuchTableError

    from pixeltable.utils.arrow import to_record_batches

    if if_exists not in ('error', 'overwrite', 'append'):
        raise excs.RequestError(
            excs.ErrorCode.INVALID_ARGUMENT,
            "export_iceberg(): 'if_exists' must be one of: ['error', 'overwrite', 'append']",
        )

    query: pxt.Query
    if isinstance(table_or_query, pxt.catalog.Table):
        query = table_or_query.select()
    else:
        query = table_or_query

    iceberg_tbl = None
    try:
        iceberg_tbl = catalog.load_table(table_name)
    except NoSuchTableError:
        pass

    if iceberg_tbl is not None:
        if if_exists == 'error':
            raise excs.AlreadyExistsError(
                excs.ErrorCode.PATH_ALREADY_EXISTS, f'export_iceberg(): table {table_name!r} already exists'
            )
        if if_exists == 'overwrite':
            catalog.drop_table(table_name)
            iceberg_tbl = None

    batch_iter = to_record_batches(query, batch_size_bytes)
    first_batch = next(batch_iter, None)
    if first_batch is None:
        return

    if iceberg_tbl is None:
        if '.' in table_name:
            catalog.create_namespace_if_not_exists(table_name.rsplit('.', 1)[0])
        iceberg_tbl = catalog.create_table(table_name, schema=first_batch.schema)
    else:
        target_schema = iceberg_tbl.schema().as_arrow()
        try:
            # Cast a zero-row slice to the target schema: pyarrow validates field names match
            # exactly and that source types can be promoted to target types (e.g. string -> large_string).
            pa.Table.from_batches([first_batch.slice(0, 0)]).cast(target_schema)
        except (pa.ArrowInvalid, ValueError) as e:
            raise excs.RequestError(
                excs.ErrorCode.TYPE_MISMATCH,
                f'export_iceberg(): source schema is not compatible with existing table {table_name!r}: {e}\n'
                f'Source schema:\n{first_batch.schema}\n'
                f'Target schema:\n{target_schema}',
            ) from e

    with iceberg_tbl.transaction() as tx:
        tx.append(pa.Table.from_batches([first_batch]))
        for batch in batch_iter:
            tx.append(pa.Table.from_batches([batch]))
