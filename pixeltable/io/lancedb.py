from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal

import pyarrow as pa

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable.env import Env


def export_lancedb(
    table_or_query: pxt.Table | pxt.Query,
    db_uri: Path,
    table_name: str,
    batch_size_bytes: int = 128 * 2**20,
    if_exists: Literal['error', 'overwrite', 'append'] = 'error',
) -> None:
    """
    Exports a Query's data to a LanceDB table.

    This writes the Query as in-memory pyarrow `RecordBatches`. `batch_size_bytes` caps the size of each batch.

    __Requirements:__

    - `pip install lancedb pylance`

    Args:
        table_or_query : Table or Query to export.
        db_uri: Local Path to the LanceDB database.
        table_name : Name of the table in the LanceDB database.
        batch_size_bytes : Maximum size in bytes for each batch.
        if_exists: Determines the behavior if the table already exists. Must be one of the following:

            - `'error'`: raise an error
            - `'overwrite'`: overwrite the existing table
            - `'append'`: append to the existing table
    """
    Env.get().require_package('lance')
    Env.get().require_package('lancedb')

    import lancedb  # type: ignore[import-untyped]

    from pixeltable.utils.arrow import to_record_batches

    if if_exists not in ('error', 'overwrite', 'append'):
        raise excs.RequestError(
            excs.ErrorCode.INVALID_ARGUMENT,
            "export_lancedb(): 'if_exists' must be one of: ['error', 'overwrite', 'append']",
        )

    query: pxt.Query
    if isinstance(table_or_query, pxt.catalog.Table):
        query = table_or_query.select()
    else:
        query = table_or_query

    db_exists = False
    if db_uri.exists():
        if not db_uri.is_dir():
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION, f"export_lancedb(): '{db_uri!s}' exists and is not a directory"
            )
        db_exists = True

    try:
        db = lancedb.connect(str(db_uri))
        lance_tbl: lancedb.LanceTable | None = None
        try:
            lance_tbl = db.open_table(table_name)
            if if_exists == 'error':
                raise excs.AlreadyExistsError(
                    excs.ErrorCode.PATH_ALREADY_EXISTS,
                    f'export_lancedb(): table {table_name!r} already exists in {db_uri!r}',
                )
        except ValueError:
            # table doesn't exist
            pass

        # lancedb 0.41+ prepares input on a thread-pool worker (`asyncio.to_thread`). Iterating a
        # Pixeltable query there creates a thread-local Runtime event loop that nothing closes.
        # Consume the query here, then pass a Table so add() does not treat RecordBatches as rows.
        batches = list(to_record_batches(query, batch_size_bytes))
        data = pa.Table.from_batches(batches) if batches else batches
        if lance_tbl is None or if_exists == 'overwrite':
            mode = 'overwrite' if lance_tbl is not None else 'create'
            _ = db.create_table(table_name, data, mode=mode)
        else:
            lance_tbl.add(data)

    except Exception as e:
        # cleanup
        if not db_exists:
            shutil.rmtree(db_uri)
        raise e
