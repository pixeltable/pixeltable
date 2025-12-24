from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Literal

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable.catalog import Catalog
from pixeltable.env import Env

_logger = logging.getLogger('pixeltable')


def export_lancedb(
    table_or_query: pxt.Table | pxt.Query,
    db_uri: Path,
    table_name: str,
    batch_size_bytes: int = 128 * 2**20,
    if_exists: Literal['error', 'overwrite', 'append'] = 'error',
) -> None:
    """
    Exports a Query's data to a LanceDB table.

    This utilizes LanceDB's streaming interface for efficient table creation, via a sequence of in-memory pyarrow
    `RecordBatches`, the size of which can be controlled with the `batch_size_bytes` parameter.

    __Requirements:__

    - `pip install lancedb`

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
    Env.get().require_package('lancedb')

    import lancedb  # type: ignore[import-untyped]

    from pixeltable.utils.arrow import to_record_batches

    if if_exists not in ('error', 'overwrite', 'append'):
        raise excs.Error("export_lancedb(): 'if_exists' must be one of: ['error', 'overwrite', 'append']")

    query: pxt.Query
    if isinstance(table_or_query, pxt.catalog.Table):
        query = table_or_query.select()
    else:
        query = table_or_query

    db_exists = False
    if db_uri.exists():
        if not db_uri.is_dir():
            raise excs.Error(f"export_lancedb(): '{db_uri!s}' exists and is not a directory")
        db_exists = True

    try:
        db = lancedb.connect(str(db_uri))
        lance_tbl: lancedb.LanceTable | None = None
        try:
            lance_tbl = db.open_table(table_name)
            if if_exists == 'error':
                raise excs.Error(f'export_lancedb(): table {table_name!r} already exists in {db_uri!r}')
        except ValueError:
            # table doesn't exist
            pass

        with Catalog.get().begin_xact(for_write=False):
            if lance_tbl is None or if_exists == 'overwrite':
                mode = 'overwrite' if lance_tbl is not None else 'create'
                _ = db.create_table(table_name, to_record_batches(query, batch_size_bytes), mode=mode)
            else:
                lance_tbl.add(to_record_batches(query, batch_size_bytes))

    except Exception as e:
        # cleanup
        if not db_exists:
            shutil.rmtree(db_uri)
        raise e
