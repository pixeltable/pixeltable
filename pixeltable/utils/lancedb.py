from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Literal

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable.catalog import Catalog

_logger = logging.getLogger('pixeltable')


def export_lancedb(
    table_or_df: pxt.Table | pxt.DataFrame,
    db_uri: Path,
    table_name: str = 'pixeltable_export',
    partition_size_bytes: int = 128 * 2**20,
    if_exists: Literal['error', 'overwrite', 'append'] = 'error',
) -> None:
    """
    Exports a dataframe's data to a LanceDB table.

    __Requirements:__
    - `pip install lancedb`

    Args:
        table_or_df : Table or Dataframe to export.
        db_uri: Local Path to the LanceDB database.
        table_name : Name of the table in the LanceDB database.
        partition_size_bytes : Maximum size in bytes for each batch.
        if_exists: Determines the behavior if the table already exists. Must be one of the following:

            - `'error'`: raise an error
            - `'overwrite'`: overwrite the existing table
            - `'append'`: append to the existing table
    """
    try:
        import lancedb
    except ImportError as e:
        raise excs.Error(f"export_lancedb(): requires 'lancedb' and 'pyarrow' packages: {e}") from None
    if if_exists not in ('error', 'overwrite', 'append'):
        raise excs.Error("export_lancedb(): 'if_exists' must be one of: ['error', 'overwrite', 'append']")

    from pixeltable.utils.arrow import to_arrow_schema, to_record_batches

    df: pxt.DataFrame
    if isinstance(table_or_df, pxt.catalog.Table):
        df = table_or_df._df()
    else:
        df = table_or_df

    db_exists = False
    if db_uri.exists():
        if not db_uri.is_dir():
            raise excs.Error(f'export_lancedb(): {db_uri!r} exists and is not a directory')
        db_exists = True

    try:
        db = lancedb.connect(str(db_uri))
        lance_tbl: lancedb.LanceTable | None = None
        try:
            lance_tbl = db.open_table(table_name)
            if if_exists == 'error':
                raise excs.Error(f'export_lancedb(): table {table_name!r} already exists in {db_uri!r}')
            elif if_exists == 'overwrite':
                lance_tbl.delete()
                lance_tbl = None
        except lancedb.LanceDBError:
            pass

        with Catalog.get().begin_xact(for_write=False):
            if lance_tbl is None:
                arrow_schema = to_arrow_schema(df.schema)
                _ = db.create_table(table_name, to_record_batches(df, partition_size_bytes), schema=arrow_schema)
            else:
                lance_tbl.add(to_record_batches(df, partition_size_bytes))

    finally:
        # cleanup
        if not db_exists:
            shutil.rmtree(db_uri)
