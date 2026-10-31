from __future__ import annotations

import contextlib
import queue
import shutil
from concurrent import futures
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable.env import Env

if TYPE_CHECKING:
    import pyarrow as pa

_END = object()


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

    import lancedb

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

        if lance_tbl is None or if_exists == 'overwrite':
            mode = 'overwrite' if lance_tbl is not None else 'create'
            _write_batches(query, batch_size_bytes, lambda batches: db.create_table(table_name, batches, mode=mode))
        else:
            _write_batches(query, batch_size_bytes, lance_tbl.add)

    except Exception as e:
        # cleanup
        if not db_exists:
            shutil.rmtree(db_uri)
        raise e


def _write_batches(query: pxt.Query, batch_size_bytes: int, write: Callable[[Iterator[pa.RecordBatch]], Any]) -> None:
    """Compute the query's record batches and pass an iterator over them to write(), which runs on a new thread.

    lancedb advances the iterator on its own threads, so the query runs on the calling thread instead.
    """
    from pixeltable.utils.arrow import to_record_batches

    # maxsize=2: overlap computing the next batch with lancedb's processing of the current one
    batches: queue.Queue = queue.Queue(maxsize=2)

    def queued_batches() -> Iterator[pa.RecordBatch]:
        while (batch := batches.get()) is not _END:
            if batch is None:
                raise RuntimeError('export_lancedb() stopped')
            yield batch

    def drain(_: futures.Future) -> None:
        # write() no longer reads the queue; unblock put()
        with contextlib.suppress(queue.Empty):
            while True:
                batches.get_nowait()

    with futures.ThreadPoolExecutor(max_workers=1) as executor:
        writer = executor.submit(write, queued_batches())
        writer.add_done_callback(drain)
        try:
            for batch in to_record_batches(query, batch_size_bytes):
                if writer.done():
                    break
                batches.put(batch)
            batches.put(_END)
        except BaseException:
            batches.put(None)  # makes write() fail
            raise
        writer.result()
