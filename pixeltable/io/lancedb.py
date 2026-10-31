from __future__ import annotations

import contextlib
import queue
import shutil
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable import telemetry
from pixeltable.env import Env
from pixeltable.runtime import get_runtime

if TYPE_CHECKING:
    import pyarrow as pa

_SENTINEL = object()


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

        # lancedb advances the iterator on its own threads
        with contextlib.closing(_ThreadedRecordBatches(query, batch_size_bytes)) as batches:
            if lance_tbl is None or if_exists == 'overwrite':
                mode = 'overwrite' if lance_tbl is not None else 'create'
                _ = db.create_table(table_name, batches, mode=mode)
            else:
                lance_tbl.add(batches)

    except Exception as e:
        # cleanup
        if not db_exists:
            shutil.rmtree(db_uri)
        raise e


class _ThreadedRecordBatches:
    """Iterator over a query's record batches, computed on a new thread.

    Any thread can advance or close the iterator, also while another thread is advancing it.
    """

    _queue: queue.Queue
    _stop: threading.Event
    _thread: threading.Thread

    def __init__(self, query: pxt.Query, batch_size_bytes: int):
        # maxsize=2: overlap computing the next batch with the consumer's processing of the current one
        self._queue = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        hooks_ctx = telemetry.capture_context()
        self._thread = threading.Thread(target=self._produce, args=(query, batch_size_bytes, hooks_ctx), daemon=True)
        self._thread.start()

    def _produce(self, query: pxt.Query, batch_size_bytes: int, hooks_ctx: telemetry.CtxSnapshot | None) -> None:
        from pixeltable.utils.arrow import to_record_batches

        hooks_token = telemetry.restore_context(hooks_ctx)
        try:
            for batch in to_record_batches(query, batch_size_bytes):
                self._queue.put(batch)
                if self._stop.is_set():
                    break
            self._queue.put(_SENTINEL)
        except BaseException as e:
            self._queue.put(e)
        finally:
            # the query created this thread's event loop and clients
            get_runtime().close()
            telemetry.exit_context(hooks_token)

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> pa.RecordBatch:
        item = self._get()
        if item is _SENTINEL:
            raise StopIteration
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self) -> None:
        """Stop the query after its current batch, and wait for its thread to finish."""
        self._stop.set()
        # the producer can be blocked in put(), so drain the queue up to its final item
        while not self._is_final(self._get()):
            pass
        self._thread.join(timeout=30)

    def _get(self) -> Any:
        item = self._queue.get()
        if self._is_final(item):
            # leave the final item for the other readers
            self._queue.put(item)
        return item

    def _is_final(self, item: Any) -> bool:
        return item is _SENTINEL or isinstance(item, BaseException)
