from __future__ import annotations

import queue
import shutil
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Iterator, Literal

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable.env import Env
from pixeltable.runtime import Runtime, get_runtime


@contextmanager
def _record_batches_on_thread(query: pxt.Query, batch_size_bytes: int) -> Iterator[Iterator[Any]]:
    """Yield RecordBatches from a producer thread and close that thread's Runtime on context exit.

    lancedb 0.41 reads the iterator on asyncio.to_thread workers, which never close a Runtime they create.
    get_runtime() and Thread.start() run when the context is entered, not on first next().
    """
    from pixeltable.utils.arrow import to_record_batches

    result_queue: queue.Queue = queue.Queue(maxsize=2)
    stop = threading.Event()
    caller_runtime = get_runtime()
    sentinel = object()

    def _put(item: object) -> bool:
        while not stop.is_set():
            try:
                result_queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def _produce_batches(thread_runtime: Runtime) -> None:
        # query execution needs the caller's catalogs
        thread_runtime.copy_db_context(caller_runtime)
        batch_iter = to_record_batches(query, batch_size_bytes)
        assert isinstance(batch_iter, Generator)
        try:
            for batch in batch_iter:
                if not _put(batch):
                    return
            _put(sentinel)
        finally:
            batch_iter.close()

    def run() -> None:
        thread_runtime = get_runtime()
        try:
            _produce_batches(thread_runtime)
        except BaseException as e:
            _put(e)
        finally:
            thread_runtime.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    def batches() -> Iterator[Any]:
        while True:
            item = result_queue.get()
            if item is sentinel:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    try:
        yield batches()
    finally:
        stop.set()
        try:
            result_queue.put_nowait(sentinel)
        except queue.Full:
            pass
        thread.join()


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

        with _record_batches_on_thread(query, batch_size_bytes) as batches:
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
