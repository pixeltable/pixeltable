from __future__ import annotations

import asyncio
import dataclasses
import io
import os
from collections import deque
from concurrent import futures
from pathlib import Path
from typing import Any, AsyncIterator

import numpy as np
import PIL.Image
import sqlalchemy as sql

import pixeltable.type_system as ts
import pixeltable.utils.image as image_utils
from pixeltable import catalog, exprs
from pixeltable.env import Env
from pixeltable.utils.local_store import LocalStore, TempStore
from pixeltable.utils.object_stores import FileDestination, ObjectOps, ObjectStoreBase

from .data_row_batch import DataRowBatch
from .exec_node import ExecNode
from .globals import INLINED_OBJECT_MD_KEY, InlinedObjectMd


class CellMaterializationNode(ExecNode):
    """
    Node to populate DataRow.cell_vals/cell_md.

    For now, the scope is limited to populating DataRow.cells_vals for json and array columns.

    Array values:
    - Arrays < MAX_DB_ARRAY_SIZE are stored inline in the db column
    - Larger arrays are written to chunks
    - Bool arrays are stored as packed bits (uint8)
    - cell_md: holds the url of the file, plus start and end offsets, plus bool flag and shape for bool arrays
      (this allows us to query cell_md to get the total external storage size of an array column)

    Json values:
    - Inlined images and arrays are written to chunks and replaced with a dict containing the object location
    - Bool arrays are also stored as packed bits; the dict also contains the shape and bool flag
    - cell_md contains the list of urls for the inlined objects.

    Chunks:
    - Without Env.cell_materialization_dest, chunks are written in place in the local media dir. Otherwise, they are
      written to the TempStore and uploaded once complete; their urls are known up front, so rows are passed on
      before their uploads finish, and the iteration only ends once all uploads have succeeded.
    - A value of at least MIN_FILE_SIZE gets a chunk of its own, so that reading a small value never requires fetching
      a large one.

    TODO:
    - execute file IO via asyncio Tasks in a thread pool?
      (we already seem to be getting 90% of hardware IO throughput)
    - subsume all cell materialization
    """

    @dataclasses.dataclass
    class Chunk:
        url: str  # recorded in cell_md
        path: Path  # local file being written
        dest: FileDestination | None  # if not None, path is in the TempStore and gets uploaded to dest

    output_col_info: dict[catalog.Column, int]  # value: slot idx
    dest: str | None  # destination of the chunks; None: the local media dir
    store: ObjectStoreBase | None  # store for dest, created when the first chunk is opened

    # execution state
    chunks: list[Chunk]  # chunks referenced by the current cell; only [-1] can be open for writing
    buffered_writer: io.BufferedWriter | None  # BufferedWriter for chunks[-1]
    executor: futures.ThreadPoolExecutor | None
    uploads: deque[futures.Future]  # oldest first

    MIN_FILE_SIZE = 8 * 2**20  # 8MB
    MAX_DB_BINARY_SIZE = 512  # max size of binary data stored in table column; in bytes
    # each pending upload holds a chunk in the TempStore; checked between rows, so a row's own chunks can exceed it
    MAX_PENDING_UPLOADS = 4

    def __init__(self, input: ExecNode):
        super().__init__(input.row_builder, [], [], input)
        self.output_col_info = {
            col: slot_idx
            for col, slot_idx in input.row_builder.table_columns.items()
            if slot_idx is not None and col.col_type.needs_cell_materialization()
        }
        self.dest = Env.get().cell_materialization_dest
        self.store = None
        self._init_exec_state()

    def _init_exec_state(self) -> None:
        self.chunks = []
        self.buffered_writer = None
        self.executor = None
        self.uploads = deque()

    def _open(self) -> None:
        self._init_exec_state()

    async def __aiter__(self) -> AsyncIterator[DataRowBatch]:
        async for batch in self.input:
            for row in batch:
                for col, slot_idx in self.output_col_info.items():
                    if row.has_exc(slot_idx):
                        # Nulls in JSONB columns need to be stored as sql.sql.null(), otherwise it stores a json 'null'
                        row.cell_vals[col.id] = sql.sql.null() if col.col_type.is_json_type() else None
                        exc = row.get_exc(slot_idx)
                        row.cell_md[col.id] = exprs.CellMd(errortype=type(exc).__name__, errormsg=str(exc))
                        continue

                    val = row[slot_idx]
                    if val is None:
                        row.cell_vals[col.id] = sql.sql.null() if col.col_type.is_json_type() else None
                        row.cell_md[col.id] = None
                        continue

                    if col.col_type.is_json_type():
                        self._materialize_json_cell(row, col, val)
                    elif col.col_type.is_array_type():
                        assert isinstance(val, np.ndarray)
                        self._materialize_array_cell(row, col, val)
                    else:
                        assert col.col_type.is_binary_type()
                        assert isinstance(val, bytes)
                        self._materialize_binary_cell(row, col, val)

                    # continue with only the currently open chunk
                    self.chunks = self.chunks[-1:] if self.buffered_writer is not None else []

                await self._wait_for_uploads(self.MAX_PENDING_UPLOADS)

            yield batch

        self._flush_buffer(finalize=True)
        # the rows reference the chunks' urls, so they can't be committed before all uploads have succeeded
        await self._wait_for_uploads(0)

    async def _wait_for_uploads(self, max_pending: int) -> None:
        """Waits until at most max_pending uploads are in flight; raises the exception of a failed upload."""
        while len(self.uploads) > max_pending:
            await asyncio.wrap_future(self.uploads.popleft())

    def init_writer(self, size: int) -> None:
        """Prepare chunks[-1] for writing a value of approximately `size` bytes."""
        if self.buffered_writer is not None and size >= self.MIN_FILE_SIZE and self.buffered_writer.tell() > 0:
            self._close_chunk()
        if self.buffered_writer is None:
            self._open_chunk()
            assert self.buffered_writer is not None

    def _close(self) -> None:
        if self.buffered_writer is not None:
            # there must have been an error, otherwise _flush_buffer(finalize=True) would have set this to None
            self.buffered_writer.close()
            self.buffered_writer = None
            if self.chunks[-1].dest is not None:
                TempStore.delete_media_file(self.chunks[-1].path)
        if self.executor is not None:
            # each upload deletes its TempStore file
            self.executor.shutdown(wait=True)
            self.executor = None

    def _materialize_json_cell(self, row: exprs.DataRow, col: catalog.Column, val: Any) -> None:
        if self._json_has_inlined_objs(val):
            row.cell_vals[col.id] = self._rewrite_json(val)
            row.cell_md[col.id] = exprs.CellMd(file_urls=[chunk.url for chunk in self.chunks])
        else:
            row.cell_vals[col.id] = val
            row.cell_md[col.id] = None

    def _materialize_array_cell(self, row: exprs.DataRow, col: catalog.Column, val: np.ndarray) -> None:
        if col.has_sa_vector_type():
            # this is a vector column (ie, used for a vector index): store the array itself
            row.cell_vals[col.id] = val
            row.cell_md[col.id] = None
        elif val.nbytes <= self.MAX_DB_BINARY_SIZE:
            # this array is small enough to store in the db column (type: binary) directly
            buffer = io.BytesIO()
            np.save(buffer, val, allow_pickle=False)
            row.cell_vals[col.id] = buffer.getvalue()
            row.cell_md[col.id] = None
        else:
            # append this array to the buffer and store its location in the cell md
            ar: np.ndarray
            if np.issubdtype(val.dtype, np.bool_):
                # for bool arrays, store as packed bits, otherwise it's 1 byte per element
                ar = np.packbits(val)
            else:
                ar = val
            self.init_writer(ar.nbytes)
            start = self.buffered_writer.tell()
            np.save(self.buffered_writer, ar, allow_pickle=False)
            end = self.buffered_writer.tell()
            row.cell_vals[col.id] = None
            cell_md = exprs.CellMd(file_urls=[self.chunks[-1].url], array_md=exprs.ArrayMd(start=start, end=end))
            if np.issubdtype(val.dtype, np.bool_):
                cell_md.array_md.is_bool = True
                cell_md.array_md.shape = val.shape
            row.cell_md[col.id] = cell_md
            self._flush_buffer()

        assert row.cell_vals[col.id] is not None or row.cell_md[col.id] is not None

    def _materialize_binary_cell(self, row: exprs.DataRow, col: catalog.Column, val: bytes) -> None:
        if len(val) <= self.MAX_DB_BINARY_SIZE:
            # this `bytes` object is small enough to store in the db column (type: binary) directly
            row.cell_vals[col.id] = val
            row.cell_md[col.id] = None
        else:
            self.init_writer(len(val))
            start = self.buffered_writer.tell()
            self.buffered_writer.write(val)
            end = self.buffered_writer.tell()
            row.cell_vals[col.id] = None
            cell_md = exprs.CellMd(file_urls=[self.chunks[-1].url], binary_md=exprs.BinaryMd(start=start, end=end))
            row.cell_md[col.id] = cell_md
            self._flush_buffer()

        assert row.cell_vals[col.id] is not None or row.cell_md[col.id] is not None

    def _json_has_inlined_objs(self, element: Any) -> bool:
        if isinstance(element, list):
            return any(self._json_has_inlined_objs(v) for v in element)
        if isinstance(element, dict):
            return any(self._json_has_inlined_objs(v) for v in element.values())
        return isinstance(element, (np.ndarray, PIL.Image.Image, bytes))

    def _rewrite_json(self, element: Any) -> Any:
        """Recursively rewrites a JSON structure by writing any inlined arrays or images to self.buffered_writer."""
        if isinstance(element, list):
            return [self._rewrite_json(v) for v in element]
        if isinstance(element, dict):
            return {k: self._rewrite_json(v) for k, v in element.items()}
        if isinstance(element, np.ndarray):
            obj_md = self._write_inlined_array(element)
            return {INLINED_OBJECT_MD_KEY: obj_md.as_dict()}
        if isinstance(element, PIL.Image.Image):
            obj_md = self._write_inlined_image(element)
            return {INLINED_OBJECT_MD_KEY: obj_md.as_dict()}
        if isinstance(element, bytes):
            obj_md = self._write_inlined_bytes(element)
            return {INLINED_OBJECT_MD_KEY: obj_md.as_dict()}
        return element

    def _write_inlined_array(self, ar: np.ndarray) -> InlinedObjectMd:
        """Write an ndarray to buffered_writer and return its metadata."""
        shape: tuple[int, ...] | None
        is_bool_array: bool
        if np.issubdtype(ar.dtype, np.bool_):
            shape = ar.shape
            ar = np.packbits(ar)
            is_bool_array = True
        else:
            shape = None
            is_bool_array = False
        self.init_writer(ar.nbytes)
        url_idx = len(self.chunks) - 1
        start = self.buffered_writer.tell()
        np.save(self.buffered_writer, ar, allow_pickle=False)
        end = self.buffered_writer.tell()
        self._flush_buffer()
        return InlinedObjectMd(
            type=ts.ColumnType.Type.ARRAY.name,
            url_idx=url_idx,
            array_md=exprs.ArrayMd(start=start, end=end, is_bool=is_bool_array, shape=shape),
        )

    def _write_inlined_image(self, img: PIL.Image.Image) -> InlinedObjectMd:
        """Write a PIL image to buffered_writer and return: index into chunks, start offset, end offset"""
        # the encoded size isn't known up front; the uncompressed size approximates it
        self.init_writer(img.width * img.height * len(img.getbands()))
        url_idx = len(self.chunks) - 1
        start = self.buffered_writer.tell()
        img.save(self.buffered_writer, format=image_utils.default_format(img))
        end = self.buffered_writer.tell()
        self._flush_buffer()
        return InlinedObjectMd(type=ts.ColumnType.Type.IMAGE.name, url_idx=url_idx, img_start=start, img_end=end)

    def _write_inlined_bytes(self, data: bytes) -> InlinedObjectMd:
        """Write raw bytes to buffered_writer and return: index into chunks, start offset, end offset"""
        self.init_writer(len(data))
        url_idx = len(self.chunks) - 1
        start = self.buffered_writer.tell()
        self.buffered_writer.write(data)
        end = self.buffered_writer.tell()
        self._flush_buffer()
        return InlinedObjectMd(
            type=ts.ColumnType.Type.BINARY.name, url_idx=url_idx, binary_md=exprs.BinaryMd(start, end)
        )

    def _open_chunk(self) -> None:
        tbl = self.row_builder.tbl
        if self.dest is None:
            local_path = LocalStore(Env.get().media_dir)._prepare_path_raw(tbl.id, 0, tbl.version)
            chunk = self.Chunk(url=local_path.as_uri(), path=local_path, dest=None)
        else:
            if self.store is None:
                self.store = ObjectOps.get_store(self.dest, False)
            file_dest = self.store.resolve_destination(tbl.id, 0, tbl.version)
            chunk = self.Chunk(url=file_dest.url, path=TempStore.create_path(), dest=file_dest)
        self.chunks.append(chunk)
        fh = open(chunk.path, 'wb', buffering=self.MIN_FILE_SIZE * 2)  # noqa: SIM115
        assert isinstance(fh, io.BufferedWriter)
        self.buffered_writer = fh

    def _close_chunk(self) -> None:
        assert self.buffered_writer is not None
        self.buffered_writer.flush()
        os.fsync(self.buffered_writer.fileno())  # needed to force bytes cached by OS to storage
        self.buffered_writer.close()
        self.buffered_writer = None
        chunk = self.chunks[-1]
        if chunk.dest is None:
            return
        if self.executor is None:
            self.executor = futures.ThreadPoolExecutor(
                max_workers=self.MAX_PENDING_UPLOADS, thread_name_prefix='pxt-chunk-upload'
            )
        self.uploads.append(self.executor.submit(ObjectOps.put_file_resolved, self.store, chunk.path, chunk.dest, True))

    def _flush_buffer(self, finalize: bool = False) -> None:
        """Close chunks[-1] if it exceeds its minimum size or finalize is True."""
        if self.buffered_writer is None:
            return
        if self.buffered_writer.tell() < self.MIN_FILE_SIZE and not finalize:
            return
        self._close_chunk()
