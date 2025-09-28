from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Any, AsyncIterator

import numpy as np
import pgvector.sqlalchemy  # type: ignore[import-untyped]
import PIL.Image
import sqlalchemy as sql

import pixeltable.type_system as ts
from pixeltable import catalog, exprs
from pixeltable.env import Env
from pixeltable.utils.local_store import LocalStore

from .data_row_batch import DataRowBatch
from .exec_node import ExecNode

_logger = logging.getLogger('pixeltable')


class CellMaterializationNode(ExecNode):
    """
    Node to populate DataRow.cell_vals/cell_md.

    For now, the scope is limited to populating DataRow.cells_vals for json and array columns.

    Array values:
    - Arrays < MAX_DB_ARRAY_SIZE are stored inline in the db column
    - Larger arrays are written to inlined_obj_files
    - Bool arrays are stored as packed bits (uint8)
    - cell_md: holds the url of the file, plus start and end offsets, plus bool flag and shape for bool arrays
      (this allows us to query cell_md to get the total external storage size of an array column)

    Json values:
    - Inlined images and arrays are written to inlined_obj_files and replaced with a dict containing the object
      location
    - Bool arrays are also stored as packed bits; the dict also contains the shape and bool flag
    - cell_md contains the list of urls for the inlined objects.

    TODO:
    - execute file IO via asyncio Tasks in a thread pool?
      (we already seem to be getting 90% of hardware IO throughput)
    - subsume all cell materialization
    """

    output_col_info: dict[catalog.Column, int]  # value: slot idx

    # execution state
    inlined_obj_files: list[Path]  # only [-1] is open for writing
    buffered_writer: io.BufferedWriter | None  # BufferedWriter for inlined_obj_files[-1]

    MIN_FILE_SIZE = 8 * 2**20  # 8MB
    MAX_DB_ARRAY_SIZE = 512  # max size of array stored in table column; in bytes

    def __init__(self, input: ExecNode):
        super().__init__(input.row_builder, [], [], input)
        self.output_col_info = {
            col: slot_idx
            for col, slot_idx in input.row_builder.table_columns.items()
            if slot_idx is not None and col.col_type.is_json_type() or col.col_type.is_array_type()
        }
        self.inlined_obj_files = []
        self._reset_buffer()

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
                    else:
                        assert col.col_type.is_array_type()
                        assert isinstance(val, np.ndarray)
                        self._materialize_array_cell(row, col, val)

                    # continue with only the currently open file
                    self.inlined_obj_files = self.inlined_obj_files[-1:]

            yield batch

        self._flush_full_buffer(finalize=True)

    def close(self) -> None:
        if self.buffered_writer is not None:
            # there must have been an error, otherwise _flush_full_buffer(finalize=True) would have set this to None
            self.buffered_writer.close()

    def _materialize_json_cell(self, row: exprs.DataRow, col: catalog.Column, val: Any) -> None:
        if self._json_has_inlined_objs(val):
            row.cell_vals[col.id] = self._rewrite_json(val)
            row.cell_md[col.id] = exprs.CellMd(file_urls=[local_path.as_uri() for local_path in self.inlined_obj_files])
        else:
            row.cell_vals[col.id] = val
            row.cell_md[col.id] = None

    def _materialize_array_cell(self, row: exprs.DataRow, col: catalog.Column, val: np.ndarray) -> None:
        if isinstance(col.sa_col_type, pgvector.sqlalchemy.Vector):
            # this is a vector column (ie, used for a vector index): store the array itself
            row.cell_vals[col.id] = val
            row.cell_md[col.id] = None
        elif val.nbytes <= self.MAX_DB_ARRAY_SIZE:
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
            start = self.buffered_writer.tell()
            np.save(self.buffered_writer, ar, allow_pickle=False)
            end = self.buffered_writer.tell()
            row.cell_vals[col.id] = None
            cell_md = exprs.CellMd(file_urls=[self.inlined_obj_files[-1].as_uri()], array_start=start, array_end=end)
            if np.issubdtype(val.dtype, np.bool_):
                cell_md.array_is_bool = True
                cell_md.array_shape = val.shape
            row.cell_md[col.id] = cell_md
            self._flush_full_buffer()

        assert row.cell_vals[col.id] is not None or row.cell_md[col.id] is not None

    def _json_has_inlined_objs(self, element: Any) -> bool:
        if isinstance(element, list):
            return any(self._json_has_inlined_objs(v) for v in element)
        if isinstance(element, dict):
            return any(self._json_has_inlined_objs(v) for v in element.values())
        return isinstance(element, (np.ndarray, PIL.Image.Image))

    def _rewrite_json(self, element: Any) -> Any:
        """Recursively rewrites a JSON structure by writing any inlined arrays or images to self.buffered_writer."""
        if isinstance(element, list):
            return [self._rewrite_json(v) for v in element]
        if isinstance(element, dict):
            return {k: self._rewrite_json(v) for k, v in element.items()}
        if isinstance(element, np.ndarray):
            url_idx, start, end, is_bool_array, shape = self._write_inlined_array(element)
            return {
                '__pxttype__': ts.ColumnType.Type.ARRAY.name,
                '__pxturlidx__': url_idx,
                '__pxtstart__': start,
                '__pxtend__': end,
                '__pxtisboolarray__': is_bool_array,
                '__pxtarrayshape__': shape,
            }
        if isinstance(element, PIL.Image.Image):
            url_idx, start, end = self._write_inlined_image(element)
            return {
                '__pxttype__': ts.ColumnType.Type.IMAGE.name,
                '__pxturlidx__': url_idx,
                '__pxtstart__': start,
                '__pxtend__': end,
            }
        return element

    def _write_inlined_array(self, ar: np.ndarray) -> tuple[int, int, int, bool, tuple[int, ...] | None]:
        """
        Write an ndarray to buffered_writer and return:
        index into inlined_obj_files, start offset, end offset, is_bool flag, shape
        """
        url_idx = len(self.inlined_obj_files) - 1
        start = self.buffered_writer.tell()
        shape: tuple[int, ...] | None
        is_bool_array: bool
        if np.issubdtype(ar.dtype, np.bool_):
            shape = ar.shape
            ar = np.packbits(ar)
            is_bool_array = True
        else:
            shape = None
            is_bool_array = False
        np.save(self.buffered_writer, ar, allow_pickle=False)
        end = self.buffered_writer.tell()
        self._flush_full_buffer()
        return url_idx, start, end, is_bool_array, shape

    def _write_inlined_image(self, img: PIL.Image.Image) -> tuple[int, int, int]:
        """Write a PIL image to buffered_writer and return: index into inlined_obj_files, start offset, end offset"""
        url_idx = len(self.inlined_obj_files) - 1
        start = self.buffered_writer.tell()
        format = 'webp' if img.has_transparency_data else 'jpeg'
        img.save(self.buffered_writer, format=format)
        end = self.buffered_writer.tell()
        self._flush_full_buffer()
        return url_idx, start, end

    def _reset_buffer(self) -> None:
        local_path = LocalStore(Env.get().media_dir)._prepare_path_raw(
            self.row_builder.tbl.id, 0, self.row_builder.tbl.version
        )
        self.inlined_obj_files.append(local_path)
        fh = open(local_path, 'wb', buffering=self.MIN_FILE_SIZE * 2)  # noqa: SIM115
        assert isinstance(fh, io.BufferedWriter)
        self.buffered_writer = fh

    def _flush_full_buffer(self, finalize: bool = False) -> None:
        if self.buffered_writer.tell() < self.MIN_FILE_SIZE and not finalize:
            return
        self.buffered_writer.flush()
        os.fsync(self.buffered_writer.fileno())  # needed to force bytes cached by OS to storage
        self.buffered_writer.close()
        if finalize:
            self.buffered_writer = None
        else:
            self._reset_buffer()
