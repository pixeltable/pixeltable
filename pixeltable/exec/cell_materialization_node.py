from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Any, AsyncIterator

import numpy as np
import pgvector.sqlalchemy  # type: ignore[import-untyped]
import PIL.Image

import pixeltable.type_system as ts
from pixeltable import exprs
from pixeltable.utils.media_store import MediaStore

from .data_row_batch import DataRowBatch
from .exec_node import ExecNode

_logger = logging.getLogger('pixeltable')


class CellMaterializationNode(ExecNode):
    """
    Node to populate DataRow.cell_vals/cell_md.

    For now, the scope is limited to populating DataRow.cells_vals for json and array columns.

    TODO:
    - execute file IO via asyncio Tasks in a thread pool
    """

    output_col_info: list[exprs.ColumnSlotIdx]

    # execution state
    embedded_obj_files: list[Path]
    buffered_writer: io.BufferedWriter

    MIN_FILE_SIZE = 8 * 2**20  # 8MB
    MAX_DB_ARRAY_SIZE = 512  # max size of array stored in table column; in bytes

    def __init__(self, row_builder: exprs.RowBuilder, input: ExecNode | None = None):
        super().__init__(row_builder, [], [], input)
        self.output_col_info = [
            col_info
            for col_info in row_builder.table_columns
            if col_info.col.col_type.is_json_type() or col_info.col.col_type.is_array_type()
        ]
        self.embedded_obj_files = []
        self._reset_buffer()

    async def __aiter__(self) -> AsyncIterator[DataRowBatch]:
        async for batch in self.input:
            for row in batch:
                for col, slot_idx in self.output_col_info:
                    if row.has_exc(slot_idx):
                        row.cell_vals[col.id] = None
                        exc = row.get_exc(slot_idx)
                        row.cell_md[col.qualified_id] = exprs.CellMd(errortype=type(exc).__name__, errormsg=str(exc))
                        continue

                    val = row[slot_idx]

                    if col.col_type.is_json_type():
                        if self._json_has_embedded_objs(val):
                            row.cell_vals[col.id] = self._rewrite_json(val)
                            row.cell_md[col.qualified_id] = exprs.CellMd(
                                embedded_object_urls=[local_path.as_uri() for local_path in self.embedded_obj_files]
                            )
                        else:
                            row.cell_vals[col.id] = val
                            row.cell_md[col.qualified_id] = None

                    else:
                        assert col.col_type.is_array_type()
                        assert isinstance(val, np.ndarray)
                        row.cell_vals[col.id] = val

                        if isinstance(col.sa_col_type, pgvector.sqlalchemy.Vector):
                            # this is a vector column (ie, used for a vector index): provide the array itself
                            row.cell_vals[col.id] = val
                            row.cell_md[col.qualified_id] = None
                        elif val.nbytes <= self.MAX_DB_ARRAY_SIZE:
                            # this array is small enough to store in the db column (type: binary) directly
                            buffer = io.BytesIO()
                            np.save(buffer, val, allow_pickle=False)
                            row.cell_vals[col.id] = buffer.getvalue()
                            row.cell_md[col.qualified_id] = None
                        else:
                            # append this array to the buffer and store its location in the cell md
                            ar: np.ndarray
                            if np.issubdtype(val.dtype, np.bool_):
                                # for bool arrays, store as packed bits, otherwise it's 1 byte per bit
                                ar = np.packbits(val)
                            else:
                                ar = val
                            start = self.buffered_writer.tell()
                            np.save(self.buffered_writer, ar, allow_pickle=False)
                            end = self.buffered_writer.tell()
                            row.cell_vals[col.id] = None
                            cell_md = exprs.CellMd(
                                embedded_object_urls=[self.embedded_obj_files[-1].as_uri()],
                                array_start=start,
                                array_end=end,
                            )
                            if np.issubdtype(val.dtype, np.bool_):
                                cell_md.array_is_bool = True
                                cell_md.array_shape = val.shape
                            row.cell_md[col.qualified_id] = cell_md
                            self._flush_full_buffer()

                        assert row.cell_vals[col.id] is not None or row.cell_md[col.qualified_id] is not None

                    # continue with only the last file
                    self.embedded_obj_files = self.embedded_obj_files[-1:]

            yield batch
        self._flush_full_buffer(force=True)

    def _json_has_embedded_objs(self, element: Any) -> bool:
        if isinstance(element, list):
            return any(self._json_has_embedded_objs(v) for v in element)
        if isinstance(element, dict):
            return any(self._json_has_embedded_objs(v) for v in element.values())
        return isinstance(element, (np.ndarray, PIL.Image.Image))

    def _rewrite_json(self, element: Any) -> Any:
        """Recursively rewrites a JSON structure by writing any arrays or images to self.bytes_buffer."""
        if isinstance(element, list):
            return [self._rewrite_json(v) for v in element]
        if isinstance(element, dict):
            return {k: self._rewrite_json(v) for k, v in element.items()}
        if isinstance(element, np.ndarray):
            url_idx, begin, end = self._write_ndarray(element)
            return {
                '__pxttype__': ts.ColumnType.Type.ARRAY.name,
                '__pxturlidx__': url_idx,
                '__pxtbegin__': begin,
                '__pxtend__': end,
            }
        if isinstance(element, PIL.Image.Image):
            url_idx, begin, end = self._write_image(element)
            return {
                '__pxttype__': ts.ColumnType.Type.IMAGE.name,
                '__pxturlidx__': url_idx,
                '__pxtbegin__': begin,
                '__pxtend__': end,
            }
        return element

    def _write_ndarray(self, ar: np.ndarray) -> tuple[int, int, int]:
        """Write an ndarray to bytes_buffer and return: index into embedded_obj_urls, start offset, end offset"""
        url_idx = len(self.embedded_obj_files) - 1
        begin = self.buffered_writer.tell()
        np.save(self.buffered_writer, ar, allow_pickle=False)
        end = self.buffered_writer.tell()
        self._flush_full_buffer()
        return url_idx, begin, end

    def _write_image(self, img: PIL.Image.Image) -> tuple[int, int, int]:
        """Write a PIL image to bytes_buffer and return: index into embedded_obj_urls, start offset, end offset"""
        url_idx = len(self.embedded_obj_files) - 1
        begin = self.buffered_writer.tell()
        format = 'webp' if img.has_transparency_data else 'jpeg'
        img.save(self.buffered_writer, format=format)
        end = self.buffered_writer.tell()
        self._flush_full_buffer()
        return url_idx, begin, end

    def _reset_buffer(self) -> None:
        local_path = MediaStore.get()._prepare_media_path_raw(self.row_builder.tbl.id, 0, self.row_builder.tbl.version)
        self.embedded_obj_files.append(local_path)
        fh = open(local_path, 'wb', buffering=self.MIN_FILE_SIZE * 2)  # noqa: SIM115
        assert isinstance(fh, io.BufferedWriter)
        self.buffered_writer = fh

    def _flush_full_buffer(self, force: bool = False) -> None:
        if self.buffered_writer.tell() < self.MIN_FILE_SIZE and not force:
            return
        self.buffered_writer.flush()
        os.fsync(self.buffered_writer.fileno())
        self.buffered_writer.close()
        self._reset_buffer()
