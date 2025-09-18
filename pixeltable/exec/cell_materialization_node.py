from __future__ import annotations

import io
import json
import logging
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
    embedded_obj_buffer: io.BytesIO

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

                    else:
                        assert col.col_type.is_array_type()
                        assert isinstance(val, np.ndarray)
                        row.cell_vals[col.id] = val

                        if isinstance(col.sa_col_type, pgvector.sqlalchemy.Vector):
                            # this is a vector column (ie, used for a vector index): provide the array itself
                            row.cell_vals[col.id] = val
                        elif val.nbytes <= self.MAX_DB_ARRAY_SIZE:
                            # this array is small enough to store in the db column (type: binary) directly
                            buffer = io.BytesIO()
                            np.save(buffer, val, allow_pickle=False)
                            row.cell_vals[col.id] = buffer.getvalue()
                        else:
                            # append this array to the buffer and store its location in the cell
                            begin = self.embedded_obj_buffer.tell()
                            np.save(self.embedded_obj_buffer, val, allow_pickle=False)
                            end = self.embedded_obj_buffer.tell()
                            location_dict = {'url': self.embedded_obj_files[-1].as_uri(), 'begin': begin, 'end': end}
                            # we need to store location_dict in the array table column, which has type binary:
                            # we binary-encode the serialized dict, with a magic prefix to make sure it's
                            # distinguishable from a serialized numpy array.
                            row.cell_vals[col.id] = b'\x01' + json.dumps(location_dict).encode('utf-8')
                            self._flush_full_buffer()

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

    def _write_ndarray(self, element: np.ndarray) -> tuple[int, int, int]:
        """Write an ndarray to bytes_buffer and return: index into embedded_obj_urls, start offset, end offset"""
        url_idx = len(self.embedded_obj_files) - 1
        begin = self.embedded_obj_buffer.tell()
        np.save(self.embedded_obj_buffer, element, allow_pickle=False)
        end = self.embedded_obj_buffer.tell()
        self._flush_full_buffer()
        return url_idx, begin, end

    def _write_image(self, element: PIL.Image.Image) -> tuple[int, int, int]:
        """Write a PIL image to bytes_buffer and return: index into embedded_obj_urls, start offset, end offset"""
        url_idx = len(self.embedded_obj_files) - 1
        begin = self.embedded_obj_buffer.tell()
        format = 'webp' if element.has_transparency_data else 'jpeg'
        element.save(self.embedded_obj_buffer, format=format)
        end = self.embedded_obj_buffer.tell()
        self._flush_full_buffer()
        return url_idx, begin, end

    def _reset_buffer(self) -> None:
        local_path = MediaStore.get()._prepare_media_path_raw(self.row_builder.tbl.id, 0, self.row_builder.tbl.version)
        self.embedded_obj_files.append(local_path)
        self.embedded_obj_buffer = io.BytesIO()

    def _flush_full_buffer(self, force: bool = False) -> None:
        if self.embedded_obj_buffer.tell() < self.MIN_FILE_SIZE and not force:
            return
        self.embedded_obj_buffer.seek(0)
        with open(self.embedded_obj_files[-1], 'wb') as f:
            f.write(self.embedded_obj_buffer.getbuffer())
        self._reset_buffer()
