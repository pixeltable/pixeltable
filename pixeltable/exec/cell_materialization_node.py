from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, AsyncIterator
from copy import copy

import numpy as np
import PIL.Image

import pixeltable.type_system as ts
from pixeltable import exprs
from pixeltable.utils.media_store import MediaStore

from .data_row_batch import DataRowBatch

_logger = logging.getLogger('pixeltable')


from .exec_node import ExecNode


class CellMaterializationNode(ExecNode):
    """
    Node to populate DataRow.cell_vals/cell_md.

    For now, the scope is limited to populating DataRow.cells_vals for json and array columns.
    """

    output_col_info: list[exprs.ColumnSlotIdx]

    # execution state
    embedded_obj_urls: list[Path]
    embedded_obj_buffer: io.BytesIO

    MIN_FILE_SIZE = 8 * 2**20  # 8MB

    def __init__(self, row_builder: exprs.RowBuilder, input: ExecNode | None = None):
        super().__init__(row_builder, [], [], input)
        self.output_col_info = [
            col_info for col_info in row_builder.table_columns if col_info.col.col_type.is_json_type()
        ]
        self.embedded_obj_urls = []
        self._reset_buffer()

    async def __aiter__(self) -> AsyncIterator[DataRowBatch]:
        async for batch in self.input:
            for row in batch:
                for col, slot_idx in self.output_col_info:
                    row.cell_has_val[col.id] = True
                    if row.has_exc(slot_idx):
                        exc = row.get_exc(slot_idx)
                        row.cell_md[slot_idx] = exprs.CellMd(errortype=type(exc).__name__, errormsg=str(exc))
                    else:
                        val = row[slot_idx]
                        if self._has_embedded_objs(val):
                            row.cell_vals[col.id] = self._rewrite_json(val)
                            row.cell_md[slot_idx] = exprs.CellMd(
                                embedded_object_file_urls=[str(url) for url in self.embedded_obj_urls]
                            )
                            # discard all completed urls
                            self.embedded_obj_urls = self.embedded_obj_urls[-1:]
                        else:
                            row.cell_vals[col.id] = val

            yield batch
        self._flush_buffer(force=True)

    def _has_embedded_objs(self, element: Any) -> bool:
        if isinstance(element, list):
            return any(self._has_embedded_objs(v) for v in element)
        if isinstance(element, dict):
            return any(self._has_embedded_objs(v) for v in element.values())
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
        url_idx = len(self.embedded_obj_urls) - 1
        begin = self.embedded_obj_buffer.tell()
        np.save(self.embedded_obj_buffer, element, allow_pickle=False)
        end = self.embedded_obj_buffer.tell()
        return url_idx, begin, end

    def _write_image(self, element: PIL.Image.Image) -> tuple[int, int, int]:
        """Write a PIL image to bytes_buffer and return: index into embedded_obj_urls, start offset, end offset"""
        url_idx = len(self.embedded_obj_urls) - 1
        begin = self.embedded_obj_buffer.tell()
        format = 'webp' if element.has_transparency_data else 'jpeg'
        element.save(self.embedded_obj_buffer, format=format)
        end = self.embedded_obj_buffer.tell()
        return url_idx, begin, end

    def _reset_buffer(self) -> None:
        url = MediaStore.get()._prepare_media_path_raw(self.row_builder.tbl.id, 0, self.row_builder.tbl.version)
        self.embedded_obj_urls.append(url)
        self.embedded_obj_buffer = io.BytesIO()

    def _flush_buffer(self, force: bool = False) -> None:
        if self.embedded_obj_buffer.tell() >= self.MIN_FILE_SIZE or force:
            self.embedded_obj_buffer.seek(0)
            with open(self.embedded_obj_urls[-1], 'wb') as f:
                f.write(self.embedded_obj_buffer.getbuffer())
            self._reset_buffer()
