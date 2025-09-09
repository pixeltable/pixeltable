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
    bytes_buffer: io.BytesIO

    MIN_FILE_SIZE = 8 * 2**20  # 8MB

    def __init__(self, row_builder: exprs.RowBuilder, input: ExecNode | None = None):
        super().__init__(row_builder, [], [], input)
        self.output_col_info = [
            col_info for col_info in row_builder.table_columns if col_info.col.col_type.is_json_type()
        ]
        self.embedded_obj_urls = []
        self.bytes_buffer = io.BytesIO()

    def __aiter__(self) -> AsyncIterator[DataRowBatch]:
        async for batch in self.input:
            for row in batch:
                for col, slot_idx in self.output_col_info:
                    val = row[slot_idx]
                    if self._has_embedded_objs(val):
                        row.cell_vals[slot_idx] = self._rewrite_json(val)
                        if row.cell_md[slot_idx] is None:
                            row.cell_md[slot_idx] = exprs.CellMd()
                        row.cell_md[slot_idx].embedded_object_file_urls = copy(self.embedded_obj_urls)
                        # discard all completed urls
                        self.embedded_obj_urls = self.embedded_obj_urls[-1:]

            yield batch

    def _has_embedded_objs(self, element: Any) -> bool:
        if isinstance(element, list):
            return any(self._has_embedded_objs(v) for v in element)
        if isinstance(element, dict):
            return any(self._has_embedded_objs(v) for v in element.values())
        return isinstance(element, (np.ndarray, PIL.Image.Image))

    def _rewrite_json(self, element: Any) -> Any:
        """Recursively rewrites a JSON structure by exporting any arrays or images to a binary file."""
        if isinstance(element, list):
            return [self._rewrite_json(v) for v in element]
        if isinstance(element, dict):
            return {k: self._rewrite_json(v) for k, v in element.items()}
        if isinstance(element, np.ndarray):
            output_url, begin, end = self._write_ndarray(element)
            return {
                '__pxttype__': ts.ColumnType.Type.ARRAY.name,
                '__pxturl__': output_url,
                '__pxtbegin__': begin,
                '__pxtend__': end,
            }
        if isinstance(element, PIL.Image.Image):
            output_url, begin, end = self._write_image(element)
            return {
                '__pxttype__': ts.ColumnType.Type.IMAGE.name,
                '__pxturl__': output_url,
                '__pxtbegin__': begin,
                '__pxtend__': end,
            }
        return element

    def _write_ndarray(self, element: np.ndarray) -> tuple[int, int, int]:
        """Write an ndarray to bytes_buffer and return: index into embedded_obj_urls, start offset, end offset"""
        url_idx = len(self.embedded_obj_urls) - 1
        begin = self.bytes_buffer.tell()
        np.save(self.bytes_buffer, element, allow_pickle=False)
        end = self.bytes_buffer.tell()
        return url_idx, begin, end

    def _write_image(self, element: PIL.Image.Image) -> tuple[Path, int, int]:
        """Write a PIL image to bytes_buffer and return: index into embedded_obj_urls, start offset, end offset"""
        url_idx = len(self.embedded_obj_urls) - 1
        begin = self.bytes_buffer.tell()
        format = 'webp' if element.has_transparency_data else 'jpeg'
        element.save(self.bytes_buffer, format=format)
        end = self.bytes_buffer.tell()
        return url_idx, begin, end
    def _flush_buffer(self) -> None:
        if self.bytes_buffer.tell() >= self.MIN_FILE_SIZE:
            self.bytes_buffer.flush()
            self.embedded_obj_urls.append(self.bytes_buffer.name)
