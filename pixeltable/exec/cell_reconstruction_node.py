from __future__ import annotations

import io
import json
import logging
from typing import Any, AsyncIterator

import numpy as np
import PIL.Image

import pixeltable.type_system as ts
from pixeltable import exprs
from pixeltable.catalog import QColumnId
from pixeltable.utils import parse_local_file_path

from .data_row_batch import DataRowBatch
from .exec_node import ExecNode

_logger = logging.getLogger('pixeltable')


class CellReconstructionNode(ExecNode):
    """ """

    json_expr_info: dict[int, QColumnId]
    array_refs: list[exprs.ColumnRef]
    file_handles: dict[str, io.BufferedReader]  # key: file path

    def __init__(
        self,
        json_expr_info: dict[int, QColumnId],
        array_refs: list[exprs.ColumnRef],
        row_builder: exprs.RowBuilder,
        input: ExecNode | None = None,
    ):
        super().__init__(row_builder, [], [], input)
        self.json_expr_info = json_expr_info
        self.array_refs = array_refs
        self.file_handles = {}

    async def __aiter__(self) -> AsyncIterator[DataRowBatch]:
        try:
            async for batch in self.input:
                for row in batch:
                    for slot_idx, q_id in self.json_expr_info.items():
                        if (
                            row.cell_md[q_id] is None
                            or row.cell_md[q_id].embedded_object_urls is None
                            or not self._json_has_embedded_objs(row[slot_idx])
                        ):
                            continue
                        row[slot_idx] = self._reconstruct_json(row[slot_idx], row.cell_md[q_id].embedded_object_urls)

                    for col_ref in self.array_refs:
                        data = row[col_ref.slot_idx]
                        if data.startswith(b'\x93NUMPY'):  # npy magic string
                            row[col_ref.slot_idx] = np.load(io.BytesIO(data), allow_pickle=False)
                        else:
                            assert data.startswith(b'\x01')
                            row[col_ref.slot_idx] = self._reconstruct_array(data[1:])

                yield batch
        finally:
            for fp in self.file_handles.values():
                fp.close()

    def _reconstruct_array(self, data: bytes) -> np.ndarray:
        location_dict = json.loads(data[1:].decode('utf-8'))
        local_path = parse_local_file_path(location_dict['url'])
        if local_path not in self.file_handles:
            self.file_handles[local_path] = open(local_path, 'rb')  # noqa: SIM115
        fp = self.file_handles[local_path]
        fp.seek(location_dict['begin'])
        assert fp.tell() == location_dict['begin']
        arr = np.load(fp, allow_pickle=False)
        assert fp.tell() == location_dict['end']
        return arr

    def _json_has_embedded_objs(self, element: Any) -> bool:
        if isinstance(element, list):
            return any(self._json_has_embedded_objs(v) for v in element)
        if isinstance(element, dict):
            if '__pxturlidx__' in element:
                assert '__pxttype__' in element
                assert '__pxtbegin__' in element
                assert '__pxtend__' in element
                return True
            return any(self._json_has_embedded_objs(v) for v in element.values())
        return False

    def _reconstruct_json(self, element: Any, urls: list[str]) -> Any:
        if isinstance(element, list):
            return [self._reconstruct_json(v, urls) for v in element]
        if isinstance(element, dict):
            if '__pxttype__' in element:
                assert '__pxturlidx__' in element
                url_idx, begin, end = element['__pxturlidx__'], element['__pxtbegin__'], element['__pxtend__']
                assert isinstance(url_idx, int) and url_idx < len(urls)
                assert isinstance(begin, int)
                assert isinstance(end, int)
                url = urls[url_idx]
                local_path = parse_local_file_path(url)
                if local_path not in self.file_handles:
                    self.file_handles[local_path] = open(local_path, 'rb')  # noqa: SIM115
                fp = self.file_handles[local_path]

                fp.seek(begin)
                assert fp.tell() == begin
                if element['__pxttype__'] == ts.ColumnType.Type.ARRAY.name:
                    arr = np.load(fp, allow_pickle=False)
                    assert fp.tell() == end
                    return arr
                else:
                    assert element['__pxttype__'] == ts.ColumnType.Type.IMAGE.name
                    bytesio = io.BytesIO(fp.read(end - begin))
                    img = PIL.Image.open(bytesio)
                    img.load()
                    assert fp.tell() == end, f'{fp.tell()} != {end} / {begin}'
                    return img
            else:
                return {k: self._reconstruct_json(v, urls) for k, v in element.items()}
        return element
