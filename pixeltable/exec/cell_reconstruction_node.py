from __future__ import annotations

import io
from types import NoneType
from typing import Any, AsyncIterator

import numpy as np
import PIL.Image

import pixeltable.type_system as ts
from pixeltable import catalog, exprs
from pixeltable.utils import parse_local_file_path
from pixeltable.utils.filecache import FileCache
from pixeltable.utils.http import fetch_url

from .data_row_batch import DataRowBatch
from .exec_node import ExecNode
from .globals import INLINED_OBJECT_MD_KEY, InlinedObjectMd


def json_has_inlined_objs(element: Any) -> bool:
    """Returns True if element contains inlined objects produced by CellMaterializationNode."""
    if isinstance(element, list):
        return any(json_has_inlined_objs(v) for v in element)
    if isinstance(element, dict):
        if INLINED_OBJECT_MD_KEY in element:
            return True
        return any(json_has_inlined_objs(v) for v in element.values())
    return False


def open_cell_file(url: str, col: catalog.Column, file_handles: dict[str, io.BufferedReader]) -> io.BufferedReader:
    """Returns a handle for a file written by CellMaterializationNode; a remote file is fetched into the file cache."""
    fp = file_handles.get(url)
    if fp is None:
        local_path = parse_local_file_path(url)
        if local_path is None:
            file_cache = FileCache.get()
            local_path = file_cache.lookup(url)
            if local_path is None:
                local_path = file_cache.add(col.get_tbl().id, col.id, url, fetch_url(url))
        fp = open(local_path, 'rb')  # noqa: SIM115
        file_handles[url] = fp
    return fp


def reconstruct_json(
    element: Any, urls: list[str], col: catalog.Column, file_handles: dict[str, io.BufferedReader]
) -> Any:
    """Recursively reconstructs inlined objects in a json structure."""
    if isinstance(element, list):
        return [reconstruct_json(v, urls, col, file_handles) for v in element]
    if isinstance(element, dict):
        if INLINED_OBJECT_MD_KEY in element:
            obj_md = InlinedObjectMd.from_dict(element[INLINED_OBJECT_MD_KEY])
            fp = open_cell_file(urls[obj_md.url_idx], col, file_handles)

            if obj_md.type == ts.ColumnType.Type.ARRAY.name:
                fp.seek(obj_md.array_md.start)
                ar = load_array(
                    fp, obj_md.array_md.start, obj_md.array_md.end, obj_md.array_md.is_bool, obj_md.array_md.shape
                )
                return ar
            elif obj_md.type == ts.ColumnType.Type.IMAGE.name:
                fp.seek(obj_md.img_start)
                bytesio = io.BytesIO(fp.read(obj_md.img_end - obj_md.img_start))
                img = PIL.Image.open(bytesio)
                img.load()
                assert fp.tell() == obj_md.img_end, f'{fp.tell()} != {obj_md.img_end} ({obj_md.img_start})'
                return img
            else:
                assert obj_md.type == ts.ColumnType.Type.BINARY.name
                assert obj_md.binary_md is not None
                fp.seek(obj_md.binary_md.start)
                data = fp.read(obj_md.binary_md.end - obj_md.binary_md.start)
                assert fp.tell() == obj_md.binary_md.end, (
                    f'{fp.tell()} != {obj_md.binary_md.end} ({obj_md.binary_md.start})'
                )
                return data
        else:
            return {k: reconstruct_json(v, urls, col, file_handles) for k, v in element.items()}
    return element


def load_array(
    fh: io.BufferedReader, start: int, end: int, is_bool_array: bool, shape: tuple[int, ...] | None
) -> np.ndarray:
    """Loads an array from a section of a file."""
    fh.seek(start)
    ar = np.load(fh, allow_pickle=False)
    assert fh.tell() == end
    if is_bool_array:
        assert shape is not None
        ar = np.unpackbits(ar, count=np.prod(shape)).reshape(shape).astype(bool)
    return ar


class CellReconstructionNode(ExecNode):
    """
    Reconstruction of stored json and array cells that were produced by CellMaterializationNode.
    """

    json_refs: exprs.ExprSet[exprs.ColumnRef]
    array_refs: exprs.ExprSet[exprs.ColumnRef]
    binary_refs: exprs.ExprSet[exprs.ColumnRef]
    file_handles: dict[str, io.BufferedReader]  # key: file url

    def __init__(
        self,
        json_refs: list[exprs.ColumnRef],
        array_refs: list[exprs.ColumnRef],
        binary_refs: list[exprs.ColumnRef],
        row_builder: exprs.RowBuilder,
        input: ExecNode | None = None,
    ):
        super().__init__(row_builder, [], [], input)
        self.json_refs = exprs.ExprSet(json_refs)
        self.array_refs = exprs.ExprSet(array_refs)
        self.binary_refs = exprs.ExprSet(binary_refs)
        self._init_exec_state()

    def _init_exec_state(self) -> None:
        self.file_handles = {}

    def _open(self) -> None:
        self._init_exec_state()

    async def __aiter__(self) -> AsyncIterator[DataRowBatch]:
        async for batch in self.input:
            for row in batch:
                for col_ref in self.json_refs:
                    val = row[col_ref.slot_idx]
                    if val is None:
                        continue
                    cell_md = row.slot_md.get(col_ref.slot_idx)
                    if cell_md is None or cell_md.file_urls is None or not json_has_inlined_objs(row[col_ref.slot_idx]):
                        continue
                    row[col_ref.slot_idx] = reconstruct_json(val, cell_md.file_urls, col_ref.col, self.file_handles)

                for col_ref in self.array_refs:
                    cell_md = row.slot_md.get(col_ref.slot_idx)
                    if cell_md is not None and cell_md.array_md is not None:
                        assert row[col_ref.slot_idx] is None
                        row[col_ref.slot_idx] = self._reconstruct_array(cell_md, col_ref.col)
                    else:
                        assert isinstance(row[col_ref.slot_idx], (NoneType, np.ndarray))

                for col_ref in self.binary_refs:
                    cell_md = row.slot_md.get(col_ref.slot_idx)
                    if cell_md is not None and cell_md.binary_md is not None:
                        assert row[col_ref.slot_idx] is None
                        row[col_ref.slot_idx] = self._reconstruct_binary(cell_md, col_ref.col)
                    else:
                        assert isinstance(row[col_ref.slot_idx], (NoneType, bytes))

            yield batch

    def _close(self) -> None:
        for fp in self.file_handles.values():
            fp.close()

    def _reconstruct_array(self, cell_md: exprs.CellMd, col: catalog.Column) -> np.ndarray:
        assert cell_md.array_md is not None
        assert cell_md.file_urls is not None and len(cell_md.file_urls) == 1
        fp = open_cell_file(cell_md.file_urls[0], col, self.file_handles)
        ar = load_array(
            fp, cell_md.array_md.start, cell_md.array_md.end, bool(cell_md.array_md.is_bool), cell_md.array_md.shape
        )
        return ar

    def _reconstruct_binary(self, cell_md: exprs.CellMd, col: catalog.Column) -> bytes:
        assert cell_md.binary_md is not None
        assert cell_md.file_urls is not None and len(cell_md.file_urls) == 1
        fp = open_cell_file(cell_md.file_urls[0], col, self.file_handles)
        fp.seek(cell_md.binary_md.start)
        data = fp.read(cell_md.binary_md.end - cell_md.binary_md.start)
        assert fp.tell() == cell_md.binary_md.end
        return data
