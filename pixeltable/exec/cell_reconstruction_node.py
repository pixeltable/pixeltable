from __future__ import annotations

import asyncio
import io
import itertools
from concurrent import futures
from pathlib import Path
from types import NoneType
from typing import IO, Any, AsyncIterator, Iterable

import numpy as np
import PIL.Image

import pixeltable.type_system as ts
from pixeltable import exprs
from pixeltable.utils import parse_local_file_path
from pixeltable.utils.local_store import TempStore
from pixeltable.utils.object_stores import ObjectOps

from .data_row_batch import DataRowBatch
from .exec_node import ExecNode
from .globals import INLINED_OBJECT_BYTES_KEY, INLINED_OBJECT_MD_KEY, InlinedObjectMd


class CellFileReader:
    """Opens the chunk files written by CellMaterializationNode, downloading remote ones.

    A remote chunk is downloaded to a TempStore file that the reader owns and deletes in release() or close(). Downloads
    bypass the FileCache: a chunk can be larger than the cache, and the cache doesn't evict files that are in use, so a
    scan over more chunks than it holds would fail.
    """

    MAX_CONCURRENT_DOWNLOADS = 8

    file_handles: dict[str, io.BufferedReader]  # key: chunk url
    downloads: dict[str, Path]  # TempStore files of downloaded remote chunks; key: chunk url
    executor: futures.ThreadPoolExecutor | None

    def __init__(self) -> None:
        self.file_handles = {}
        self.downloads = {}
        self.executor = None

    def open_file(self, url: str) -> io.BufferedReader:
        """Returns a handle for the chunk at url; a remote chunk that wasn't prefetched is downloaded first."""
        fp = self.file_handles.get(url)
        if fp is None:
            path = parse_local_file_path(url)
            if path is None:
                path = self.downloads.get(url) or self._download(url)
            fp = open(path, 'rb')  # noqa: SIM115
            self.file_handles[url] = fp
        return fp

    async def prefetch(self, urls: Iterable[str]) -> None:
        """Downloads the remote chunks among urls that haven't been downloaded yet, concurrently."""
        missing = [url for url in set(urls) if url not in self.downloads and parse_local_file_path(url) is None]
        if len(missing) == 0:
            return
        if self.executor is None:
            self.executor = futures.ThreadPoolExecutor(
                max_workers=self.MAX_CONCURRENT_DOWNLOADS, thread_name_prefix='pxt-chunk-download'
            )
        loop = asyncio.get_running_loop()
        await asyncio.gather(*(loop.run_in_executor(self.executor, self._download, url) for url in missing))

    def _download(self, url: str) -> Path:
        path = TempStore.create_path()
        try:
            ObjectOps.copy_object_to_local_file(url, path)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        self.downloads[url] = path
        return path

    def release(self, keep: set[str]) -> None:
        """Closes the chunks not in keep and deletes their downloads."""
        for url in [url for url in self.file_handles if url not in keep]:
            self.file_handles.pop(url).close()
        for url in [url for url in self.downloads if url not in keep]:
            self.downloads.pop(url).unlink(missing_ok=True)

    def close(self) -> None:
        if self.executor is not None:
            # let in-flight downloads finish, so that release() deletes them
            self.executor.shutdown(wait=True)
            self.executor = None
        self.release(keep=set())


def json_has_inlined_objs(element: Any) -> bool:
    """Returns True if element contains inlined objects produced by CellMaterializationNode."""
    if isinstance(element, list):
        return any(json_has_inlined_objs(v) for v in element)
    if isinstance(element, dict):
        if INLINED_OBJECT_MD_KEY in element:
            return True
        return any(json_has_inlined_objs(v) for v in element.values())
    return False


def collect_json_urls(element: Any, urls: list[str], result: set[str]) -> None:
    """Adds the urls of the chunks that hold the inlined objects in a json structure to result."""
    if isinstance(element, list):
        for v in element:
            collect_json_urls(v, urls, result)
    elif isinstance(element, dict):
        if INLINED_OBJECT_MD_KEY in element:
            result.add(urls[element[INLINED_OBJECT_MD_KEY]['url_idx']])
        else:
            for v in element.values():
                collect_json_urls(v, urls, result)


def _object_range(obj_md: InlinedObjectMd) -> tuple[int, int]:
    """Returns the start and end offsets of an inlined object in its chunk."""
    if obj_md.type == ts.ColumnType.Type.ARRAY.name:
        assert obj_md.array_md is not None
        return obj_md.array_md.start, obj_md.array_md.end
    if obj_md.type == ts.ColumnType.Type.IMAGE.name:
        assert obj_md.img_start is not None and obj_md.img_end is not None
        return obj_md.img_start, obj_md.img_end
    assert obj_md.type == ts.ColumnType.Type.BINARY.name
    assert obj_md.binary_md is not None
    return obj_md.binary_md.start, obj_md.binary_md.end


def load_remote_objects(element: Any, urls: list[str], reader: CellFileReader) -> Any:
    """Adds the bytes of the inlined objects stored in remote chunks to a json structure, so that reconstruct_json()
    doesn't need reader's chunks."""
    if isinstance(element, list):
        return [load_remote_objects(v, urls, reader) for v in element]
    if isinstance(element, dict):
        if INLINED_OBJECT_MD_KEY not in element:
            return {k: load_remote_objects(v, urls, reader) for k, v in element.items()}
        obj_md = InlinedObjectMd.from_dict(element[INLINED_OBJECT_MD_KEY])
        url = urls[obj_md.url_idx]
        if parse_local_file_path(url) is not None:
            return element
        start, end = _object_range(obj_md)
        fp = reader.open_file(url)
        fp.seek(start)
        return {**element, INLINED_OBJECT_BYTES_KEY: fp.read(end - start)}
    return element


def reconstruct_json(element: Any, urls: list[str], reader: CellFileReader) -> Any:
    """Recursively reconstructs inlined objects in a json structure."""
    if isinstance(element, list):
        return [reconstruct_json(v, urls, reader) for v in element]
    if isinstance(element, dict):
        if INLINED_OBJECT_MD_KEY in element:
            obj_md = InlinedObjectMd.from_dict(element[INLINED_OBJECT_MD_KEY])
            start, end = _object_range(obj_md)
            fp: IO[bytes]
            data = element.get(INLINED_OBJECT_BYTES_KEY)
            if data is None:
                fp = reader.open_file(urls[obj_md.url_idx])
            else:
                # load_remote_objects() already read the object
                fp, start, end = io.BytesIO(data), 0, len(data)

            if obj_md.type == ts.ColumnType.Type.ARRAY.name:
                assert obj_md.array_md is not None
                return load_array(fp, start, end, bool(obj_md.array_md.is_bool), obj_md.array_md.shape)
            elif obj_md.type == ts.ColumnType.Type.IMAGE.name:
                fp.seek(start)
                bytesio = io.BytesIO(fp.read(end - start))
                img = PIL.Image.open(bytesio)
                img.load()
                assert fp.tell() == end, f'{fp.tell()} != {end} ({start})'
                return img
            else:
                assert obj_md.type == ts.ColumnType.Type.BINARY.name
                fp.seek(start)
                data = fp.read(end - start)
                assert fp.tell() == end, f'{fp.tell()} != {end} ({start})'
                return data
        else:
            return {k: reconstruct_json(v, urls, reader) for k, v in element.items()}
    return element


def load_array(fh: IO[bytes], start: int, end: int, is_bool_array: bool, shape: tuple[int, ...] | None) -> np.ndarray:
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

    A json column that is only accessed through JsonPaths isn't reconstructed here: each JsonPath reconstructs the
    elements it selects. JsonPaths are evaluated downstream, after this node may have released the chunks of their
    rows, so for those columns this node adds the bytes of the objects in remote chunks to the json value.

    The node holds only the chunks of its current batch, downloading remote ones concurrently.
    """

    json_refs: exprs.ExprSet[exprs.ColumnRef]
    array_refs: exprs.ExprSet[exprs.ColumnRef]
    binary_refs: exprs.ExprSet[exprs.ColumnRef]
    json_path_anchors: exprs.ExprSet[exprs.ColumnRef]
    reader: CellFileReader

    def __init__(
        self,
        json_refs: list[exprs.ColumnRef],
        array_refs: list[exprs.ColumnRef],
        binary_refs: list[exprs.ColumnRef],
        json_path_anchors: list[exprs.ColumnRef],
        row_builder: exprs.RowBuilder,
        input: ExecNode | None = None,
    ):
        super().__init__(row_builder, [], [], input)
        self.json_refs = exprs.ExprSet(json_refs)
        self.array_refs = exprs.ExprSet(array_refs)
        self.binary_refs = exprs.ExprSet(binary_refs)
        self.json_path_anchors = exprs.ExprSet(json_path_anchors)
        self._init_exec_state()

    def _init_exec_state(self) -> None:
        self.reader = CellFileReader()

    def _open(self) -> None:
        self._init_exec_state()

    async def __aiter__(self) -> AsyncIterator[DataRowBatch]:
        async for batch in self.input:
            urls = self._chunk_urls(batch)
            self.reader.release(keep=urls)
            await self.reader.prefetch(urls)
            for row in batch:
                for col_ref in self.json_refs:
                    val = row[col_ref.slot_idx]
                    if val is None:
                        continue
                    cell_md = row.slot_md.get(col_ref.slot_idx)
                    if cell_md is None or cell_md.file_urls is None or not json_has_inlined_objs(row[col_ref.slot_idx]):
                        continue
                    row[col_ref.slot_idx] = reconstruct_json(val, cell_md.file_urls, self.reader)

                for col_ref in self.json_path_anchors:
                    val = row[col_ref.slot_idx]
                    if val is None:
                        continue
                    cell_md = row.slot_md.get(col_ref.slot_idx)
                    if cell_md is None or cell_md.file_urls is None or not json_has_inlined_objs(val):
                        continue
                    row[col_ref.slot_idx] = load_remote_objects(val, cell_md.file_urls, self.reader)

                for col_ref in self.array_refs:
                    cell_md = row.slot_md.get(col_ref.slot_idx)
                    if cell_md is not None and cell_md.array_md is not None:
                        assert row[col_ref.slot_idx] is None
                        row[col_ref.slot_idx] = self._reconstruct_array(cell_md)
                    else:
                        assert isinstance(row[col_ref.slot_idx], (NoneType, np.ndarray))

                for col_ref in self.binary_refs:
                    cell_md = row.slot_md.get(col_ref.slot_idx)
                    if cell_md is not None and cell_md.binary_md is not None:
                        assert row[col_ref.slot_idx] is None
                        row[col_ref.slot_idx] = self._reconstruct_binary(cell_md)
                    else:
                        assert isinstance(row[col_ref.slot_idx], (NoneType, bytes))

            yield batch

    def _close(self) -> None:
        self.reader.close()

    def _chunk_urls(self, batch: DataRowBatch) -> set[str]:
        """Returns the urls of the chunks referenced by the rows in batch."""
        urls: set[str] = set()
        for row in batch:
            for col_ref in itertools.chain(self.array_refs, self.binary_refs):
                cell_md = row.slot_md.get(col_ref.slot_idx)
                if cell_md is not None and (cell_md.array_md is not None or cell_md.binary_md is not None):
                    assert cell_md.file_urls is not None
                    urls.add(cell_md.file_urls[0])
            for col_ref in itertools.chain(self.json_refs, self.json_path_anchors):
                val = row[col_ref.slot_idx]
                cell_md = row.slot_md.get(col_ref.slot_idx)
                if val is not None and cell_md is not None and cell_md.file_urls is not None:
                    collect_json_urls(val, cell_md.file_urls, urls)
        return urls

    def _reconstruct_array(self, cell_md: exprs.CellMd) -> np.ndarray:
        assert cell_md.array_md is not None
        assert cell_md.file_urls is not None and len(cell_md.file_urls) == 1
        fp = self.reader.open_file(cell_md.file_urls[0])
        ar = load_array(
            fp, cell_md.array_md.start, cell_md.array_md.end, bool(cell_md.array_md.is_bool), cell_md.array_md.shape
        )
        return ar

    def _reconstruct_binary(self, cell_md: exprs.CellMd) -> bytes:
        assert cell_md.binary_md is not None
        assert cell_md.file_urls is not None and len(cell_md.file_urls) == 1
        fp = self.reader.open_file(cell_md.file_urls[0])
        fp.seek(cell_md.binary_md.start)
        data = fp.read(cell_md.binary_md.end - cell_md.binary_md.start)
        assert fp.tell() == cell_md.binary_md.end
        return data
