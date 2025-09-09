from __future__ import annotations

import logging
from typing import AsyncIterator, Optional

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

    def __init__(
        self,
        row_builder: exprs.RowBuilder,
        input: Optional[ExecNode] = None,
    ):
        super().__init__(row_builder, [], [], input)
        self.output_col_info = [col_info for col_info in row_builder.table_columns if col_info.col.col_type.is_json_type() or
            col_info.col.col_type.is_array_type()]

    def __aiter__(self) -> AsyncIterator[DataRowBatch]:
        async for batch in self.input:
            for row in batch:
                for col, slot_idx in self.output_col_info:
                    row.set_cell_val(slot_idx, row[slot_idx])
                row.set_cell_md()

            yield batch

    def flush_json(self, index: int, col: catalog.Column) -> None:
        """If the JSON object contains images or arrays, save them to a media store and rewrite the JSON object."""
        # TODO: Do the same thing for lists that exceed a certain length?
        # TODO: Also allow datetimes?
        if self.__has_media(self.vals[index]):
            path = TempStore.create_path(extension='.bin')
            with open(path, 'wb') as fp:
                self.vals[index] = self.__rewrite_json(self.vals[index], fp)
            # Now move the temp file to the media store and update the JSON structure with the new URL
            url = MediaStore.get().relocate_local_media_file(path, col)
            self.__add_url_to_json(self.vals[index], url)
            self.file_urls[index] = url

    @classmethod
    def __has_media(cls, element: Any) -> bool:
        if isinstance(element, list):
            return any(cls.__has_media(v) for v in element)
        if isinstance(element, dict):
            return any(cls.__has_media(v) for v in element.values())
        return isinstance(element, (np.ndarray, PIL.Image.Image))

    @classmethod
    def __rewrite_json(cls, element: Any, fp: io.BufferedWriter) -> Any:
        """Recursively rewrites a JSON structure by exporting any arrays or images to a binary file."""
        if isinstance(element, list):
            return [cls.__rewrite_json(v, fp) for v in element]
        if isinstance(element, dict):
            return {k: cls.__rewrite_json(v, fp) for k, v in element.items()}
        if isinstance(element, np.ndarray):
            begin = fp.tell()
            np.save(fp, element, allow_pickle=False)
            end = fp.tell()
            return {'__pxttype__': ts.ColumnType.Type.ARRAY.name, '__pxtbegin__': begin, '__pxtend__': end}
        if isinstance(element, PIL.Image.Image):
            format = 'webp' if element.has_transparency_data else 'jpeg'
            begin = fp.tell()
            element.save(fp, format=format)
            end = fp.tell()
            return {'__pxttype__': ts.ColumnType.Type.IMAGE.name, '__pxtbegin__': begin, '__pxtend__': end}
        return element

    @classmethod
    def __add_url_to_json(cls, element: Any, url: str) -> None:
        if isinstance(element, list):
            for v in element:
                cls.__add_url_to_json(v, url)
        if isinstance(element, dict):
            if '__pxttype__' in element:
                element['__pxturl__'] = url
            else:
                for v in element.values():
                    cls.__add_url_to_json(v, url)

    @classmethod
    def reconstruct_json(cls, element: Any) -> Any:
        """
        Recursively reconstructs a JSON structure that may contain references to image or array
        data stored in a binary file.
        """
        url = cls.__find_pxturl(element)
        if url is None:
            return element
        parsed = urllib.parse.urlparse(url)
        assert parsed.scheme == 'file'
        path = urllib.parse.unquote(urllib.request.url2pathname(parsed.path))
        with open(path, 'rb') as fp:
            return cls.__reconstruct_json(element, fp)

    @classmethod
    def __find_pxturl(cls, element: Any) -> Optional[str]:
        if isinstance(element, list):
            for v in element:
                url = cls.__find_pxturl(v)
                if url is not None:
                    return url

        if isinstance(element, dict):
            if '__pxturl__' in element:
                return element['__pxturl__']
            for v in element.values():
                url = cls.__find_pxturl(v)
                if url is not None:
                    return url

        return None

