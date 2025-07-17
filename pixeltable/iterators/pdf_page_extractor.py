import dataclasses
import io
from typing import Any, Optional

import PIL.Image

from pixeltable.iterators.base import ComponentIterator
from pixeltable.iterators.document import ChunkMetadata, DocumentSplitter, _parse_metadata
from pixeltable.type_system import ColumnType, DocumentType, ImageType, IntType, StringType
from pixeltable.utils.documents import get_document_handle


@dataclasses.dataclass
class PageChunk:
    text: str
    page: int
    image_bytes: bytes
    image: PIL.Image.Image


class PdfPageExtractor(ComponentIterator):
    def __init__(
        self,
        document: str,
        dpi: int = 300,
        image_format: str = 'png',
        metadata: str = '',
        limit: Optional[int] = None,
        overlap: Optional[int] = None,
        separators: str = 'page',
    ):
        self._doc_handle = get_document_handle(document)
        assert self._doc_handle.pdf_doc is not None
        self._pdf_doc = self._doc_handle.pdf_doc
        self._page_iter = iter(self._pdf_doc)
        self._dpi = dpi
        self._image_format = image_format
        self._metadata_fields = _parse_metadata(metadata)

        # Validate output schema to ensure metadata is supported
        self.output_schema(metadata=metadata)

    @classmethod
    def input_schema(cls) -> dict[str, ColumnType]:
        return {
            'document': DocumentType(nullable=False),
            'dpi': IntType(nullable=True),
            'image_format': StringType(nullable=True),
            'metadata': StringType(nullable=True),
            'limit': IntType(nullable=True),
            'overlap': IntType(nullable=True),
            'separators': StringType(nullable=True),
        }

    @classmethod
    def output_schema(cls, *args: Any, **kwargs: Any) -> tuple[dict[str, ColumnType], list[str]]:
        schema: dict[str, ColumnType] = {
            'text': StringType(nullable=False),
            'page': IntType(nullable=False),
            'image': ImageType(nullable=False),
        }
        metadata_fields = _parse_metadata(kwargs.get('metadata', ''))
        for md_field in metadata_fields:
            schema[md_field.name.lower()] = DocumentSplitter.METADATA_COLUMN_TYPES[md_field]
        return schema, []

    def __next__(self) -> dict[str, Any]:
        page = next(self._page_iter)
        page_number = page.number
        text = page.get_text()
        clean_text = text.replace('\x00', '')

        pix = page.get_pixmap(dpi=self._dpi)
        image = PIL.Image.open(io.BytesIO(pix.tobytes(self._image_format)))

        result = {'text': clean_text, 'page': page_number, 'image': image}

        # Dynamically include metadata fields
        for md_field in self._metadata_fields:
            if md_field == ChunkMetadata.PAGE:
                result['page'] = page_number
            elif md_field == ChunkMetadata.BOUNDING_BOX:
                result['bounding_box'] = {'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}  # placeholder if needed
            # Add more metadata types here as needed

        return result

    def close(self) -> None:
        self._pdf_doc.close()

    def set_pos(self, pos: int) -> None:
        pass  # Not implemented yet
