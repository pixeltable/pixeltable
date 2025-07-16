import dataclasses
import io
from typing import Any

from PIL import Image

from pixeltable.iterators.base import ComponentIterator
from pixeltable.type_system import ColumnType, ImageType, IntType, StringType
from pixeltable.utils.documents import get_document_handle


@dataclasses.dataclass
class PageChunk:
    text: str
    page: int
    image_bytes: bytes
    image: Image


class PdfPageExtractor(ComponentIterator):
    def __init__(self, document: str):
        self._doc_handle = get_document_handle(document)
        assert self._doc_handle.pdf_doc is not None
        self._pdf_doc = self._doc_handle.pdf_doc
        self._page_iter = iter(self._pdf_doc)

    @classmethod
    def input_schema(cls) -> dict[str, ColumnType]:
        return {'document': StringType(nullable=False)}

    @classmethod
    def output_schema(cls, *args: Any, **kwargs: Any) -> tuple[dict[str, ColumnType], list[str]]:
        return {
            'text': StringType(nullable=False),
            'page': IntType(nullable=False),
            'image': ImageType(nullable=False),
        }, []

    def __next__(self) -> dict[str, Any]:
        page = next(self._page_iter)
        page_number = page.number
        text = page.get_text()

        # Remove NUL bytes
        clean_text = text.replace('\x00', '')

        # Render image
        pix = page.get_pixmap(dpi=300)
        image = Image.open(io.BytesIO(pix.tobytes('png')))

        return {'text': clean_text, 'page': page_number, 'image': image}

    def close(self) -> None:
        self._pdf_doc.close()

    def set_pos(self, pos: int) -> None:
        # Optional: seek support
        pass
