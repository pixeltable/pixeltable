"""HTML rendering of `ResultSet` values in notebooks (upstream `pixeltable.utils.formatter.Formatter`)."""

from __future__ import annotations

import base64
import html
import io
import json
import logging
import mimetypes
import uuid
from typing import Any, Callable

import numpy as np
import PIL.Image

from pixeltable import _http_server
from pixeltable.type_system import ColumnType

_logger = logging.getLogger('pixeltable')


class Formatter:
    """HTML formatters for the values of a `ResultSet` with `num_rows` rows and `num_cols` columns."""

    __FLOAT_PRECISION = 3
    __LIST_THRESHOLD = 16
    __LIST_EDGEITEMS = 6
    __STRING_SEP = ' ...... '
    __STRING_MAX_LEN = 1000
    __NESTED_STRING_MAX_LEN = 300

    def __init__(self, num_rows: int, num_cols: int):
        self.__num_rows = num_rows
        self.__num_cols = num_cols

    def get_pandas_formatter(self, col_type: ColumnType) -> Callable | None:
        if col_type.is_string_type():
            return self.format_string
        if col_type.is_uuid_type():
            return self.format_uuid
        if col_type.is_binary_type():
            return self.format_binary
        if col_type.is_float_type():
            return self.format_float
        if col_type.is_json_type():
            return self.format_json
        if col_type.is_array_type():
            return self.format_array
        if col_type.is_image_type():
            return self.format_img
        if col_type.is_video_type():
            return self.format_video
        if col_type.is_audio_type():
            return self.format_audio
        if col_type.is_document_type():
            return self.format_document
        return None

    @classmethod
    def format_string(cls, val: str) -> str:
        return cls.__escape(cls.abbreviate(val))

    @classmethod
    def format_uuid(cls, val: uuid.UUID | None) -> str:
        return '' if val is None else cls.format_string(str(val))

    @classmethod
    def format_binary(cls, val: bytes) -> str:
        return cls.format_string(str(val))

    @classmethod
    def abbreviate(cls, val: str, max_len: int = __STRING_MAX_LEN) -> str:
        if len(val) > max_len:
            edgeitems = (max_len - len(cls.__STRING_SEP)) // 2
            return f'{val[:edgeitems]}{cls.__STRING_SEP}{val[-edgeitems:]}'
        return val

    @classmethod
    def __escape(cls, val: str) -> str:
        # escape $ to suppress MathJax formatting
        return html.escape(val).replace('$', r'\$')

    @classmethod
    def format_float(cls, val: float) -> str:
        return np.array2string(np.array(val), precision=cls.__FLOAT_PRECISION)

    @classmethod
    def format_array(cls, arr: np.ndarray) -> str:
        return np.array2string(
            arr,
            precision=cls.__FLOAT_PRECISION,
            threshold=cls.__LIST_THRESHOLD,
            edgeitems=cls.__LIST_EDGEITEMS,
            max_line_width=1000000,
        )

    @classmethod
    def format_json(cls, val: Any, escape_strings: bool = True) -> str:
        if isinstance(val, str):
            return cls.format_string(val) if escape_strings else cls.abbreviate(val)
        return cls.__format_json_rec(val, escape_strings)

    @classmethod
    def __format_json_rec(cls, val: Any, escape_strings: bool) -> str:
        if isinstance(val, str):
            formatted = json.dumps(cls.abbreviate(val, cls.__NESTED_STRING_MAX_LEN))
            return cls.__escape(formatted) if escape_strings else formatted
        if isinstance(val, float):
            return cls.format_float(val)
        if isinstance(val, np.ndarray):
            return cls.format_array(val)
        if isinstance(val, list):
            if len(val) < cls.__LIST_THRESHOLD:
                components = [cls.__format_json_rec(x, escape_strings) for x in val]
            else:
                components = [cls.__format_json_rec(x, escape_strings) for x in val[: cls.__LIST_EDGEITEMS]]
                components.append('...')
                components.extend(cls.__format_json_rec(x, escape_strings) for x in val[-cls.__LIST_EDGEITEMS :])
            return '[' + ', '.join(components) + ']'
        if isinstance(val, dict):
            kv_pairs = (
                f'{cls.__format_json_rec(k, escape_strings)}: {cls.__format_json_rec(v, escape_strings)}'
                for k, v in val.items()
            )
            return '{' + ', '.join(kv_pairs) + '}'
        try:
            return json.dumps(val)
        except TypeError:
            return cls.__escape(str(val))

    def format_img(self, img: PIL.Image.Image) -> str:
        if self.__num_rows > 1:
            width = min(240, img.width)
        elif self.__num_cols > 1:
            width = min(480, img.width)
        else:
            width = min(640, img.width)
        with io.BytesIO() as buffer:
            img.save(buffer, 'webp')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
        return f"""
            <div class="pxt_image" style="width:{width}px;">
                <img src="data:image/webp;base64,{img_base64}" width="{width}" />
            </div>
            """

    def format_video(self, file_path: str) -> str:
        # the first frame as the poster, so that an exported notebook shows something without the video
        thumb = self.extract_first_video_frame(file_path)
        if thumb is None:
            thumb_tag = ''
        else:
            with io.BytesIO() as buffer:
                thumb.save(buffer, 'jpeg')
                thumb_tag = f'poster="data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode()}"'
        if self.__num_rows > 1:
            width = 320
        elif self.__num_cols > 1:
            width = 480
        else:
            width = 800
        return f"""
        <div class="pxt_video" style="width:{width}px;">
            <video controls width="{width}" {thumb_tag}>
                {self.__create_source_tag(file_path)}
            </video>
        </div>
        """

    @classmethod
    def extract_first_video_frame(cls, file_path: str) -> PIL.Image.Image | None:
        import av

        try:
            with av.open(file_path) as container:
                img = next(container.decode(video=0)).to_image()
                return img if isinstance(img, PIL.Image.Image) else None
        except Exception:
            return None

    def format_audio(self, file_path: str) -> str:
        return f"""
        <div class="pxt_audio">
            <audio controls>
                {self.__create_source_tag(file_path)}
            </audio>
        </div>
        """

    def format_document(self, file_path: str, max_width: int = 320, max_height: int = 320) -> str:
        inner_element = html.escape(file_path)
        thumb = self.make_document_thumbnail(file_path, max_width, max_height)
        if thumb is not None:
            with io.BytesIO() as buffer:
                thumb.save(buffer, 'webp')
                thumb_tag = f'data:image/webp;base64,{base64.b64encode(buffer.getvalue()).decode()}'
            inner_element = f'<img style="object-fit: contain; border: 1px solid black;" src="{thumb_tag}" />'
        return f"""
        <div class="pxt_document" style="width:{max_width}px;">
            <a href="{_media_url(file_path)}">
                {inner_element}
            </a>
        </div>
        """

    @classmethod
    def make_document_thumbnail(
        cls, file_path: str, max_width: int = 320, max_height: int = 320
    ) -> PIL.Image.Image | None:
        """A thumbnail of the first page of a PDF (needs `pypdfium2`), else None."""
        if file_path.lower().endswith('.pdf'):
            try:
                from pypdfium2 import PdfDocument  # type: ignore[import-untyped]

                doc = PdfDocument(file_path)
                if len(doc) == 0:
                    return None
                img = doc[0].render().to_pil()
                img.thumbnail((max_width, max_height), PIL.Image.LANCZOS)
                return img
            except Exception:
                _logger.warning(f'Failed to produce PDF thumbnail {file_path}. Make sure you have pypdfium2 installed.')
        return None

    @classmethod
    def __create_source_tag(cls, file_path: str) -> str:
        src_url = _media_url(file_path)
        mime = mimetypes.guess_type(src_url)[0]
        mime_attr = f'type="{mime}"' if mime is not None else ''
        return f'<source src="{src_url}" {mime_attr} />'


def _media_url(path_or_url: str) -> str:
    """The URL a browser loads a media value from: local files through the local file server."""
    if '://' in path_or_url and not path_or_url.startswith('file://'):
        return path_or_url
    if path_or_url.startswith('file://'):
        import urllib.parse
        import urllib.request

        path_or_url = urllib.request.url2pathname(urllib.parse.urlparse(path_or_url).path)
    return _http_server.file_uri(path_or_url)
