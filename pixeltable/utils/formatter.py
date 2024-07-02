import base64
import html
import json
import logging
import mimetypes
from typing import Any, Callable, Optional

import PIL
import PIL.Image as Image
import cv2
import numpy as np

import io
import pixeltable.type_system as ts
from pixeltable.utils.http_server import get_file_uri

_logger = logging.getLogger('pixeltable')

_FLOAT_PRECISION = 3
_LIST_THRESHOLD = 16
_LIST_EDGEITEMS = 6
_STRING_SEP = ' ...... '
_STRING_MAX_LEN = 1000
_NESTED_STRING_MAX_LEN = 300


class PixeltableFormatter:
    """
    A factory for constructing Pandas HTML formatters for Pixeltable data. The formatters are used to customize
    the rendering of `DataFrameResultSet`s in notebooks.

    Args:
        num_rows: Number of rows in the DataFrame being rendered.
        num_cols: Number of columns in the DataFrame being rendered.
        http_address: Root address of the Pixeltable HTTP server (used to construct URLs for media references).
    """
    def __init__(self, num_rows: int, num_cols: int, http_address: str):
        self.__num_rows = num_rows
        self.__num_cols = num_cols
        self.__http_address = http_address

    def get_pandas_formatter(self, col_type: ts.ColumnType) -> Optional[Callable]:
        if col_type.is_string_type():
            return self.format_string
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
        """
        Escapes special characters in `val`, and abbreviates `val` if its length exceeds `_STRING_MAX_LEN`.
        """
        return cls.__escape(cls.__abbreviate(val, _STRING_MAX_LEN))

    @classmethod
    def __abbreviate(cls, val: str, max_len: int) -> str:
        if len(val) > max_len:
            edgeitems = (max_len - len(_STRING_SEP)) // 2
            return f'{val[:edgeitems]}{_STRING_SEP}{val[-edgeitems:]}'
        return val

    @classmethod
    def __escape(cls, val: str) -> str:
        # HTML-escape the specified string, then escape $ signs to suppress MathJax formatting
        # TODO(aaron-siegel): The '$' escaping isn't perfect; it will fail on '$' that are already escaped
        return html.escape(val).replace('$', r'\$')

    @classmethod
    def format_float(cls, val: float) -> str:
        # stay consistent with numpy formatting (0-D array has no brackets)
        return np.array2string(np.array(val), precision=_FLOAT_PRECISION)

    @classmethod
    def format_array(cls, arr: np.ndarray) -> str:
        return np.array2string(
            arr, precision=_FLOAT_PRECISION, threshold=_LIST_THRESHOLD, edgeitems=_LIST_EDGEITEMS,
            max_line_width=1000000)

    @classmethod
    def format_json(cls, val: Any) -> str:
        if isinstance(val, str):
            # JSON-like formatting will be applied to strings that appear nested within a list or dict
            # (quote the string; escape any quotes inside the string; shorter abbreviations).
            # However, if the string appears in top-level position (i.e., the entire JSON value is a
            # string), then we format it like an ordinary string.
            return cls.format_string(val)
        # In all other cases, dump the JSON struct recursively.
        return cls.__format_json_rec(val)

    @classmethod
    def __format_json_rec(cls, val: Any) -> str:
        if isinstance(val, str):
            return cls.__escape(json.dumps(cls.__abbreviate(val, _NESTED_STRING_MAX_LEN)))
        if isinstance(val, float):
            return cls.format_float(val)
        if isinstance(val, np.ndarray):
            return cls.format_array(val)
        if isinstance(val, list):
            if len(val) < _LIST_THRESHOLD:
                components = [cls.__format_json_rec(x) for x in val]
            else:
                components = [cls.__format_json_rec(x) for x in val[:_LIST_EDGEITEMS]]
                components.append('...')
                components.extend(cls.__format_json_rec(x) for x in val[-_LIST_EDGEITEMS:])
            return '[' + ', '.join(components) + ']'
        if isinstance(val, dict):
            kv_pairs = (f'{cls.__format_json_rec(k)}: {cls.__format_json_rec(v)}' for k, v in val.items())
            return '{' + ', '.join(kv_pairs) + '}'
        return json.dumps(val)

    def format_img(self, img: Image.Image) -> str:
        """
        Create <img> tag for Image object.
        """
        assert isinstance(img, Image.Image), f'Wrong type: {type(img)}'
        # Try to make it look decent in a variety of display scenarios
        if self.__num_rows > 1:
            width = 240  # Multiple rows: display small images
        elif self.__num_cols > 1:
            width = 480  # Multiple columns: display medium images
        else:
            width = 640  # A single image: larger display
        with io.BytesIO() as buffer:
            img.save(buffer, 'jpeg')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            return f"""
            <div class="pxt_image" style="width:{width}px;">
                <img src="data:image/jpeg;base64,{img_base64}" width="{width}" />
            </div>
            """

    def format_video(self, file_path: str) -> str:
        thumb_tag = ''
        # Attempt to extract the first frame of the video to use as a thumbnail,
        # so that the notebook can be exported as HTML and viewed in contexts where
        # the video itself is not accessible.
        # TODO(aaron-siegel): If the video is backed by a concrete external URL,
        # should we link to that instead?
        video_reader = cv2.VideoCapture(str(file_path))
        if video_reader.isOpened():
            status, img_array = video_reader.read()
            if status:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                thumb = PIL.Image.fromarray(img_array)
                with io.BytesIO() as buffer:
                    thumb.save(buffer, 'jpeg')
                    thumb_base64 = base64.b64encode(buffer.getvalue()).decode()
                    thumb_tag = f'poster="data:image/jpeg;base64,{thumb_base64}"'
            video_reader.release()
        if self.__num_rows > 1:
            width = 320
        elif self.__num_cols > 1:
            width = 480
        else:
            width = 800
        return f"""
        <div class="pxt_video" style="width:{width}px;">
            <video controls width="{width}" {thumb_tag}>
                {self.__create_source_tag(self.__http_address, file_path)}
            </video>
        </div>
        """

    def format_audio(self, file_path: str) -> str:
        return f"""
        <div class="pxt_audio">
            <audio controls>
                {self.__create_source_tag(self.__http_address, file_path)}
            </audio>
        </div>
        """

    def format_document(self, file_path: str) -> str:
        max_width = max_height = 320
        # by default, file path will be shown as a link
        inner_element = file_path
        inner_element = html.escape(inner_element)
        # try generating a thumbnail for different types and use that if successful
        if file_path.lower().endswith('.pdf'):
            try:
                import fitz

                doc = fitz.open(file_path)
                p = doc.get_page_pixmap(0)
                while p.width > max_width or p.height > max_height:
                    # shrink(1) will halve each dimension
                    p.shrink(1)
                data = p.tobytes(output='jpeg')
                thumb_base64 = base64.b64encode(data).decode()
                img_src = f'data:image/jpeg;base64,{thumb_base64}'
                inner_element = f"""
                    <img style="object-fit: contain; border: 1px solid black;" src="{img_src}" />
                """
            except:
                logging.warning(f'Failed to produce PDF thumbnail {file_path}. Make sure you have PyMuPDF installed.')

        return f"""
        <div class="pxt_document" style="width:{max_width}px;">
            <a href="{get_file_uri(self.__http_address, file_path)}">
                {inner_element}
            </a>
        </div>
        """

    @classmethod
    def __create_source_tag(cls, http_address: str, file_path: str) -> str:
        src_url = get_file_uri(http_address, file_path)
        mime = mimetypes.guess_type(src_url)[0]
        # if mime is None, the attribute string would not be valid html.
        mime_attr = f'type="{mime}"' if mime is not None else ''
        return f'<source src="{src_url}" {mime_attr} />'
