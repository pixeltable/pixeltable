import base64
import numpy as np
import PIL
import PIL.Image as Image
import io
import cv2
import mimetypes
from typing import Any, Optional
from pixeltable.utils.http_server import get_file_uri
import logging

_logger = logging.getLogger('pixeltable')

FLOAT_PRECISION = 3
NP_THRESHOLD = 16
NP_EDGEITEMS = 6
STRING_THRESHOLD = 250
STRING_EDGEITEMS = 120
INDENT = 1


def _create_source_tag(http_address: str, file_path: str) -> str:
    src_url = get_file_uri(http_address, file_path)
    mime = mimetypes.guess_type(src_url)[0]
    # if mime is None, the attribute string would not be valid html.
    mime_attr = f'type="{mime}"' if mime is not None else ''
    return f'<source src="{src_url}" {mime_attr} />'


def escape_string_for_json(input_string: str) -> str:
    """ the output here can be printed to screen, and copy-pasted, to get the same string value """
    # Use encode to escape special characters
    escaped_string = input_string.encode('unicode_escape').decode('utf-8')
    # Escape double quotes
    escaped_string = escaped_string.replace('"', '\\"')
    return escaped_string


def as_simple_ndarray(val: list[Any]) -> Optional[np.ndarray]:
    """attempts to parse list as a numerical np.ndarray, if not possible returns None"""
    arr = None
    try:
        arr = np.array(val)
    except TypeError:
        return None
    if not np.issubdtype(arr.dtype, np.number):
        arr = None
    return arr


def is_simple_array(val: list[Any]) -> bool:
    for v in val:
        if isinstance(v,list) or isinstance(v,dict):
            return False
    return True


def escape_string_for_json(input_string):
    """ the output here can be printed to screen, and copy-pasted, to get the same string value """
    # Use encode to escape special characters
    escaped_string = input_string.encode('unicode_escape').decode('utf-8')
    # Escape double quotes
    escaped_string = escaped_string.replace('"', '\\"')
    return escaped_string


class PixeltableFormatter:
    def __init__(self, num_rows: int, num_cols: int, http_address: str):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.http_address = http_address

    def _format_float(self, val: float) -> str:
        # stay consistent with numpy formatting (0-D array has no brackets)
        return np.array2string(np.array(val), precision=FLOAT_PRECISION)

    def _format_array(self, arr: list[Any]) -> str:
        return np.array2string(arr, precision=FLOAT_PRECISION, threshold=NP_THRESHOLD, separator=',', edgeitems=NP_EDGEITEMS)

    def _format_string(self, val: str) -> str:
        if len(val) > STRING_THRESHOLD:
            return f'"{escape_string_for_json(val[:STRING_EDGEITEMS])} ...... {escape_string_for_json(val[-STRING_EDGEITEMS:])}"'
        return f'"{escape_string_for_json(val)}"'

    def _format_json_helper(self, obj: Any, level: int) -> str:
        def get_prefix(level):
            return ' ' * (level * INDENT)

        if isinstance(obj, list):
            # we will distinguish 3 separate cases:
            # 1. numerical arrays (potentially multiple levels of nesting, often seen as model outputs)
            #   => treat just like array columns for consistency.
            # 2. list of elementary types (anything elementary not covered by the above, like strings)
            #   => (eg category names) print in one line (ie, avoid line per element)
            # 3. list with some complex types, eg list of dicts.
            #   => insert newlines between elements to help read them.
            arr = as_simple_ndarray(obj)
            if arr is not None:
                arrstr = self._format_array(arr)
                return '\n'.join([f'{get_prefix(level)}{line}' for line in arrstr.splitlines()])
            out_pieces = []
            if is_simple_array(obj):
                for elt in obj:
                    fmt_elt = self._format_json_helper(elt, level=0)
                    out_pieces.append(fmt_elt)
                contents = ','.join(out_pieces)
                return f'{get_prefix(level)}[{contents}]'
            else:
                for elt in obj:
                    fmt_elt = self._format_json_helper(elt, level=level+1)
                    out_pieces.append(fmt_elt)
                joiner = ',\n'
                contents = joiner.join(out_pieces)
                return f'{get_prefix(level)}[\n{contents}\n{get_prefix(level)}]'
        elif isinstance(obj, dict):
            out_pieces = []
            for key, value in obj.items():
                fmt_value = self._format_json_helper(value, level=level+1)
                fmt_key = self._format_string(key)
                out_pieces.append(f'{get_prefix(level+1)}{fmt_key}: {fmt_value.lstrip()}')
            contents = ',\n'.join(out_pieces)
            return f'{get_prefix(level)}{{\n{contents}\n{get_prefix(level)}}}'
        elif isinstance(obj, float):
            return self._format_float(obj)
        elif isinstance(obj, str):
            return self._format_string(obj)
        elif isinstance(obj, int):
            return str(int)
        elif isinstance(obj, None):
            return 'null'
        else:
            assert False, f'Unexpected type: {type(obj)}'

    def _format_json(self, obj: Any) -> str:
        return self._format_json_helper(obj, level=0)

    def _format_img(self, img: Image.Image) -> str:
        """
        Create <img> tag for Image object.
        """
        assert isinstance(img, Image.Image), f'Wrong type: {type(img)}'
        # Try to make it look decent in a variety of display scenarios
        if self.num_rows > 1:
            width = 240  # Multiple rows: display small images
        elif self.num_cols > 1:
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

    def _format_video(self, file_path: str) -> str:
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
        if self.num_rows > 1:
            width = 320
        elif self.num_cols > 1:
            width = 480
        else:
            width = 800
        return f"""
        <div class="pxt_video" style="width:{width}px;">
            <video controls width="{width}" {thumb_tag}>
                {_create_source_tag(self.http_address, file_path)}
            </video>
        </div>
        """

    def _format_document(self, file_path: str) -> str:
        max_width = max_height = 320
        # by default, file path will be shown as a link
        inner_element = file_path
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
            <a href="{get_file_uri(self.http_address, file_path)}">
                {inner_element}
            </a>
        </div>
        """

    def _format_audio(self, file_path: str) -> str:
        return f"""
        <div class="pxt_audio">
            <audio controls>
                {_create_source_tag(self.http_address, file_path)}
            </audio>
        </div>
        """