import base64
import html
import io
import logging
import mimetypes
from typing import Any, Optional

import cv2
import numpy as np
import PIL
import PIL.Image as Image

from pixeltable.utils.http_server import get_file_uri

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


def as_numeric_ndarray(val: list[Any]) -> Optional[np.ndarray]:
    """
    Attempts to parse list as a numerical np.ndarray, if not,
    either because the shape is not consistent or because the elements are not numerical,
    returns None
    """
    arr = None
    try:
        arr = np.array(val)
    except TypeError:
        return None
    if not np.issubdtype(arr.dtype, np.number):
        arr = None
    return arr


from typing import Union


def contains_only_simple_elements(values: Union[list[Any], dict[Any]]) -> bool:
    """whether all values of this array or dict are themselves elementary (no nested lists or dicts)"""
    if isinstance(values, dict):
        values = values.values()

    return not any(isinstance(v, (list, dict)) for v in values)


class PixeltableFormatter:
    def __init__(self, num_rows: int, num_cols: int, http_address: str):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.http_address = http_address

    def format_float(self, val: float) -> str:
        # stay consistent with numpy formatting (0-D array has no brackets)
        return np.array2string(np.array(val), precision=FLOAT_PRECISION)

    def _format_array(self, arr: list[Any]) -> str:
        """for numerical arrays only. to format an array type, use format_json, which also includes styling"""
        return np.array2string(
            arr, precision=FLOAT_PRECISION, threshold=NP_THRESHOLD, separator=',', edgeitems=NP_EDGEITEMS
        )

    def _format_string(self, val: str) -> str:
        """
            Main goal:
            escape special characters like <, >, &, etc. using html.escape()
            this avoids rendering artifacts, or worse, corrupting the html, or worse,
            some kind of injection attack.
            (NB escape=False in our call DataFrame.to_html() so that we can insert our own HTML for other types,
            hence we must escape the strings ourselves)

            Secondary goal:
            If the string is too long, show only the first and last STRING_EDGEITEMS characters.

            TODO: render enclosing quotes also? but need to escape inner quotes
            TODO: enable user to somehow see the full string if they want to and eg copy it
        """
        if len(val) > STRING_THRESHOLD:
            return f'{html.escape(val[:STRING_EDGEITEMS])} ...... {html.escape(val[-STRING_EDGEITEMS:])}'
        return f'{html.escape(val)}'

    def _format_json_helper(self, obj: Any, level: int) -> str:
        def get_prefix(level):
            return ' ' * (level * INDENT)

        if obj is None:
            return f'{get_prefix(level)}None'

        if isinstance(obj, np.ndarray):  # sometimes we get a numpy array, which is technically not a list
            return self._format_json_helper(obj.tolist(), level=level)
        elif isinstance(obj, list):
            # we will distinguish 3 separate cases:
            # 1. numerical arrays (potentially multiple levels of nesting, often seen as model outputs)
            #   => treat just like array columns for consistency.
            # 2. list of elementary types (anything elementary not covered by the above, like strings)
            #   => (eg category names) print in one line (ie, avoid line per element)
            # 3. list with at least one nested list or dict
            #   => insert newlines between elements to help read them.
            arr = as_numeric_ndarray(obj)
            if arr is not None:
                arrstr = self._format_array(arr)
                return '\n'.join([f'{get_prefix(level)}{line}' for line in arrstr.splitlines()])
            out_pieces = []
            if contains_only_simple_elements(obj):
                for elt in obj:
                    fmt_elt = self._format_json_helper(elt, level=0)
                    out_pieces.append(fmt_elt)
                contents = ', '.join(out_pieces)
                return f'{get_prefix(level)}[{contents}]'
            else:
                for elt in obj:
                    fmt_elt = self._format_json_helper(elt, level=level + 1)
                    out_pieces.append(fmt_elt)
                contents = ',\n'.join(out_pieces)
                return f'{get_prefix(level)}[\n{contents}\n{get_prefix(level)}]'
        elif isinstance(obj, dict):
            # similar consideration to lists, but there is no case 1 (numerical arrays)
            # we will distinguish 2 separate cases, based on the complexity of the values
            out_pieces = []
            if contains_only_simple_elements(obj):
                for key, value in obj.items():
                    fmt_value = self._format_json_helper(value, level=0)
                    # NB. json keys are always strings
                    # but we will accept any type for the value, as this does seem to happen
                    # (eg int keys in some demos), though it is not clear if this can be stored
                    # in the backend.
                    fmt_key = self._format_string(str(key))
                    out_pieces.append(f'{fmt_key}: {fmt_value}')
                contents = ', '.join(out_pieces)
                return f'{get_prefix(level)}{{{contents}}}'
            else:
                for key, value in obj.items():
                    fmt_value = self._format_json_helper(value, level=level + 1)
                    fmt_key = self._format_string(key)
                    out_pieces.append(f'{get_prefix(level + 1)}{fmt_key}: {fmt_value.lstrip()}')
                contents = ',\n'.join(out_pieces)
                return f'{get_prefix(level)}{{\n{contents}\n{get_prefix(level)}}}'
        elif isinstance(obj, bool):
            return f'{get_prefix(level)}{str(obj)}'
        elif isinstance(obj, float):
            return f'{get_prefix(level)}{self.format_float(obj)}'
        elif isinstance(obj, str):
            return f'{get_prefix(level)}{self._format_string(obj)}'
        elif isinstance(obj, int):
            return f'{get_prefix(level)}{obj}'
        else:
            raise AssertionError(f'Unexpected type within json: {type(obj)}')

    def format_json(self, obj: Any) -> str:
        """ Formats a json object for display in a notebook cell so that it is easier to read, and
        so that sub-objects are consistent with the top-level elementary types like int, float and arrays.

        Implementation notes:
            Uses <pre> tag so that :
                1) json is displayed in a monospace font
                2) whitespace is respected (newlines and spaces)
            Set style so that:
                1) json is left-aligned  (overrides jupyter default to right-aligned)
                2) background is transparent (so that jupyter alternate row coloring remains the same)
            Unlike json.dumps, this function will treat numerical arrays like numpy arrays, and render them using
            the same formatting as an array column.
            Also unlike json.dumps, we will not print one item per line for lists of elementary types (eg strings),
            to avoid excessive vertical space.
        """
        return f"""<pre style="text-align:left; background-color:transparent; margin:0;">{self._format_json_helper(obj, level=0)}</pre>"""

    def format_img(self, img: Image.Image) -> str:
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
            <a href="{get_file_uri(self.http_address, file_path)}">
                {inner_element}
            </a>
        </div>
        """

    def format_audio(self, file_path: str) -> str:
        return f"""
        <div class="pxt_audio">
            <audio controls>
                {_create_source_tag(self.http_address, file_path)}
            </audio>
        </div>
        """