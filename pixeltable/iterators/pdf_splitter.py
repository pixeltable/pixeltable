import ctypes
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from PIL import ImageDraw

DEBUG = False


@dataclass
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class PdfChar:
    value: str
    bounding_box: BoundingBox
    center_x: float
    center_y: float


# @dataclass
class PdfSplitter:
    path: str
    num_pages: int
    page_num: int
    pdf: pdfium.PdfDocument
    page: pdfium.PdfPage
    textpage: pdfium.PdfTextPage

    def __init__(self, path: str) -> None:
        self.path = path
        self.pdf = pdfium.PdfDocument(path)
        self.num_pages = len(self.pdf)

    def split_page(self, page_num: int) -> None:
        self.page_num = page_num
        self.page = self.pdf[page_num]
        self.textpage = self.page.get_textpage()

        # Get the whole text. It may or may not be an improvement over GetUnicode
        num_chars = len(self.textpage.get_text_range())
        buffer_size = pdfium_c.FPDFText_CountChars(self.textpage) + 1
        text = (pdfium_c.FPDF_WCHAR * buffer_size)()
        pdfium_c.FPDFText_GetText(self.textpage, 0, num_chars, text)

        chars = []
        x0, y0, x1, y1 = ctypes.c_double(), ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
        for i in range(0, pdfium_c.FPDFText_CountChars(self.textpage)):
            # code = pdfium_c.FPDFText_GetUnicode(self.textpage, i)
            code = text[i]
            assert pdfium_c.FPDFText_GetCharBox(self.textpage, i, x0, x1, y0, y1)
            assert x0.value <= x1.value
            assert y0.value <= y1.value
            chars.append(
                PdfChar(
                    value=chr(code),
                    bounding_box=BoundingBox(x0.value, y0.value, x1.value, y1.value),
                    center_x=(x0.value + x1.value) / 2,
                    center_y=(y0.value + y1.value) / 2,
                )
            )
        assert len(chars) == len(self.textpage.get_text_range())
        if DEBUG:
            print('\n====== entire textpage ======\n')
            print(self.textpage.get_text_range())
            print('\n====== / entire textpage ======\n')
            print('====== char by char comparison ======\n')
            for i, c1, c2 in zip(range(len(chars)), chars, self.textpage.get_text_range()):
                code = pdfium_c.FPDFText_GetUnicode(self.textpage, i)
                match = c1.value == c2
                print(
                    f'index {i:6}/{len(chars)}: code={code}, char: {c1.value} vs textpage char: {c2} {"" if match else "<<< MISMATCH >>>"}'
                )
            print('====== / char by char comparison ======\n')

        non_whitespace_chars = [c for c in chars if not c.value.isspace()]
        min_x = 0
        min_y = 0
        max_x = self.page.get_width()
        max_y = self.page.get_height()

        bounds = self._split_page(self.textpage, non_whitespace_chars, BoundingBox(min_x, min_y, max_x, max_y))
        assert bounds is not None
        if DEBUG:
            print('\n====== split bounds ======\n')
            for b in bounds:
                print(f'Box: ({b.x0}, {b.y0}, {b.x1}, {b.y1})')
                chars_in_box = self._chars_in_bound(chars, b)
                box_text = ''.join(c.value for c in chars_in_box)
                print(f'Text: {box_text}')
                print('-----')
            print('\n====== / split bounds ======\n')

        scores = np.zeros((len(self.textpage.get_text_range()),), dtype=np.uint8)
        current_box: BoundingBox | None = None
        # give a higher score to the first character in each box
        for i in range(len(self.textpage.get_text_range())):
            c = chars[i]
            box = self._find_bounding_box(c, bounds)
            if box is None:
                # possibly whitespace character that we ignored when finding bounding boxes
                continue
            if box != current_box:
                scores[i] = 100
                current_box = box

        # detect paragraphs
        import unicodedata

        previous_line_x: float | None = None
        previous_line_y: float | None = None
        is_new_line = True
        for i in range(len(self.textpage.get_text_range())):
            c = chars[i]
            if c.value == '\n':
                is_new_line = True
                continue
            is_printable = not unicodedata.category(c.value).startswith('C') and not c.value.isspace()
            if not is_printable:
                continue
            if previous_line_x is None:
                # first line
                previous_line_x = c.center_x
                previous_line_y = c.center_y
                continue
            if is_new_line:
                # encountered a printable character on a new line
                is_new_line = False
                x = c.center_x
                y = c.center_y
                if y > previous_line_y:
                    # text jumped up, possibly to a new column
                    previous_line_x = x
                    previous_line_y = y
                    continue
                gap_vs_last_line = x - previous_line_x
                char_width = self._char_width(c)
                if gap_vs_last_line > char_width and gap_vs_last_line < 5 * char_width:
                    # a possible paragraph indent
                    scores[i] += 50
                previous_line_x = x
                previous_line_y = y

        if DEBUG:
            self._visualize_results(bounds, chars, scores)

    def _visualize_results(self, bounds: list[BoundingBox], chars: list[PdfChar], scores: np.ndarray) -> None:
        # visualize results
        page_img = self.page.render()
        pil_img = page_img.to_pil()
        # draw bounding boxes
        draw = ImageDraw.Draw(pil_img)
        for b in bounds:
            x0, y0 = self._page_coords_to_image_coords(self.page, pil_img, b.x0, b.y0)
            x1, y1 = self._page_coords_to_image_coords(self.page, pil_img, b.x1, b.y1)
            box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            draw.rectangle(box, outline='red', width=1)

        # draw segmentation scores
        for i in range(len(scores)):
            if scores[i] > 0:
                c = chars[i]
                c_x, c_y = self._page_coords_to_image_coords(self.page, pil_img, c.center_x, c.center_y)
                radius = 5
                color = 'red' if scores[i] >= 100 else 'blue'
                box = (c_x - radius, c_y - radius, c_x + radius, c_y + radius)
                draw.ellipse(box, outline=color, fill=color, width=1)

        pil_img.save(f'./output_{self.page_num}.png')

    def _page_coords_to_image_coords(self, page: pdfium.PdfPage, image, x: float, y: float) -> tuple[float, float]:
        img_x = x * (image.width / page.get_width())
        img_y = y * (image.height / page.get_height())
        img_y = image.height - img_y  # PDF coords start at bottom-left
        return (img_x, img_y)

    def _chars_in_bound(self, chars, bound: BoundingBox) -> list[PdfChar]:
        return [c for c in chars if self._char_is_in_bound(c, bound)]

    def _char_is_in_bound(self, c: PdfChar, bound: BoundingBox) -> bool:
        return bound.x0 <= c.center_x <= bound.x1 and bound.y0 <= c.center_y <= bound.y1

    def _find_bounding_box(self, c: PdfChar, boxes: list[BoundingBox]) -> BoundingBox | None:
        for b in boxes:
            if self._char_is_in_bound(c, b):
                return b
        return None

    def _split_page(self, textpage: pdfium.PdfTextPage, chars, bound: BoundingBox) -> list[BoundingBox] | None:
        if bound.x0 >= bound.x1 or bound.y0 >= bound.y1:
            return None

        chars_in_bound = self._chars_in_bound(chars, bound)
        if len(chars_in_bound) < 200:
            # too few chars to split further
            return [bound]

        # find biggest vertical gap
        chars_in_bound.sort(key=lambda c: c.center_y)
        vert_gap = self._biggest_gap(chars_in_bound, lambda c: c.center_y)
        avg_char_height = self._avg([self._char_height(c) for c in chars_in_bound])
        vert_gap_significance = None
        if vert_gap is not None:
            vert_gap_significance = vert_gap[0] / avg_char_height
        if vert_gap_significance is not None and vert_gap_significance < 2.0:
            vert_gap_significance = None

        # find biggest horizontal gap
        chars_in_bound.sort(key=lambda c: c.center_x)
        horiz_gap = self._biggest_gap(chars_in_bound, lambda c: c.center_x)
        avg_char_width = self._avg([self._char_width(c) for c in chars_in_bound])
        horiz_gap_significance = None
        if horiz_gap is not None:
            horiz_gap_significance = horiz_gap[0] / avg_char_width
        if horiz_gap_significance is not None and horiz_gap_significance < 4.0:
            horiz_gap_significance = None

        if vert_gap_significance is None and horiz_gap_significance is None:
            # No significant gaps, can't split further
            return [bound]
        do_vert_split: bool
        if vert_gap_significance is not None and horiz_gap_significance is not None:
            if vert_gap_significance >= horiz_gap_significance:
                do_vert_split = True
            else:
                do_vert_split = False
        else:
            do_vert_split = vert_gap_significance is not None

        if do_vert_split:
            box1 = BoundingBox(bound.x0, bound.y0, bound.x1, vert_gap[1])
            box2 = BoundingBox(bound.x0, vert_gap[1], bound.x1, bound.y1)
            return self._split_page(textpage, chars, box1) + self._split_page(textpage, chars, box2)
        else:
            box1 = BoundingBox(bound.x0, bound.y0, horiz_gap[1], bound.y1)
            box2 = BoundingBox(horiz_gap[1], bound.y0, bound.x1, bound.y1)
            return self._split_page(textpage, chars, box1) + self._split_page(textpage, chars, box2)

    def _biggest_gap(self, chars, coord_func: Callable[[PdfChar], float]) -> tuple[float, float] | None:
        # Finds the biggest gap between consecutive chars based on the provided coordinate function.
        # Returns a tuple of (gap_size, gap_center)
        assert len(chars) > 1
        biggest_gap = -1
        biggest_gap_center = -1
        for i in range(len(chars) - 1):
            coord_i = coord_func(chars[i])
            coord_iplus1 = coord_func(chars[i + 1])
            gap = coord_iplus1 - coord_i
            assert gap >= 0
            if gap > biggest_gap:
                biggest_gap = gap
                biggest_gap_center = (coord_i + coord_iplus1) / 2
        if biggest_gap >= 0:
            return (biggest_gap, biggest_gap_center)
        return None

    def _char_width(self, char: PdfChar) -> float:
        return char.bounding_box.x1 - char.bounding_box.x0

    def _char_height(self, char: PdfChar) -> float:
        return char.bounding_box.y1 - char.bounding_box.y0

    def _avg(self, vals: list[float]) -> float:
        assert len(vals) > 0
        return sum(vals) / len(vals)
