import numpy as np
import PIL

from pixeltable.type_system import ImageType, ArrayType, ColumnType
from pixeltable.function import Function

def _draw_boxes(img: PIL.Image.Image, boxes: np.ndarray) -> PIL.Image.Image:
    result = img.copy()
    d = PIL.ImageDraw.Draw(result)
    for i in range(boxes.shape[0]):
        d.rectangle(list(boxes[i, :4]), width=3)
    return result

draw_boxes = Function(
    ImageType(), [ImageType(), ArrayType((None, 6), dtype=ColumnType.Type.FLOAT)],
    module_name=__name__, symbol='_draw_boxes')

__all__ = [
    draw_boxes,
]