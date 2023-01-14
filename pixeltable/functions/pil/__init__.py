import numpy as np
import PIL

from pixeltable.type_system import ImageType, ArrayType, ColumnType
from pixeltable.function import Function
from pixeltable.exceptions import Error

def _draw_boxes(img: PIL.Image.Image, boxes: np.ndarray) -> PIL.Image.Image:
    if len(boxes.shape) != 2 or boxes.shape[1] != 4:
        raise Error(f'draw(): boxes needs to have shape (None, 4) but instead has shape {boxes.shape}')
    result = img.copy()
    d = PIL.ImageDraw.Draw(result)
    for i in range(boxes.shape[0]):
        d.rectangle(list(boxes[i]), width=3)
    return result

draw_boxes = Function(
    ImageType(), [ImageType(), ArrayType((None, 6), dtype=ColumnType.Type.FLOAT)],
    module_name=__name__, eval_symbol='_draw_boxes')

__all__ = [
    draw_boxes,
]