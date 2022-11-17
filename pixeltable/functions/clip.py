from pixeltable.utils import clip
from pixeltable.functions import Function
from pixeltable.type_system import ColumnType


encode_image = Function(clip.encode_image, ColumnType.VECTOR, [ColumnType.IMAGE])
encode_text = Function(clip.encode_text, ColumnType.VECTOR, [ColumnType.STRING])
