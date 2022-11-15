from pixeltable.utils import clip
from pixeltable.functions import Function
from pixeltable.type_system import ColumnType


encode_image = Function(clip.encode_image, ColumnType.VECTOR, [ColumnType.IMAGE])