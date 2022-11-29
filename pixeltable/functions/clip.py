from pixeltable.utils import clip
from pixeltable.functions import Function
from pixeltable.type_system import StringType, ImageType, ArrayType


encode_image = Function(clip.encode_image, ArrayType(), [ImageType()])
encode_text = Function(clip.encode_text, ArrayType(), [StringType()])
