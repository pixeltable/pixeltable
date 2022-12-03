from pixeltable.utils import clip
from pixeltable.type_system import StringType, ImageType, ArrayType, Function


encode_image = Function(clip.encode_image, ArrayType(), [ImageType()])
encode_text = Function(clip.encode_text, ArrayType(), [StringType()])
