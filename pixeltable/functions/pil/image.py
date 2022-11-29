import PIL.Image

from pixeltable.functions import Function
from pixeltable.type_system import FloatType, ImageType


alpha_composite = Function(PIL.Image.alpha_composite, ImageType(), [ImageType(), ImageType()])
blend = Function(PIL.Image.blend, ImageType(), [ImageType(), ImageType(), FloatType()])
composite = Function(PIL.Image.composite, ImageType(), [ImageType(), ImageType(), ImageType()])
