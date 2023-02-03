from pixeltable.type_system import FloatType, ImageType
from pixeltable.function import Function


alpha_composite = Function.make_library_function(
    ImageType(), [ImageType(), ImageType()], 'PIL.Image', 'alpha_composite')
blend = Function.make_library_function(ImageType(), [ImageType(), ImageType(), FloatType()], 'PIL.Image', 'blend')
composite = Function.make_library_function(
    ImageType(), [ImageType(), ImageType(), ImageType()], 'PIL.Image', 'composite')
