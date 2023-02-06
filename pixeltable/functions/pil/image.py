from pixeltable.type_system import FloatType, ImageType
from pixeltable.function import Function, FunctionRegistry


alpha_composite = Function.make_library_function(
    ImageType(), [ImageType(), ImageType()], 'PIL.Image', 'alpha_composite')
FunctionRegistry.get().register_function(__name__, 'alpha_composite', alpha_composite)
blend = Function.make_library_function(ImageType(), [ImageType(), ImageType(), FloatType()], 'PIL.Image', 'blend')
FunctionRegistry.get().register_function(__name__, 'blend', blend)
composite = Function.make_library_function(
    ImageType(), [ImageType(), ImageType(), ImageType()], 'PIL.Image', 'composite')
FunctionRegistry.get().register_function(__name__, 'composite', composite)
