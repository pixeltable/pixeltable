from pixeltable.type_system import FloatType, ImageType
from pixeltable.function import Function


alpha_composite = Function(
    ImageType(), [ImageType(), ImageType()], module_name='PIL.Image', eval_symbol='alpha_composite')
blend = Function(ImageType(), [ImageType(), ImageType(), FloatType()], module_name='PIL.Image', eval_symbol='blend')
composite = Function(
    ImageType(), [ImageType(), ImageType(), ImageType()], module_name='PIL.Image', eval_symbol='composite')
