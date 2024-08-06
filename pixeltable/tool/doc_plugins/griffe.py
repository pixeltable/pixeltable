import ast
from typing import Optional, Union
import warnings

import griffe
import griffe.expressions
from griffe import Extension, Object, ObjectNode

import pixeltable as pxt

logger = griffe.get_logger(__name__)

class PxtGriffeExtension(Extension):
    """Implementation of a Pixeltable custom griffe extension."""

    def on_instance(self, node: Union[ast.AST, ObjectNode], obj: Object) -> None:
        if obj.docstring is None:
            # Skip over entities without a docstring
            return

        if isinstance(obj, griffe.Function):
            # See if the (Python) function has a @pxt.udf decorator
            if any(
                isinstance(dec.value, griffe.expressions.Expr) and dec.value.canonical_path in ['pixeltable.func.udf', 'pixeltable.udf']
                for dec in obj.decorators
            ):
                # Update the template
                self.__modify_pxt_udf(obj)

    def __modify_pxt_udf(self, func: griffe.Function) -> None:
        """
        Instructs the doc snippet for `func` to use the custom Pixeltable UDF jinja template, and
        converts all type hints to Pixeltable column type references, in accordance with the @udf
        decorator behavior.
        """
        func.extra['mkdocstrings']['template'] = 'udf.html.jinja'
        # Dynamically load the UDF reference so we can inspect the Pixeltable signature directly
        warnings.simplefilter("ignore")
        udf = griffe.dynamic_import(func.path)
        assert isinstance(udf, pxt.Function)
        # Convert the return type to a Pixeltable type reference
        func.returns = self.__column_type_to_display_str(udf.signature.get_return_type())
        # Convert the parameter types to Pixeltable type references
        for griffe_param in func.parameters:
            assert isinstance(griffe_param.annotation, griffe.expressions.Expr)
            if griffe_param.name not in udf.signature.parameters:
                logger.warning(f'Parameter `{griffe_param.name}` not found in signature for UDF: {udf.display_name}')
                continue
            pxt_param = udf.signature.parameters[griffe_param.name]
            griffe_param.annotation = self.__column_type_to_display_str(pxt_param.col_type)

    def __column_type_to_display_str(self, column_type: Optional[pxt.ColumnType]) -> str:
        # TODO: When we enhance the Pixeltable type system, we may want to refactor some of this logic out.
        #   I'm putting it here for now though.
        if column_type is None:
            return 'None'
        if column_type.is_string_type():
            base = 'str'
        elif column_type.is_int_type():
            base = 'int'
        elif column_type.is_float_type():
            base = 'float'
        elif column_type.is_bool_type():
            base = 'bool'
        elif column_type.is_timestamp_type():
            base = 'datetime'
        elif column_type.is_array_type():
            base = 'ArrayT'
        elif column_type.is_json_type():
            base = 'JsonT'
        elif column_type.is_image_type():
            base = 'ImageT'
        elif column_type.is_video_type():
            base = 'VideoT'
        elif column_type.is_audio_type():
            base = 'AudioT'
        elif column_type.is_document_type():
            base = 'DocumentT'
        else:
            assert False
        return f'Optional[{base}]' if column_type.nullable else base
