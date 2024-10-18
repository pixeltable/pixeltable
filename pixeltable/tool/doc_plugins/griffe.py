import ast
import warnings
from typing import Optional, Union

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
        func.returns = str(udf.signature.get_return_type())
        # Convert the parameter types to Pixeltable type references
        for griffe_param in func.parameters:
            assert isinstance(griffe_param.annotation, griffe.expressions.Expr)
            if griffe_param.name not in udf.signature.parameters:
                logger.warning(f'Parameter `{griffe_param.name}` not found in signature for UDF: {udf.display_name}')
                continue
            pxt_param = udf.signature.parameters[griffe_param.name]
            griffe_param.annotation = str(pxt_param.col_type)
