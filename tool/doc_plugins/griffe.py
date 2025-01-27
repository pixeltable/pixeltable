import ast
import inspect
import warnings
from typing import Union

import griffe.expressions
from griffe import Extension, Function, Object, ObjectNode, dynamic_import  # type: ignore[attr-defined]
from mkdocstrings_handlers.python import rendering

import pixeltable as pxt

logger = griffe.get_logger(__name__)  # type: ignore[attr-defined]

class PxtGriffeExtension(Extension):
    """Implementation of a Pixeltable custom griffe extension."""

    def on_instance(self, node: Union[ast.AST, ObjectNode], obj: Object) -> None:
        if obj.docstring is None:
            # Skip over entities without a docstring
            return

        if isinstance(obj, Function):
            # See if the (Python) function has a @pxt.udf decorator
            if any(
                isinstance(dec.value, griffe.expressions.Expr) and dec.value.canonical_path in ['pixeltable.func.udf', 'pixeltable.udf']
                for dec in obj.decorators
            ):
                # Update the template
                self.__modify_pxt_udf(obj)

    def __modify_pxt_udf(self, func: Function) -> None:
        """
        Instructs the doc snippet for `func` to use the custom Pixeltable UDF jinja template, and
        converts all type hints to Pixeltable column type references, in accordance with the @udf
        decorator behavior.
        """
        func.extra['mkdocstrings']['template'] = 'udf.html.jinja'
        # Dynamically load the UDF reference so we can inspect the Pixeltable signature directly
        warnings.simplefilter("ignore")
        udf = dynamic_import(func.path)
        assert isinstance(udf, pxt.Function)
        # Convert the return type to a Pixeltable type reference
        func.returns = str(udf.signatures[0].get_return_type())
        # Convert the parameter types to Pixeltable type references
        for griffe_param in func.parameters:
            assert isinstance(griffe_param.annotation, griffe.expressions.Expr)
            if griffe_param.name not in udf.signatures[0].parameters:
                logger.warning(f'Parameter `{griffe_param.name}` not found in signature for UDF: {udf.display_name}')
                continue
            pxt_param = udf.signatures[0].parameters[griffe_param.name]
            griffe_param.annotation = str(pxt_param.col_type)
        # Document additional signatures for polymorphic functions
        if len(udf.signatures) > 1:
            polymorphic_signatures = [self.__signature_str(udf.name, sig) for sig in udf.signatures[1:]]
            func.docstring.value = '\n'.join(polymorphic_signatures + [func.docstring.value])

    def __signature_str(self, name: str, sig: pxt.func.Signature) -> str:
        """
        Constructs a signature block for a Pixeltable UDF. This is used to document additional signatures
        beyond the first for polymorphic UDFs. (Mkdocstrings will only generate the first.)
        """
        param_strs = []
        printed_varargs = False
        for param in sig.parameters.values():
            if param.kind == inspect._ParameterKind.KEYWORD_ONLY and not printed_varargs:
                param_strs.append('*')
                printed_varargs = True
            param_strs.append(self.__param_str(param))
            if param.kind == inspect._ParameterKind.VAR_POSITIONAL:
                printed_varargs = True
        params_str = f'({", ".join(param_strs)}) -> {sig.get_return_type()}'
        signature_str = rendering._format_signature(name, params_str, line_length=80)  # type: ignore[arg-type]
        return f'```python\n{signature_str}\n```\n'

    def __param_str(self, param: pxt.func.Parameter) -> str:
        prec: str
        default: str
        if param.kind == inspect._ParameterKind.VAR_POSITIONAL:
            prec = '*'
        elif param.kind == inspect._ParameterKind.VAR_KEYWORD:
            prec = '**'
        else:
            prec = ''
        if param.default is inspect.Parameter.empty:
            default = ''
        else:
            default = f' = {param.default}'
        return f'{prec}{param.name}: {param.col_type}{default}'
