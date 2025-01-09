from dataclasses import dataclass
import dataclasses
import json
from typing import TYPE_CHECKING, Any, Optional

import pydantic

from .function import Function
from .signature import Parameter
from .udf import udf

if TYPE_CHECKING:
    from pixeltable import exprs


# The Tool and Tools classes are containers that hold Pixeltable UDFs and related metadata, so that they can be
# realized as LLM tools. They are implemented as Pydantic models in order to provide a canonical way of converting
# to JSON, via the Pydantic `model_serializer` interface. In this way, they can be passed directly as UDF
# parameters as described in the `pixeltable.tools` and `pixeltable.tool` docstrings.
#
# (The dataclass dict serializer is insufficiently flexible for this purpose: `Tool` contains a member of type
# `Function`, which is not natively JSON-serializable; Pydantic provides a way of customizing its default
# serialization behavior, whereas dataclasses do not.)

class Tool(pydantic.BaseModel):
    # Allow arbitrary types so that we can include a Pixeltable function in the schema.
    # We will implement a model_serializer to ensure the Tool model can be serialized.
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    fn: Function
    name: Optional[str] = None
    description: Optional[str] = None

    @property
    def parameters(self) -> dict[str, Parameter]:
        return self.fn.signature.parameters

    @pydantic.model_serializer
    def ser_model(self) -> dict[str, Any]:
        return {
            'name': self.name or self.fn.name,
            'description': self.description or self.fn._docstring(),
            'parameters': {
                'type': 'object',
                'properties': {
                    param.name: param.col_type._to_json_schema()
                    for param in self.parameters.values()
                }
            },
            'required': [
                param.name for param in self.parameters.values() if not param.col_type.nullable
            ],
            'additionalProperties': False,  # TODO Handle kwargs?
        }

    # `tool_calls` must be in standardized tool invocation format:
    # {tool_name: {'args': {name1: value1, name2: value2, ...}}, ...}
    def invoke(self, tool_calls: 'exprs.Expr') -> 'exprs.FunctionCall':
        kwargs = {
            param.name: self.__extract_tool_arg(param, tool_calls)
            for param in self.parameters.values()
        }
        return self.fn(**kwargs)

    def __extract_tool_arg(self, param: Parameter, tool_calls: 'exprs.Expr') -> 'exprs.Expr':
        func_name = self.name or self.fn.name
        if param.col_type.is_string_type():
            return _extract_str_tool_arg(tool_calls, func_name=func_name, param_name=param.name)
        if param.col_type.is_int_type():
            return _extract_int_tool_arg(tool_calls, func_name=func_name, param_name=param.name)
        if param.col_type.is_float_type():
            return _extract_float_tool_arg(tool_calls, func_name=func_name, param_name=param.name)
        if param.col_type.is_bool_type():
            return _extract_bool_tool_arg(tool_calls, func_name=func_name, param_name=param.name)
        assert False


class Tools(pydantic.BaseModel):
    tools: list[Tool]

    @pydantic.model_serializer
    def ser_model(self) -> list[dict[str, Any]]:
        return [tool.ser_model() for tool in self.tools]

    # `tool_calls` must be in standardized tool invocation format:
    # {tool_name: {'args': {name1: value1, name2: value2, ...}}, ...}
    def _invoke(self, tool_calls: 'exprs.Expr') -> 'exprs.InlineDict':
        from pixeltable import exprs

        return exprs.InlineDict({
            tool.name or tool.fn.name: tool.invoke(tool_calls)
            for tool in self.tools
        })


@udf
def _extract_str_tool_arg(tool_calls: dict, func_name: str, param_name: str) -> Optional[str]:
    return str(_extract_arg(tool_calls, func_name, param_name))

@udf
def _extract_int_tool_arg(tool_calls: dict, func_name: str, param_name: str) -> Optional[int]:
    return int(_extract_arg(tool_calls, func_name, param_name))

@udf
def _extract_float_tool_arg(tool_calls: dict, func_name: str, param_name: str) -> Optional[float]:
    return float(_extract_arg(tool_calls, func_name, param_name))

@udf
def _extract_bool_tool_arg(tool_calls: dict, func_name: str, param_name: str) -> Optional[bool]:
    return bool(_extract_arg(tool_calls, func_name, param_name))

def _extract_arg(tool_calls: dict, func_name: str, param_name: str) -> Any:
    if func_name in tool_calls:
        arguments = tool_calls[func_name]['args']
        return arguments.get(param_name)
    return None
