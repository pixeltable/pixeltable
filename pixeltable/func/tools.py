import typing
from dataclasses import dataclass
from typing import Any, Optional

import pydantic
from pydantic import json_schema
from pydantic_core import core_schema

from .function import Function
from .signature import Parameter


class _FunctionAdapter:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: pydantic.GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        return core_schema.is_instance_schema(
            Function,
            serialization=core_schema.plain_serializer_function_ser_schema(cls.__serialize_fn),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: pydantic.GetJsonSchemaHandler
    ) -> json_schema.JsonSchemaValue:
        # Use the same schema that would be used for `int`
        return handler(core_schema.dict_schema())


@dataclass(frozen=True)
class Tool:
    fn: typing.Annotated[Function, _FunctionAdapter]
    name: Optional[str] = None
    description: Optional[str] = None

    @property
    def parameters(self) -> dict[str, Parameter]:
        return self.fn.signature.parameters

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: pydantic.GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        return core_schema.is_instance_schema(
            Tool,
            serialization=core_schema.plain_serializer_function_ser_schema(cls._serialize),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: pydantic.GetJsonSchemaHandler
    ) -> pydantic.json_schema.JsonSchemaValue:
        # Use the same schema that would be used for `int`
        return handler(core_schema.dict_schema())

    @classmethod
    def _serialize(cls, instance: 'Tool') -> dict[str, Any]:
        return {
            'type': 'function',
            'function': {
                'name': instance.name or instance.fn.name,
                'description': instance.description or instance.fn._docstring(),
                'parameters': {
                    'type': 'object',
                    'properties': {
                        param.name: param.as_tool_dict()
                        for param in instance.parameters.values()
                    }
                },
                'required': [
                    param.name for param in instance.parameters.values() if not param.col_type.nullable
                ],
                'additionalProperties': False,  # TODO Handle kwargs?
            }
        }


class Tools(pydantic.BaseModel):
    tools: list[Tool]

    @pydantic.model_serializer
    def ser_model(self) -> list[dict[str, Any]]:
        return [Tool._serialize(tool) for tool in self.tools]
