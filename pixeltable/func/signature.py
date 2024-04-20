from __future__ import annotations

import dataclasses
import enum
import inspect
import logging
import typing
from typing import Optional, Callable, Dict, List, Any, Union, Tuple

import pixeltable.exceptions as excs
import pixeltable.type_system as ts

_logger = logging.getLogger('pixeltable')


@dataclasses.dataclass
class Parameter:
    name: str
    col_type: Optional[ts.ColumnType]  # None for variable parameters
    kind: enum.Enum  # inspect.Parameter.kind; inspect._ParameterKind is private
    is_batched: bool = False  # True if the parameter is a batched parameter (eg, Batch[dict])


T = typing.TypeVar('T')
Batch = typing.Annotated[list[T], 'pxt-batch']


class Signature:
    """
    Represents the signature of a Pixeltable function.

    Regarding return type:
    - most functions will have a fixed return type, which is specified directly
    - some functions will have a return type that depends on the argument values;
      ex.: PIL.Image.Image.resize() returns an image with dimensions specified as a parameter
    - in the latter case, the 'return_type' field is a function that takes the bound arguments and returns the
      return type; if no bound arguments are specified, a generic return type is returned (eg, ImageType() without a
      size)
    - self.is_batched: return type is a Batch[...] type
    """
    SPECIAL_PARAM_NAMES = ['group_by', 'order_by']

    def __init__(
            self,
            return_type: Union[ts.ColumnType, Callable[[Dict[str, Any]], ts.ColumnType]],
            parameters: List[Parameter], is_batched: bool = False):
        self.return_type = return_type
        self.is_batched = is_batched
        # we rely on the ordering guarantee of dicts in Python >=3.7
        self.parameters = {p.name: p for p in parameters}
        self.parameters_by_pos = parameters.copy()
        self.constant_parameters = [p for p in parameters if not p.is_batched]
        self.batched_parameters = [p for p in parameters if p.is_batched]

    def get_return_type(self, bound_args: Optional[Dict[str, Any]] = None) -> ts.ColumnType:
        if isinstance(self.return_type, ts.ColumnType):
            return self.return_type
        return self.return_type(bound_args)

    def as_dict(self) -> Dict[str, Any]:
        result = {
            'return_type': self.get_return_type().as_dict(),
            'parameters': [
                [p.name, p.col_type.as_dict() if p.col_type is not None else None, p.kind, p.is_batched]
                for p in self.parameters.values()
            ]
        }
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Signature:
        parameters = [Parameter(p[0], ts.ColumnType.from_dict(p[1]), p[2], p[3]) for p in d['parameters']]
        return cls(ts.ColumnType.from_dict(d['return_type']), parameters)

    def __eq__(self, other: Signature) -> bool:
        if self.get_return_type() != other.get_return_type():
            return False
        if len(self.parameters) != len(other.parameters):
            return False
        # ignore the parameter name
        for param, other_param in zip(self.parameters.values(), other.parameters.values()):
            if param.col_type != other_param.col_type or param.kind != other_param.kind:
                return False
        return True

    def __str__(self) -> str:
        param_strs: List[str] = []
        for p in self.parameters.values():
            if p.kind == inspect.Parameter.VAR_POSITIONAL:
                param_strs.append(f'*{p.name}')
            elif p.kind == inspect.Parameter.VAR_KEYWORD:
                param_strs.append(f'**{p.name}')
            else:
                param_strs.append(f'{p.name}: {str(p.col_type)}')
        return f'({", ".join(param_strs)}) -> {str(self.get_return_type())}'

    @classmethod
    def _infer_type(cls, annotation: Optional[type]) -> Tuple[Optional[ts.ColumnType], Optional[bool]]:
        """Returns: (column type, is_batched) or (None, ...) if the type cannot be inferred"""
        if annotation is None:
            return (None, None)
        py_type: Optional[type] = None
        is_batched = False
        if typing.get_origin(annotation) == typing.Annotated:
            type_args = typing.get_args(annotation)
            if len(type_args) == 2 and type_args[1] == 'pxt-batch':
                # this is our Batch
                assert typing.get_origin(type_args[0]) == list
                is_batched = True
                py_type = typing.get_args(type_args[0])[0]
        if py_type is None:
            py_type = annotation
        col_type = ts.ColumnType.from_python_type(py_type)
        return (col_type, is_batched)

    @classmethod
    def create_parameters(
            cls, c: Callable, param_types: Optional[List[ts.ColumnType]] = None) -> List[Parameter]:
        sig = inspect.signature(c)
        py_parameters = list(sig.parameters.values())
        parameters: List[Parameter] = []

        for idx, param in enumerate(py_parameters):
            if param.name in cls.SPECIAL_PARAM_NAMES:
                raise excs.Error(f"'{param.name}' is a reserved parameter name")
            if param.kind == inspect.Parameter.VAR_POSITIONAL or param.kind == inspect.Parameter.VAR_KEYWORD:
                parameters.append(Parameter(param.name, None, param.kind, False))
                continue

            # check non-var parameters for name collisions and default value compatibility
            if param_types is not None:
                if idx >= len(param_types):
                    raise excs.Error(f'Missing type for parameter {param.name}')
                param_type = param_types[idx]
                is_batched = False
            else:
                param_type, is_batched = cls._infer_type(param.annotation)
                if param_type is None:
                    raise excs.Error(f'Cannot infer pixeltable type for parameter {param.name}')

            # check default value compatibility
            default_val = sig.parameters[param.name].default
            if default_val != inspect.Parameter.empty and default_val is not None:
                try:
                    _ = param_type.create_literal(default_val)
                except TypeError as e:
                    raise excs.Error(f'Default value for parameter {param.name}: {str(e)}')

            parameters.append(Parameter(param.name, param_type, param.kind, is_batched))

        return parameters

    @classmethod
    def create(
            cls, c: Callable,
            param_types: Optional[List[ts.ColumnType]] = None,
            return_type: Optional[Union[ts.ColumnType, Callable]] = None
    ) -> Signature:
        """Create a signature for the given Callable.
        Infer the parameter and return types, if none are specified.
        Raises an exception if the types cannot be inferred.
        """
        parameters = cls.create_parameters(c, param_types)
        sig = inspect.signature(c)
        if return_type is None:
            return_type, return_is_batched = cls._infer_type(sig.return_annotation)
            if return_type is None:
                raise excs.Error('Cannot infer pixeltable return type')
        else:
            _, return_is_batched = cls._infer_type(sig.return_annotation)

        return Signature(return_type, parameters, return_is_batched)
