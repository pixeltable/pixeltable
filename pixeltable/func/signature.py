from __future__ import annotations

import dataclasses
import enum
import inspect
import logging
from typing import Optional, Callable, Dict, List, Any, Union

import pixeltable.exceptions as excs
import pixeltable.type_system as ts

_logger = logging.getLogger('pixeltable')


@dataclasses.dataclass
class Parameter:
    name: str
    col_type: Optional[ts.ColumnType]  # None for variable parameters
    kind: enum.Enum  # inspect.Parameter.kind; inspect._ParameterKind is private


class Signature:
    """
    Return type:
    - most functions will have a fixed return type, which is specified directly
    - some functions will have a return type that depends on the argument values;
      ex.: PIL.Image.Image.resize() returns an image with dimensions specified as a parameter
    - in the latter case, the 'return_type' field is a function that takes the bound arguments and returns the
      return type; if no bound arguments are specified, a generic return type is returned (eg, ImageType() without a
      size)
    """
    SPECIAL_PARAM_NAMES = ['group_by', 'order_by']

    def __init__(
            self,
            return_type: Union[ts.ColumnType, Callable[[Dict[str, Any]], ts.ColumnType]],
            parameters: List[Parameter]):
        self.return_type = return_type
        # we rely on the ordering guarantee of dicts in Python >=3.7
        self.parameters = {p.name: p for p in parameters}
        self.parameters_by_pos = parameters.copy()

    def get_return_type(self, bound_args: Optional[Dict[str, Any]] = None) -> ts.ColumnType:
        if isinstance(self.return_type, ts.ColumnType):
            return self.return_type
        return self.return_type(bound_args)

    def as_dict(self) -> Dict[str, Any]:
        result = {
            'return_type': self.get_return_type().as_dict(),
            'parameters': [
                [p.name, p.col_type.as_dict() if p.col_type is not None else None, p.kind]
                for p in self.parameters.values()
            ]
        }
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Signature:
        parameters = [Parameter(p[0], ts.ColumnType.from_dict(p[1]), p[2]) for p in d['parameters']]
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
    def create(
            cls, c: Callable, param_types: List[ts.ColumnType],
            return_type: Union[ts.ColumnType, Callable]
    ) -> Signature:
        sig = inspect.signature(c)
        py_parameters = list(sig.parameters.values())
        # check non-var parameters for name collisions and default value compatibility
        num_nonvar_params = 0
        parameters: List[Parameter] = []
        for idx, param in enumerate(py_parameters):
            if param.name in cls.SPECIAL_PARAM_NAMES:
                raise excs.Error(f"'{param.name}' is a reserved parameter name")
            if param.kind == inspect.Parameter.VAR_POSITIONAL or param.kind == inspect.Parameter.VAR_KEYWORD:
                parameters.append(Parameter(param.name, None, param.kind))
                continue
            if idx >= len(param_types):
                raise excs.Error(f'Missing type for parameter {param.name}')

            num_nonvar_params += 1
            default_val = sig.parameters[param.name].default
            if default_val != inspect.Parameter.empty and default_val is not None:
                try:
                    _ = param_types[idx].create_literal(default_val)
                except TypeError as e:
                    raise excs.Error(f'Default value for parameter {param.name}: {str(e)}')
            parameters.append(Parameter(param.name, param_types[idx], param.kind))

        if len(param_types) != num_nonvar_params:
            raise excs.Error(f'Expected {num_nonvar_params} parameter types, got {len(param_types)}')

        return Signature(return_type, parameters)
