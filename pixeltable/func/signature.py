from __future__ import annotations
from typing import Optional, Callable, Dict, List, Any, Tuple, Union
import inspect
import logging

import pixeltable.type_system as ts
import pixeltable.exceptions as excs

_logger = logging.getLogger('pixeltable')


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
            parameters: List[Tuple[str, ts.ColumnType]]):
        self.return_type = return_type
        # we rely on the ordering guarantee of dicts in Python >=3.7
        self.parameters = {param_name: param_type for param_name, param_type in parameters}
        self.parameter_types_by_pos = [param_type for _, param_type in parameters]

    def get_return_type(self, bound_args: Optional[Dict[str, Any]] = None) -> ts.ColumnType:
        if isinstance(self.return_type, ts.ColumnType):
            return self.return_type
        return self.return_type(bound_args)

    def as_dict(self) -> Dict[str, Any]:
        result = {
            'return_type': self.get_return_type().as_dict(),
        }
        if self.parameters is not None:
            result['parameters'] = [[name, col_type.as_dict()] for name, col_type in self.parameters.items()]
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Signature:
        parameters = [(p[0], ts.ColumnType.from_dict(p[1])) for p in d['parameters']]
        return cls(ts.ColumnType.from_dict(d['return_type']), parameters)

    def __eq__(self, other: Signature) -> bool:
        if self.get_return_type() != other.get_return_type() or (self.parameters is None) != (other.parameters is None):
            return False
        if self.parameters is None:
            return True
        if len(self.parameters) != len(other.parameters):
            return False
        # ignore the parameter name
        for param_type, other_param_type in zip(self.parameter_types_by_pos, other.parameter_types_by_pos):
            if param_type != other_param_type:
                return False
        return True

    def __str__(self) -> str:
        return (
            f'({", ".join([name + ": " + str(col_type) for name, col_type in self.parameters.items()])})'
            f'-> {str(self.get_return_type())}'
        )

    @classmethod
    def create(
            cls, c: Callable, is_agg: bool, param_types: List[ts.ColumnType],
            return_type: Union[ts.ColumnType, Callable], check_params: bool = True
    ) -> Signature:
        if param_types is None:
            return Signature(return_type, None)
        sig = inspect.signature(c)
        param_names = list(sig.parameters.keys())
        if is_agg:
            param_names = param_names[1:]  # the first parameter is the state returned by init()
        if check_params and len(param_names) != len(param_types):
            raise excs.Error(
                f"The number of parameters of '{getattr(c, '__name__', 'anonymous')}' is not the same as "
                f"the number of provided parameter types: "
                f"{len(param_names)} ({', '.join(param_names)}) vs "
                f"{len(param_types)} ({', '.join([str(t) for t in param_types])})")
        # check parameters for name collisions and default value compatibility
        for idx, param_name in enumerate(param_names):
            if param_name in cls.SPECIAL_PARAM_NAMES:
                raise excs.Error(f"'{param_name}' is a reserved parameter name")
            default_val = sig.parameters[param_name].default
            if default_val == inspect.Parameter.empty or default_val is None:
                continue
            try:
                _ = param_types[idx].create_literal(default_val)
            except TypeError as e:
                raise excs.Error(f'Default value for parameter {param_name}: {str(e)}')

        parameters = [(param_names[i], param_types[i]) for i in range(len(param_names))]
        return Signature(return_type, parameters)

