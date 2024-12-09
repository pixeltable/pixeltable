from __future__ import annotations

import dataclasses
import enum
import inspect
import json
import logging
import typing
from typing import Any, Callable, Optional, Union

import pixeltable.exceptions as excs
import pixeltable.type_system as ts

_logger = logging.getLogger('pixeltable')


@dataclasses.dataclass
class Parameter:
    name: str
    col_type: Optional[ts.ColumnType]  # None for variable parameters
    kind: inspect._ParameterKind
    # for some reason, this needs to precede is_batched in the dataclass definition,
    # otherwise Python complains that an argument with a default is followed by an argument without a default
    default: Any = inspect.Parameter.empty  # default value for the parameter
    is_batched: bool = False  # True if the parameter is a batched parameter (eg, Batch[dict])

    def __post_init__(self) -> None:
        # make sure that default is json-serializable and of the correct type
        if self.default is inspect.Parameter.empty or self.default is None:
            return
        try:
            _ = json.dumps(self.default)
        except TypeError:
            raise excs.Error(f'Default value for parameter {self.name} is not JSON-serializable: {str(self.default)}')
        if self.col_type is not None:
            try:
                self.col_type.validate_literal(self.default)
            except TypeError as e:
                raise excs.Error(f'Default value for parameter {self.name}: {str(e)}')

    def has_default(self) -> bool:
        return self.default is not inspect.Parameter.empty

    def as_dict(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'col_type': self.col_type.as_dict() if self.col_type is not None else None,
            'kind': self.kind.name,
            'is_batched': self.is_batched,
            'has_default': self.has_default(),
            'default': self.default if self.has_default() else None,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Parameter:
        has_default = d['has_default']
        if has_default:
            default = d['default']
        else:
            default = inspect.Parameter.empty
        return cls(
            name=d['name'],
            col_type=ts.ColumnType.from_dict(d['col_type']) if d['col_type'] is not None else None,
            kind=getattr(inspect.Parameter, d['kind']),
            is_batched=d['is_batched'],
            default=default
        )

    def to_py_param(self) -> inspect.Parameter:
        return inspect.Parameter(self.name, self.kind, default=self.default)


T = typing.TypeVar('T')
Batch = typing.Annotated[list[T], 'pxt-batch']


class Signature:
    """
    Represents the signature of a Pixeltable function.

    - self.is_batched: return type is a Batch[...] type
    """
    SPECIAL_PARAM_NAMES = ['group_by', 'order_by']

    def __init__(self, return_type: ts.ColumnType, parameters: list[Parameter], is_batched: bool = False):
        assert isinstance(return_type, ts.ColumnType)
        self.return_type = return_type
        self.is_batched = is_batched
        # we rely on the ordering guarantee of dicts in Python >=3.7
        self.parameters = {p.name: p for p in parameters}
        self.parameters_by_pos = parameters.copy()
        self.constant_parameters = [p for p in parameters if not p.is_batched]
        self.batched_parameters = [p for p in parameters if p.is_batched]
        self.required_parameters = [p for p in parameters if not p.has_default()]
        self.py_signature = inspect.Signature([p.to_py_param() for p in self.parameters_by_pos])

    def get_return_type(self) -> ts.ColumnType:
        assert isinstance(self.return_type, ts.ColumnType)
        return self.return_type

    def as_dict(self) -> dict[str, Any]:
        result = {
            'return_type': self.get_return_type().as_dict(),
            'parameters': [p.as_dict() for p in self.parameters.values()],
            'is_batched': self.is_batched,
        }
        return result

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Signature:
        parameters = [Parameter.from_dict(param_dict) for param_dict in d['parameters']]
        return cls(ts.ColumnType.from_dict(d['return_type']), parameters, d['is_batched'])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Signature):
            return False
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
        param_strs: list[str] = []
        for p in self.parameters.values():
            if p.kind == inspect.Parameter.VAR_POSITIONAL:
                param_strs.append(f'*{p.name}')
            elif p.kind == inspect.Parameter.VAR_KEYWORD:
                param_strs.append(f'**{p.name}')
            else:
                param_strs.append(f'{p.name}: {str(p.col_type)}')
        return f'({", ".join(param_strs)}) -> {str(self.get_return_type())}'

    @classmethod
    def _infer_type(cls, annotation: Optional[type]) -> tuple[Optional[ts.ColumnType], Optional[bool]]:
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
        cls,
        py_fn: Optional[Callable] = None,
        py_params: Optional[list[inspect.Parameter]] = None,
        param_types: Optional[list[ts.ColumnType]] = None,
        type_substitutions: Optional[dict] = None,
        is_cls_method: bool = False
    ) -> list[Parameter]:
        assert (py_fn is None) != (py_params is None)
        if py_fn is not None:
            sig = inspect.signature(py_fn)
            py_params = list(sig.parameters.values())
        parameters: list[Parameter] = []

        if type_substitutions is None:
            type_substitutions = {}

        for idx, param in enumerate(py_params):
            if is_cls_method and idx == 0:
                continue  # skip 'self' or 'cls' parameter
            if param.name in cls.SPECIAL_PARAM_NAMES:
                raise excs.Error(f"'{param.name}' is a reserved parameter name")
            if param.kind == inspect.Parameter.VAR_POSITIONAL or param.kind == inspect.Parameter.VAR_KEYWORD:
                parameters.append(Parameter(param.name, col_type=None, kind=param.kind))
                continue

            # check non-var parameters for name collisions and default value compatibility
            if param_types is not None:
                if idx >= len(param_types):
                    raise excs.Error(f'Missing type for parameter {param.name}')
                param_type = param_types[idx]
                is_batched = False
            else:
                py_type: Optional[type]
                if param.annotation in type_substitutions:
                    py_type = type_substitutions[param.annotation]
                else:
                    py_type = param.annotation
                param_type, is_batched = cls._infer_type(py_type)
                if param_type is None:
                    raise excs.Error(f'Cannot infer pixeltable type for parameter {param.name}')

            parameters.append(Parameter(
                param.name, col_type=param_type, kind=param.kind, is_batched=is_batched, default=param.default))

        return parameters

    @classmethod
    def create(
        cls,
        py_fn: Callable,
        param_types: Optional[list[ts.ColumnType]] = None,
        return_type: Optional[ts.ColumnType] = None,
        type_substitutions: Optional[dict] = None,
        is_cls_method: bool = False
    ) -> Signature:
        """Create a signature for the given Callable.
        Infer the parameter and return types, if none are specified.
        Raises an exception if the types cannot be inferred.
        """
        if type_substitutions is None:
            type_substitutions = {}

        parameters = cls.create_parameters(py_fn=py_fn, param_types=param_types, is_cls_method=is_cls_method, type_substitutions=type_substitutions)
        sig = inspect.signature(py_fn)
        if return_type is None:
            py_type: Optional[type]
            if sig.return_annotation in type_substitutions:
                py_type = type_substitutions[sig.return_annotation]
            else:
                py_type = sig.return_annotation
            return_type, return_is_batched = cls._infer_type(py_type)
            if return_type is None:
                raise excs.Error('Cannot infer pixeltable return type')
        else:
            _, return_is_batched = cls._infer_type(sig.return_annotation)

        return Signature(return_type, parameters, return_is_batched)
