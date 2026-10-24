"""Signatures of Python functions, built by introspecting their type hints (upstream `pixeltable.func.signature`).

Binding calls to signatures, type checking and result types are computed in the backend; `Signature._to_core()`
is the form it takes.
"""

from __future__ import annotations

import dataclasses
import inspect
import typing
from typing import Any, Callable, ClassVar

import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable import _core

T = typing.TypeVar('T')
Batch = typing.Annotated[list[T], 'pxt-batch']


def _literal(val: Any) -> _core.Expr | None:
    """`exprs.Expr.from_object(val)` if it is a constant, else None."""
    e = _core.expr_from_object(val)
    if e is None or not e._literal_value()[0]:
        return None
    return e


@dataclasses.dataclass
class Parameter:
    name: str
    col_type: ts.ColumnType | None  # None for variable parameters
    kind: inspect._ParameterKind
    default: _core.Expr | None = None  # a constant expression
    is_batched: bool = False

    def __post_init__(self) -> None:
        if self.default is not None:
            if self.col_type is None:
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'Cannot have a default value for variable parameter {self.name!r}',
                )
            if not self.default._literal_value()[0]:
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION, f'Default value for parameter {self.name!r} is not a constant'
                )
            if not self.col_type.is_supertype_of(self.default.col_type):
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'Default value for parameter {self.name!r} has type `{self.default.col_type}`, '
                    f'which is not of type `{self.col_type!r}`: {self.default}',
                )

    def has_default(self) -> bool:
        return self.default is not None

    @property
    def default_value(self) -> Any:
        """Unwrapped Python value of this parameter's default. Caller must check has_default()."""
        assert self.default is not None
        return self.default._literal_value()[1]

    def as_dict(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'col_type': self.col_type.as_dict() if self.col_type is not None else None,
            'kind': self.kind.name,
            'is_batched': self.is_batched,
            'default': None if self.default is None else self.default.as_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Parameter:
        default = None if d['default'] is None else _core.Expr.from_dict(d['default'])
        return cls(
            name=d['name'],
            col_type=ts.ColumnType.from_dict(d['col_type']) if d['col_type'] is not None else None,
            kind=getattr(inspect.Parameter, d['kind']),
            is_batched=d['is_batched'],
            default=default,
        )

    def to_py_param(self) -> inspect.Parameter:
        py_default = self.default_value if self.default is not None else inspect.Parameter.empty
        return inspect.Parameter(self.name, self.kind, default=py_default)

    def _to_core(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'kind': self.kind.name,
            'col_type': self.col_type._core if self.col_type is not None else None,
            'default': self.default,
            'is_batched': self.is_batched,
        }

    def __hash__(self) -> int:
        return hash((self.name, self.col_type, self.kind, self.is_batched))


class Signature:
    """
    The signature of a Pixeltable function.

    - self.is_batched: return type is a Batch[...] type
    """

    SPECIAL_PARAM_NAMES: ClassVar[list[str]] = ['group_by', 'order_by']
    SYSTEM_PARAM_NAMES: ClassVar[list[str]] = ['_runtime_ctx']

    return_type: ts.ColumnType
    is_batched: bool
    parameters: dict[str, Parameter]
    parameters_by_pos: list[Parameter]
    constant_parameters: list[Parameter]
    batched_parameters: list[Parameter]
    required_parameters: list[Parameter]
    system_parameters: list[str]

    def __init__(
        self,
        return_type: ts.ColumnType,
        parameters: list[Parameter],
        is_batched: bool = False,
        system_parameters: list[str] | None = None,
    ):
        assert isinstance(return_type, ts.ColumnType)
        self.return_type = return_type
        self.is_batched = is_batched
        self.parameters = {p.name: p for p in parameters}
        self.parameters_by_pos = parameters.copy()
        self.constant_parameters = [p for p in parameters if not p.is_batched]
        self.batched_parameters = [p for p in parameters if p.is_batched]
        self.required_parameters = [p for p in parameters if not p.has_default()]
        self.system_parameters = system_parameters if system_parameters is not None else []

    @property
    def py_signature(self) -> inspect.Signature:
        return inspect.Signature([p.to_py_param() for p in self.parameters_by_pos])

    def get_return_type(self) -> ts.ColumnType:
        return self.return_type

    def as_dict(self) -> dict[str, Any]:
        return {
            'return_type': self.get_return_type().as_dict(),
            'parameters': [p.as_dict() for p in self.parameters.values()],
            'is_batched': self.is_batched,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Signature:
        parameters = [Parameter.from_dict(param_dict) for param_dict in d['parameters']]
        return cls(ts.ColumnType.from_dict(d['return_type']), parameters, d['is_batched'])

    def _to_core(self) -> dict[str, Any]:
        """The form `pixeltable._core.Udf` takes."""
        return {
            'parameters': [p._to_core() for p in self.parameters_by_pos],
            'return_type': self.return_type._core,
            'is_batched': self.is_batched,
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Signature):
            return False
        if self.get_return_type() != other.get_return_type():
            return False
        if len(self.parameters) != len(other.parameters):
            return False
        # ignore the parameter names
        for param, other_param in zip(self.parameters.values(), other.parameters.values()):
            if param.col_type != other_param.col_type or param.kind != other_param.kind:
                return False
        return True

    def __hash__(self) -> int:
        return hash((self.return_type, tuple(self.parameters)))

    def params_str(self) -> str:
        """A user-friendly rendering of the parameters."""
        param_strs: list[str] = []
        pending_pos_only_marker = False
        needs_kw_only_marker = True
        for p in self.parameters.values():
            if p.kind == inspect.Parameter.POSITIONAL_ONLY:
                pending_pos_only_marker = True
            elif pending_pos_only_marker:
                param_strs.append('/')
                pending_pos_only_marker = False
            if p.kind == inspect.Parameter.VAR_POSITIONAL:
                needs_kw_only_marker = False
            elif p.kind == inspect.Parameter.KEYWORD_ONLY and needs_kw_only_marker:
                param_strs.append('*')
                needs_kw_only_marker = False

            if p.kind == inspect.Parameter.VAR_POSITIONAL:
                param_strs.append(f'*{p.name}')
            elif p.kind == inspect.Parameter.VAR_KEYWORD:
                param_strs.append(f'**{p.name}')
            else:
                param_strs.append(f'{p.name}: pxt.{p.col_type}')
        if pending_pos_only_marker:
            param_strs.append('/')
        return ', '.join(param_strs)

    def return_str(self, pretty_print_json: bool = False) -> str:
        return f'pxt.{self.get_return_type()}'

    def _to_str(self, pretty_print_json: bool = False) -> str:
        return f'({self.params_str()}) -> {self.return_str(pretty_print_json)}'

    def __str__(self) -> str:
        return self._to_str()

    def __repr__(self) -> str:
        return self._to_str()

    @classmethod
    def _infer_type(cls, annotation: Any) -> tuple[ts.ColumnType | None, bool | None]:
        """Returns: (column type, is_batched) or (None, ...) if the type cannot be inferred"""
        if annotation is None or annotation is inspect.Parameter.empty:
            return (None, None)
        py_type: Any = None
        is_batched = False
        if typing.get_origin(annotation) == typing.Annotated:
            type_args = typing.get_args(annotation)
            if len(type_args) == 2 and type_args[1] == 'pxt-batch':
                assert typing.get_origin(type_args[0]) is list
                is_batched = True
                py_type = typing.get_args(type_args[0])[0]
        if py_type is None:
            py_type = annotation
        col_type = ts.ColumnType.from_python_type(py_type)
        return (col_type, is_batched)

    @classmethod
    def create_parameters(
        cls,
        py_fn: Callable | None = None,
        py_params: list[inspect.Parameter] | None = None,
        param_types: list[ts.ColumnType] | None = None,
        type_substitutions: dict | None = None,
        is_cls_method: bool = False,
    ) -> list[Parameter]:
        assert (py_fn is None) != (py_params is None)
        if py_fn is not None:
            sig = inspect.signature(py_fn, eval_str=True)
            py_params = list(sig.parameters.values())
        assert py_params is not None
        parameters: list[Parameter] = []
        if type_substitutions is None:
            type_substitutions = {}

        for idx, param in enumerate(py_params):
            if is_cls_method and idx == 0:
                continue  # skip 'self' or 'cls'
            if param.name in cls.SYSTEM_PARAM_NAMES:
                continue
            if param.name.startswith('_'):
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION, f"{param.name!r}: parameters starting with '_' are reserved"
                )
            if param.name in cls.SPECIAL_PARAM_NAMES:
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION, f'{param.name!r} is a reserved parameter name'
                )
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                parameters.append(Parameter(param.name, col_type=None, kind=param.kind))
                continue

            if param_types is not None:
                if idx >= len(param_types):
                    raise excs.RequestError(
                        excs.ErrorCode.MISSING_REQUIRED, f'Missing type for parameter {param.name!r}'
                    )
                param_type: ts.ColumnType | None = param_types[idx]
                is_batched: bool | None = False
            else:
                py_type = type_substitutions.get(param.annotation, param.annotation)
                param_type, is_batched = cls._infer_type(py_type)
                if param_type is None:
                    raise excs.RequestError(
                        excs.ErrorCode.INVALID_TYPE, f'Cannot infer pixeltable type for parameter {param.name!r}'
                    )

            default = None
            if param.default is not inspect.Parameter.empty:
                default = _literal(param.default)
                if default is None:
                    raise excs.RequestError(
                        excs.ErrorCode.INVALID_ARGUMENT,
                        f'Default value for parameter {param.name!r} must be a constant',
                    )

            parameters.append(
                Parameter(param.name, col_type=param_type, kind=param.kind, is_batched=bool(is_batched), default=default)
            )

        return parameters

    @classmethod
    def create(
        cls,
        py_fn: Callable,
        param_types: list[ts.ColumnType] | None = None,
        return_type: ts.ColumnType | None = None,
        type_substitutions: dict | None = None,
        is_cls_method: bool = False,
    ) -> Signature:
        """Create a signature for the given Callable, inferring parameter and return types if not given."""
        if type_substitutions is None:
            type_substitutions = {}
        parameters = cls.create_parameters(
            py_fn=py_fn, param_types=param_types, is_cls_method=is_cls_method, type_substitutions=type_substitutions
        )
        sig = inspect.signature(py_fn, eval_str=True)
        if return_type is None:
            py_type = type_substitutions.get(sig.return_annotation, sig.return_annotation)
            return_type, return_is_batched = cls._infer_type(py_type)
            if return_type is None:
                raise excs.RequestError(excs.ErrorCode.INVALID_TYPE, 'Cannot infer pixeltable return type')
        else:
            _, return_is_batched = cls._infer_type(sig.return_annotation)
        system_params = [param_name for param_name in sig.parameters if param_name in cls.SYSTEM_PARAM_NAMES]
        return Signature(return_type, parameters, bool(return_is_batched), system_parameters=system_params)
