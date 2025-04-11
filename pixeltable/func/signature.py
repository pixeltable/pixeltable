from __future__ import annotations

import dataclasses
import inspect
import logging
import typing
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Optional

import pixeltable.exceptions as excs
import pixeltable.type_system as ts

if TYPE_CHECKING:
    from pixeltable import exprs

_logger = logging.getLogger('pixeltable')


@dataclasses.dataclass
class Parameter:
    name: str
    col_type: Optional[ts.ColumnType]  # None for variable parameters
    kind: inspect._ParameterKind
    # for some reason, this needs to precede is_batched in the dataclass definition,
    # otherwise Python complains that an argument with a default is followed by an argument without a default
    default: Optional['exprs.Literal'] = None  # default value for the parameter
    is_batched: bool = False  # True if the parameter is a batched parameter (eg, Batch[dict])

    def __post_init__(self) -> None:
        from pixeltable import exprs

        if self.default is not None:
            if self.col_type is None:
                raise excs.Error(f'Cannot have a default value for variable parameter {self.name!r}')
            if not isinstance(self.default, exprs.Literal):
                raise excs.Error(f'Default value for parameter {self.name!r} is not a constant')
            if not self.col_type.is_supertype_of(self.default.col_type):
                raise excs.Error(
                    f'Default value for parameter {self.name!r} is not of type {self.col_type!r}: {self.default}'
                )

    def has_default(self) -> bool:
        return self.default is not None

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
        from pixeltable import exprs

        assert d['default'] is None or isinstance(d['default'], dict), d
        default = None if d['default'] is None else exprs.Literal.from_dict(d['default'])
        return cls(
            name=d['name'],
            col_type=ts.ColumnType.from_dict(d['col_type']) if d['col_type'] is not None else None,
            kind=getattr(inspect.Parameter, d['kind']),
            is_batched=d['is_batched'],
            default=default,
        )

    def to_py_param(self) -> inspect.Parameter:
        py_default = self.default.val if self.default is not None else inspect.Parameter.empty
        return inspect.Parameter(self.name, self.kind, default=py_default)

    def __hash__(self) -> int:
        return hash((self.name, self.col_type, self.kind, self.default, self.is_batched))


T = typing.TypeVar('T')
Batch = typing.Annotated[list[T], 'pxt-batch']


class Signature:
    """
    Represents the signature of a Pixeltable function.

    - self.is_batched: return type is a Batch[...] type
    """

    SPECIAL_PARAM_NAMES: ClassVar[list[str]] = ['group_by', 'order_by']

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

    def is_consistent_with(self, other: Signature) -> bool:
        """
        Returns True if this signature is consistent with the other signature.
        S is consistent with T if we could safely replace S by T in any call where S is used. Specifically:
        (i) S.return_type is a supertype of T.return_type
        (ii) For each parameter p in S, there is a parameter q in T such that:
            - p and q have the same name and kind
            - q.col_type is a supertype of p.col_type
        (iii) For each *required* parameter q in T, there is a parameter p in S with the same name (in which
            case the kinds and types must also match, by condition (ii)).
        """
        # Check (i)
        if not self.get_return_type().is_supertype_of(other.get_return_type(), ignore_nullable=True):
            return False

        # Check (ii)
        for param_name, param in self.parameters.items():
            if param_name not in other.parameters:
                return False
            other_param = other.parameters[param_name]
            if (
                param.kind != other_param.kind
                or (param.col_type is None) != (other_param.col_type is None)  # this can happen if they are varargs
                or (
                    param.col_type is not None
                    and not other_param.col_type.is_supertype_of(param.col_type, ignore_nullable=True)
                )
            ):
                return False

        # Check (iii)
        for other_param in other.required_parameters:  # noqa: SIM110
            if other_param.name not in self.parameters:
                return False

        return True

    def validate_args(self, bound_args: dict[str, Optional['exprs.Expr']], context: str = '') -> None:
        if context:
            context = f' ({context})'

        for param_name, arg in bound_args.items():
            assert param_name in self.parameters
            param = self.parameters[param_name]
            is_var_param = param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
            if is_var_param:
                continue
            assert param.col_type is not None

            if arg is None:
                raise excs.Error(f'Parameter {param_name!r}{context}: invalid argument')

            # Check that the argument is consistent with the expected parameter type, with the allowance that
            # non-nullable parameters can still accept nullable arguments (since in that event, FunctionCall.eval()
            # detects the Nones and skips evaluation).
            if not (
                param.col_type.is_supertype_of(arg.col_type, ignore_nullable=True)
                # TODO: this is a hack to allow JSON columns to be passed to functions that accept scalar
                # types. It's necessary to avoid littering notebooks with `apply(str)` calls or equivalent.
                # (Previously, this wasn't necessary because `is_supertype_of()` was improperly implemented.)
                # We need to think through the right way to handle this scenario.
                or (arg.col_type.is_json_type() and param.col_type.is_scalar_type())
            ):
                raise excs.Error(
                    f'Parameter {param_name!r}{context}: argument type {arg.col_type} does not'
                    f' match parameter type {param.col_type}'
                )

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

    def __hash__(self) -> int:
        return hash((self.return_type, self.parameters))

    def __str__(self) -> str:
        param_strs: list[str] = []
        for p in self.parameters.values():
            if p.kind == inspect.Parameter.VAR_POSITIONAL:
                param_strs.append(f'*{p.name}')
            elif p.kind == inspect.Parameter.VAR_KEYWORD:
                param_strs.append(f'**{p.name}')
            else:
                param_strs.append(f'{p.name}: {p.col_type}')
        return f'({", ".join(param_strs)}) -> {self.get_return_type()}'

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
        py_fn: Optional[Callable] = None,
        py_params: Optional[list[inspect.Parameter]] = None,
        param_types: Optional[list[ts.ColumnType]] = None,
        type_substitutions: Optional[dict] = None,
        is_cls_method: bool = False,
    ) -> list[Parameter]:
        from pixeltable import exprs

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
                raise excs.Error(f'{param.name!r} is a reserved parameter name')
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                parameters.append(Parameter(param.name, col_type=None, kind=param.kind))
                continue

            # check non-var parameters for name collisions and default value compatibility
            if param_types is not None:
                if idx >= len(param_types):
                    raise excs.Error(f'Missing type for parameter {param.name!r}')
                param_type = param_types[idx]
                is_batched = False
            else:
                # Look up the substitution for param.annotation, defaulting to param.annotation if there is none
                py_type = type_substitutions.get(param.annotation, param.annotation)
                param_type, is_batched = cls._infer_type(py_type)
                if param_type is None:
                    raise excs.Error(f'Cannot infer pixeltable type for parameter {param.name!r}')

            default = None if param.default is inspect.Parameter.empty else exprs.Expr.from_object(param.default)
            if not (default is None or isinstance(default, exprs.Literal)):
                raise excs.Error(f'Default value for parameter {param.name!r} must be a constant')

            parameters.append(
                Parameter(param.name, col_type=param_type, kind=param.kind, is_batched=is_batched, default=default)
            )

        return parameters

    @classmethod
    def create(
        cls,
        py_fn: Callable,
        param_types: Optional[list[ts.ColumnType]] = None,
        return_type: Optional[ts.ColumnType] = None,
        type_substitutions: Optional[dict] = None,
        is_cls_method: bool = False,
    ) -> Signature:
        """Create a signature for the given Callable.
        Infer the parameter and return types, if none are specified.
        Raises an exception if the types cannot be inferred.
        """
        if type_substitutions is None:
            type_substitutions = {}

        parameters = cls.create_parameters(
            py_fn=py_fn, param_types=param_types, is_cls_method=is_cls_method, type_substitutions=type_substitutions
        )
        sig = inspect.signature(py_fn)
        if return_type is None:
            # Look up the substitution for sig.return_annotation, defaulting to return_annotation if there is none
            py_type = type_substitutions.get(sig.return_annotation, sig.return_annotation)
            return_type, return_is_batched = cls._infer_type(py_type)
            if return_type is None:
                raise excs.Error('Cannot infer pixeltable return type')
        else:
            _, return_is_batched = cls._infer_type(sig.return_annotation)

        return Signature(return_type, parameters, return_is_batched)
