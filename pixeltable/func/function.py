from __future__ import annotations

import abc
import importlib
import inspect
from copy import copy
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import sqlalchemy as sql
from typing_extensions import Self

import pixeltable as pxt
import pixeltable.exceptions as excs
import pixeltable.type_system as ts

from .globals import resolve_symbol
from .signature import Signature

if TYPE_CHECKING:
    from .expr_template_function import ExprTemplate, ExprTemplateFunction


class Function(abc.ABC):
    """Base class for Pixeltable's function interface.

    A function in Pixeltable is an object that has a signature and implements __call__().
    This base class provides a default serialization mechanism for Function instances provided by Python modules,
    via the member self_path.
    """

    signatures: list[Signature]
    self_path: Optional[str]
    is_method: bool
    is_property: bool
    _conditional_return_type: Optional[Callable[..., ts.ColumnType]]
    __resolutions: list[Self]  # Cached overload resolutions of this function

    # Translates a call to this function with the given arguments to its SQLAlchemy equivalent.
    # Overriden for specific Function instances via the to_sql() decorator. The override must accept the same
    # parameter names as the original function. Each parameter is going to be of type sql.ColumnElement.
    _to_sql: Callable[..., Optional[sql.ColumnElement]]

    def __init__(
        self,
        signatures: list[Signature],
        self_path: Optional[str] = None,
        is_method: bool = False,
        is_property: bool = False
    ):
        # Check that stored functions cannot be declared using `is_method` or `is_property`:
        assert not ((is_method or is_property) and self_path is None)
        assert isinstance(signatures, list)
        assert len(signatures) > 0
        self.signatures = signatures
        self.self_path = self_path  # fully-qualified path to self
        self.is_method = is_method
        self.is_property = is_property
        self._conditional_return_type = None
        self._to_sql = self.__default_to_sql

        self.__resolutions = []

    @property
    def name(self) -> str:
        assert self.self_path is not None
        return self.self_path.split('.')[-1]

    @property
    def display_name(self) -> str:
        if self.self_path is None:
            return '<anonymous>'
        ptf_prefix = 'pixeltable.functions.'
        if self.self_path.startswith(ptf_prefix):
            return self.self_path[len(ptf_prefix):]
        return self.self_path

    @property
    def is_polymorphic(self) -> bool:
        return len(self.signatures) > 1

    @property
    def signature(self) -> Signature:
        assert not self.is_polymorphic
        return self.signatures[0]

    @property
    def arity(self) -> int:
        assert not self.is_polymorphic
        return len(self.signature.parameters)

    def _overload_resolution(self, signature_idx: int) -> Function:
        """
        Return a new  `Function` that retains just the single signature at index `signature_idx`, and is otherwise
        identical to this `Function`.
        """
        assert 0 <= signature_idx < len(self.signatures)

        # We cache the resolutions in self.__resolutions. This ensures that each resolution is represented globally
        # by a single Python object. We do this dynamically rather than pre-constructing them in order to avoid
        # circular complexity in the `Function` initialization logic.

        # Note that it is necessary to create a resolution even if this Function has only one signature, because
        # there is no guarantee we have seen all the Function's @overload declarations at the time this is called.

        while len(self.__resolutions) <= signature_idx:
            idx = len(self.__resolutions)
            resolution = copy(self)
            resolution.signatures = [self.signatures[idx]]
            resolution._update_as_overload_resolution(idx)
            self.__resolutions.append(resolution)

        return self.__resolutions[signature_idx]

    def _update_as_overload_resolution(self, signature_idx: int) -> None:
        """
        Subclasses must implement this in order to do any additional work when creating a resolution, beyond
        simply updating `self.signatures`.
        """
        raise NotImplementedError()

    def help_str(self) -> str:
        return self.display_name + str(self.signatures[0])

    def __call__(self, *args: Any, **kwargs: Any) -> 'pxt.exprs.FunctionCall':
        from pixeltable import exprs

        resolved_fn, bound_args = self._bind_to_matching_signature(args, kwargs)
        return_type = resolved_fn.call_return_type(args, kwargs)
        return exprs.FunctionCall(resolved_fn, bound_args, return_type)

    def _bind_to_matching_signature(self, args: Sequence[Any], kwargs: dict[str, Any]) -> tuple[Function, dict[str, Any]]:
        result: int = -1
        bound_args: Optional[dict[str, Any]] = None
        assert len(self.signatures) > 0
        if len(self.signatures) == 1:
            # Only one signature: call _bind_to_signature() and surface any errors directly
            result = 0
            bound_args = self._bind_to_signature(0, args, kwargs)
        else:
            # Multiple signatures: try each signature in declaration order and trap any errors.
            # If none of them succeed, raise a generic error message.
            for i in range(len(self.signatures)):
                try:
                    bound_args = self._bind_to_signature(i, args, kwargs)
                except (TypeError, excs.Error):
                    continue
                result = i
                break
            if result == -1:
                raise excs.Error(f'Function {self.name!r} has no matching signature for arguments')
        assert result >= 0
        assert bound_args is not None
        return self._overload_resolution(result), bound_args

    def _bind_to_signature(self, signature_idx: int, args: Sequence[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        from pixeltable import exprs

        signature = self.signatures[signature_idx]
        bound_args = signature.py_signature.bind(*args, **kwargs).arguments
        self._overload_resolution(signature_idx).validate_call(bound_args)
        exprs.FunctionCall.normalize_args(self.name, signature, bound_args)
        return bound_args

    def validate_call(self, bound_args: dict[str, Any]) -> None:
        """Override this to do custom validation of the arguments"""
        assert not self.is_polymorphic

    def call_return_type(self, args: Sequence[Any], kwargs: dict[str, Any]) -> ts.ColumnType:
        """Return the type of the value returned by calling this function with the given arguments"""
        assert not self.is_polymorphic
        if self._conditional_return_type is None:
            return self.signature.return_type
        bound_args = self.signature.py_signature.bind(*args, **kwargs).arguments
        kw_args: dict[str, Any] = {}
        sig = inspect.signature(self._conditional_return_type)
        for param in sig.parameters.values():
            if param.name in bound_args:
                kw_args[param.name] = bound_args[param.name]
        return self._conditional_return_type(**kw_args)

    def conditional_return_type(self, fn: Callable[..., ts.ColumnType]) -> Callable[..., ts.ColumnType]:
        """Instance decorator for specifying a conditional return type for this function"""
        if self.is_polymorphic:
            raise excs.Error('`conditional_return_type` is not supported for functions with multiple signatures')
        # verify that call_return_type only has parameters that are also present in the signature
        sig = inspect.signature(fn)
        for param in sig.parameters.values():
            if param.name not in self.signature.parameters:
                raise ValueError(f'`conditional_return_type` has parameter `{param.name}` that is not in the signature')
        self._conditional_return_type = fn
        return fn

    def using(self, **kwargs: Any) -> 'ExprTemplateFunction':
        from .expr_template_function import ExprTemplate, ExprTemplateFunction

        assert len(self.signatures) > 0
        if len(self.signatures) == 1:
            # Only one signature: call _bind_to_template() and surface any errors directly
            template = self._bind_and_create_template(kwargs)
            return ExprTemplateFunction([template])
        else:
            # Multiple signatures: iterate over each signature and generate a template for each
            # successful binding. If there are no successful bindings, raise a generic error.
            # (Note that the resulting ExprTemplateFunction may have strictly fewer signatures than
            # this Function, in the event that only some of the signatures are successfully bound.)
            templates: list[ExprTemplate] = []
            for resolved_fn in self.__resolutions:
                try:
                    template = resolved_fn._bind_and_create_template(kwargs)
                    templates.append(template)
                except (TypeError, excs.Error):
                    continue
            if len(templates) == 0:
                raise excs.Error(f'Function {self.name!r} has no matching signature for arguments')
            return ExprTemplateFunction(templates)

    def _bind_and_create_template(self, kwargs: dict[str, Any]) -> 'ExprTemplate':
        from pixeltable import exprs

        from .expr_template_function import ExprTemplate

        assert not self.is_polymorphic

        # Resolve each kwarg into a parameter binding
        bindings: dict[str, exprs.Expr] = {}
        for k, v in kwargs.items():
            if k not in self.signature.parameters:
                raise excs.Error(f'Unknown parameter: {k}')
            param = self.signature.parameters[k]
            expr = exprs.Expr.from_object(v)
            if not param.col_type.is_supertype_of(expr.col_type):
                raise excs.Error(f'Expected type `{param.col_type}` for parameter `{k}`; got `{expr.col_type}`')
            bindings[k] = v  # Use the original value, not the Expr (The Expr is only for validation)

        residual_params = [
            p for p in self.signature.parameters.values() if p.name not in bindings
        ]

        # Bind each remaining parameter to a like-named variable
        for param in residual_params:
            bindings[param.name] = exprs.Variable(param.name, param.col_type)

        return_type = self.call_return_type([], bindings)
        call = exprs.FunctionCall(self, bindings, return_type)

        # Construct the (n-k)-ary signature of the new function. We use `call.col_type` for this, rather than
        # `self.signature.return_type`, because the return type of the new function may be specialized via a
        # conditional return type.
        new_signature = Signature(call.col_type, residual_params, self.signature.is_batched)

        return ExprTemplate(call, new_signature)

    @abc.abstractmethod
    def exec(self, args: Sequence[Any], kwargs: dict[str, Any]) -> Any:
        """Execute the function with the given arguments and return the result."""
        pass

    def to_sql(self, fn: Callable[..., Optional[sql.ColumnElement]]) -> Callable[..., Optional[sql.ColumnElement]]:
        """Instance decorator for specifying the SQL translation of this function"""
        self._to_sql = fn
        return fn

    def __default_to_sql(self, *args: Any, **kwargs: Any) -> Optional[sql.ColumnElement]:
        """The default implementation of SQL translation, which provides no translation"""
        return None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.self_path == other.self_path

    def source(self) -> None:
        """Print source code"""
        print('source not available')

    def as_dict(self) -> dict:
        """
        Return a serialized reference to the instance that can be passed to json.dumps() and converted back
        to an instance with from_dict().
        Subclasses can override _as_dict().
        """
        # We currently only ever serialize a function that has a specific signature (not a polymorphic form).
        assert not self.is_polymorphic
        classpath = f'{self.__class__.__module__}.{self.__class__.__qualname__}'
        return {'_classpath': classpath, **self._as_dict()}

    def _as_dict(self) -> dict:
        """Default serialization: store the path to self (which includes the module path) and signature."""
        assert self.self_path is not None
        return {
            'path': self.self_path,
            'signature': self.signature.as_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Function:
        """
        Turn dict that was produced by calling as_dict() into an instance of the correct Function subclass.
        """
        assert '_classpath' in d
        module_path, class_name = d['_classpath'].rsplit('.', 1)
        class_module = importlib.import_module(module_path)
        func_class = getattr(class_module, class_name)
        assert isinstance(func_class, type) and issubclass(func_class, Function)
        return func_class._from_dict(d)

    @classmethod
    def _from_dict(cls, d: dict) -> Function:
        """Default deserialization: load the symbol indicated by the stored symbol_path"""
        assert 'path' in d and d['path'] is not None
        assert 'signature' in d and d['signature'] is not None
        instance = resolve_symbol(d['path'])
        assert isinstance(instance, Function)

        # Load the signature from the DB and check that it is still valid (i.e., is still consistent with a signature
        # in the code).
        signature = Signature.from_dict(d['signature'])
        idx = instance.__find_matching_overload(signature)
        if idx is None:
            # No match; generate an informative error message.
            signature_note_str = 'any of its signatures' if instance.is_polymorphic else 'its signature as'
            instance_signature_str = (
                f'{len(instance.signatures)} signatures' if instance.is_polymorphic else str(instance.signature)
            )
            # TODO: Handle this more gracefully (instead of failing the DB load, allow the DB load to succeed, but
            #       mark any enclosing FunctionCall as unusable). It's the same issue as dealing with a renamed UDF or
            #       FunctionCall return type mismatch.
            raise excs.Error(
                f'The signature stored in the database for the UDF `{instance.self_path}` no longer matches '
                f'{signature_note_str} as currently defined in the code.\nThis probably means that the code for '
                f'`{instance.self_path}` has changed in a backward-incompatible way.\n'
                f'Signature in database: {signature}\n'
                f'Signature in code: {instance_signature_str}'
            )
        # We found a match; specialize to the appropriate overload resolution (non-polymorphic form) and return that.
        return instance._overload_resolution(idx)

    def __find_matching_overload(self, sig: Signature) -> Optional[int]:
        for idx, overload_sig in enumerate(self.signatures):
            if sig.is_consistent_with(overload_sig):
                return idx
        return None

    def to_store(self) -> tuple[dict, bytes]:
        """
        Serialize the function to a format that can be stored in the Pixeltable store
        Returns:
            - a dict that can be passed to json.dumps()
            - additional binary data
        Only Function subclasses that can be stored need to override this.
        """
        raise NotImplementedError()

    @classmethod
    def from_store(cls, name: Optional[str], md: dict, binary_obj: bytes) -> Function:
        """
        Create a Function instance from the serialized representation returned by to_store()
        """
        raise NotImplementedError()
