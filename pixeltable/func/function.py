from __future__ import annotations

import importlib
import inspect
from abc import ABC, abstractmethod
from copy import copy
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, cast

import sqlalchemy as sql
from typing_extensions import Self

import pixeltable as pxt
import pixeltable.exceptions as excs
import pixeltable.type_system as ts

from .globals import resolve_symbol
from .signature import Signature

if TYPE_CHECKING:
    from pixeltable import exprs

    from .expr_template_function import ExprTemplate, ExprTemplateFunction


class Function(ABC):
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

    # We cache the overload resolutions in self._resolutions. This ensures that each resolution is represented
    # globally by a single Python object. We do this dynamically rather than pre-constructing them in order to
    # avoid circular complexity in the `Function` initialization logic.
    __resolved_fns: list[Self]

    # Translates a call to this function with the given arguments to its SQLAlchemy equivalent.
    # Overriden for specific Function instances via the to_sql() decorator. The override must accept the same
    # parameter names as the original function. Each parameter is going to be of type sql.ColumnElement.
    _to_sql: Callable[..., Optional[sql.ColumnElement]]

    # Returns the resource pool to use for calling this function with the given arguments.
    # Overriden for specific Function instances via the resource_pool() decorator. The override must accept a subset
    # of the parameters of the original function, with the same type.
    _resource_pool: Callable[..., Optional[str]]

    def __init__(
        self,
        signatures: list[Signature],
        self_path: Optional[str] = None,
        is_method: bool = False,
        is_property: bool = False,
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
        self.__resolved_fns = []
        self._to_sql = self.__default_to_sql
        self._resource_pool = self.__default_resource_pool

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
            return self.self_path[len(ptf_prefix) :]
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

    @property
    @abstractmethod
    def is_async(self) -> bool: ...

    def _docstring(self) -> Optional[str]:
        return None

    def help_str(self) -> str:
        docstring = self._docstring()
        display = self.display_name + str(self.signatures[0])
        if docstring is None:
            return display
        return f'{display}\n\n{docstring}'

    @property
    def _resolved_fns(self) -> list[Self]:
        """
        Return the list of overload resolutions for this `Function`, constructing it first if necessary.
        Each resolution is a new `Function` instance that retains just the single signature at index `signature_idx`,
        and is otherwise identical to this `Function`.
        """
        if len(self.__resolved_fns) == 0:
            # The list of overload resolutions hasn't been constructed yet; do so now.
            if len(self.signatures) == 1:
                # Only one signature: no need to construct separate resolutions
                self.__resolved_fns.append(self)
            else:
                # Multiple signatures: construct a resolution for each signature
                for idx in range(len(self.signatures)):
                    resolution = cast(Self, copy(self))
                    resolution.signatures = [self.signatures[idx]]
                    resolution.__resolved_fns = [resolution]  # Resolves to itself
                    resolution._update_as_overload_resolution(idx)
                    self.__resolved_fns.append(resolution)

        return self.__resolved_fns

    @property
    def _has_resolved_fns(self) -> bool:
        """
        Returns true if the resolved_fns for this `Function` have been constructed (i.e., if self._resolved_fns
        has been accessed).
        """
        return len(self.__resolved_fns) > 0

    def _update_as_overload_resolution(self, signature_idx: int) -> None:
        """
        Subclasses must implement this in order to do any additional work when creating a resolution, beyond
        simply updating `self.signatures`.
        """
        raise NotImplementedError()

    def __call__(self, *args: Any, **kwargs: Any) -> 'pxt.exprs.FunctionCall':
        from pixeltable import exprs

        args = [exprs.Expr.from_object(arg) for arg in args]
        kwargs = {k: exprs.Expr.from_object(v) for k, v in kwargs.items()}

        resolved_fn, bound_args = self._bind_to_matching_signature(args, kwargs)
        return_type = resolved_fn.call_return_type(bound_args)

        return exprs.FunctionCall(resolved_fn, args, kwargs, return_type)

    def _bind_to_matching_signature(self, args: Sequence[Any], kwargs: dict[str, Any]) -> tuple[Self, dict[str, Any]]:
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
        return self._resolved_fns[result], bound_args

    def _bind_to_signature(self, signature_idx: int, args: Sequence[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        from pixeltable import exprs

        signature = self.signatures[signature_idx]
        bound_args = signature.py_signature.bind(*args, **kwargs).arguments
        normalized_args = {k: exprs.Expr.from_object(v) for k, v in bound_args.items()}
        self._resolved_fns[signature_idx].validate_call(normalized_args)
        return normalized_args

    def validate_call(self, bound_args: dict[str, Optional['exprs.Expr']]) -> None:
        """Override this to do custom validation of the arguments"""
        assert not self.is_polymorphic
        self.signature.validate_args(bound_args, context=f'in function {self.name!r}')

    def call_resource_pool(self, bound_args: dict[str, 'exprs.Expr']) -> str:
        """Return the resource pool to use for calling this function with the given arguments"""
        rp_kwargs = self._assemble_callable_args(self._resource_pool, bound_args)
        if rp_kwargs is None:
            # TODO: What to do in this case? An example where this can happen is if model_id is not a constant
            #   in a call to one of the OpenAI endpoints.
            raise excs.Error('Could not determine resource pool')
        return self._resource_pool(**rp_kwargs)

    def call_return_type(self, bound_args: dict[str, 'exprs.Expr']) -> ts.ColumnType:
        """Return the type of the value returned by calling this function with the given arguments"""
        if self._conditional_return_type is None:
            # No conditional return type specified; use the default return type
            return_type = self.signature.return_type
        else:
            crt_kwargs = self._assemble_callable_args(self._conditional_return_type, bound_args)
            if crt_kwargs is None:
                # A conditional return type is specified, but one of its arguments is not a constant.
                # Use the default return type
                return_type = self.signature.return_type
            else:
                # A conditional return type is specified and all its arguments are constants; use the specific
                # call return type
                return_type = self._conditional_return_type(**crt_kwargs)

        if return_type.nullable:
            return return_type

        # If `return_type` is non-nullable, but the function call has a nullable input to any of its non-nullable
        # parameters, then we need to make it nullable. This is because Pixeltable defaults a function output to
        # `None` when any of its non-nullable inputs are `None`.
        for arg_name, arg in bound_args.items():
            param = self.signature.parameters[arg_name]
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if arg.col_type.nullable and not param.col_type.nullable:
                return_type = return_type.copy(nullable=True)
                break

        return return_type

    def _assemble_callable_args(
        self, callable: Callable, bound_args: dict[str, 'exprs.Expr']
    ) -> Optional[dict[str, Any]]:
        """
        Return the kwargs to pass to callable, given bound_args passed to this function.
        """
        from pixeltable import exprs

        assert not self.is_polymorphic

        callable_signature = inspect.signature(callable)
        callable_args: dict[str, Any] = {}

        for param in callable_signature.parameters.values():
            arg: Any
            if param.name in bound_args:
                arg = bound_args[param.name]
            elif self.signature.parameters[param.name].has_default():
                arg = self.signature.parameters[param.name].default
            else:
                return None

            if isinstance(arg, exprs.Literal):
                callable_args[param.name] = arg.val
            elif isinstance(arg, exprs.Expr):
                return None
            else:
                callable_args[param.name] = arg

        return callable_args

    def conditional_return_type(self, fn: Callable[..., ts.ColumnType]) -> Callable[..., ts.ColumnType]:
        """Instance decorator for specifying a conditional return type for this function"""
        # verify that call_return_type only has parameters that are also present in the signature
        fn_sig = inspect.signature(fn)
        for param in fn_sig.parameters.values():
            for self_sig in self.signatures:
                if param.name not in self_sig.parameters:
                    raise ValueError(
                        f'`conditional_return_type` has parameter `{param.name}` that is not in a signature'
                    )
        self._conditional_return_type = fn
        return fn

    def using(self, **kwargs: Any) -> 'ExprTemplateFunction':
        from .expr_template_function import ExprTemplateFunction

        assert len(self.signatures) > 0
        if len(self.signatures) == 1:
            # Only one signature: call _bind_and_create_template() and surface any errors directly
            template = self._bind_and_create_template(kwargs)
            return ExprTemplateFunction([template])
        else:
            # Multiple signatures: iterate over each signature and generate a template for each
            # successful binding. If there are no successful bindings, raise a generic error.
            # (Note that the resulting ExprTemplateFunction may have strictly fewer signatures than
            # this Function, in the event that only some of the signatures are successfully bound.)
            templates: list['ExprTemplate'] = []
            for i in range(len(self.signatures)):
                try:
                    template = self._resolved_fns[i]._bind_and_create_template(kwargs)
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
            bindings[k] = expr

        residual_params = [p for p in self.signature.parameters.values() if p.name not in bindings]

        # Bind each remaining parameter to a like-named variable.
        # Also construct the call arguments for the template function call. Variables become args when possible;
        # otherwise, they are passed as kwargs.
        template_args: list[exprs.Expr] = []
        template_kwargs: dict[str, exprs.Expr] = {}
        args_ok = True
        for name, param in self.signature.parameters.items():
            if name in bindings:
                template_kwargs[name] = bindings[name]
                args_ok = False
            else:
                var = exprs.Variable(name, param.col_type)
                bindings[name] = var
                if args_ok and param.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                ):
                    template_args.append(var)
                else:
                    template_kwargs[name] = var
                    args_ok = False

        return_type = self.call_return_type(bindings)
        call = exprs.FunctionCall(self, template_args, template_kwargs, return_type)

        # Construct the (n-k)-ary signature of the new function. We use `call.col_type` for this, rather than
        # `self.signature.return_type`, because the return type of the new function may be specialized via a
        # conditional return type.
        new_signature = Signature(call.col_type, residual_params, self.signature.is_batched)

        return ExprTemplate(call, new_signature)

    def exec(self, args: Sequence[Any], kwargs: dict[str, Any]) -> Any:
        """Execute the function with the given arguments and return the result."""
        raise NotImplementedError()

    async def aexec(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the function with the given arguments and return the result."""
        raise NotImplementedError()

    def to_sql(self, fn: Callable[..., Optional[sql.ColumnElement]]) -> Callable[..., Optional[sql.ColumnElement]]:
        """Instance decorator for specifying the SQL translation of this function"""
        self._to_sql = fn
        return fn

    def __default_to_sql(self, *args: Any, **kwargs: Any) -> Optional[sql.ColumnElement]:
        """The default implementation of SQL translation, which provides no translation"""
        return None

    def resource_pool(self, fn: Callable[..., str]) -> Callable[..., str]:
        """Instance decorator for specifying the resource pool of this function"""
        # TODO: check that fn's parameters are a subset of our parameters
        self._resource_pool = fn
        return fn

    def __default_resource_pool(self) -> Optional[str]:
        return None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.self_path == other.self_path

    def source(self) -> None:
        """Print source code"""
        print('source not available')

    def as_dict(self) -> dict[str, Any]:
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
        return {'path': self.self_path, 'signature': self.signature.as_dict()}

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
        return instance

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
