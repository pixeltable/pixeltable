from __future__ import annotations

import abc
import importlib
import inspect
from typing import TYPE_CHECKING, Any, Callable, Optional

import sqlalchemy as sql

import pixeltable as pxt
import pixeltable.exceptions as excs
import pixeltable.type_system as ts

from .globals import resolve_symbol
from .signature import Signature

if TYPE_CHECKING:
    from .expr_template_function import ExprTemplateFunction


class Function(abc.ABC):
    """Base class for Pixeltable's function interface.

    A function in Pixeltable is an object that has a signature and implements __call__().
    This base class provides a default serialization mechanism for Function instances provided by Python modules,
    via the member self_path.
    """

    signature: Signature
    self_path: Optional[str]
    is_method: bool
    is_property: bool
    _conditional_return_type: Optional[Callable[..., ts.ColumnType]]

    # Translates a call to this function with the given arguments to its SQLAlchemy equivalent.
    # Overriden for specific Function instances via the to_sql() decorator. The override must accept the same
    # parameter names as the original function. Each parameter is going to be of type sql.ColumnElement.
    _to_sql: Callable[..., Optional[sql.ColumnElement]]


    def __init__(
        self, signature: Signature, self_path: Optional[str] = None, is_method: bool = False, is_property: bool = False
    ):
        # Check that stored functions cannot be declared using `is_method` or `is_property`:
        assert not ((is_method or is_property) and self_path is None)
        self.signature = signature
        self.self_path = self_path  # fully-qualified path to self
        self.is_method = is_method
        self.is_property = is_property
        self._conditional_return_type = None
        self._to_sql = self.__default_to_sql

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
    def arity(self) -> int:
        return len(self.signature.parameters)

    def _docstring(self) -> Optional[str]:
        return None

    def help_str(self) -> str:
        docstring = self._docstring()
        display = self.display_name + str(self.signature)
        if docstring is None:
            return display
        return f'{display}\n\n{docstring}'

    def __call__(self, *args: Any, **kwargs: Any) -> 'pxt.exprs.FunctionCall':
        from pixeltable import exprs
        bound_args = self.signature.py_signature.bind(*args, **kwargs)
        self.validate_call(bound_args.arguments)
        return exprs.FunctionCall(self, bound_args.arguments)

    def validate_call(self, bound_args: dict[str, Any]) -> None:
        """Override this to do custom validation of the arguments"""
        pass

    def call_return_type(self, kwargs: dict[str, Any]) -> ts.ColumnType:
        """Return the type of the value returned by calling this function with the given arguments"""
        if self._conditional_return_type is None:
            return self.signature.return_type
        bound_args = self.signature.py_signature.bind(**kwargs)
        kw_args: dict[str, Any] = {}
        sig = inspect.signature(self._conditional_return_type)
        for param in sig.parameters.values():
            if param.name in bound_args.arguments:
                kw_args[param.name] = bound_args.arguments[param.name]
        return self._conditional_return_type(**kw_args)

    def conditional_return_type(self, fn: Callable[..., ts.ColumnType]) -> Callable[..., ts.ColumnType]:
        """Instance decorator for specifying a conditional return type for this function"""
        # verify that call_return_type only has parameters that are also present in the signature
        sig = inspect.signature(fn)
        for param in sig.parameters.values():
            if param.name not in self.signature.parameters:
                raise ValueError(f'`conditional_return_type` has parameter `{param.name}` that is not in the signature')
        self._conditional_return_type = fn
        return fn

    def using(self, **kwargs: Any) -> 'ExprTemplateFunction':
        from pixeltable import exprs

        from .expr_template_function import ExprTemplateFunction

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

        call = exprs.FunctionCall(self, bindings)

        # Construct the (n-k)-ary signature of the new function. We use `call.col_type` for this, rather than
        # `self.signature.return_type`, because the return type of the new function may be specialized via a
        # conditional return type.
        new_signature = Signature(call.col_type, residual_params, self.signature.is_batched)
        return ExprTemplateFunction(call, new_signature)

    @abc.abstractmethod
    def exec(self, *args: Any, **kwargs: Any) -> Any:
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

    def as_dict(self) -> dict[str, Any]:
        """
        Return a serialized reference to the instance that can be passed to json.dumps() and converted back
        to an instance with from_dict().
        Subclasses can override _as_dict().
        """
        classpath = f'{self.__class__.__module__}.{self.__class__.__qualname__}'
        return {'_classpath': classpath, **self._as_dict()}

    def _as_dict(self) -> dict:
        """Default serialization: store the path to self (which includes the module path)"""
        assert self.self_path is not None
        return {'path': self.self_path}

    @classmethod
    def from_dict(cls, d: dict) -> Function:
        """
        Turn dict that was produced by calling as_dict() into an instance of the correct Function subclass.
        """
        assert '_classpath' in d
        module_path, class_name = d['_classpath'].rsplit('.', 1)
        class_module = importlib.import_module(module_path)
        func_class = getattr(class_module, class_name)
        return func_class._from_dict(d)

    @classmethod
    def _from_dict(cls, d: dict) -> Function:
        """Default deserialization: load the symbol indicated by the stored symbol_path"""
        assert 'path' in d and d['path'] is not None
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
