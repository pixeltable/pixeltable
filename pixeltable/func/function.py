from __future__ import annotations

import abc
import importlib
import inspect
from typing import Optional, Dict, Any, Tuple, Callable

import pixeltable
import pixeltable.type_system as ts
from .globals import resolve_symbol
from .signature import Signature


class Function(abc.ABC):
    """Base class for Pixeltable's function interface.

    A function in Pixeltable is an object that has a signature and implements __call__().
    This base class provides a default serialization mechanism for Function instances provided by Python modules,
    via the member self_path.
    """

    def __init__(
            self, signature: Signature, py_signature: inspect.Signature, self_path: Optional[str] = None
    ):
        self.signature = signature
        self.py_signature = py_signature
        self.self_path = self_path  # fully-qualified path to self
        self._conditional_return_type: Optional[Callable[..., ts.ColumnType]] = None

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

    def help_str(self) -> str:
        return self.display_name + str(self.signature)

    def __call__(self, *args: Any, **kwargs: Any) -> 'pixeltable.exprs.Expr':
        from pixeltable import exprs
        bound_args = self.py_signature.bind(*args, **kwargs)
        self.validate_call(bound_args.arguments)
        return exprs.FunctionCall(self, bound_args.arguments)

    def validate_call(self, bound_args: Dict[str, Any]) -> None:
        """Override this to do custom validation of the arguments"""
        pass

    def call_return_type(self, kwargs: dict[str, Any]) -> ts.ColumnType:
        """Return the type of the value returned by calling this function with the given arguments"""
        if self._conditional_return_type is None:
            return self.signature.return_type
        bound_args = self.py_signature.bind(**kwargs)
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

    @abc.abstractmethod
    def exec(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the function with the given arguments and return the result."""
        pass

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.self_path == other.self_path

    def source(self) -> None:
        """Print source code"""
        print('source not available')

    def as_dict(self) -> Dict:
        """
        Return a serialized reference to the instance that can be passed to json.dumps() and converted back
        to an instance with from_dict().
        Subclasses can override _as_dict().
        """
        classpath = f'{self.__class__.__module__}.{self.__class__.__qualname__}'
        return {'_classpath': classpath, **self._as_dict()}

    def _as_dict(self) -> Dict:
        """Default serialization: store the path to self (which includes the module path)"""
        assert self.self_path is not None
        return {'path': self.self_path}

    @classmethod
    def from_dict(cls, d: Dict) -> Function:
        """
        Turn dict that was produced by calling as_dict() into an instance of the correct Function subclass.
        """
        assert '_classpath' in d
        module_path, class_name = d['_classpath'].rsplit('.', 1)
        class_module = importlib.import_module(module_path)
        func_class = getattr(class_module, class_name)
        return func_class._from_dict(d)

    @classmethod
    def _from_dict(cls, d: Dict) -> Function:
        """Default deserialization: load the symbol indicated by the stored symbol_path"""
        assert 'path' in d and d['path'] is not None
        instance = resolve_symbol(d['path'])
        assert isinstance(instance, Function)
        return instance

    def to_store(self) -> Tuple[Dict, bytes]:
        """
        Serialize the function to a format that can be stored in the Pixeltable store
        Returns:
            - a dict that can be passed to json.dumps()
            - additional binary data
        Only Function subclasses that can be stored need to override this.
        """
        raise NotImplementedError()

    @classmethod
    def from_store(cls, name: Optional[str], md: Dict, binary_obj: bytes) -> Function:
        """
        Create a Function instance from the serialized representation returned by to_store()
        """
        raise NotImplementedError()
