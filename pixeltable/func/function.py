from __future__ import annotations

import abc
import importlib
import inspect
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import pixeltable
import pixeltable.exceptions as excs
from .signature import Signature


class Function(abc.ABC):
    """Base class for Pixeltable's function interface.

    A function in Pixeltable is an object that has a signature and implements __call__().
    This base class provides a default serialization mechanism for Function instances provided by Python modules,
    via the member self_path.
    """

    def __init__(self, signature: Signature, py_signature: inspect.Signature, self_path: Optional[FunctionReference] = None):
        self.signature = signature
        self.py_signature = py_signature
        self.self_path = self_path  # fully-qualified path to self

    @property
    def name(self) -> str:
        assert self.self_path is not None
        return self.self_path.qualname.split('.')[-1]

    @property
    def display_name(self) -> str:
        if self.self_path is None:
            return '<anonymous>'
        if self.self_path.module.startswith('pixeltable.functions.'):
            module_display_name = self.self_path.module.lstrip('pixeltable.functions.')
        else:
            module_display_name = self.self_path.module
        return f'{module_display_name}.{self.self_path.qualname}'

    def help_str(self) -> str:
        return self.display_name + str(self.signature)

    def __call__(self, *args: object, **kwargs: object) -> 'pixeltable.exprs.Expr':
        from pixeltable import exprs
        bound_args = self.py_signature.bind(*args, **kwargs)
        self.validate_call(bound_args.arguments)
        return exprs.FunctionCall(self, bound_args.arguments)

    def validate_call(self, bound_args: Dict[str, Any]) -> None:
        """Override this to do custom validation of the arguments"""
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
        return {'ref': self.self_path.as_dict()}

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
        assert 'ref' in d and d['ref'] is not None
        instance = FunctionReference.from_dict(d['ref']).resolve()
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


@dataclass(frozen=True)
class FunctionReference:
    module: str
    qualname: str

    @property
    def full_path(self) -> str:
        return f'{self.module}.{self.qualname}'

    def as_dict(self) -> dict:
        return {'module': self.module, 'qualname': self.qualname}

    @classmethod
    def from_dict(cls, d: dict) -> FunctionReference:
        assert 'module' in d and 'qualname' in d
        return FunctionReference(module=d['module'], qualname=d['qualname'])

    def resolve(self) -> Optional[object]:
        module = importlib.import_module(self.module)
        path_elems = self.qualname.split('.')
        obj = module
        for el in path_elems:
            obj = getattr(obj, el)
        return obj

    def validate(self) -> None:
        path_elems = self.qualname.split('.')
        if any(el == '<locals>' for el in path_elems):
            raise excs.Error(
                f'{self.qualname}(): nested functions are not supported. '
                'Move the function to the module level or into a class.'
            )
        if any(not el.isidentifier() for el in path_elems):
            raise excs.Error(
                f'{self.qualname}(): cannot resolve symbol path {self.module}.{self.qualname}. '
                'Move the function to the module level or into a class.'
            )
