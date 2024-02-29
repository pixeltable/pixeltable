from __future__ import annotations
from typing import Optional, Dict, Any, Callable
from uuid import UUID
import inspect
import importlib


from .function_md import FunctionMd
from .globals import resolve_symbol
import pixeltable.exceptions as excs

class Function:
    """Wrapper for a Python function.

    A Function's executable function is specified either directly or as module/symbol.
    In the former case, the function needs to be pickled and stored for serialization.
    In the latter case, the executable function is resolved in init().
    self.id is only set for non-module functions that are in the backing store.
    requires_order_by: if True, the first parameter to an aggregate function defines the order in which the function
    sees rows in update()
    allows_std_agg: if True, the aggregate function can be used as a standard aggregate function w/o a window
    allows_window: if True, the aggregate function can be used with a window
    """

    def __init__(
            self, md: FunctionMd, id: Optional[UUID] = None,
            module_name: Optional[str] = None, eval_symbol: Optional[str] = None,
            init_symbol: Optional[str] = None, update_symbol: Optional[str] = None, value_symbol: Optional[str] = None,
            eval_fn: Optional[Callable] = None,
            init_fn: Optional[Callable] = None, update_fn: Optional[Callable] = None,
            value_fn: Optional[Callable] = None,
            py_signature: Optional[inspect.Signature] = None
    ):
        self.id = id
        self.module_name = module_name
        self.eval_symbol = eval_symbol
        self.eval_fn = eval_fn
        self.init_symbol = init_symbol
        self.init_fn = init_fn
        self.update_symbol = update_symbol
        self.update_fn = update_fn
        self.value_symbol = value_symbol
        self.value_fn = value_fn
        self.md = md

        if module_name is not None:
            # resolve symbols
            if eval_symbol is not None:
                self.eval_fn = resolve_symbol(module_name, eval_symbol)
            if init_symbol is not None:
                self.init_fn = resolve_symbol(module_name, init_symbol)
            if update_symbol is not None:
                self.update_fn = resolve_symbol(module_name, update_symbol)
            if value_symbol is not None:
                self.value_fn = resolve_symbol(module_name, value_symbol)

        # NOS functions don't have an eval_fn and specify their Python signature directly
        if py_signature is not None:
            self.py_signature = py_signature
        else :
            # for everything else, we infer the Python signature
            assert (md.is_agg and self.update_fn is not None) or (not md.is_agg and self.eval_fn is not None)
            if md.is_agg:
                # the Python signature is the signature of 'update', but without self
                sig = inspect.signature(self.update_fn)
                self.py_signature = \
                    inspect.Signature(list(sig.parameters.values())[1:], return_annotation=sig.return_annotation)
            else:
                self.py_signature = inspect.signature(self.eval_fn)

    @property
    def name(self) -> bool:
        return self.md.fqn.split('.')[-1]

    @property
    def requires_order_by(self) -> bool:
        return self.md.requires_order_by

    @property
    def allows_std_agg(self) -> bool:
        return self.md.allows_std_agg

    @property
    def allows_window(self) -> bool:
        return self.md.allows_window

    @property
    def is_aggregate(self) -> bool:
        return self.init_fn is not None

    @property
    def is_library_function(self) -> bool:
        return self.module_name is not None

    @property
    def display_name(self) -> str:
        if self.md.fqn is None:
            if self.eval_fn is not None:
                return self.eval_fn.__name__
            else:
                return ''
        ptf_prefix = 'pixeltable.functions.'
        if self.md.fqn.startswith(ptf_prefix):
            return self.md.fqn[len(ptf_prefix):]
        return self.md.fqn

    def help_str(self) -> str:
        res = self.display_name + str(self.md.signature)
        if self.eval_fn is not None:
            res += '\n\n' + inspect.getdoc(self.eval_fn)
        elif self.update_fn is not None:
            res += '\n\n' + inspect.getdoc(self.update_fn)
        return res

    def __call__(self, *args: object, **kwargs: object) -> 'pixeltable.exprs.FunctionCall':
        from pixeltable import exprs

        # perform semantic analysis of special parameters 'order_by' and 'group_by'
        order_by_clause: Optional[Any] = None
        if 'order_by' in kwargs:
            if self.requires_order_by:
                raise excs.Error(
                    f'Order_by invalid, this function requires the first argument to be the ordering expression')
            if not self.is_aggregate:
                raise excs.Error(f'Order_by invalid with a non-aggregate function')
            if not self.allows_window:
                raise excs.Error(f'Order_by invalid with an aggregate function that does not allow windows')
            order_by_clause = kwargs['order_by']
            del kwargs['order_by']
        elif self.requires_order_by:
            # the first argument is the order-by expr
            if len(args) == 0:
                raise excs.Error(f'Function requires an ordering expression as its first argument')
            order_by_clause = args[0]
            if not isinstance(order_by_clause, exprs.Expr):
                raise excs.Error(
                    f'The first argument needs to be a Pixeltable expression, but instead is a {type(order_by_clause)}')
            # don't pass the first parameter on, the Function doesn't get to see it
            args = args[1:]

        group_by_clause: Optional[Any] = None
        if 'group_by' in kwargs:
            if not self.is_aggregate:
                raise excs.Error(f'Group_by invalid with a non-aggregate function')
            if not self.allows_window:
                raise excs.Error(f'Group_by invalid with an aggregate function that does not allow windows')
            group_by_clause = kwargs['group_by']
            del kwargs['group_by']

        bound_args = self.py_signature.bind(*args, **kwargs)
        self.verify_call(bound_args.arguments)
        return exprs.FunctionCall(
            self, bound_args.arguments,
            order_by_clause=[order_by_clause] if order_by_clause is not None else [],
            group_by_clause=[group_by_clause] if group_by_clause is not None else [])

    def verify_call(self, bound_args: Dict[str, Any]) -> None:
        """Override this to do custom verification of the arguments"""
        pass

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if self.is_library_function != other.is_library_function:
            return False
        if self.is_library_function:
            # this is a library function, which is uniquely identified by its fqn
            return self.md.fqn == other.md.fqn
        return self.eval_fn == other.eval_fn and self.init_fn == other.init_fn \
               and self.update_fn == other.update_fn and self.value_fn == other.value_fn

    def source(self) -> None:
        """
        Print source code
        """
        if self.is_library_function:
            raise excs.Error(f'source() not valid for library functions: {self.display_name}')
        if self.md.src == '':
            print('sources not available')
        print(self.md.src)

    def as_dict(self) -> Dict:
        """
        Turn Function object into a dict that can be passed to json.dumps().
        Subclasses override _as_dict().
        """
        return {
            '_classname': self.__class__.__name__,
            **self._as_dict(),
        }

    def _as_dict(self) -> Dict:
        if not self.is_library_function and self.id is None:
            # this is not a library function and the absence of an assigned id indicates that it's not in the store yet
            from .function_registry import FunctionRegistry
            FunctionRegistry.get().create_function(self)
            assert self.id is not None
        if self.id is not None:
            # this is a stored function, we only need the id to reconstruct it
            return {'id': self.id.hex, 'fqn': None}
        else:
            # this is a library function, the fqn serves as the id
            return {'id': None, 'fqn': self.md.fqn}

    @classmethod
    def from_dict(cls, d: Dict) -> Function:
        """
        Turn dict that was produced by calling as_dict() into an instance of the correct Function subclass.
        """
        assert '_classname' in d
        func_module = importlib.import_module(cls.__module__.rsplit('.', 1)[0])
        type_class = getattr(func_module, d['_classname'])
        return type_class._from_dict(d)

    @classmethod
    def _from_dict(cls, d: Dict) -> Function:
        assert 'id' in d
        assert 'fqn' in d
        from .function_registry import FunctionRegistry
        if d['id'] is not None:
            return FunctionRegistry.get().get_function(id=UUID(hex=d['id']))
        else:
            assert d['fqn'] is not None
            # this is a library function; make sure we have the module loaded
            fqn_elems = d['fqn'].split('.')
            module_name = '.'.join(fqn_elems[:-1])
            fn = resolve_symbol(module_name, fqn_elems[-1])
            assert isinstance(fn, Function)
            return fn
