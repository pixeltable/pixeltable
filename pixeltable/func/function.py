from __future__ import annotations
from typing import Optional, Dict, Any, Callable, List, Union
from uuid import UUID
import inspect
import importlib


from .signature import Signature
from .globals import resolve_symbol
import pixeltable.type_system as ts
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
    SPECIAL_PARAM_NAMES = ['group_by', 'order_by']

    class Metadata:
        def __init__(self, signature: Signature, is_agg: bool, is_library_fn: bool):
            self.signature = signature
            self.is_agg = is_agg
            self.is_library_fn = is_library_fn
            # the following are set externally
            self.fqn: Optional[str] = None  # fully-qualified name
            self.src: str = ''  # source code shown in list()
            self.requires_order_by = False
            self.allows_std_agg = False
            self.allows_window = False

        def as_dict(self) -> Dict[str, Any]:
            # we leave out fqn, which is reconstructed externally
            return {
                'signature': self.signature.as_dict(),
                'is_agg': self.is_agg, 'is_library_fn': self.is_library_fn, 'src': self.src,
                'requires_order_by': self.requires_order_by, 'allows_std_agg': self.allows_std_agg,
                'allows_window': self.allows_window,
            }

        @classmethod
        def from_dict(cls, d: Dict[str, Any]) -> Function.Metadata:
            result = cls(Signature.from_dict(d['signature']), d['is_agg'], d['is_library_fn'])
            result.requires_order_by = d['requires_order_by']
            result.allows_std_agg = d['allows_std_agg']
            result.allows_window = d['allows_window']
            if 'src' in d:
                result.src = d['src']
            return result


    def __init__(
            self, md: Function.Metadata, id: Optional[UUID] = None,
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
            return
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

    @classmethod
    def _create_signature(
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

    @classmethod
    def make_function(cls, return_type: ts.ColumnType, param_types: List[ts.ColumnType], eval_fn: Callable) -> Function:
        assert eval_fn is not None
        signature = cls._create_signature(eval_fn, False, param_types, return_type)
        md = cls.Metadata(signature, False, False)
        try:
            md.src = inspect.getsource(eval_fn)
        except OSError as e:
            pass
        return Function(md, eval_fn=eval_fn)

    @classmethod
    def make_aggregate_function(
            cls, return_type: ts.ColumnType, param_types: List[ts.ColumnType],
            init_fn: Callable, update_fn: Callable, value_fn: Callable,
            requires_order_by: bool = False, allows_std_agg: bool = False, allows_window: bool = False
    ) -> Function:
        assert init_fn is not None and update_fn is not None and value_fn is not None
        signature = cls._create_signature(update_fn, True, param_types, return_type)
        md = cls.Metadata(signature, True, False)
        md.requires_order_by = requires_order_by
        md.allows_std_agg = allows_std_agg
        md.allows_window = allows_window
        try:
            md.src = (
                f'init:\n{inspect.getsource(init_fn)}\n\n'
                f'update:\n{inspect.getsource(update_fn)}\n\n'
                f'value:\n{inspect.getsource(value_fn)}\n'
            )
        except OSError as e:
            pass
        return Function(md, init_fn=init_fn, update_fn=update_fn, value_fn=value_fn)

    @classmethod
    def make_library_function(
            cls, return_type: Union[ts.ColumnType, Callable], param_types: List[ts.ColumnType], module_name: str, eval_symbol: str
    ) -> Function:
        assert module_name is not None and eval_symbol is not None
        eval_fn = resolve_symbol(module_name, eval_symbol)
        signature = cls._create_signature(eval_fn, False, param_types, return_type, check_params=True)
        md = cls.Metadata(signature, False, True)
        return Function(md, module_name=module_name, eval_symbol=eval_symbol)

    @classmethod
    def make_library_aggregate_function(
            cls, return_type: ts.ColumnType, param_types: List[ts.ColumnType],
            module_name: str, init_symbol: str, update_symbol: str, value_symbol: str,
            requires_order_by: bool = False, allows_std_agg: bool = False, allows_window: bool = False
    ) -> Function:
        assert module_name is not None and init_symbol is not None and update_symbol is not None \
               and value_symbol is not None
        update_fn = resolve_symbol(module_name, update_symbol)
        signature = cls._create_signature(update_fn, True, param_types, return_type)
        md = cls.Metadata(signature, True, True)
        md.requires_order_by = requires_order_by
        md.allows_std_agg = allows_std_agg
        md.allows_window = allows_window
        return Function(
            md, module_name=module_name, init_symbol=init_symbol, update_symbol=update_symbol,
            value_symbol=value_symbol)

    @classmethod
    def make_nos_function(
            cls, return_type: ts.ColumnType, param_types: List[ts.ColumnType], param_names: List[str], module_name: str
    ) -> Function:
        assert len(param_names) == len(param_types)
        signature = Signature(return_type, [(name, col_type) for name, col_type in zip(param_names, param_types)])
        md = cls.Metadata(signature, False, True)
        # construct inspect.Signature
        params = [
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for name, col_type in zip(param_names, param_types)
        ]
        py_signature = inspect.Signature(params)
        # we pass module_name to indicate that it's a library function
        return Function(md, module_name=module_name, py_signature=py_signature)

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
        return exprs.FunctionCall(
            self, bound_args.arguments,
            order_by_clause=[order_by_clause] if order_by_clause is not None else [],
            group_by_clause=[group_by_clause] if group_by_clause is not None else [])

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
        assert 'id' in d
        assert 'fqn' in d
        from .function_registry import FunctionRegistry
        if d['id'] is not None:
            return FunctionRegistry.get().get_function(id=UUID(hex=d['id']))
        else:
            assert d['fqn'] is not None
            # this is a library function; make sure we have the module loaded
            module_name = '.'.join(d['fqn'].split('.')[:-1])
            _ = importlib.import_module(module_name)
            return FunctionRegistry.get().get_function(fqn=d['fqn'])

