from __future__ import annotations
import sys
import types
from typing import Optional, Callable, Dict, List, Any, Tuple, Union
from types import ModuleType
import importlib
import sqlalchemy as sql
import cloudpickle
import inspect
import logging
from uuid import UUID
import dataclasses

import nos

from pixeltable.type_system import ColumnType, JsonType
from pixeltable.metadata import schema
from pixeltable.env import Env
from pixeltable import exceptions as exc
import pixeltable


_logger = logging.getLogger('pixeltable')

def _resolve_symbol(module_name: str, symbol: str) -> object:
    module = importlib.import_module(module_name)
    obj = module
    for el in symbol.split('.'):
        obj = getattr(obj, el)
    return obj


class Signature:
    """
    Return type:
    - most functions will have a fixed return type, which is specified directly
    - some functions will have a return type that depends on the argument values;
      ex.: PIL.Image.Image.resize() returns an image with dimensions specified as a parameter
    - in the latter case, the 'return_type' field is a function that takes the bound arguments and returns the
      return type; if no bound arguments are specified, a generic return type is returned (eg, ImageType() without a
      size)
    """
    def __init__(
            self,
            return_type: Union[ColumnType, Callable[[Dict[str, Any]], ColumnType]],
            parameters: List[Tuple[str, ColumnType]]):
        self.return_type = return_type
        # we rely on the ordering guarantee of dicts in Python >=3.7
        self.parameters = {param_name: param_type for param_name, param_type in parameters}
        self.parameter_types_by_pos = [param_type for _, param_type in parameters]

    def get_return_type(self, bound_args: Optional[Dict[str, Any]] = None) -> ColumnType:
        if isinstance(self.return_type, ColumnType):
            return self.return_type
        return self.return_type(bound_args)

    def as_dict(self) -> Dict[str, Any]:
        result = {
            'return_type': self.get_return_type().as_dict(),
        }
        if self.parameters is not None:
            result['parameters'] = [[name, col_type.as_dict()] for name, col_type in self.parameters.items()]
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Signature:
        parameters = [(p[0], ColumnType.from_dict(p[1])) for p in d['parameters']]
        return cls(ColumnType.from_dict(d['return_type']), parameters)

    def __eq__(self, other: Signature) -> bool:
        if self.get_return_type() != other.get_return_type() or (self.parameters is None) != (other.parameters is None):
            return False
        if self.parameters is None:
            return True
        if len(self.parameters) != len(other.parameters):
            return False
        # ignore the parameter name
        for param_type, other_param_type in zip(self.parameter_types_by_pos, other.parameter_types_by_pos):
            if param_type != other_param_type:
                return False
        return True

    def __str__(self) -> str:
        return (
            f'({", ".join([name + ": " + str(col_type) for name, col_type in self.parameters.items()])})'
            f'-> {str(self.get_return_type())}'
        )


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
                self.eval_fn = _resolve_symbol(module_name, eval_symbol)
            if init_symbol is not None:
                self.init_fn = _resolve_symbol(module_name, init_symbol)
            if update_symbol is not None:
                self.update_fn = _resolve_symbol(module_name, update_symbol)
            if value_symbol is not None:
                self.value_fn = _resolve_symbol(module_name, value_symbol)

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
            cls, c: Callable, is_agg: bool, param_types: List[ColumnType], return_type: Union[ColumnType, Callable],
            check_params: bool = True
    ) -> Signature:
        if param_types is None:
            return Signature(return_type, None)
        sig = inspect.signature(c)
        param_names = list(sig.parameters.keys())
        if is_agg:
            param_names = param_names[1:]  # the first parameter is the state returned by init()
        if check_params and len(param_names) != len(param_types):
            raise exc.Error(
                f"The number of parameters of '{getattr(c, '__name__', 'anonymous')}' is not the same as "
                f"the number of provided parameter types: "
                f"{len(param_names)} ({', '.join(param_names)}) vs "
                f"{len(param_types)} ({', '.join([str(t) for t in param_types])})")
        # check parameters for name collisions and default value compatibility
        for idx, param_name in enumerate(param_names):
            if param_name in cls.SPECIAL_PARAM_NAMES:
                raise exc.Error(f"'{param_name}' is a reserved parameter name")
            default_val = sig.parameters[param_name].default
            if default_val == inspect.Parameter.empty or default_val is None:
                continue
            try:
                param_types[idx].validate_literal(default_val)
            except TypeError as e:
                raise exc.Error(f'Default value for parameter {param_name}: {str(e)}')

        parameters = [(param_names[i], param_types[i]) for i in range(len(param_names))]
        return Signature(return_type, parameters)

    @classmethod
    def make_function(cls, return_type: ColumnType, param_types: List[ColumnType], eval_fn: Callable) -> Function:
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
            cls, return_type: ColumnType, param_types: List[ColumnType],
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
            cls, return_type: Union[ColumnType, Callable], param_types: List[ColumnType], module_name: str, eval_symbol: str
    ) -> Function:
        assert module_name is not None and eval_symbol is not None
        eval_fn = _resolve_symbol(module_name, eval_symbol)
        signature = cls._create_signature(eval_fn, False, param_types, return_type, check_params=True)
        md = cls.Metadata(signature, False, True)
        return Function(md, module_name=module_name, eval_symbol=eval_symbol)

    @classmethod
    def make_library_aggregate_function(
            cls, return_type: ColumnType, param_types: List[ColumnType],
            module_name: str, init_symbol: str, update_symbol: str, value_symbol: str,
            requires_order_by: bool = False, allows_std_agg: bool = False, allows_window: bool = False
    ) -> Function:
        assert module_name is not None and init_symbol is not None and update_symbol is not None \
               and value_symbol is not None
        update_fn = _resolve_symbol(module_name, update_symbol)
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
            cls, return_type: ColumnType, param_types: List[ColumnType], param_names: List[str], module_name: str
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

        order_by_expr: Optional[exprs.Expr] = None
        if 'order_by' in kwargs:
            if self.requires_order_by:
                raise exc.Error(
                    f'Order_by invalid, this function requires the first argument to be the ordering expression')
            if not self.is_aggregate:
                raise exc.Error(f'Order_by invalid with a non-aggregate function')
            if not self.allows_window:
                raise exc.Error(f'Order_by invalid with an aggregate function that does not allow windows')
            order_by_expr = kwargs['order_by']
            if not isinstance(order_by_expr, exprs.Expr):
                raise exc.Error(
                    f'order_by argument needs to be a Pixeltable expression, but instead is a {type(order_by_expr)}')
            del kwargs['order_by']
        elif self.requires_order_by:
            # the first argument is the order-by expr
            if len(args) == 0:
                raise exc.Error(f'Function requires an ordering expression as its first argument')
            order_by_expr = args[0]
            if not isinstance(order_by_expr, exprs.Expr):
                raise exc.Error(
                    f'The first argument needs to be a Pixeltable expression, but instead is a {type(order_by_expr)}')
            # don't pass the first parameter on, the Function doesn't get to see it
            args = args[1:]

        group_by_expr: Optional[exprs.Expr] = None
        if 'group_by' in kwargs:
            if not self.is_aggregate:
                raise exc.Error(f'Group_by invalid with a non-aggregate function')
            if not self.allows_window:
                raise exc.Error(f'Group_by invalid with an aggregate function that does not allow windows')
            group_by_expr = kwargs['group_by']
            if not isinstance(group_by_expr, exprs.Expr):
                raise exc.Error(
                    f'group_by argument needs to be a Pixeltable expression, but instead is a {type(order_by_expr)}')
            del kwargs['group_by']

        bound_args = self.py_signature.bind(*args, **kwargs)
        return exprs.FunctionCall(
            self, bound_args.arguments,
            order_by_exprs=[order_by_expr] if order_by_expr is not None else [],
            group_by_exprs=[group_by_expr] if group_by_expr is not None else [])

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
            raise exc.Error(f'source() not valid for library functions: {self.display_name}')
        if self.md.src == '':
            print('sources not available')
        print(self.md.src)

    def as_dict(self) -> Dict:
        if not self.is_library_function and self.id is None:
            # this is not a library function and the absence of an assigned id indicates that it's not in the store yet
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
        if d['id'] is not None:
            return FunctionRegistry.get().get_function(id=UUID(hex=d['id']))
        else:
            assert d['fqn'] is not None
            # this is a library function; make sure we have the module loaded
            module_name = '.'.join(d['fqn'].split('.')[:-1])
            _ = importlib.import_module(module_name)
            return FunctionRegistry.get().get_function(fqn=d['fqn'])


def function(*, return_type: ColumnType, param_types: List[ColumnType]) -> Callable:
    """Returns decorator to create a Function from a function definition.

    Example:
        >>> @pt.function(param_types=[pt.IntType()], return_type=pt.IntType())
        ... def my_function(x):
        ...    return x + 1
    """
    def decorator(fn: Callable) -> Function:
        return Function.make_function(return_type, param_types, fn)
    return decorator


class FunctionRegistry:
    """
    A central registry for all Functions. Handles interactions with the backing store.
    Function are loaded from the store on demand.
    """
    _instance: Optional[FunctionRegistry] = None

    @classmethod
    def get(cls) -> FunctionRegistry:
        if cls._instance is None:
            cls._instance = FunctionRegistry()
            #cls._instance.register_nos_functions()
        return cls._instance

    def __init__(self):
        self.stored_fns_by_id: Dict[UUID, Function] = {}
        self.library_fns: Dict[str, Function] = {}  # fqn -> Function
        self.has_registered_nos_functions = False
        self.nos_functions: Dict[str, nos.common.ModelSpec] = {}

    def clear_cache(self) -> None:
        """
        Useful during testing
        """
        self.stored_fns_by_id: Dict[UUID, Function] = {}

    def register_module(self, module: ModuleType) -> None:
        """Register all Functions in a module"""
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, Function):
                fqn = f'{module.__name__}.{name}'  # fully-qualified name
                self.library_fns[fqn] = obj
                obj.md.fqn = fqn

    def register_function(self, module_name: str, fn_name: str, fn: Function) -> None:
        fqn = f'{module_name}.{fn_name}'  # fully-qualified name
        self.library_fns[fqn] = fn
        fn.md.fqn = fqn

    def get_library_fn(self, fqn: str) -> Function:
        return self.library_fns[fqn]

    def _convert_nos_signature(self, sig: nos.common.spec.FunctionSignature) -> Tuple[ColumnType, List[ColumnType]]:
        if len(sig.get_outputs_spec()) > 1:
            return_type = JsonType()
        else:
            return_type = ColumnType.from_nos(list(sig.get_outputs_spec().values())[0])
        param_types: List[ColumnType] = []
        for _, type_info in sig.get_inputs_spec().items():
            # if there are multiple input shapes we leave them out of the ColumnType and deal with them in FunctionCall
            if isinstance(type_info, list):
                param_types.append(ColumnType.from_nos(type_info[0], ignore_shape=True))
            else:
                param_types.append(ColumnType.from_nos(type_info, ignore_shape=False))
        return return_type, param_types

    def register_nos_functions(self) -> None:
        """Register all models supported by the NOS backend as library functions"""
        if self.has_registered_nos_functions:
            return
        self.has_registered_nos_functions = True
        models = Env.get().nos_client.ListModels()
        model_info = [Env.get().nos_client.GetModelInfo(model) for model in models]
        model_info.sort(key=lambda info: info.task.value)

        prev_task = ''
        new_modules: List[types.ModuleType] = []
        pt_module: Optional[types.ModuleType] = None
        for info in model_info:
            if info.task.value != prev_task:
                # we construct one submodule of pixeltable.functions per task
                module_name = f'pixeltable.functions.{info.task.name.lower()}'
                pt_module = types.ModuleType(module_name)
                pt_module.__package__ = 'pixeltable.functions'
                new_modules.append(pt_module)
                sys.modules[module_name] = pt_module
                prev_task = info.task.value

            # add a Function for this model to the module
            model_id = info.name.replace("/", "_").replace("-", "_")
            return_type, param_types = self._convert_nos_signature(info.signature)
            pt_func = Function.make_nos_function(
                return_type, param_types, list(info.signature.get_inputs_spec().keys()), module_name)
            setattr(pt_module, model_id, pt_func)
            fqn = f'{module_name}.{model_id}'
            self.nos_functions[fqn] = info

        for module in new_modules:
            self.register_module(module)

    def get_nos_info(self, fn: Function) -> Optional[nos.common.ModelSpec]:
        return self.nos_functions.get(fn.md.fqn)

    def list_functions(self) -> List[Function.Metadata]:
        # retrieve Function.Metadata data for all existing stored functions from store directly
        # (self.stored_fns_by_id isn't guaranteed to contain all functions)
        # TODO: have the client do this, once the client takes over the Db functionality
        # stmt = sql.select(
        #         schema.Function.name, schema.Function.md,
        #         schema.Db.name, schema.Dir.path, sql_func.length(schema.Function.init_obj))\
        #     .where(schema.Function.db_id == schema.Db.id)\
        #     .where(schema.Function.dir_id == schema.Dir.id)
        # stored_fn_md: List[Function.Metadata] = []
        # with Env.get().engine.begin() as conn:
        #     rows = conn.execute(stmt)
        #     for name, md_dict, db_name, dir_path, init_obj_len in rows:
        #         md = Function.Metadata.from_dict(md_dict)
        #         md.fqn = f'{db_name}{"." + dir_path if dir_path != "" else ""}.{name}'
        #         stored_fn_md.append(md)
        return [fn.md for fn in self.library_fns.values()]

    def get_function(self, *, id: Optional[UUID] = None, fqn: Optional[str] = None) -> Function:
        assert (id is not None) != (fqn is not None)
        if id is not None:
            if id not in self.stored_fns_by_id:
                stmt = sql.select(
                        schema.Function.md, schema.Function.eval_obj, schema.Function.init_obj,
                        schema.Function.update_obj, schema.Function.value_obj) \
                    .where(schema.Function.id == id)
                with Env.get().engine.begin() as conn:
                    rows = conn.execute(stmt)
                    row = next(rows)
                    schema_md = schema.md_from_dict(schema.FunctionMd, row[0])
                    name = schema_md.name
                    md = Function.Metadata.from_dict(schema_md.md)
                    # md.fqn is set by caller
                    eval_fn = cloudpickle.loads(row[1]) if row[1] is not None else None
                    # TODO: are these checks needed?
                    if row[1] is not None and eval_fn is None:
                        raise exc.Error(f'Could not load eval_fn for function {name}')
                    init_fn = cloudpickle.loads(row[2]) if row[2] is not None else None
                    if row[2] is not None and init_fn is None:
                        raise exc.Error(f'Could not load init_fn for aggregate function {name}')
                    update_fn = cloudpickle.loads(row[3]) if row[3] is not None else None
                    if row[3] is not None and update_fn is None:
                        raise exc.Error(f'Could not load update_fn for aggregate function {name}')
                    value_fn = cloudpickle.loads(row[4]) if row[4] is not None else None
                    if row[4] is not None and value_fn is None:
                        raise exc.Error(f'Could not load value_fn for aggregate function {name}')

                    func = Function(
                        md, id=id,
                        eval_fn=eval_fn, init_fn=init_fn, update_fn=update_fn, value_fn=value_fn)
                    _logger.info(f'Loaded function {name} from store')
                    self.stored_fns_by_id[id] = func
            assert id in self.stored_fns_by_id
            return self.stored_fns_by_id[id]
        else:
            # this is an already-registered library function
            assert fqn in self.library_fns, f'{fqn} not found'
            return self.library_fns[fqn]

    def get_type_methods(self, name: str, base_type: ColumnType.Type) -> List[Function]:
        return [
            fn for fn in self.library_fns.values()
            if fn.md.fqn.endswith('.' + name)
               and fn.md.signature.parameter_types_by_pos[0].type_enum == base_type
        ]

    def create_function(self, fn: Function, dir_id: Optional[UUID] = None, name: Optional[str] = None) -> None:
        with Env.get().engine.begin() as conn:
            _logger.debug(f'Pickling function {name}')
            eval_fn_str = cloudpickle.dumps(fn.eval_fn) if fn.eval_fn is not None else None
            init_fn_str = cloudpickle.dumps(fn.init_fn) if fn.init_fn is not None else None
            update_fn_str = cloudpickle.dumps(fn.update_fn) if fn.update_fn is not None else None
            value_fn_str = cloudpickle.dumps(fn.value_fn) if fn.value_fn is not None else None
            total_size = \
                (len(eval_fn_str) if eval_fn_str is not None else 0) + \
                (len(init_fn_str) if init_fn_str is not None else 0) + \
                (len(update_fn_str) if update_fn_str is not None else 0) + \
                (len(value_fn_str) if value_fn_str is not None else 0)
            _logger.debug(f'Pickled function {name} ({total_size} bytes)')

            schema_md = schema.FunctionMd(name=name, md=fn.md.as_dict())
            res = conn.execute(
                sql.insert(schema.Function.__table__)
                    .values(
                        dir_id=dir_id, md=dataclasses.asdict(schema_md),
                        eval_obj=eval_fn_str, init_obj=init_fn_str, update_obj=update_fn_str, value_obj=value_fn_str))
            fn.id = res.inserted_primary_key[0]
            self.stored_fns_by_id[fn.id] = fn
            _logger.info(f'Created function {name} in store')

    def update_function(self, id: UUID, new_fn: Function) -> None:
        """
        Updates the callables for the Function with the given id in the store and in the cache, if present.
        """
        assert not new_fn.is_library_function
        with Env.get().engine.begin() as conn:
            updates = {}
            if new_fn.eval_fn is not None:
                updates[schema.Function.eval_obj] = cloudpickle.dumps(new_fn.eval_fn)
            if new_fn.init_fn is not None:
                updates[schema.Function.init_obj] = cloudpickle.dumps(new_fn.init_fn)
            if new_fn.update_fn is not None:
                updates[schema.Function.update_obj] = cloudpickle.dumps(new_fn.update_fn)
            if new_fn.value_fn is not None:
                updates[schema.Function.value_obj] = cloudpickle.dumps(new_fn.value_fn)
            conn.execute(
                sql.update(schema.Function.__table__)
                    .values(updates)
                    .where(schema.Function.id == id))
            _logger.info(f'Updated function {new_fn.md.fqn} (id={id}) in store')
        if id in self.stored_fns_by_id:
            if new_fn.eval_fn is not None:
                self.stored_fns_by_id[id].eval_fn = new_fn.eval_fn
            if new_fn.init_fn is not None:
                self.stored_fns_by_id[id].init_fn = new_fn.init_fn
            if new_fn.update_fn is not None:
                self.stored_fns_by_id[id].update_fn = new_fn.update_fn
            if new_fn.value_fn is not None:
                self.stored_fns_by_id[id].value_fn = new_fn.value_fn

    def delete_function(self, id: UUID) -> None:
        assert id is not None
        with Env.get().engine.begin() as conn:
            conn.execute(
                sql.delete(schema.Function.__table__)
                    .where(schema.Function.id == id))
            _logger.info(f'Deleted function with id {id} from store')
