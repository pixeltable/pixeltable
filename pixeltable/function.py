from __future__ import annotations
import sys
import types
from typing import Optional, Callable, Dict, List, Any, Tuple
import importlib
import sqlalchemy as sql
from sqlalchemy.sql.expression import func as sql_func
import cloudpickle
import inspect
import logging

import nos

from pixeltable.type_system import ColumnType, JsonType
from pixeltable import store
from pixeltable.env import Env
from pixeltable import exceptions as exc
import pixeltable


_logger = logging.getLogger('pixeltable')

class Signature:
    def __init__(self, return_type: ColumnType, parameters: Optional[List[Tuple[str, ColumnType]]]):
        self.return_type = return_type
        self.parameters = parameters

    def as_dict(self) -> Dict[str, Any]:
        result = {
            'return_type': self.return_type.as_dict(),
        }
        if self.parameters is not None:
            result['parameters'] = [[p[0], p[1].as_dict()] for p in self.parameters]
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Signature:
        parameters = [(p[0], ColumnType.from_dict(p[1])) for p in d['parameters']]
        return cls(ColumnType.from_dict(d['return_type']), parameters)

    def __eq__(self, other: Signature) -> bool:
        if self.return_type != other.return_type or (self.parameters is None) != (other.parameters is None):
            return False
        if self.parameters is None:
            return True
        if len(self.parameters) != len(other.parameters):
            return False
        for i in range(len(self.parameters)):
            # TODO: ignore the parameter name?
            if self.parameters[i] != other.parameters[i]:
                return False
        return True

    def __str__(self) -> str:
        return f'({", ".join([p[0] + ": " + str(p[1]) for p in self.parameters])}) -> {str(self.return_type)}'


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
            self, md: Function.Metadata, id: Optional[int] = None,
            module_name: Optional[str] = None, eval_symbol: Optional[str] = None, init_symbol: Optional[str] = None,
            update_symbol: Optional[str] = None, value_symbol: Optional[str] = None,
            eval_fn: Optional[Callable] = None, init_fn: Optional[Callable] = None,
            update_fn: Optional[Callable] = None, value_fn: Optional[Callable] = None
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
                self.eval_fn = self._resolve_symbol(module_name, eval_symbol)
            if init_symbol is not None:
                self.init_fn = self._resolve_symbol(module_name, init_symbol)
            if update_symbol is not None:
                self.update_fn = self._resolve_symbol(module_name, update_symbol)
            if value_symbol is not None:
                self.value_fn = self._resolve_symbol(module_name, value_symbol)

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
            cls, c: Callable, is_agg: bool, param_types: List[ColumnType], return_type: ColumnType,
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
            cls, return_type: ColumnType, param_types: List[ColumnType], module_name: str, eval_symbol: str
    ) -> Function:
        assert module_name is not None and eval_symbol is not None
        eval_fn = cls._resolve_symbol(module_name, eval_symbol)
        signature = cls._create_signature(eval_fn, False, param_types, return_type, check_params=False)
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
        update_fn = cls._resolve_symbol(module_name, update_symbol)
        signature = cls._create_signature(update_fn, True, param_types, return_type)
        md = cls.Metadata(signature, True, True)
        md.requires_order_by = requires_order_by
        md.allows_std_agg = allows_std_agg
        md.allows_window = allows_window
        return Function(
            md, module_name=module_name,
            init_symbol=init_symbol, update_symbol=update_symbol, value_symbol=value_symbol)

    @classmethod
    def _resolve_symbol(cls, module_name: str, symbol: str) -> object:
        module = importlib.import_module(module_name)
        obj = module
        for el in symbol.split('.'):
            obj = getattr(obj, el)
        return obj

    @property
    def is_aggregate(self) -> bool:
        return self.init_fn is not None

    @property
    def is_library_function(self) -> bool:
        return self.module_name is not None

    @property
    def display_name(self) -> str:
        if self.md.fqn is None:
            return ''
        ptf_prefix = 'pixeltable.functions'
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

        return exprs.FunctionCall(
            self, args,
            order_by_exprs=[order_by_expr] if order_by_expr is not None else [],
            group_by_exprs=[group_by_expr] if group_by_expr is not None else [])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.md.signature == other.md.signature \
               and self.id == other.id and self.module_name == other.module_name \
               and self.eval_symbol == other.eval_symbol and self.init_symbol == other.init_symbol \
               and self.update_symbol == other.update_symbol and self.value_symbol == other.value_symbol \
               and self.eval_fn == other.eval_fn and self.init_fn == other.init_fn \
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
        if self.id is not None:
            # this is a stored function, we only need the id to reconstruct it
            return {'id': self.id}

        if not self.is_library_function and self.id is None:
            # this is not a library function and the absence of an assigned id indicates that it's not in the store yet
            FunctionRegistry.get().create_function(self)
            assert self.id is not None
        return {
            'id': self.id,
            'md': self.md.as_dict(),
            'module_name': self.module_name,
            'eval_symbol': self.eval_symbol,
            'init_symbol': self.init_symbol,
            'update_symbol': self.update_symbol,
            'value_symbol': self.value_symbol,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> Function:
        assert 'id' in d
        if d['id'] is not None:
            return FunctionRegistry.get().get_function(d['id'])
        else:
            assert 'module_name' in d
            assert 'eval_symbol' in d and 'init_symbol' in d and 'update_symbol' in d and 'value_symbol' in d
            md = cls.Metadata.from_dict(d['md'])
            return cls(
                md, module_name=d['module_name'], eval_symbol=d['eval_symbol'],
                init_symbol=d['init_symbol'], update_symbol=d['update_symbol'], value_symbol=d['value_symbol'])


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
        self.stored_fns_by_id: Dict[int, Function] = {}
        self.library_fns: Dict[str, Function] = {}  # fqn -> Function
        self.has_registered_nos_functions = False

    def clear_cache(self) -> None:
        """
        Useful during testing
        """
        self.stored_fns_by_id: Dict[int, Function] = {}

    def register_function(self, module_name: str, fn_name: str, fn: Function) -> None:
        fqn = f'{module_name}.{fn_name}'  # fully-qualified name
        self.library_fns[fqn] = fn
        fn.md.fqn = fqn

    def _convert_nos_signature(self, sig: nos.common.spec.FunctionSignature) -> Tuple[ColumnType, List[ColumnType]]:
        if len(sig.get_outputs_spec()) > 1:
            return_type = JsonType()
        else:
            return_type = ColumnType.from_nos(list(sig.get_outputs_spec().values())[0])
        param_types: List[ColumnType] = []
        for _, type_info in sig.get_inputs_spec().items():
            # TODO: deal with multiple input shapes
            if isinstance(type_info, list):
                type_info = type_info[0]
            param_types.append(ColumnType.from_nos(type_info))
        return return_type, param_types

    def register_nos_functions(self) -> None:
        """Register all models supported by the NOS backend as library functions"""
        if self.has_registered_nos_functions:
            return
        self.has_registered_nos_functions = True
        models = Env.get().nos_client.ListModels()
        model_info = [Env.get().nos_client.GetModelInfo(model) for model in models]
        model_info.sort(key=lambda info: info.task.value)

        def create_nos_udf(task: str, model_name: str, param_names: List[str]) -> Callable:
            def func(*args: Any) -> Any:
                kwargs = {param_name: val for param_name, val in zip(param_names, args)}
                result = Env.get().nos_client.Run(task=task, model_name=model_name, **kwargs)
                if len(result) == 1:
                    return list(result.values())[0]
                else:
                    return result

            return func

        prev_task = ''
        pt_module: Optional[types.ModuleType] = None
        for info in model_info:
            if info.task.value != prev_task:
                # we construct one submodule of pixeltable.functions per task
                module_name = f'pixeltable.functions.{info.task.name.lower()}'
                pt_module = types.ModuleType(module_name)
                pt_module.__package__ = 'pixeltable.functions'
                sys.modules[module_name] = pt_module
                prev_task = info.task.value

            # add a Function and its implementation for this model to the module
            model_id = info.name.replace("/", "_").replace("-", "_")
            eval_symbol = f'{model_id}_impl'
            inputs = info.signature.get_inputs_spec()
            # create the eval function
            setattr(pt_module, eval_symbol, create_nos_udf(info.task, info.name, list(inputs.keys())))
            return_type, param_types = self._convert_nos_signature(info.signature)
            pt_func = Function.make_library_function(
                return_type, param_types, module_name=module_name, eval_symbol=eval_symbol)
            setattr(pt_module, model_id, pt_func)
            self.register_function(module_name, model_id, pt_func)

    def list_functions(self) -> List[Function.Metadata]:
        # retrieve Function.Metadata data for all existing stored functions from store directly
        # (self.stored_fns_by_id isn't guaranteed to contain all functions)
        stmt = sql.select(
                store.Function.name, store.Function.md,
                store.Db.name, store.Dir.path, sql_func.length(store.Function.init_obj))\
            .where(store.Function.db_id == store.Db.id)\
            .where(store.Function.dir_id == store.Dir.id)
        stored_fn_md: List[Function.Metadata] = []
        with Env.get().engine.begin() as conn:
            rows = conn.execute(stmt)
            for name, md_dict, db_name, dir_path, init_obj_len in rows:
                md = Function.Metadata.from_dict(md_dict)
                md.fqn = f'{db_name}{"." + dir_path if dir_path != "" else ""}.{name}'
                stored_fn_md.append(md)
        return [fn.md for fn in self.library_fns.values()] + stored_fn_md

    def get_function(self, id: int) -> Function:
        if id not in self.stored_fns_by_id:
            stmt = sql.select(
                    store.Function.name, store.Function.md,
                    store.Function.eval_obj, store.Function.init_obj, store.Function.update_obj,
                    store.Function.value_obj) \
                .where(store.Function.id == id)
            with Env.get().engine.begin() as conn:
                rows = conn.execute(stmt)
                row = next(rows)
                name = row[0]
                md = Function.Metadata.from_dict(row[1])
                # md.fqn is set by caller
                eval_fn = cloudpickle.loads(row[2]) if row[2] is not None else None
                # TODO: are these checks needed?
                if row[2] is not None and eval_fn is None:
                    raise exc.Error(f'Could not load eval_fn for function {name}')
                init_fn = cloudpickle.loads(row[3]) if row[3] is not None else None
                if row[3] is not None and init_fn is None:
                    raise exc.Error(f'Could not load init_fn for aggregate function {name}')
                update_fn = cloudpickle.loads(row[4]) if row[4] is not None else None
                if row[4] is not None and update_fn is None:
                    raise exc.Error(f'Could not load update_fn for aggregate function {name}')
                value_fn = cloudpickle.loads(row[5]) if row[5] is not None else None
                if row[5] is not None and value_fn is None:
                    raise exc.Error(f'Could not load value_fn for aggregate function {name}')

                func = Function(
                    md, id=id,
                    eval_fn=eval_fn, init_fn=init_fn, update_fn=update_fn, value_fn=value_fn)
                _logger.info(f'Loaded function {name} from store')
                self.stored_fns_by_id[id] = func
        assert id in self.stored_fns_by_id
        return self.stored_fns_by_id[id]

    def create_function(
            self, fn: Function, db_id: Optional[int] = None, dir_id: Optional[int] = None,
            name: Optional[str] = None
    ) -> None:
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

            res = conn.execute(
                sql.insert(store.Function.__table__)
                    .values(
                        db_id=db_id, dir_id=dir_id, name=name, md=fn.md.as_dict(),
                        eval_obj=eval_fn_str, init_obj=init_fn_str, update_obj=update_fn_str, value_obj=value_fn_str))
            fn.id = res.inserted_primary_key[0]
            self.stored_fns_by_id[fn.id] = fn
            _logger.info(f'Created function {name} in store')

    def update_function(self, id: int, new_fn: Function) -> None:
        """
        Updates the callables for the Function with the given id in the store and in the cache, if present.
        """
        assert not new_fn.is_library_function
        with Env.get().engine.begin() as conn:
            updates = {}
            if new_fn.eval_fn is not None:
                updates[store.Function.eval_obj] = cloudpickle.dumps(new_fn.eval_fn)
            if new_fn.init_fn is not None:
                updates[store.Function.init_obj] = cloudpickle.dumps(new_fn.init_fn)
            if new_fn.update_fn is not None:
                updates[store.Function.update_obj] = cloudpickle.dumps(new_fn.update_fn)
            if new_fn.value_fn is not None:
                updates[store.Function.value_obj] = cloudpickle.dumps(new_fn.value_fn)
            conn.execute(
                sql.update(store.Function.__table__)
                    .values(updates)
                    .where(store.Function.id == id))
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

    def delete_function(self, id: int) -> None:
        assert id is not None
        with Env.get().engine.begin() as conn:
            conn.execute(
                sql.delete(store.Function.__table__)
                    .where(store.Function.id == id))
            _logger.info(f'Deleted function with id {id} from store')


# def create_module_list() -> None:
#     """
#     Generate file standard_modules.py, which contains a list of modules available after 'import pixeltable'.
#     These are the modules we don't want to pickle.
#     TODO: move this elsewhere?
#     """
#     with open('standard_modules.py', 'w') as f:
#         f.write('module_names = set([\n    ')
#         line_len = 0
#         module_names = sys.modules.keys()
#         for name in module_names:
#             str = f"'{name}', "
#             line_len += len(str)
#             if line_len >= 80:
#                 f.write('\n    ')
#                 line_len = 4  # spaces
#             f.write(str)
#         f.write('\n])')


# make create_module_list() callable from the commandline
if __name__ == '__main__':
    globals()[sys.argv[1]]()
