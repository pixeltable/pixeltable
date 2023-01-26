import sys
from typing import Optional, Callable, Dict, List, Any
import importlib
import sqlalchemy as sql
import cloudpickle

from pixeltable.type_system import ColumnType
from pixeltable import store
from pixeltable.env import Env
from pixeltable import exceptions as exc
#from pixeltable import function_pickle


class Function:
    """
    A Function's executable function is specified either directly or as module/symbol.
    In the former case, the function needs to be pickled and stored for serialization.
    In the latter case, the executable function is resolved in init().
    self.id is only set for non-module functions that are in the backing store.
    """
    def __init__(
            self, return_type: ColumnType, param_types: Optional[List[ColumnType]], id: Optional[int] = None,
            module_name: Optional[str] = None, eval_symbol: Optional[str] = None, init_symbol: Optional[str] = None,
            update_symbol: Optional[str] = None, value_symbol: Optional[str] = None,
            eval_fn: Optional[Callable] = None, init_fn: Optional[Callable] = None,
            update_fn: Optional[Callable] = None, value_fn: Optional[Callable] = None,
            order_by: List[int] = []
    ):
        has_agg_symbols = init_symbol is not None and update_symbol is not None and value_symbol is not None
        has_agg_fns = init_fn is not None and update_fn is not None and value_fn is not None
        assert (module_name is not None) == (eval_symbol is not None or has_agg_symbols)
        assert (module_name is None) == (eval_fn is not None or has_agg_fns)
        # exactly one of the 4 scenarios (is agg fn x is library fn) is specified
        assert int(eval_symbol is not None) + int(eval_fn is not None) + int(has_agg_symbols) + int(has_agg_fns) == 1
        self.return_type = return_type
        self.param_types = param_types
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

        if module_name is not None:
            # resolve module_name and symbol
            module = importlib.import_module(module_name)
            if eval_symbol is not None:
                self.eval_fn = self._resolve_symbol(module, eval_symbol)
            if init_symbol is not None:
                self.init_fn = self._resolve_symbol(module, init_symbol)
            if update_symbol is not None:
                self.update_fn = self._resolve_symbol(module, update_symbol)
            if value_symbol is not None:
                self.value_fn = self._resolve_symbol(module, value_symbol)

        if len(order_by) > 0:
            if self.init_fn is None:
                raise exc.Error(f'order_by parameter only valid for aggregate functions')
            for idx in order_by:
                if not isinstance(idx, int) or idx >= len(param_types):
                    raise exc.Error(f'order_by element not a valid index into param_types: {idx}')
        self.order_by = order_by

    def _resolve_symbol(self, module: Any, symbol: str) -> object:
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

    def __call__(self, *args: object) -> 'pixeltable.exprs.FunctionCall':
        from pixeltable import exprs
        return exprs.FunctionCall(self, args)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.return_type == other.return_type and self.param_types == other.param_types \
            and self.id == other.id and self.module_name == other.module_name \
            and self.eval_symbol == other.eval_symbol and self.init_symbol == other.init_symbol \
            and self.update_symbol == other.update_symbol and self.value_symbol == other.value_symbol \
            and self.eval_fn == other.eval_fn and self.init_fn == other.init_fn \
            and self.update_fn == other.update_fn and self.value_fn == other.value_fn


    def as_dict(self) -> Dict:
        if self.module_name is None and self.id is None:
            # this is not a library function and the absence of an assigned id indicates that it's not in the store yet
            FunctionRegistry.get().create_function(self)
            assert self.id is not None
        return {
            'return_type': self.return_type.as_dict(),
            'param_types': [t.as_dict() for t in self.param_types] if self.param_types is not None else None,
            'id': self.id,
            'module_name': self.module_name,
            'eval_symbol': self.eval_symbol,
            'init_symbol': self.init_symbol,
            'update_symbol': self.update_symbol,
            'value_symbol': self.value_symbol,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'Function':
        assert 'return_type' in d
        return_type = ColumnType.from_dict(d['return_type'])
        assert 'param_types' in d
        if d['param_types'] is None:
            param_types = None
        else:
            param_types = [ColumnType.from_dict(type_dict) for type_dict in d['param_types']]
        assert 'id' in d
        assert 'module_name' in d
        assert 'eval_symbol' in d and 'init_symbol' in d and 'update_symbol' in d and 'value_symbol' in d

        if d['id'] is not None:
            assert d['module_name'] is None
            return FunctionRegistry.get().get_function(d['id'])
        else:
            return cls(
                return_type, param_types, module_name=d['module_name'], eval_symbol=d['eval_symbol'],
                init_symbol=d['init_symbol'], update_symbol=d['update_symbol'], value_symbol=d['value_symbol'])


class FunctionRegistry:
    """
    A central registry for all Functions. Handles interactions with the backing store.
    Function are loaded from the store on demand.
    """
    _instance: Optional['FunctionRegistry'] = None

    @classmethod
    def get(cls) -> 'FunctionRegistry':
        if cls._instance is None:
            cls._instance = FunctionRegistry()
        return cls._instance

    def __init__(self):
        self.fns_by_id: Dict[int, Function] = {}

    def clear_cache(self) -> None:
        """
        Useful during testing
        """
        self.fns_by_id: Dict[int, Function] = {}

    def get_function(self, id: int) -> Function:
        if id not in self.fns_by_id:
            stmt = sql.select(
                store.Function.name, store.Function.return_type, store.Function.param_types,
                store.Function.eval_obj, store.Function.init_obj, store.Function.update_obj, store.Function.value_obj) \
                .where(store.Function.id == id)
            with Env.get().engine.begin() as conn:
                rows = conn.execute(stmt)
                row = next(rows)
                name = row[0]
                return_type = ColumnType.deserialize(row[1])
                param_types = ColumnType.deserialize_list(row[2])
                eval_fn = cloudpickle.loads(row[3]) if row[3] is not None else None
                # TODO: are these checks needed?
                if row[3] is not None and eval_fn is None:
                    raise exc.Error(f'Could not load eval_fn for function {name}')
                init_fn = cloudpickle.loads(row[4]) if row[4] is not None else None
                if row[4] is not None and init_fn is None:
                    raise exc.Error(f'Could not load init_fn for aggregate function {name}')
                update_fn = cloudpickle.loads(row[5]) if row[5] is not None else None
                if row[5] is not None and update_fn is None:
                    raise exc.Error(f'Could not load update_fn for aggregate function {name}')
                value_fn = cloudpickle.loads(row[6]) if row[6] is not None else None
                if row[6] is not None and value_fn is None:
                    raise exc.Error(f'Could not load value_fn for aggregate function {name}')

                func = Function(
                    return_type, param_types, id=id, eval_fn=eval_fn, init_fn=init_fn, update_fn=update_fn,
                    value_fn=value_fn)
                self.fns_by_id[id] = func
        assert id in self.fns_by_id
        return self.fns_by_id[id]

    def create_function(
            self, fn: Function, db_id: Optional[int] = None, dir_id: Optional[int] = None,
            name: Optional[str] = None
    ) -> None:
        with Env.get().engine.begin() as conn:
            eval_fn_str = cloudpickle.dumps(fn.eval_fn) if fn.eval_fn is not None else None
            init_fn_str = cloudpickle.dumps(fn.init_fn) if fn.init_fn is not None else None
            update_fn_str = cloudpickle.dumps(fn.update_fn) if fn.update_fn is not None else None
            value_fn_str = cloudpickle.dumps(fn.value_fn) if fn.value_fn is not None else None
            res = conn.execute(
                sql.insert(store.Function.__table__)
                    .values(
                        db_id=db_id, dir_id=dir_id, name=name, return_type=fn.return_type.serialize(),
                        param_types=ColumnType.serialize_list(fn.param_types),
                        eval_obj=eval_fn_str, init_obj=init_fn_str, update_obj=update_fn_str, value_obj=value_fn_str))
            fn.id = res.inserted_primary_key[0]
            self.fns_by_id[fn.id] = fn

    def update_function(
            self, id: int, eval_fn: Optional[Callable] = None, init_fn: Optional[Callable] = None,
            update_fn: Optional[Callable] = None, value_fn: Optional[Callable] = None
    ) -> None:
        """
        Updates the callable for the function with the given id in the store and in the cache, if present.
        """
        with Env.get().engine.begin() as conn:
            updates = {}
            if eval_fn is not None:
                updates[store.Function.eval_obj] = cloudpickle.dumps(eval_fn)
            if init_fn is not None:
                updates[store.Function.init_obj] = cloudpickle.dumps(init_fn)
            if update_fn is not None:
                updates[store.Function.update_obj] = cloudpickle.dumps(update_fn)
            if value_fn is not None:
                updates[store.Function.value_obj] = cloudpickle.dumps(value_fn)
            conn.execute(
                sql.update(store.Function.__table__)
                    .values(updates)
                    .where(store.Function.id == id))
        if id in self.fns_by_id:
            if eval_fn is not None:
                self.fns_by_id[id].eval_fn = eval_fn
            if init_fn is not None:
                self.fns_by_id[id].init_fn = init_fn
            if update_fn is not None:
                self.fns_by_id[id].update_fn = update_fn
            if value_fn is not None:
                self.fns_by_id[id].value_fn = value_fn

    def delete_function(self, id: int) -> None:
        assert id is not None
        with Env.get().engine.begin() as conn:
            conn.execute(
                sql.delete(store.Function.__table__)
                    .where(store.Function.id == id))


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
