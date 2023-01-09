from typing import Optional, Callable, Dict, List, Any
import importlib
import sqlalchemy as sql
import cloudpickle

from pixeltable.type_system import ColumnType
from pixeltable import env, store


class Function:
    """
    A Function's executable function is specified either directly or as module/symbol.
    In the former case, the function needs to be pickled and stored for serialization.
    In the latter case, the executable function is resolved in init().
    self.id is only set for non-module functions that are in the backing store.
    """
    def __init__(
            self, return_type: ColumnType, param_types: Optional[List[ColumnType]],
            id: Optional[int] = None, module_name: Optional[str] = None, symbol: Optional[str] = None,
            eval_fn: Optional[Callable] = None):
        assert (module_name is None) == (symbol is None)
        assert (module_name is None) != (eval_fn is None)
        self.return_type = return_type
        self.param_types = param_types
        self.module_name = module_name
        self.symbol = symbol
        if module_name is not None:
            # resolve module_name and symbol
            obj = importlib.import_module(module_name)
            for el in symbol.split('.'):
                obj = getattr(obj, el)
            self.eval_fn = obj
        else:
            self.eval_fn = eval_fn
        self.id = id

    def is_library_function(self) -> bool:
        return self.module_name is not None

    def __call__(self, *args: object) -> 'pixeltable.exprs.FunctionCall':
        from pixeltable import exprs
        return exprs.FunctionCall(self, args)

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
            'symbol': self.symbol
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
        assert 'symbol' in d

        if d['id'] is not None:
            assert d['module_name'] is None
            return FunctionRegistry.get().get_function(d['id'])
        else:
            return cls(return_type, param_types, module_name=d['module_name'], symbol=d['symbol'])


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

    @classmethod
    def register_pickled_module(cls, module: Any) -> None:
        cloudpickle.register_pickle_by_value(module)

    def clear_cache(self) -> None:
        """
        Useful during testing
        """
        self.fns_by_id: Dict[int, Function] = {}

    def get_function(self, id: int) -> Function:
        if id not in self.fns_by_id:
            stmt = sql.select(store.Function.return_type, store.Function.param_types, store.Function.pickled_obj) \
                .where(store.Function.id == id)
            with env.get_engine().begin() as conn:
                rows = conn.execute(stmt)
                row = next(rows)
                return_type = ColumnType.deserialize(row[0])
                param_types = ColumnType.deserialize_list(row[1])
                eval_fn = cloudpickle.loads(row[2])
                func = Function(return_type, param_types, eval_fn=eval_fn)
                self.fns_by_id[id] = func
        assert id in self.fns_by_id
        return self.fns_by_id[id]

    def create_function(
            self, fn: Function, db_id: Optional[int] = None, dir_id: Optional[int] = None,
            name: Optional[str] = None
    ) -> None:
        with env.get_engine().begin() as conn:
            pickled_obj = cloudpickle.dumps(fn.eval_fn)
            res = conn.execute(
                sql.insert(store.Function.__table__)
                    .values(
                        db_id=db_id, dir_id=dir_id, name=name, return_type=fn.return_type.serialize(),
                        param_types=ColumnType.serialize_list(fn.param_types), pickled_obj=pickled_obj))
            fn.id = res.inserted_primary_key[0]
            self.fns_by_id[fn.id] = fn

    def update_function(self, id: int, eval_fn: Callable) -> None:
        """
        Updates the callable for the function with the given id in the store and in the cache, if present.
        """
        with env.get_engine().begin() as conn:
            pickled_obj = cloudpickle.dumps(eval_fn)
            conn.execute(
                sql.update(store.Function.__table__)
                    .values({store.Function.pickled_obj: pickled_obj})
                    .where(store.Function.id == id))
        if id in self.fns_by_id:
            self.fns_by_id[id].eval_fn = eval_fn

    def delete_function(self, id: int) -> None:
        assert id is not None
        with env.get_engine().begin() as conn:
            conn.execute(
                sql.delete(store.Function.__table__)
                    .where(store.Function.id == id))
