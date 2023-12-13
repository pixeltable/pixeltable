from __future__ import annotations
from typing import Optional, Dict, Tuple, List
from uuid import UUID
import types
import sys
import cloudpickle
import logging
import dataclasses

import sqlalchemy as sql
import nos

from .function import Function
import pixeltable.type_system as ts
from pixeltable.metadata import schema
import pixeltable.env as env
import pixeltable.exceptions as excs


_logger = logging.getLogger('pixeltable')

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

    def register_module(self, module: types.ModuleType) -> None:
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

    def _convert_nos_signature(self, sig: nos.common.spec.FunctionSignature) -> Tuple[ts.ColumnType, List[ts.ColumnType]]:
        if len(sig.get_outputs_spec()) > 1:
            return_type = ts.JsonType()
        else:
            return_type = ts.ColumnType.from_nos(list(sig.get_outputs_spec().values())[0])
        param_types: List[ts.ColumnType] = []
        for _, type_info in sig.get_inputs_spec().items():
            # if there are multiple input shapes we leave them out of the ColumnType and deal with them in FunctionCall
            if isinstance(type_info, list):
                param_types.append(ts.ColumnType.from_nos(type_info[0], ignore_shape=True))
            else:
                param_types.append(ts.ColumnType.from_nos(type_info, ignore_shape=False))
        return return_type, param_types

    def register_nos_functions(self) -> None:
        """Register all models supported by the NOS backend as library functions"""
        if self.has_registered_nos_functions:
            return
        self.has_registered_nos_functions = True
        models = env.Env.get().nos_client.ListModels()
        model_info = [env.Env.get().nos_client.GetModelInfo(model) for model in models]
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
                with env.Env.get().engine.begin() as conn:
                    rows = conn.execute(stmt)
                    row = next(rows)
                    schema_md = schema.md_from_dict(schema.FunctionMd, row[0])
                    name = schema_md.name
                    md = Function.Metadata.from_dict(schema_md.md)
                    # md.fqn is set by caller
                    eval_fn = cloudpickle.loads(row[1]) if row[1] is not None else None
                    # TODO: are these checks needed?
                    if row[1] is not None and eval_fn is None:
                        raise excs.Error(f'Could not load eval_fn for function {name}')
                    init_fn = cloudpickle.loads(row[2]) if row[2] is not None else None
                    if row[2] is not None and init_fn is None:
                        raise excs.Error(f'Could not load init_fn for aggregate function {name}')
                    update_fn = cloudpickle.loads(row[3]) if row[3] is not None else None
                    if row[3] is not None and update_fn is None:
                        raise excs.Error(f'Could not load update_fn for aggregate function {name}')
                    value_fn = cloudpickle.loads(row[4]) if row[4] is not None else None
                    if row[4] is not None and value_fn is None:
                        raise excs.Error(f'Could not load value_fn for aggregate function {name}')

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

    def get_type_methods(self, name: str, base_type: ts.ColumnType.Type) -> List[Function]:
        return [
            fn for fn in self.library_fns.values()
            if fn.md.fqn.endswith('.' + name)
               and fn.md.signature.parameter_types_by_pos[0].type_enum == base_type
        ]

    def create_function(self, fn: Function, dir_id: Optional[UUID] = None, name: Optional[str] = None) -> None:
        with env.Env.get().engine.begin() as conn:
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
        with env.Env.get().engine.begin() as conn:
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
        with env.Env.get().engine.begin() as conn:
            conn.execute(
                sql.delete(schema.Function.__table__)
                    .where(schema.Function.id == id))
            _logger.info(f'Deleted function with id {id} from store')
