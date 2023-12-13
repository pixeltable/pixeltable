from __future__ import annotations
import importlib
import logging
from typing import List, Optional, Type, Dict, Set
from uuid import UUID
import inspect

import sqlalchemy.orm as orm

from .table_version import TableVersion
from .mutable_table import MutableTable
from .column import Column
from .globals import POS_COLUMN_NAME
from pixeltable.env import Env
from pixeltable.iterators import ComponentIterator
from pixeltable.exceptions import Error
import pixeltable.func as func
from pixeltable.type_system import InvalidType, IntType


_logger = logging.getLogger('pixeltable')

class View(MutableTable):
    """A :py:class:`MutableTable` that presents a virtual view of another table (or view).
    """
    def __init__(self, dir_id: UUID, tbl_version: TableVersion):
        super().__init__(tbl_version.id, dir_id, tbl_version)

    @classmethod
    def display_name(cls) -> str:
        return 'view'

    @classmethod
    def create(
            cls, dir_id: UUID, name: str, base: TableVersion, cols: List[Column], predicate: 'exprs.Predicate',
            num_retained_versions: int, iterator_cls: Optional[Type[ComponentIterator]], iterator_args: Optional[Dict]
    ) -> View:
        cls._verify_user_columns(cols)

        if iterator_cls is not None:
            assert iterator_args is not None

            # validate iterator_args
            py_signature = inspect.signature(iterator_cls.__init__)
            try:
                # make sure iterator_args can be used to instantiate iterator_cls
                bound_args = py_signature.bind(None, **iterator_args).arguments  # None: arg for self
                # we ignore 'self'
                first_param_name = next(iter(py_signature.parameters))  # can't guarantee it's actually 'self'
                del bound_args[first_param_name]

                # construct Signature and type-check bound_args
                params = [(param_name, param_type) for param_name, param_type in iterator_cls.input_schema().items()]
                sig = func.Signature(InvalidType(), params)
                from pixeltable.exprs import FunctionCall
                FunctionCall.check_args(sig, bound_args)
            except TypeError as e:
                raise Error(f'Cannot instantiate iterator with given arguments: {e}')

            # prepend pos and output_schema columns to cols:
            # a component view exposes the pos column of its rowid;
            # we create that column here, so it gets assigned a column id;
            # stored=False: it is not stored separately (it's already stored as part of the rowid)
            iterator_cols = [Column(POS_COLUMN_NAME, IntType(), stored=False)]
            output_dict, unstored_cols = iterator_cls.output_schema()
            iterator_cols.extend([
                Column(col_name, col_type, stored=col_name not in unstored_cols)
                for col_name, col_type in output_dict.items()
            ])

            iterator_col_names = {col.name for col in iterator_cols}
            for col in cols:
                if col.name in iterator_col_names:
                    raise Error(f'Duplicate name: column {col.name} is already present in the iterator output schema')
            cols = iterator_cols + cols

        with orm.Session(Env.get().engine, future=True) as session:
            from pixeltable.exprs import InlineDict
            tbl_version = TableVersion.create(
                dir_id, name, cols, base, predicate, num_retained_versions, iterator_cls,
                InlineDict(iterator_args) if iterator_args is not None else None, session)
            view = cls(dir_id, tbl_version)

            from pixeltable.plan import Planner
            plan, num_values_per_row = Planner.create_view_load_plan(tbl_version)
            num_rows, num_excs, cols_with_excs = tbl_version.store_tbl.insert_rows(
                plan, session.connection(), v_min=tbl_version.version)
            session.commit()
            _logger.info(f'created view {name}, id={tbl_version.id}')
            msg = f'created view {name} with {num_rows} rows, {num_excs} exceptions'
            print(msg)
            return view

    @classmethod
    def _verify_column(cls, col: Column, existing_column_names: Set[str]) -> None:
        # make sure that columns are nullable or have a default
        if not col.col_type.nullable and not col.is_computed:
            raise Error(f'Column {col.name}: non-computed columns in views must be nullable')
        super()._verify_column(col, existing_column_names)
