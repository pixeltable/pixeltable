from __future__ import annotations
import abc
import copy
import datetime
import enum
import sys
import typing
from typing import Union, Optional, List, Callable, Any, Dict, Tuple, Set, Generator, Iterator, Type
import operator
import json
import io
from collections.abc import Iterable
import time
import inspect
from uuid import UUID
import urllib.parse
import urllib.request
from dataclasses import dataclass
import hashlib

import PIL.Image
import jmespath
import numpy as np
import sqlalchemy as sql

from pixeltable import catalog
from pixeltable.type_system import \
    ColumnType, InvalidType, StringType, IntType, FloatType, BoolType, JsonType, ArrayType
from pixeltable.function import Function, FunctionRegistry, Signature
from pixeltable.exceptions import Error, ExprEvalError
from pixeltable.utils import print_perf_counter_delta
from pixeltable.utils.clip import embed_image, embed_text
from pixeltable.catalog import is_valid_identifier
from pixeltable.iterators import ComponentIterator

# Python types corresponding to our literal types
LiteralPythonTypes = Union[str, int, float, bool, datetime.datetime, datetime.date]

def _print_slice(s: slice) -> str:
    start_str = f'{str(s.start) if s.start is not None else ""}'
    stop_str = f'{str(s.stop) if s.stop is not None else ""}'
    step_str = f'{str(s.step) if s.step is not None else ""}'
    return f'{start_str}:{stop_str}{":" if s.step is not None else ""}{step_str}'


class ComparisonOperator(enum.Enum):
    LT = 0
    LE = 1
    EQ = 2
    NE = 3
    GT = 4
    GE = 5

    def __str__(self) -> str:
        if self == self.LT:
            return '<'
        if self == self.LE:
            return '<='
        if self == self.EQ:
            return '=='
        if self == self.GT:
            return '>'
        if self == self.GE:
            return '>='


class LogicalOperator(enum.Enum):
    AND = 0
    OR = 1
    NOT = 2

    def __str__(self) -> str:
        if self == self.AND:
            return '&'
        if self == self.OR:
            return '|'
        if self == self.NOT:
            return '~'


class ArithmeticOperator(enum.Enum):
    ADD = 0
    SUB = 1
    MUL = 2
    DIV = 3
    MOD = 4

    def __str__(self) -> str:
        if self == self.ADD:
            return '+'
        if self == self.SUB:
            return '-'
        if self == self.MUL:
            return '*'
        if self == self.DIV:
            return '/'
        if self == self.MOD:
            return '%'

class ExprScope:
    """
    Representation of the scope in which an Expr needs to be evaluated. Used to determine nesting of scopes.
    parent is None: outermost scope
    """
    def __init__(self, parent: Optional[ExprScope]):
        self.parent = parent

    def is_contained_in(self, other: ExprScope) -> bool:
        if self == other:
            return True
        if self.parent is None:
            return False
        return self.parent.is_contained_in(other)


_GLOBAL_SCOPE = ExprScope(None)


class Expr(abc.ABC):
    """
    Rules for using state in subclasses:
    - all state except for components and slot_idx is shared between copies of an Expr
    - slot_idx is set during analysis (DataFrame.show())
    - during eval(), components can only be accessed via self.components; any Exprs outside of that won't
      have slot_idx set
    """
    def __init__(self, col_type: ColumnType):
        self.col_type = col_type

        # each instance has an id that is used for equality comparisons
        # - set by the subclass's __init__()
        # - produced by _create_id()
        # - not expected to survive a serialize()/deserialize() roundtrip
        self.id: Optional[int] = None

        # index of the expr's value in the data row:
        # - set for all materialized exprs
        # - -1: not executable
        # - not set for subexprs that don't need to be materialized because the parent can be materialized via SQL
        self.slot_idx = -1
        self.components: List[Expr] = []  # the subexprs that are needed to construct this expr

    def dependencies(self) -> List[Expr]:
        """
        Returns all exprs that need to have been evaluated before eval() can be called on this one.
        """
        return self.components

    def scope(self) -> ExprScope:
        # by default this is the innermost scope of any of our components
        result = _GLOBAL_SCOPE
        for c in self.components:
            c_scope = c.scope()
            if c_scope.is_contained_in(result):
                result = c_scope
        return result

    def bind_rel_paths(self, mapper: Optional[JsonMapper] = None) -> None:
        """
        Binds relative JsonPaths to mapper.
        This needs to be done in a separate phase after __init__(), because RelativeJsonPath()(-1) cannot be resolved
        by the immediately containing JsonMapper during initialization.
        """
        for c in self.components:
            c.bind_rel_paths(mapper)

    def default_column_name(self) -> Optional[str]:
        """
        Returns: 
            None if this expression lacks a default name, 
            or a valid identifier (according to catalog.is_valid_identifer) otherwise.
        """
        return None

    def equals(self, other: Expr) -> bool:
        """
        Subclass-specific comparison. Implemented as a function because __eq__() is needed to construct Comparisons.
        """
        if type(self) != type(other):
            return False
        if len(self.components) != len(other.components):
            return False
        for i in range(len(self.components)):
            if not self.components[i].equals(other.components[i]):
                return False
        return self._equals(other)

    def _id_attrs(self) -> List[Tuple[str, Any]]:
        """Returns attribute/value pairs that are used to construct the instance id.

        Attribute values must be immutable and have str() defined.
        """
        return [('classname', self.__class__.__name__)]

    def _create_id(self) -> int:
        hasher = hashlib.sha256()
        for attr, value in self._id_attrs():
            hasher.update(attr.encode('utf-8'))
            hasher.update(str(value).encode('utf-8'))
        for expr in self.components:
            hasher.update(str(expr.id).encode('utf-8'))
        # truncate to machine's word size
        return int(hasher.hexdigest(), 16) & sys.maxsize

    @classmethod
    def list_equals(cls, a: List[Expr], b: List[Expr]) -> bool:
        if len(a) != len(b):
            return False
        for i in range(len(a)):
            if not a[i].equals(b[i]):
                return False
        return True

    def copy(self) -> Expr:
        """
        Creates a copy that can be evaluated separately: it doesn't share any eval context (slot_idx)
        but shares everything else (catalog objects, etc.)
        """
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        result.slot_idx = -1
        result.components = [c.copy() for c in self.components]
        return result

    @classmethod
    def copy_list(cls, expr_list: List[Expr]) -> List[Expr]:
        return [e.copy() for e in expr_list]

    def __deepcopy__(self, memo={}) -> Expr:
        # we don't need to create an actual deep copy because all state other than execution state is read-only
        result = self.copy()
        memo[id(self)] = result
        return result

    def substitute(self, old: Expr, new: Expr) -> Expr:
        """
        Replace 'old' with 'new' recursively.
        """
        if self.equals(old):
            return new.copy()
        for i in range(len(self.components)):
            self.components[i] = self.components[i].substitute(old, new)
        return self

    def resolve_computed_cols(self, unstored_only: bool) -> Expr:
        """
        Recursively replace ColRefs to computed columns with their value exprs.

        Args:
            unstored_only: if True, only replace references to unstored computed columns
        """
        result = self
        while True:
            computed_col_refs = [
                e for e in result.subexprs()
                if isinstance(e, ColumnRef) and e.col.is_computed and (not e.col.is_stored or not unstored_only)
            ]
            if len(computed_col_refs) == 0:
                return result
            for ref in computed_col_refs:
                assert ref.col.value_expr is not None
                result = result.substitute(ref, ref.col.value_expr)

    def is_bound_by(self, tbl: catalog.TableVersion) -> bool:
        """Returns True if this expr can be evaluated in the context of tbl."""
        col_refs = self.subexprs(ColumnRef)
        for col_ref in col_refs:
            if not tbl.has_column(col_ref.col):
                return False
        return True

    @classmethod
    def list_substitute(cls, expr_list: List[Expr], old: Expr, new: Expr) -> None:
        for i in range(len(expr_list)):
            expr_list[i] = expr_list[i].substitute(old, new)

    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    def display_str(self, inline: bool = True) -> str:
        """
        inline: if False, use line breaks where appropriate; otherwise don't use linebreaks
        """
        return str(self)

    @classmethod
    def print_list(cls, expr_list: List[Expr]) -> str:
        if len(expr_list) == 1:
            return str(expr_list[0])
        return f'({", ".join([str(e) for e in expr_list])})'

    def subexprs(
            self, expr_class: Optional[Type[Expr]] = None, filter: Optional[Callable[[Expr], bool]] = None,
            traverse_matches: bool = True
    ) -> Generator[Expr, None, None]:
        """
        Iterate over all subexprs, including self.
        """
        assert expr_class is None or filter is None  # at most one of them
        if expr_class is not None:
            filter = lambda e: isinstance(e, expr_class)
        is_match = filter is None or filter(self)
        if not is_match or traverse_matches:
            for c in self.components:
                yield from c.subexprs(filter=filter, traverse_matches=traverse_matches)
        if is_match:
            yield self

    def contains(self, cls: Optional[Type[Expr]] = None, filter: Optional[Callable[[Expr], bool]] = None) -> bool:
        """
        Returns True if any subexpr is an instance of cls.
        """
        assert (cls is not None) != (filter is not None)  # need one of them
        if cls is not None:
            filter = lambda e: isinstance(e, cls)
        try:
            _ = next(self.subexprs(filter=filter, traverse_matches=False))
            return True
        except StopIteration:
            return False

    @classmethod
    def list_subexprs(
            cls, expr_list: List[Expr], expr_class: Optional[Type[Expr]] = None,
            filter: Optional[Callable[[Expr], bool]] = None, traverse_matches: bool = True
    ) -> Generator[Expr, None, None]:
        """Produce subexprs for all exprs in list. Can contain duplicates."""
        for e in expr_list:
            yield from e.subexprs(expr_class=expr_class, filter=filter, traverse_matches=traverse_matches)

    def tbl_ids(self) -> Set[UUID]:
        """Returns table ids referenced by this expr."""
        return {ref.col.tbl.id for ref in self.subexprs(ColumnRef)} | {ref.tbl.id for ref in self.subexprs(RowidRef)}

    @classmethod
    def list_tbl_ids(cls, expr_list: List[Expr]) -> Set[UUID]:
        ids: Set[UUID] = set()
        for e in expr_list:
            ids.update(e.tbl_ids())
        return ids

    @classmethod
    def from_object(cls, o: object) -> Optional[Expr]:
        """
        Try to turn a literal object into an Expr.
        """
        if isinstance(o, Expr):
            return o
        # try to create a literal
        obj_type = ColumnType.infer_literal_type(o)
        if obj_type is not None:
            return Literal(o, col_type=obj_type)
        if isinstance(o, dict):
            return InlineDict(o)
        elif isinstance(o, list):
            return InlineArray(tuple(o))
        return None

    @abc.abstractmethod
    def _equals(self, other: Expr) -> bool:
        pass

    @abc.abstractmethod
    def sql_expr(self) -> Optional[sql.ClauseElement]:
        """
        If this expr can be materialized directly in SQL:
        - returns a ClauseElement
        - eval() will not be called (exception: Literal)
        Otherwise
        - returns None
        - eval() will be called
        """
        pass

    @abc.abstractmethod
    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        """
        Compute the expr value for data_row and store the result in data_row[slot_idx].
        Not called if sql_expr() != None (exception: Literal).
        """
        pass

    def release(self) -> None:
        """
        Allow Expr class to tear down execution state. This is called after the last eval() call.
        """
        for c in self.components:
            c.release()

    @classmethod
    def release_list(cls, expr_list: List[Expr]) -> None:
        for e in expr_list:
            e.release()

    def serialize(self) -> str:
        return json.dumps(self.as_dict())

    def as_dict(self) -> Dict:
        """
        Turn Expr object into a dict that can be passed to json.dumps().
        Subclasses override _as_dict().
        """
        return {
            '_classname': self.__class__.__name__,
            **self._as_dict(),
        }

    @classmethod
    def as_dict_list(self, expr_list: List[Expr]) -> List[Dict]:
        return [e.as_dict() for e in expr_list]

    def _as_dict(self) -> Dict:
        if len(self.components) > 0:
            return {'components': [c.as_dict() for c in self.components]}
        return {}

    @classmethod
    def deserialize(cls, dict_str: str, t: catalog.TableVersion) -> Expr:
        return cls.from_dict(json.loads(dict_str), t)

    @classmethod
    def from_dict(cls, d: Dict, t: catalog.TableVersion) -> Expr:
        """
        Turn dict that was produced by calling Expr.as_dict() into an instance of the correct Expr subclass.
        """
        assert '_classname' in d
        type_class = globals()[d['_classname']]
        components: List[Expr] = []
        if 'components' in d:
            components = [cls.from_dict(component_dict, t) for component_dict in d['components']]
        return type_class._from_dict(d, components, t)

    @classmethod
    def from_dict_list(cls, dict_list: List[Dict], t: catalog.TableVersion) -> List[Expr]:
        return [cls.from_dict(d, t) for d in dict_list]

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr], t: catalog.TableVersion) -> Expr:
        assert False, 'not implemented'

    def __getitem__(self, index: object) -> Expr:
        if self.col_type.is_json_type():
            return JsonPath(self).__getitem__(index)
        if self.col_type.is_array_type():
            return ArraySlice(self, index)
        raise Error(f'Type {self.col_type} is not subscriptable')

    def __getattr__(self, name: str) -> Union[ImageMemberAccess, JsonPath]:
        """
        ex.: <img col>.rotate(60)
        """
        if self.col_type.is_image_type():
            return ImageMemberAccess(name, self)
        if self.col_type.is_json_type():
            return JsonPath(self).__getattr__(name)
        raise Error(f'Member access not supported on type {self.col_type}: {name}')

    def __lt__(self, other: object) -> Comparison:
        return self._make_comparison(ComparisonOperator.LT, other)

    def __le__(self, other: object) -> Comparison:
        return self._make_comparison(ComparisonOperator.LE, other)

    def __eq__(self, other: object) -> Comparison:
        if other is None:
            return IsNull(self)
        return self._make_comparison(ComparisonOperator.EQ, other)

    def __ne__(self, other: object) -> Comparison:
        if other is None:
            return CompoundPredicate(LogicalOperator.NOT, [IsNull(self)])
        return self._make_comparison(ComparisonOperator.NE, other)

    def __gt__(self, other: object) -> Comparison:
        return self._make_comparison(ComparisonOperator.GT, other)

    def __ge__(self, other: object) -> Comparison:
        return self._make_comparison(ComparisonOperator.GE, other)

    def _make_comparison(self, op: ComparisonOperator, other: object) -> Comparison:
        """
        other: Union[Expr, LiteralPythonTypes]
        """
        # TODO: check for compatibility
        if isinstance(other, Expr):
            return Comparison(op, self, other)
        if isinstance(other, typing.get_args(LiteralPythonTypes)):
            return Comparison(op, self, Literal(other))  # type: ignore[arg-type]
        raise TypeError(f'Other must be Expr or literal: {type(other)}')

    def __add__(self, other: object) -> ArithmeticExpr:
        return self._make_arithmetic_expr(ArithmeticOperator.ADD, other)

    def __sub__(self, other: object) -> ArithmeticExpr:
        return self._make_arithmetic_expr(ArithmeticOperator.SUB, other)

    def __mul__(self, other: object) -> ArithmeticExpr:
        return self._make_arithmetic_expr(ArithmeticOperator.MUL, other)

    def __truediv__(self, other: object) -> ArithmeticExpr:
        return self._make_arithmetic_expr(ArithmeticOperator.DIV, other)

    def __mod__(self, other: object) -> ArithmeticExpr:
        return self._make_arithmetic_expr(ArithmeticOperator.MOD, other)

    def _make_arithmetic_expr(self, op: ArithmeticOperator, other: object) -> ArithmeticExpr:
        """
        other: Union[Expr, LiteralPythonTypes]
        """
        # TODO: check for compatibility
        if isinstance(other, Expr):
            return ArithmeticExpr(op, self, other)
        if isinstance(other, typing.get_args(LiteralPythonTypes)):
            return ArithmeticExpr(op, self, Literal(other))  # type: ignore[arg-type]
        raise TypeError(f'Other must be Expr or literal: {type(other)}')


class ColumnRef(Expr):
    """A reference to a table column

    When this reference is created in the context of a view, it can also refer to a column of the view base.
    For that reason, a ColumnRef needs to be serialized with the qualifying table id (column ids are only
    unique in the context of a particular table).
    """
    def __init__(self, col: catalog.Column):
        super().__init__(col.col_type)
        assert col.tbl is not None
        self.col = col
        self.is_unstored_iter_col = \
            col.tbl.is_component_view() and col.tbl.is_iterator_column(col) and not col.is_stored
        self.iter_arg_ctx: Optional[RowBuilder.EvalCtx] = None
        # number of rowid columns in the base table
        self.base_rowid_len = len(col.tbl.base.store_tbl.rowid_columns()) if self.is_unstored_iter_col else 0
        self.base_rowid = [None] * self.base_rowid_len
        self.iterator: Optional[ComponentIterator] = None
        # index of the position column in the view's primary key
        self.pos_idx: Optional[int] = len(col.tbl.store_tbl.rowid_columns()) - 1 if self.is_unstored_iter_col else None
        self.id = self._create_id()

    def set_iter_arg_ctx(self, iter_arg_ctx: RowBuilder.EvalCtx) -> None:
        self.iter_arg_ctx = iter_arg_ctx
        assert len(self.iter_arg_ctx.target_slot_idxs) == 1  # a single inline dict

    def _id_attrs(self) -> List[Tuple[str, Any]]:
        return super()._id_attrs() + [('tbl_id', self.col.tbl.id), ('col_id', self.col.id)]

    def __getattr__(self, name: str) -> Expr:
        # resolve column properties
        if name == ColumnPropertyRef.Property.ERRORTYPE.name.lower() \
                or name == ColumnPropertyRef.Property.ERRORMSG.name.lower():
            if not self.col.is_computed or not self.col.is_stored:
                raise Error(f'{name} not valid for a non-computed or unstored column: {self}')
            return ColumnPropertyRef(self, ColumnPropertyRef.Property[name.upper()])
        if name == ColumnPropertyRef.Property.FILEURL.name.lower() \
                or name == ColumnPropertyRef.Property.LOCALPATH.name.lower():
            if not self.col.col_type.is_image_type() and not self.col.col_type.is_video_type():
                raise Error(f'{name} only valid for image and video columns: {self}')
            if self.col.is_computed and not self.col.is_stored:
                raise Error(f'{name} not valid for computed unstored columns: {self}')
            return ColumnPropertyRef(self, ColumnPropertyRef.Property[name.upper()])

        if self.col_type.is_json_type():
            return JsonPath(self, [name])

        return super().__getattr__(name)

    def default_column_name(self) -> Optional[str]:
        return str(self)

    def _equals(self, other: ColumnRef) -> bool:
        return self.col == other.col

    def __str__(self) -> str:
        return self.col.name

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        return self.col.sa_col

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        if not self.is_unstored_iter_col:
            return
        # if this is a new base row, we need to instantiate a new iterator
        if self.base_rowid != data_row.pk[:self.base_rowid_len]:
            row_builder.eval(data_row, self.iter_arg_ctx)
            iterator_args = data_row[self.iter_arg_ctx.target_slot_idxs[0]]
            self.iterator = self.col.tbl.iterator_cls(**iterator_args)
            self.base_rowid = data_row.pk[:self.base_rowid_len]
        self.iterator.set_pos(data_row.pk[self.pos_idx])
        res = next(self.iterator)
        data_row[self.slot_idx] = res[self.col.name]

    def _as_dict(self) -> Dict:
        return {'tbl_id': str(self.col.tbl.id), 'col_id': self.col.id}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr], t: catalog.TableVersion) -> Expr:
        # resolve d['tbl_id'], which is either t or a base of t
        origin = t.find_tbl(UUID(d['tbl_id']))
        col_id = d['col_id']
        assert col_id in origin.cols_by_id
        return cls(origin.cols_by_id[col_id])


class ColumnPropertyRef(Expr):
    """A reference to a property of a table column

    The properties themselves are type-specific and may or may not need to reference the underlying column data.
    """
    class Property(enum.Enum):
        ERRORTYPE = 0
        ERRORMSG = 1
        FILEURL = 2
        LOCALPATH = 3

    def __init__(self, col_ref: ColumnRef, prop: Property):
        super().__init__(StringType(nullable=True))
        self.components = [col_ref]
        self.prop = prop
        self.id = self._create_id()

    def default_column_name(self) -> Optional[str]:
        return str(self).replace('.', '_')

    def _equals(self, other: ColumnRef) -> bool:
        return self.prop == other.prop

    def _id_attrs(self) -> List[Tuple[str, Any]]:
        return super()._id_attrs() + [('prop', self.prop.value)]

    @property
    def _col_ref(self) -> ColumnRef:
        return self.components[0]

    def __str__(self) -> str:
        return f'{self._col_ref}.{self.prop.name.lower()}'

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        if not self._col_ref.col.is_stored:
            return None
        if self.prop == self.Property.ERRORTYPE:
            assert self._col_ref.col.sa_errortype_col is not None
            return self._col_ref.col.sa_errortype_col
        if self.prop == self.Property.ERRORMSG:
            assert self._col_ref.col.sa_errormsg_col is not None
            return self._col_ref.col.sa_errormsg_col
        return None

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        assert self.prop == self.Property.FILEURL or self.prop == self.Property.LOCALPATH
        assert data_row.has_val[self._col_ref.slot_idx]
        if self.prop == self.Property.FILEURL:
            data_row[self.slot_idx] = data_row.file_urls[self._col_ref.slot_idx]
        if self.prop == self.Property.LOCALPATH:
            data_row[self.slot_idx] = data_row.file_paths[self._col_ref.slot_idx]

    def _as_dict(self) -> Dict:
        return {'prop': self.prop.value, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr], t: catalog.TableVersion) -> Expr:
        assert 'prop' in d
        return cls(components[0], cls.Property(d['prop']))


class RowidRef(Expr):
    """A reference to (a part of) a table rowid column

    This is used internally to support grouping by a base table and for references to the 'pos' column.
    """
    def __init__(self, tbl: catalog.TableVersion, idx: int):
        super().__init__(IntType(nullable=False))
        # normalize to simplify comparisons: we refer to the lowest base table that has the requested rowid idx
        # (which has the same values as all its descendent views)
        while tbl.base is not None and len(tbl.base.store_tbl.rowid_columns()) > idx:
            tbl = tbl.base
        self.tbl = tbl
        self.rowid_component_idx = idx
        self.id = self._create_id()

    def default_column_name(self) -> Optional[str]:
        return str(self)

    def _equals(self, other: ColumnRef) -> bool:
        return self.tbl is other.tbl and self.rowid_component_idx == other.rowid_component_idx

    def _id_attrs(self) -> List[Tuple[str, Any]]:
        return super()._id_attrs() + [('tbl_id', self.tbl.id), ('idx', self.rowid_component_idx)]

    def __str__(self) -> str:
        # check if this is the pos column of a component view
        if self.tbl.is_component_view() and self.rowid_component_idx == self.tbl.store_tbl.pos_col_idx:
            return catalog.globals.POS_COLUMN_NAME
        return None

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        rowid_cols = self.tbl.store_tbl.rowid_columns()
        return rowid_cols[self.rowid_component_idx]

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        data_row[self.slot_idx] = data_row.pk[self.rowid_component_idx]

    def _as_dict(self) -> Dict:
        return {'tbl_id': str(self.tbl.id), 'idx': self.rowid_component_idx}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr], t: catalog.TableVersion) -> Expr:
        tbl = t.find_tbl(UUID(d['tbl_id']))
        assert tbl is not None
        return cls(tbl, d['idx'])


class FunctionCall(Expr):
    def __init__(
            self, fn: Function, bound_args: Dict[str, Any], order_by_clause: List[Any] = [],
            group_by_clause: List[Any] = [], is_method_call: bool = False):
        signature = fn.md.signature
        super().__init__(signature.get_return_type(bound_args))
        self.fn = fn
        self.is_method_call = is_method_call
        self.check_args(signature, bound_args)

        # construct components, args, kwargs
        self.components: List[Expr] = []

        # Tuple[int, Any]:
        # - for Exprs: (index into components, None)
        # - otherwise: (-1, val)
        self.args: List[Tuple[int, Any]] = []
        self.kwargs: Dict[str, Tuple[int, Any]] = {}

        self.arg_types: List[ColumnType] = []  # needed for runtime type checks
        self.kwarg_types: Dict[str, ColumnType] = {}
        # the prefix of parameters that are bound can be passed by position
        for param in fn.py_signature.parameters.values():
            if param.name not in bound_args or param.kind == inspect.Parameter.KEYWORD_ONLY:
                break
            arg = bound_args[param.name]
            if isinstance(arg, Expr):
                self.args.append((len(self.components), None))
                self.components.append(arg.copy())
            else:
                self.args.append((-1, arg))
            self.arg_types.append(signature.parameters[param.name])

        # the remaining args are passed as keywords
        kw_param_names = set(bound_args.keys()) - set(list(fn.py_signature.parameters.keys())[:len(self.args)])
        for param_name in kw_param_names:
            arg = bound_args[param_name]
            if isinstance(arg, Expr):
                self.kwargs[param_name] = (len(self.components), None)
                self.components.append(arg.copy())
            else:
                # TODO: make sure it's json-serializable
                self.kwargs[param_name] = (-1, arg)
            self.kwarg_types[param_name] = signature.parameters[param_name]

        # window function state:
        # self.components[self.group_by_start_idx:self.group_by_stop_idx] contains group_by exprs
        self.group_by_start_idx, self.group_by_stop_idx = 0, 0
        if len(group_by_clause) > 0:
            # TODO: analyze group_by_clause
            if isinstance(group_by_clause[0], catalog.Table):
                group_by_exprs = self._create_rowid_refs(group_by_clause[0])
            else:
                assert isinstance(group_by_clause[0], Expr)
                group_by_exprs = group_by_clause
            # record grouping exprs in self.components, we need to evaluate them to get partition vals
            self.group_by_start_idx = len(self.components)
            self.group_by_stop_idx = len(self.components) + len(group_by_exprs)
            self.components.extend(group_by_exprs)

        # we want to make sure that order_by_clause get assigned slot_idxs, even though we won't need to evaluate them
        # (that's done in SQL)
        if len(order_by_clause) > 0 and not isinstance(order_by_clause[0], Expr):
            raise Error(
                f'order_by argument needs to be a Pixeltable expression, but instead is a {type(order_by_clause[0])}')
        self.order_by_start_idx = len(self.components)
        self.components.extend(order_by_clause)

        self.nos_info = FunctionRegistry.get().get_nos_info(self.fn)
        self.constant_args = {param_name for param_name, arg in bound_args.items() if not isinstance(arg, Expr)}

        # execution state for aggregate functions
        self.aggregator: Optional[Any] = None
        self.current_partition_vals: Optional[List[Any]] = None

        self.id = self._create_id()

    def _create_rowid_refs(self, tbl: catalog.Table) -> List[Expr]:
        rowid_cols = tbl.tbl_version.store_tbl.rowid_columns()
        return [RowidRef(tbl.tbl_version, i) for i in range(len(rowid_cols))] if len(rowid_cols) > 0 else []

    @classmethod
    def check_args(cls, signature: Signature, bound_args: Dict[str, Any]) -> None:
        """Checks that bound_args are compatible with signature.

        Convert literals to the correct type and update bound_args in place, if necessary.
        """
        for param_name, arg in bound_args.items():
            if not isinstance(arg, Expr):
                # make sure that non-Expr args are json-serializable and are literals of the correct type
                try:
                    _ = json.dumps(arg)
                except TypeError:
                    raise Error(f"Argument for parameter '{param_name}' is not json-serializable: {arg}")
                if arg is not None:
                    try:
                        param_type = signature.parameters[param_name]
                        bound_args[param_name] = param_type.create_literal(arg)
                    except TypeError as e:
                        msg = str(e)
                        raise Error(f"Argument for parameter '{param_name}': {msg[0].lower() + msg[1:]}")
                continue

            param_type = signature.parameters[param_name]
            if not param_type.is_supertype_of(arg.col_type):
                raise Error((
                    f'Parameter {param_name}: argument type {arg.col_type} does not match parameter type '
                    f'{param_type}'))

    def is_nos_call(self) -> bool:
        return self.nos_info is not None

    def _equals(self, other: FunctionCall) -> bool:
        if self.fn != other.fn:
            return False
        if len(self.args) != len(other.args):
            return False
        for i in range(len(self.args)):
            if self.args[i] != other.args[i]:
                return False
        if self.group_by_start_idx != other.group_by_start_idx:
            return False
        if self.group_by_stop_idx != other.group_by_stop_idx:
            return False
        if self.order_by_start_idx != other.order_by_start_idx:
            return False
        return True

    def _id_attrs(self) -> List[Tuple[str, Any]]:
        return super()._id_attrs() + [
            ('fn', self.fn.md.fqn),
            ('args', self.args),
            ('kwargs', self.kwargs),
            ('group_by_start_idx', self.group_by_start_idx),
            ('group_by_stop_idx', self.group_by_stop_idx),
            ('order_by_start_idx', self.order_by_start_idx)
        ]

    def __str__(self) -> str:
        return self.display_str()

    def display_str(self, inline: bool = True) -> str:
        if self.is_method_call:
            return f'{self.components[0]}.{self.fn.name}({self._print_args(1, inline)})'
        else:
            fn_name = self.fn.display_name if self.fn.display_name != '' else 'anonymous_fn'
            return f'{fn_name}({self._print_args()})'

    def _print_args(self, start_idx: int = 0, inline: bool = True) -> str:
        arg_strs = [
            str(arg) if idx == -1 else str(self.components[idx]) for idx, arg in self.args[start_idx:]
        ]
        arg_strs.extend([
            f'{param_name}={str(arg) if idx == -1 else str(self.components[idx])}'
            for param_name, (idx, arg) in self.kwargs.items()
        ])
        if len(self.order_by) > 0:
            if self.fn.requires_order_by:
                arg_strs.insert(0, Expr.print_list(self.order_by))
            else:
                arg_strs.append(f'order_by={Expr.print_list(self.order_by)}')
        if len(self.group_by) > 0:
            arg_strs.append(f'group_by={Expr.print_list(self.group_by)}')
        # TODO: figure out the function name
        separator = ', ' if inline else ',\n    '
        return separator.join(arg_strs)

    def has_group_by(self) -> List[Expr]:
        return self.group_by_stop_idx != 0

    @property
    def group_by(self) -> List[Expr]:
        return self.components[self.group_by_start_idx:self.group_by_stop_idx]

    @property
    def order_by(self) -> List[Expr]:
        return self.components[self.order_by_start_idx:]

    @property
    def is_window_fn_call(self) -> bool:
        return self.fn.is_aggregate and self.fn.allows_window and \
            (not self.fn.allows_std_agg \
             or self.has_group_by() \
             or (len(self.order_by) > 0 and not self.fn.requires_order_by))

    def get_window_sort_exprs(self) -> Tuple[List[Expr], List[Expr]]:
        return self.group_by, self.order_by

    @property
    def is_agg_fn_call(self) -> bool:
        return self.fn.is_aggregate and not self.is_window_fn_call

    def get_agg_order_by(self) -> List[Expr]:
        assert self.is_agg_fn_call
        return self.order_by

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        # TODO: implement for standard aggregate functions
        return None

    def reset_agg(self) -> None:
        """
        Init agg state
        """
        assert self.is_agg_fn_call
        self.aggregator = self.fn.init_fn()

    def update(self, data_row: DataRow) -> None:
        """
        Update agg state
        """
        assert self.is_agg_fn_call
        args, kwargs = self._make_args(data_row)
        self.fn.update_fn(*[self.aggregator, *args], **kwargs)

    def _make_args(self, data_row: DataRow) -> Tuple[List[Any], Dict[str, Any]]:
        args = [arg if idx == -1 else data_row[self.components[idx].slot_idx] for idx, arg in self.args]
        kwargs = {
            param_name: val if idx == -1 else data_row[self.components[idx].slot_idx]
            for param_name, (idx, val) in self.kwargs.items()
        }
        return args, kwargs

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        args, kwargs = self._make_args(data_row)
        signature = self.fn.md.signature
        if signature.parameters is not None:
            # check for nulls
            for arg, param_type in zip(args, self.arg_types):
                if arg is None and not param_type.nullable:
                    # we can't evaluate this function
                    data_row[self.slot_idx] = None
                    return
            for param_name, param_type in self.kwarg_types.items():
                if kwargs[param_name] is None and not param_type.nullable:
                    # we can't evaluate this function
                    data_row[self.slot_idx] = None
                    return

        if not self.fn.is_aggregate:
            data_row[self.slot_idx] = self.fn.eval_fn(*args, **kwargs)
        elif self.is_window_fn_call:
            if self.has_group_by():
                if self.current_partition_vals is None:
                    self.current_partition_vals = [None] * len(self.group_by)
                partition_vals = [data_row[e.slot_idx] for e in self.group_by]
                if partition_vals != self.current_partition_vals:
                    # new partition
                    self.aggregator = self.fn.init_fn()
                    self.current_partition_vals = partition_vals
            elif self.aggregator is None:
                self.aggregator = self.fn.init_fn()
            self.fn.update_fn(self.aggregator, *args)
            data_row[self.slot_idx] = self.fn.value_fn(self.aggregator)
        else:
            assert self.is_agg_fn_call
            data_row[self.slot_idx] = self.fn.value_fn(self.aggregator)

    def _as_dict(self) -> Dict:
        result = {
            'fn': self.fn.as_dict(), 'args': self.args, 'kwargs': self.kwargs,
            'group_by_start_idx': self.group_by_start_idx, 'group_by_stop_idx': self.group_by_stop_idx,
            'order_by_start_idx': self.order_by_start_idx,
            **super()._as_dict()
        }
        return result

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr], t: catalog.TableVersion) -> Expr:
        assert 'fn' in d
        assert 'args' in d
        assert 'kwargs' in d
        # reassemble bound args
        fn = Function.from_dict(d['fn'])
        param_names = list(fn.md.signature.parameters.keys())
        bound_args = {param_names[i]: arg if idx == -1 else components[idx] for i, (idx, arg) in enumerate(d['args'])}
        bound_args.update(
            {param_name: val if idx == -1 else components[idx] for param_name, (idx, val) in d['kwargs'].items()})
        group_by_exprs = components[d['group_by_start_idx']:d['group_by_stop_idx']]
        order_by_exprs = components[d['order_by_start_idx']:]
        fn_call = cls(
            Function.from_dict(d['fn']), bound_args, group_by_clause=group_by_exprs, order_by_clause=order_by_exprs)
        return fn_call


# TODO: this doesn't dig up all attrs for actual jpeg images
def _create_pil_attr_info() -> Dict[str, ColumnType]:
    # create random Image to inspect for attrs
    img = PIL.Image.new('RGB', (100, 100))
    # we're only interested in public attrs (including properties)
    result: Dict[str, ColumnType] = {}
    for name in [name for name in dir(img) if not callable(getattr(img, name)) and not name.startswith('_')]:
        if getattr(img, name) is None:
            continue
        if isinstance(getattr(img, name), str):
            result[name] = StringType()
        if isinstance(getattr(img, name), int):
            result[name] = IntType()
        if getattr(img, name) is dict:
            result[name] = JsonType()
    return result


class ImageMemberAccess(Expr):
    """
    Access of either an attribute or function member of PIL.Image.Image.
    Ex.: tbl.img_col_ref.rotate(90), tbl.img_col_ref.width
    TODO: remove this class and use FunctionCall instead (attributes to be replaced by functions)
    """
    attr_info = _create_pil_attr_info()

    def __init__(self, member_name: str, caller: Expr):
        if member_name == 'nearest':
            super().__init__(InvalidType())  # requires FunctionCall to return value
        elif member_name in self.attr_info:
            super().__init__(self.attr_info[member_name])
        else:
            candidates = FunctionRegistry.get().get_type_methods(member_name, ColumnType.Type.IMAGE)
            if len(candidates) == 0:
                raise Error(f'Unknown Image member: {member_name}')
            if len(candidates) > 1:
                raise Error(f'Ambiguous Image method: {member_name}')
            self.img_method = candidates[0]
            super().__init__(InvalidType())  # requires FunctionCall to return value
        self.member_name = member_name
        self.components = [caller]
        self.id = self._create_id()

    def default_column_name(self) -> Optional[str]:
        return self.member_name.replace('.', '_')

    @property
    def _caller(self) -> Expr:
        return self.components[0]

    def __str__(self) -> str:
        return f'{self._caller}.{self.member_name}'

    def _as_dict(self) -> Dict:
        return {'member_name': self.member_name, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr], t: catalog.TableVersion) -> Expr:
        assert 'member_name' in d
        assert len(components) == 1
        return cls(d['member_name'], components[0])

    def __call__(self, *args, **kwargs) -> Union[FunctionCall, ImageSimilarityPredicate]:
        caller = self._caller
        call_signature = f'({",".join([type(arg).__name__ for arg in args])})'
        if self.member_name == 'nearest':
            # - caller must be ColumnRef
            # - signature is (Union[PIL.Image.Image, str])
            if not isinstance(caller, ColumnRef):
                raise Error(f'nearest(): caller must be an image column')
            if len(args) != 1 or (not isinstance(args[0], PIL.Image.Image) and not isinstance(args[0], str)):
                raise Error(f'nearest(): requires a PIL.Image.Image or str, got {call_signature} instead')
            return ImageSimilarityPredicate(
                caller,
                img=args[0] if isinstance(args[0], PIL.Image.Image) else None,
                text=args[0] if isinstance(args[0], str) else None)

        result = self.img_method(*[caller, *args], **kwargs)
        result.is_method_call = True
        return result

    def _equals(self, other: ImageMemberAccess) -> bool:
        return self.member_name == other.member_name

    def _id_attrs(self) -> List[Tuple[str, Any]]:
        return super()._id_attrs() + [('member_name', self.member_name)]

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        return None

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        caller_val = data_row[self._caller.slot_idx]
        try:
            data_row[self.slot_idx] = getattr(caller_val, self.member_name)
        except AttributeError:
            data_row[self.slot_idx] = None


class JsonPath(Expr):
    def __init__(self, anchor: Optional[ColumnRef], path_elements: List[str] = [], scope_idx: int = 0):
        """
        anchor can be None, in which case this is a relative JsonPath and the anchor is set later via set_anchor().
        scope_idx: for relative paths, index of referenced JsonMapper
        (0: indicates the immediately preceding JsonMapper, -1: the parent of the immediately preceding mapper, ...)
        """
        super().__init__(JsonType())
        if anchor is not None:
            self.components = [anchor]
        self.path_elements: List[Union[str, int]] = path_elements
        self.compiled_path = jmespath.compile(self._json_path()) if len(path_elements) > 0 else None
        self.scope_idx = scope_idx
        # NOTE: the _create_id() result will change if set_anchor() gets called;
        # this is not a problem, because _create_id() shouldn't be called after init()
        self.id = self._create_id()

    def __str__(self) -> str:
        # else "R": the anchor is RELATIVE_PATH_ROOT
        return (f'{str(self._anchor) if self._anchor is not None else "R"}'
            f'{"." if isinstance(self.path_elements[0], str) else ""}{self._json_path()}')

    def _as_dict(self) -> Dict:
        return {'path_elements': self.path_elements, 'scope_idx': self.scope_idx, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr], t: catalog.TableVersion) -> Expr:
        assert 'path_elements' in d
        assert 'scope_idx' in d
        assert len(components) <= 1
        anchor = components[0] if len(components) == 1 else None
        return cls(anchor, d['path_elements'], d['scope_idx'])

    @property
    def _anchor(self) -> Optional[Expr]:
        return None if len(self.components) == 0 else self.components[0]

    def set_anchor(self, anchor: Expr) -> None:
        assert len(self.components) == 0
        self.components = [anchor]

    def is_relative_path(self) -> bool:
        return self._anchor is None

    def bind_rel_paths(self, mapper: Optional['JsonMapper'] = None) -> None:
        if not self.is_relative_path():
            return
        # TODO: take scope_idx into account
        self.set_anchor(mapper.scope_anchor)

    def __call__(self, *args: object, **kwargs: object) -> 'JsonPath':
        """
        Construct a relative path that references an ancestor of the immediately enclosing JsonMapper.
        """
        if not self.is_relative_path():
            raise Error(f'() for an absolute path is invalid')
        if len(args) != 1 or not isinstance(args[0], int) or args[0] >= 0:
            raise Error(f'R() requires a negative index')
        return JsonPath(None, [], args[0])

    def __getattr__(self, name: str) -> 'JsonPath':
        assert isinstance(name, str)
        return JsonPath(self._anchor, self.path_elements + [name])

    def __getitem__(self, index: object) -> 'JsonPath':
        if isinstance(index, str):
            if index != '*':
                raise Error(f'Invalid json list index: {index}')
        else:
            if not isinstance(index, slice) and not isinstance(index, int):
                raise Error(f'Invalid json list index: {index}')
        return JsonPath(self._anchor, self.path_elements + [index])

    def __rshift__(self, other: object) -> 'JsonMapper':
        rhs_expr = Expr.from_object(other)
        if rhs_expr is None:
            raise Error(f'>> requires an expression on the right-hand side, found {type(other)}')
        return JsonMapper(self, rhs_expr)

    def default_column_name(self) -> Optional[str]:
        anchor_name = self._anchor.default_column_name() if self._anchor is not None else ''
        ret_name = f'{anchor_name}.{self._json_path()}'
        
        def cleanup_char(s : str) -> str:
            if s == '.':
                return '_'
            elif s == '*':
                return 'star'
            elif s.isalnum():
                return s
            else:
                return ''
            
        clean_name = ''.join(map(cleanup_char, ret_name))
        clean_name = clean_name.lstrip('_') # remove leading underscore
        if clean_name == '':
            clean_name = None
        
        assert clean_name is None or is_valid_identifier(clean_name)
        return clean_name

    def _equals(self, other: JsonPath) -> bool:
        return self.path_elements == other.path_elements

    def _id_attrs(self) -> List[Tuple[str, Any]]:
        return super()._id_attrs() + [('path_elements', self.path_elements)]

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        """
        Postgres appears to have a bug: jsonb_path_query('{a: [{b: 0}, {b: 1}]}', '$.a.b') returns
        *two* rows (each containing col val 0), not a single row with [0, 0].
        We need to use a workaround: retrieve the entire dict, then use jmespath to extract the path correctly.
        """
        #path_str = '$.' + '.'.join(self.path_elements)
        #assert isinstance(self._anchor(), ColumnRef)
        #return sql.func.jsonb_path_query(self._anchor().col.sa_col, path_str)
        return None

    def _json_path(self) -> str:
        assert len(self.path_elements) > 0
        result: List[str] = []
        for element in self.path_elements:
            if element == '*':
                result.append('[*]')
            elif isinstance(element, str):
                result.append(f'{"." if len(result) > 0 else ""}{element}')
            elif isinstance(element, int):
                result.append(f'[{element}]')
            elif isinstance(element, slice):
                result.append(f'[{_print_slice(element)}]')
        return ''.join(result)

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        val = data_row[self._anchor.slot_idx]
        if self.compiled_path is not None:
            val = self.compiled_path.search(val)
        data_row[self.slot_idx] = val


RELATIVE_PATH_ROOT = JsonPath(None)


class Literal(Expr):
    def __init__(self, val: Any, col_type: Optional[ColumnType] = None):
        if col_type is not None:
            val = col_type.create_literal(val)
        else:
            # try to determine a type for val
            col_type = ColumnType.infer_literal_type(val)
            if col_type is None:
                raise TypeError(f'Not a valid literal: {val}')
        super().__init__(col_type)
        self.val = val
        self.id = self._create_id()

    def default_column_name(self) -> Optional[str]:
        return 'Literal'

    def __str__(self) -> str:
        if self.col_type.is_string_type() or self.col_type.is_timestamp_type():
            return f"'{self.val}'"
        return str(self.val)

    def _equals(self, other: Literal) -> bool:
        return self.val == other.val

    def _id_attrs(self) -> List[Tuple[str, Any]]:
        return super()._id_attrs() + [('val', self.val)]

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        # we need to return something here so that we can generate a Where clause for predicates
        # that involve literals (like Where c > 0)
        return sql.sql.expression.literal(self.val)

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        # this will be called, even though sql_expr() does not return None
        data_row[self.slot_idx] = self.val

    def _as_dict(self) -> Dict:
        return {'val': self.val, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr], t: catalog.TableVersion) -> Expr:
        assert 'val' in d
        return cls(d['val'])


class InlineDict(Expr):
    """
    Dictionary 'literal' which can use Exprs as values.
    """
    def __init__(self, d: Dict):
        super().__init__(JsonType())  # we need to call this in order to populate self.components
        # dict_items contains
        # - for Expr fields: (key, index into components, None)
        # - for non-Expr fields: (key, -1, value)
        self.dict_items: List[Tuple[str, int, Any]] = []
        for key, val in d.items():
            if not isinstance(key, str):
                raise Error(f'Dictionary requires string keys, {key} has type {type(key)}')
            val = copy.deepcopy(val)
            if isinstance(val, dict):
                val = InlineDict(val)
            if isinstance(val, Expr):
                self.dict_items.append((key, len(self.components), None))
                self.components.append(val)
            else:
                self.dict_items.append((key, -1, val))

        self.type_spec: Optional[Dict[str, ColumnType]] = {}
        for key, idx, _ in self.dict_items:
            if idx == -1:
                # TODO: implement type inference for values
                self.type_spec = None
                break
            self.type_spec[key] = self.components[idx].col_type
        self.col_type = JsonType(self.type_spec)

        self.id = self._create_id()

    def __str__(self) -> str:
        item_strs: List[str] = []
        i = 0
        for key, idx, val in self.dict_items:
            if idx != -1:
                item_strs.append(f"'{key}': {str(self.components[i])}")
                i += 1
            else:
                item_strs.append(f"'{key}': {str(val)}")
        return '{' + ', '.join(item_strs) + '}'

    def _equals(self, other: InlineDict) -> bool:
        return self.dict_items == other.dict_items

    def _id_attrs(self) -> List[Tuple[str, Any]]:
        return super()._id_attrs() + [('dict_items', self.dict_items)]

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        return None

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        result = {}
        for key, idx, val in self.dict_items:
            assert isinstance(key, str)
            if idx >= 0:
                result[key] = data_row[self.components[idx].slot_idx]
            else:
                result[key] = copy.deepcopy(val)
        data_row[self.slot_idx] = result

    def _as_dict(self) -> Dict:
        return {'dict_items': self.dict_items, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr], t: catalog.TableVersion) -> Expr:
        assert 'dict_items' in d
        arg: Dict[str, Any] = {}
        for key, idx, val in d['dict_items']:
            if idx >= 0:
                arg[key] = components[idx]
            else:
                arg[key] = val
        return cls(arg)


class InlineArray(Expr):
    """
    Array 'literal' which can use Exprs as values.
    """
    def __init__(self, elements: Tuple):
        # we need to call this in order to populate self.components
        super().__init__(ArrayType((len(elements),), IntType()))

        # elements contains
        # - for Expr elements: (index into components, None)
        # - for non-Expr elements: (-1, value)
        self.elements: List[Tuple[int, Any]] = []
        for el in elements:
            el = copy.deepcopy(el)
            if isinstance(el, list):
                el = InlineArray(tuple(el))
            if isinstance(el, Expr):
                self.elements.append((len(self.components), None))
                self.components.append(el)
            else:
                self.elements.append((-1, el))

        element_type = InvalidType()
        for idx, val in self.elements:
            if idx >= 0:
                element_type = ColumnType.supertype(element_type, self.components[idx].col_type)
            else:
                element_type = ColumnType.supertype(element_type, ColumnType.infer_literal_type(val))
            if element_type is None:
                # there is no common element type: this is a json value, not an array
                # TODO: make sure this doesn't contain Images
                self.col_type = JsonType()
                return

        if element_type.is_scalar_type():
            self.col_type = ArrayType((len(self.elements),), element_type)
        elif element_type.is_array_type():
            assert isinstance(element_type, ArrayType)
            self.col_type = ArrayType(
                (len(self.elements), *element_type.shape), ColumnType.make_type(element_type.dtype))
        elif element_type.is_json_type():
            self.col_type = JsonType()

        self.id = self._create_id()

    def __str__(self) -> str:
        elem_strs = [str(val) if val is not None else str(self.components[idx]) for idx, val in self.elements]
        return f'[{", ".join(elem_strs)}]'

    def _equals(self, other: InlineDict) -> bool:
        return self.elements == other.elements

    def _id_attrs(self) -> List[Tuple[str, Any]]:
        return super()._id_attrs() + [('elements', self.elements)]

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        return None

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        result = [None] * len(self.elements)
        for i, (child_idx, val) in enumerate(self.elements):
            if child_idx >= 0:
                result[i] = data_row[self.components[child_idx].slot_idx]
            else:
                result[i] = copy.deepcopy(val)
        data_row[self.slot_idx] = np.array(result)

    def _as_dict(self) -> Dict:
        return {'elements': self.elements, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr], t: catalog.TableVersion) -> Expr:
        assert 'elements' in d
        arg: List[Any] = []
        for idx, val in d['elements']:
            if idx >= 0:
                arg.append(components[idx])
            else:
                arg.append(val)
        return cls(tuple(arg))


class ArraySlice(Expr):
    """
    Slice operation on an array, eg, t.array_col[:, 1:2].
    """
    def __init__(self, arr: Expr, index: Tuple):
        assert arr.col_type.is_array_type()
        # determine result type
        super().__init__(arr.col_type)
        self.components = [arr]
        self.index = index
        self.id = self._create_id()

    def __str__(self) -> str:
        index_strs: List[str] = []
        for el in self.index:
            if isinstance(el, int):
                index_strs.append(str(el))
            if isinstance(el, slice):
                index_strs.append(_print_slice(el))
        return f'{self._array}[{", ".join(index_strs)}]'

    @property
    def _array(self) -> Expr:
        return self.components[0]

    def _equals(self, other: ArraySlice) -> bool:
        return self.index == other.index

    def _id_attrs(self) -> List[Tuple[str, Any]]:
        return super()._id_attrs() + [('index', self.index)]

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        return None

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        val = data_row[self._array.slot_idx]
        data_row[self.slot_idx] = val[self.index]

    def _as_dict(self) -> Dict:
        index = []
        for el in self.index:
            if isinstance(el, slice):
                index.append([el.start, el.stop, el.step])
            else:
                index.append(el)
        return {'index': index, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr], t: catalog.TableVersion) -> Expr:
        assert 'index' in d
        index = []
        for el in d['index']:
            if isinstance(el, list):
                index.append(slice(el[0], el[1], el[2]))
            else:
                index.append(el)
        return cls(components[0], tuple(index))


class Predicate(Expr):
    def __init__(self) -> None:
        super().__init__(BoolType())

    def split_conjuncts(
            self, condition: Callable[[Predicate], bool]) -> Tuple[List[Predicate], Optional[Predicate]]:
        """
        Returns clauses of a conjunction that meet condition in the first element.
        The second element contains remaining clauses, rolled into a conjunction.
        """
        if condition(self):
            return [self], None
        else:
            return [], self

    def __and__(self, other: object) -> CompoundPredicate:
        if not isinstance(other, Expr):
            raise TypeError(f'Other needs to be an expression: {type(other)}')
        if not other.col_type.is_bool_type():
            raise TypeError(f'Other needs to be an expression that returns a boolean: {other.col_type}')
        return CompoundPredicate(LogicalOperator.AND, [self, other])

    def __or__(self, other: object) -> CompoundPredicate:
        if not isinstance(other, Expr):
            raise TypeError(f'Other needs to be an expression: {type(other)}')
        if not other.col_type.is_bool_type():
            raise TypeError(f'Other needs to be an expression that returns a boolean: {other.col_type}')
        return CompoundPredicate(LogicalOperator.OR, [self, other])

    def __invert__(self) -> CompoundPredicate:
        return CompoundPredicate(LogicalOperator.NOT, [self])


class CompoundPredicate(Predicate):
    def __init__(self, operator: LogicalOperator, operands: List[Predicate]):
        super().__init__()
        self.operator = operator
        # operands are stored in self.components
        if self.operator == LogicalOperator.NOT:
            assert len(operands) == 1
            self.components = operands
        else:
            assert len(operands) > 1
            self.operands: List[Predicate] = []
            for operand in operands:
                self._merge_operand(operand)

        self.id = self._create_id()

    def __str__(self) -> str:
        if self.operator == LogicalOperator.NOT:
            return f'~({self.components[0]})'
        return f' {self.operator} '.join([f'({e})' for e in self.components])

    @classmethod
    def make_conjunction(cls, operands: List[Predicate]) -> Optional[Predicate]:
        if len(operands) == 0:
            return None
        if len(operands) == 1:
            return operands[0]
        return CompoundPredicate(LogicalOperator.AND, operands)

    def _merge_operand(self, operand: Predicate) -> None:
        """
        Merge this operand, if possible, otherwise simply record it.
        """
        if isinstance(operand, CompoundPredicate) and operand.operator == self.operator:
            # this can be merged
            for child_op in operand.components:
                self._merge_operand(child_op)
        else:
            self.components.append(operand)

    def _equals(self, other: CompoundPredicate) -> bool:
        return self.operator == other.operator

    def _id_attrs(self) -> List[Tuple[str, Any]]:
        return super()._id_attrs() + [('operator', self.operator.value)]

    def split_conjuncts(
            self, condition: Callable[[Predicate], bool]) -> Tuple[List[Predicate], Optional[Predicate]]:
        if self.operator == LogicalOperator.OR or self.operator == LogicalOperator.NOT:
            return super().split_conjuncts(condition)
        matches = [op for op in self.components if condition(op)]
        non_matches = [op for op in self.components if not condition(op)]
        return (matches, self.make_conjunction(non_matches))

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        sql_exprs = [op.sql_expr() for op in self.components]
        if any(e is None for e in sql_exprs):
            return None
        if self.operator == LogicalOperator.NOT:
            assert len(sql_exprs) == 1
            return sql.not_(sql_exprs[0])
        assert len(sql_exprs) > 1
        operator = sql.and_ if self.operator == LogicalOperator.AND else sql.or_
        combined = operator(*sql_exprs)
        return combined

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        if self.operator == LogicalOperator.NOT:
            data_row[self.slot_idx] = not data_row[self.components[0].slot_idx]
        else:
            val = True if self.operator == LogicalOperator.AND else False
            op_function = operator.and_ if self.operator == LogicalOperator.AND else operator.or_
            for op in self.components:
                val = op_function(val, data_row[op.slot_idx])
            data_row[self.slot_idx] = val

    def _as_dict(self) -> Dict:
        return {'operator': self.operator.value, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr], t: catalog.TableVersion) -> Expr:
        assert 'operator' in d
        return cls(LogicalOperator(d['operator']), components)


class Comparison(Predicate):
    def __init__(self, operator: ComparisonOperator, op1: Expr, op2: Expr):
        super().__init__()
        self.operator = operator
        self.components = [op1, op2]
        self.id = self._create_id()

    def __str__(self) -> str:
        return f'{self._op1} {self.operator} {self._op2}'

    def _equals(self, other: Comparison) -> bool:
        return self.operator == other.operator

    def _id_attrs(self) -> List[Tuple[str, Any]]:
        return super()._id_attrs() + [('operator', self.operator.value)]

    @property
    def _op1(self) -> Expr:
        return self.components[0]

    @property
    def _op2(self) -> Expr:
        return self.components[1]

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        left = self._op1.sql_expr()
        right = self._op2.sql_expr()
        if left is None or right is None:
            return None
        if self.operator == ComparisonOperator.LT:
            return left < right
        if self.operator == ComparisonOperator.LE:
            return left <= right
        if self.operator == ComparisonOperator.EQ:
            return left == right
        if self.operator == ComparisonOperator.NE:
            return left != right
        if self.operator == ComparisonOperator.GT:
            return left > right
        if self.operator == ComparisonOperator.GE:
            return left >= right

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        if self.operator == ComparisonOperator.LT:
            data_row[self.slot_idx] = data_row[self._op1.slot_idx] < data_row[self._op2.slot_idx]
        elif self.operator == ComparisonOperator.LE:
            data_row[self.slot_idx] = data_row[self._op1.slot_idx] <= data_row[self._op2.slot_idx]
        elif self.operator == ComparisonOperator.EQ:
            data_row[self.slot_idx] = data_row[self._op1.slot_idx] == data_row[self._op2.slot_idx]
        elif self.operator == ComparisonOperator.NE:
            data_row[self.slot_idx] = data_row[self._op1.slot_idx] != data_row[self._op2.slot_idx]
        elif self.operator == ComparisonOperator.GT:
            data_row[self.slot_idx] = data_row[self._op1.slot_idx] > data_row[self._op2.slot_idx]
        elif self.operator == ComparisonOperator.GE:
            data_row[self.slot_idx] = data_row[self._op1.slot_idx] >= data_row[self._op2.slot_idx]

    def _as_dict(self) -> Dict:
        return {'operator': self.operator.value, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr], t: catalog.TableVersion) -> Expr:
        assert 'operator' in d
        return cls(ComparisonOperator(d['operator']), components[0], components[1])


class ImageSimilarityPredicate(Predicate):
    def __init__(self, img_col_ref: ColumnRef, img: Optional[PIL.Image.Image] = None, text: Optional[str] = None):
        assert (img is None) != (text is None)
        super().__init__()
        self.img_col_ref = img_col_ref
        self.components = [img_col_ref]
        self.img = img
        self.text = text
        self.id = self._create_id()

    def embedding(self) -> np.ndarray:
        if self.text is not None:
            return embed_text(self.text)
        else:
            return embed_image(self.img)

    def __str__(self) -> str:
        return f'{str(self.img_col_ref)}.nearest({"<img>" if self.img is not None else self.text})'

    def _equals(self, other: ImageSimilarityPredicate) -> bool:
        return False

    def _id_attrs(self) -> List[Tuple[str, Any]]:
        return super()._id_attrs() + [('img', id(self.img)), ('text', self.text)]

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        return None

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        assert False

    def _as_dict(self) -> Dict:
        assert False, 'not implemented'
        # TODO: convert self.img into a serializable string
        return {'img': self.img, 'text': self.text, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr], t: catalog.TableVersion) -> Expr:
        assert 'img' in d
        assert 'text' in d
        assert len(components) == 1
        return cls(components[0], d['img'], d['text'])


class IsNull(Predicate):
    def __init__(self, e: Expr):
        super().__init__()
        self.components = [e]
        self.id = self._create_id()

    def __str__(self) -> str:
        return f'{str(self.components[0])} == None'

    def _equals(self, other: IsNull) -> bool:
        return True

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        e = self.components[0].sql_expr()
        if e is None:
            return None
        return e == None

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        data_row[self.slot_idx] = data_row[self.components[0].slot_idx] is None

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr], t: catalog.TableVersion) -> Expr:
        assert len(components) == 1
        return cls(components[0])


class ArithmeticExpr(Expr):
    """
    Allows arithmetic exprs on json paths
    """
    def __init__(self, operator: ArithmeticOperator, op1: Expr, op2: Expr):
        # TODO: determine most specific common supertype
        if op1.col_type.is_json_type() or op2.col_type.is_json_type():
            # we assume it's a float
            super().__init__(FloatType())
        else:
            super().__init__(ColumnType.supertype(op1.col_type, op2.col_type))
        self.operator = operator
        self.components = [op1, op2]

        # do typechecking after initialization in order for __str__() to work
        if not op1.col_type.is_numeric_type() and not op1.col_type.is_json_type():
            raise Error(f'{self}: {operator} requires numeric types, but {op1} has type {op1.col_type}')
        if not op2.col_type.is_numeric_type() and not op2.col_type.is_json_type():
            raise Error(f'{self}: {operator} requires numeric types, but {op2} has type {op2.col_type}')

        self.id = self._create_id()

    def __str__(self) -> str:
        # add parentheses around operands that are ArithmeticExprs to express precedence
        op1_str = f'({self._op1})' if isinstance(self._op1, ArithmeticExpr) else str(self._op1)
        op2_str = f'({self._op2})' if isinstance(self._op2, ArithmeticExpr) else str(self._op2)
        return f'{op1_str} {str(self.operator)} {op2_str}'

    def _equals(self, other: ArithmeticExpr) -> bool:
        return self.operator == other.operator

    def _id_attrs(self) -> List[Tuple[str, Any]]:
        return super()._id_attrs() + [('operator', self.operator.value)]

    @property
    def _op1(self) -> Expr:
        return self.components[0]

    @property
    def _op2(self) -> Expr:
        return self.components[1]

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        left = self._op1.sql_expr()
        right = self._op2.sql_expr()
        if left is None or right is None:
            return None
        if self.operator == ArithmeticOperator.ADD:
            return left + right
        if self.operator == ArithmeticOperator.SUB:
            return left - right
        if self.operator == ArithmeticOperator.MUL:
            return left * right
        if self.operator == ArithmeticOperator.DIV:
            return left / right
        if self.operator == ArithmeticOperator.MOD:
            return left % right

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        op1_val = data_row[self._op1.slot_idx]
        op2_val = data_row[self._op2.slot_idx]
        # check types if we couldn't do that prior to execution
        if self._op1.col_type.is_json_type() and not isinstance(op1_val, int) and not isinstance(op1_val, float):
            raise Error(
                f'{self.operator} requires numeric type, but {self._op1} has type {type(op1_val).__name__}')
        if self._op2.col_type.is_json_type() and not isinstance(op2_val, int) and not isinstance(op2_val, float):
            raise Error(
                f'{self.operator} requires numeric type, but {self._op2} has type {type(op2_val).__name__}')

        if self.operator == ArithmeticOperator.ADD:
            data_row[self.slot_idx] = op1_val + op2_val
        elif self.operator == ArithmeticOperator.SUB:
            data_row[self.slot_idx] = op1_val - op2_val
        elif self.operator == ArithmeticOperator.MUL:
            data_row[self.slot_idx] = op1_val * op2_val
        elif self.operator == ArithmeticOperator.DIV:
            data_row[self.slot_idx] = op1_val / op2_val
        elif self.operator == ArithmeticOperator.MOD:
            data_row[self.slot_idx] = op1_val % op2_val

    def _as_dict(self) -> Dict:
        return {'operator': self.operator.value, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr], t: catalog.TableVersion) -> Expr:
        assert 'operator' in d
        assert len(components) == 2
        return cls(ArithmeticOperator(d['operator']), components[0], components[1])


class ObjectRef(Expr):
    """
    Reference to an intermediate result, such as the "scope variable" produced by a JsonMapper.
    The object is generated/materialized elsewhere and establishes a new scope.
    """
    def __init__(self, scope: ExprScope, owner: JsonMapper):
        # TODO: do we need an Unknown type after all?
        super().__init__(JsonType())  # JsonType: this could be anything
        self._scope = scope
        self.owner = owner
        self.id = self._create_id()

    def scope(self) -> ExprScope:
        return self._scope

    def __str__(self) -> str:
        assert False

    def _equals(self, other: ObjectRef) -> bool:
        return self.owner is other.owner

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        return None

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        # this will be called, but the value has already been materialized elsewhere
        pass


class JsonMapper(Expr):
    """
    JsonMapper transforms the list output of a JsonPath by applying a target expr to every element of the list.
    The target expr would typically contain relative JsonPaths, which are bound to an ObjectRef, which in turn
    is populated by JsonMapper.eval(). The JsonMapper effectively creates a new scope for its target expr.
    """
    def __init__(self, src_expr: Expr, target_expr: Expr):
        # TODO: type spec should be List[target_expr.col_type]
        super().__init__(JsonType())

        # we're creating a new scope, but we don't know yet whether this is nested within another JsonMapper;
        # this gets resolved in bind_rel_paths(); for now we assume we're in the global scope
        self.target_expr_scope = ExprScope(_GLOBAL_SCOPE)

        scope_anchor = ObjectRef(self.target_expr_scope, self)
        self.components = [src_expr, target_expr, scope_anchor]
        self.parent_mapper: Optional[JsonMapper] = None
        self.target_expr_eval_ctx: Optional[RowBuilder.EvalCtx] = None
        self.id = self._create_id()

    def bind_rel_paths(self, mapper: Optional[JsonMapper]) -> None:
        self._src_expr.bind_rel_paths(mapper)
        self._target_expr.bind_rel_paths(self)
        self.parent_mapper = mapper
        parent_scope = _GLOBAL_SCOPE if mapper is None else mapper.target_expr_scope
        self.target_expr_scope.parent = parent_scope

    def scope(self) -> ExprScope:
        # need to ignore target_expr
        return self._src_expr.scope()

    def dependencies(self) -> List[Expr]:
        result = [self._src_expr]
        result.extend(self._target_dependencies(self._target_expr))
        return result

    def _target_dependencies(self, e: Expr) -> List[Expr]:
        """
        Return all subexprs of e of which the scope isn't contained in target_expr_scope.
        Those need to be evaluated before us.
        """
        expr_scope = e.scope()
        if not expr_scope.is_contained_in(self.target_expr_scope):
            return [e]
        result: List[Expr] = []
        for c in e.components:
            result.extend(self._target_dependencies(c))
        return result

    def equals(self, other: Expr) -> bool:
        """
        We override equals() because we need to avoid comparing our scope anchor.
        """
        if type(self) != type(other):
            return False
        return self._src_expr.equals(other._src_expr) and self._target_expr.equals(other._target_expr)

    def __str__(self) -> str:
        return f'{str(self._src_expr)} >> {str(self._target_expr)}'

    @property
    def _src_expr(self) -> Expr:
        return self.components[0]

    @property
    def _target_expr(self) -> Expr:
        return self.components[1]

    @property
    def scope_anchor(self) -> Expr:
        return self.components[2]

    def _equals(self, other: JsonMapper) -> bool:
        return True

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        return None

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        # this will be called, but the value has already been materialized elsewhere
        src = data_row[self._src_expr.slot_idx]
        if not isinstance(src, list):
            # invalid/non-list src path
            data_row[self.slot_idx] = None
            return

        result = [None] * len(src)
        if self.target_expr_eval_ctx is None:
            self.target_expr_eval_ctx = row_builder.create_eval_ctx([self._target_expr])
        for i, val in enumerate(src):
            data_row[self.scope_anchor.slot_idx] = val
            # stored target_expr
            exc_tb = row_builder.eval(data_row, self.target_expr_eval_ctx)
            assert exc_tb is None
            result[i] = data_row[self._target_expr.slot_idx]
        data_row[self.slot_idx] = result

    def _as_dict(self) -> Dict:
        """
        We need to avoid serializing component[2], which is an ObjectRef.
        """
        return {'components': [c.as_dict() for c in self.components[0:2]]}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr], t: catalog.TableVersion) -> Expr:
        assert len(components) == 2
        return cls(components[0], components[1])


class DataRow:
    """
    Encapsulates all data and execution state needed by RowBuilder and DataRowBatch:
    - state for in-memory computation
    - state for storing the data
    This is not meant to be a black-box abstraction.

    In-memory representations by column type:
    - StringType: str
    - IntType: int
    - FloatType: float
    - BoolType: bool
    - TimestampType: datetime.datetime
    - JsonType: json-serializable object
    - ArrayType: numpy.ndarray
    - ImageType: PIL.Image.Image
    - VideoType: local path if available, otherwise url
    """
    def __init__(self, size: int, img_slot_idxs: List[int], video_slot_idxs: List[int], array_slot_idxs: List[int]):
        self.vals: List[Any] = [None] * size  # either cell values or exceptions
        self.has_val = [False] * size
        self.excs: List[Optional[Exception]] = [None] * size

        # control structures that are shared across all DataRows in a batch
        self.img_slot_idxs = img_slot_idxs
        self.video_slot_idxs = video_slot_idxs
        self.array_slot_idxs = array_slot_idxs

        # the primary key of a store row is a sequence of ints (the number is different for table vs view)
        self.pk: Optional[Tuple[int, ...]] = None

        # file_urls:
        # - stored url of file for image or video in vals[i]
        # - None if vals[i] is not an image/video
        # - not None if file_paths[i] is not None
        self.file_urls: Optional[str] = [None] * size

        # file_paths:
        # - local path of file for image or video in vals[i]; points to the file cache if file_urls[i] is remote
        # - None if vals[i] is not an image/video or if there is no local file yet for file_urls[i]
        self.file_paths: Optional[str] = [None] * size

    def clear(self) -> None:
        size = len(self.vals)
        self.vals = [None] * size
        self.has_val = [False] * size
        self.excs = [None] * size
        self.pk = None
        self.file_urls = [None] * size
        self.file_paths = [None] * size

    def copy(self, target: DataRow) -> None:
        """Create a copy of the contents of this DataRow in target
        The copy shares the cell values, but not the control structures (eg, self.has_val), because these
        need to be independently updateable.
        """
        target.vals = self.vals.copy()
        target.has_val = self.has_val.copy()
        target.excs = self.excs.copy()
        target.pk = self.pk
        target.file_urls = self.file_urls.copy()
        target.file_paths = self.file_paths.copy()

    def set_pk(self, pk: Tuple[int, ...]) -> None:
        self.pk = pk

    def has_exc(self, slot_idx: int) -> bool:
        return self.excs[slot_idx] is not None

    def get_exc(self, slot_idx: int) -> Exception:
        assert self.has_val[slot_idx] is False
        assert self.excs[slot_idx] is not None
        return self.excs[slot_idx]

    def set_exc(self, slot_idx: int, exc: Exception) -> None:
        assert self.has_val[slot_idx] is False
        assert self.excs[slot_idx] is None
        self.excs[slot_idx] = exc

    def __getitem__(self, index: object) -> Any:
        """Returns in-memory value, ie, what is needed for expr evaluation"""
        if not self.has_val[index]:
            # for debugging purposes
            pass
        assert self.has_val[index], index

        if index in self.img_slot_idxs:
            # if we need to load this from a file, it should have been materialized locally
            assert not(self.file_urls[index] is not None and self.file_paths[index] is None)
            if self.file_paths[index] is not None and self.vals[index] is None:
                self.vals[index] = PIL.Image.open(self.file_paths[index])

        if index in self.video_slot_idxs:
            assert self.file_paths[index] is not None and self.file_paths[index] == self.vals[index] \
               or self.file_urls[index] is not None and self.file_urls[index] == self.vals[index]

        return self.vals[index]

    def get_stored_val(self, index: object) -> Any:
        """Return the value that gets stored in the db"""
        assert self.excs[index] is None
        if not self.has_val[index]:
            # for debugging purposes
            pass
        assert self.has_val[index]
        if index in self.img_slot_idxs or index in self.video_slot_idxs:
            # if this is an image or video we want to store, we should have a url
            assert self.file_urls[index] is not None
            return self.file_urls[index]
        if index in self.array_slot_idxs:
            assert isinstance(self.vals[index], np.ndarray)
            np_array = self.vals[index]
            buffer = io.BytesIO()
            np.save(buffer, np_array)
            return buffer.getvalue()
        return self.vals[index]

    def __setitem__(self, idx: object, val: Any) -> None:
        """Assign in-memory cell value
        This allows overwriting
        """
        assert self.excs[idx] is None

        if (idx in self.img_slot_idxs or idx in self.video_slot_idxs) and isinstance(val, str):
            # this is either a local file path or a URL
            parsed = urllib.parse.urlparse(val)
            if parsed.scheme == '' or parsed.scheme == 'file':
                # local file path
                assert self.file_urls[idx] is None and self.file_paths[idx] is None
                if parsed.scheme == '':
                    self.file_urls[idx] = urllib.parse.urljoin('file:', urllib.request.pathname2url(parsed.path))
                else:
                    self.file_urls[idx] = val
                self.file_paths[idx] = urllib.parse.unquote(parsed.path)
            else:
                # URL
                assert self.file_urls[idx] is None
                self.file_urls[idx] = val

            if idx in self.video_slot_idxs:
                self.vals[idx] = self.file_paths[idx] if self.file_paths[idx] is not None else self.file_urls[idx]
        elif idx in self.array_slot_idxs and isinstance(val, bytes):
            self.vals[idx] = np.load(io.BytesIO(val))
        else:
            self.vals[idx] = val
        self.has_val[idx] = True

    def set_file_path(self, idx: object, path: str) -> None:
        """Augment an existing url with a local file path"""
        assert self.has_val[idx]
        assert idx in self.img_slot_idxs or idx in self.video_slot_idxs
        self.file_paths[idx] = path
        if idx in self.video_slot_idxs:
            self.vals[idx] = path

    def flush_img(self, index: object, filepath: Optional[str] = None) -> None:
        """Discard the in-memory value and save it to a local file, if filepath is not None"""
        if self.vals[index] is None:
            return
        assert self.excs[index] is None
        if self.file_paths[index] is None:
            if filepath is not None:
                # we want to save this to a file
                self.file_paths[index] = filepath
                self.file_urls[index] = urllib.parse.urljoin('file:', urllib.request.pathname2url(filepath))
                self.vals[index].save(filepath, format='JPEG')
            else:
                # we discard the content of this cell
                self.has_val[index] = False
        else:
            # we already have a file for this image, nothing left to do
            pass
        self.vals[index] = None


class ExecProfile:
    def __init__(self, row_builder: RowBuilder):
        self.eval_time = [0.0] * row_builder.num_materialized
        self.eval_count = [0] * row_builder.num_materialized
        self.row_builder = row_builder

    def print(self, num_rows: int) -> str:
        for i in range(self.row_builder.num_materialized):
            if self.eval_count[i] == 0:
                continue
            per_call_time = self.eval_time[i] / self.eval_count[i]
            calls_per_row = self.eval_count[i] / num_rows
            multiple_str = f'({calls_per_row}x)' if calls_per_row > 1 else ''
            print(f'{self.row_builder.unique_exprs[i]}: {print_perf_counter_delta(per_call_time)} {multiple_str}')


@dataclass
class ColumnSlotIdx:
    """Info for how to locate materialized column in DataRow
    TODO: can this be integrated into RowBuilder directly?
    """
    col: catalog.Column
    slot_idx: int


class RowBuilder:
    """Create and populate DataRows and table rows from exprs and computed columns

    For ColumnRefs to unstored iterator columns:
    - in order for them to be executable, we also record the iterator args and pass them to the ColumnRef
    """

    @dataclass
    class EvalCtx:
        """Context for evaluating a set of target exprs"""
        slot_idxs: List[int]  # slot idxs of exprs needed to evaluate target exprs; does not contain duplicates
        exprs: List[Expr]  # exprs corresponding to slot_idxs
        target_slot_idxs: List[int]  # slot idxs of target exprs; might contain duplicates
        target_exprs: List[Expr]  # exprs corresponding to target_slot_idxs

    def __init__(
            self, output_exprs: List[Expr], columns: List[catalog.Column],
            indices: List[Tuple[catalog.Column, Function]], input_exprs: List[Expr],
            resolve_unstored_only: bool
    ):
        """
        Args:
            output_exprs: list of Exprs to be evaluated
            columns: list of columns to be materialized
            indices: list of embeddings to be materialized (Tuple[indexed column, embedding function])
        """
        self.unique_exprs = UniqueExprList()  # dependencies precede their dependents
        self.next_slot_idx = 0

        # record input and output exprs; make copies to avoid reusing execution state
        unique_input_exprs = [self._record_unique_expr(e.copy(), recursive=False) for e in input_exprs]
        self.input_expr_slot_idxs = {e.slot_idx for e in unique_input_exprs}

        # output exprs: all exprs the caller wants to materialize
        # - explicitly requested output_exprs
        # - values for computed columns
        # - embedding values for indices
        self.output_exprs = [
            self._record_unique_expr(e.copy().resolve_computed_cols(resolve_unstored_only), recursive=True)
            for e in output_exprs
        ]

        # record columns for create_table_row()
        self.table_columns: List[ColumnSlotIdx] = []
        for col in columns:
            if col.is_computed:
                assert col.value_expr is not None
                # create a copy here so we don't reuse execution state and resolve references to computed columns
                expr = col.value_expr.copy().resolve_computed_cols(unstored_only=resolve_unstored_only)
                expr = self._record_unique_expr(expr, recursive=True)
                self.add_table_column(col, expr.slot_idx)
                self.output_exprs.append(expr)
            else:
                # record a ColumnRef so that references to this column resolve to the same slot idx
                ref = ColumnRef(col)
                ref = self._record_unique_expr(ref, recursive=False)
                self.add_table_column(col, ref.slot_idx)

        # record indices; indexed by slot_idx
        self.index_columns: List[catalog.Column] = []
        for col, embedding_fn in indices:
            # we assume that the parameter of the embedding function is a ref to an image column
            assert col.col_type.is_image_type()
            # construct expr to compute embedding; explicitly resize images to the required size
            target_img_type = next(iter(embedding_fn.md.signature.parameters.values()))
            expr = embedding_fn(ColumnRef(col).resize(target_img_type.size))
            expr = self._record_unique_expr(expr, recursive=True)
            self.output_exprs.append(expr)
            if len(self.index_columns) <= expr.slot_idx:
                # pad to slot_idx
                self.index_columns.extend([None] * (expr.slot_idx - len(self.index_columns) + 1))
            self.index_columns[expr.slot_idx] = col

        # default eval ctx: all output exprs
        self.default_eval_ctx = self.create_eval_ctx(self.output_exprs, exclude=unique_input_exprs)

        # references to unstored iterator columns:
        # - those ColumnRefs need to instantiate iterators
        # - we create and record the iterator args here and pass them to their respective ColumnRefs
        col_refs = [e for e in self.unique_exprs if isinstance(e, ColumnRef)]
        def refs_unstored_iter_col(col_ref: ColumnRef) -> bool:
            tbl = col_ref.col.tbl
            return tbl.is_component_view() and tbl.is_iterator_column(col_ref.col) and not col_ref.col.is_stored
        unstored_iter_col_refs = [col_ref for col_ref in col_refs if refs_unstored_iter_col(col_ref)]
        component_views = [col_ref.col.tbl for col_ref in unstored_iter_col_refs]
        unstored_iter_args = {view.id: view.iterator_args.copy() for view in component_views}
        self.unstored_iter_args = \
            {id: self._record_unique_expr(arg, recursive=True) for id, arg in unstored_iter_args.items()}

        for col_ref in unstored_iter_col_refs:
            iter_arg_ctx = self.create_eval_ctx([unstored_iter_args[col_ref.col.tbl.id]])
            col_ref.set_iter_arg_ctx(iter_arg_ctx)

        # we guarantee that we can compute the expr DAG in a single front-to-back pass
        for i in range(1, len(self.unique_exprs)):
            assert self.unique_exprs.exprs[i].slot_idx > self.unique_exprs.exprs[i - 1].slot_idx

        # record transitive dependencies (list of set of slot_idxs, indexed by slot_idx)
        self.dependencies: List[Set[int]] = [set() for _ in range(self.num_materialized)]
        for expr in self.unique_exprs:
            if expr.slot_idx in self.input_expr_slot_idxs:
                # this is input and therefore doesn't depend on other exprs
                continue
            for d in expr.dependencies():
                self.dependencies[expr.slot_idx].add(d.slot_idx)
                self.dependencies[expr.slot_idx].update(self.dependencies[d.slot_idx])

        # derive transitive dependents
        self.dependents: List[Set[int]] = [set() for _ in range(self.num_materialized)]
        for expr in self.unique_exprs:
            for d in self.dependencies[expr.slot_idx]:
                self.dependents[d].add(expr.slot_idx)

        # records the output_expr that a subexpr belongs to
        # (a subexpr can be shared across multiple output exprs)
        self.output_expr_ids: List[Set[int]] = [set() for _ in range(self.num_materialized)]
        for e in self.output_exprs:
            self._record_output_expr_id(e, e.slot_idx)

    def add_table_column(self, col: catalog.Column, slot_idx: int) -> None:
        """Record a column that is part of the table row"""
        self.table_columns.append(ColumnSlotIdx(col, slot_idx))

    def output_slot_idxs(self) -> List[ColumnSlotIdx]:
        """Return ColumnSlotIdx for output columns"""
        return self.table_columns

    def index_slot_idxs(self) -> List[ColumnSlotIdx]:
        """Return ColumnSlotIdx for index columns"""
        return [
            ColumnSlotIdx(self.output_columns[i], i) for i in range(len(self.index_columns))
            if self.output_columns[i] is not None
        ]

    @property
    def num_materialized(self) -> int:
        return self.next_slot_idx

    def get_output_exprs(self) -> List[Expr]:
        """Returns exprs that were requested in the c'tor and require evaluation"""
        return self.output_exprs

    def _next_slot_idx(self) -> int:
        result = self.next_slot_idx
        self.next_slot_idx += 1
        return result

    def _record_unique_expr(self, expr: Expr, recursive: bool) -> Expr:
        """Records the expr if it's not a duplicate and assigns a slot idx to expr and its components"
        Returns:
            the unique expr
        """
        if expr in self.unique_exprs:
            # expr is a duplicate: we use the original instead
            return self.unique_exprs[expr]

        # expr value needs to be computed via Expr.eval()
        if recursive:
            for i, c in enumerate(expr.components):
                # make sure we only refer to components that have themselves been recorded
                expr.components[i] = self._record_unique_expr(c, True)
        assert expr.slot_idx < 0
        expr.slot_idx = self._next_slot_idx()
        self.unique_exprs.append(expr)
        return expr

    def _record_output_expr_id(self, e: Expr, output_expr_id: int) -> None:
        self.output_expr_ids[e.slot_idx].add(output_expr_id)
        for d in e.dependencies():
            self._record_output_expr_id(d, output_expr_id)

    def _compute_dependencies(self, target_slot_idxs: List[int], excluded_slot_idxs: List[int]) -> List[int]:
        """Compute exprs needed to materialize the given target slots, excluding 'excluded_slot_idxs'"""
        dependencies = [set() for _ in range(self.num_materialized)]  # indexed by slot_idx
        # doing this front-to-back ensures that we capture transitive dependencies
        max_target_slot_idx = max(target_slot_idxs)
        for expr in self.unique_exprs:
            if expr.slot_idx > max_target_slot_idx:
                # we're done
                break
            if expr.slot_idx in excluded_slot_idxs:
                continue
            if expr.slot_idx in self.input_expr_slot_idxs:
                # this is input and therefore doesn't depend on other exprs
                continue
            for d in expr.dependencies():
                if d.slot_idx in excluded_slot_idxs:
                    continue
                dependencies[expr.slot_idx].add(d.slot_idx)
                dependencies[expr.slot_idx].update(dependencies[d.slot_idx])
        # merge dependencies and convert to list
        return sorted(set().union(*[dependencies[i] for i in target_slot_idxs]))

    def substitute_exprs(self, expr_list: List[Expr], remove_duplicates: bool = True) -> None:
        """Substitutes exprs with their executable counterparts from unique_exprs and optionally removes duplicates"""
        i = 0
        unique_ids: Set[i] = set()  # slot idxs within expr_list
        while i < len(expr_list):
            unique_expr = self.unique_exprs[expr_list[i]]
            if unique_expr.slot_idx in unique_ids and remove_duplicates:
                del expr_list[i]
            else:
                expr_list[i] = unique_expr
                unique_ids.add(unique_expr.slot_idx)
                i += 1

    def get_dependencies(self, targets: List[Expr], exclude: List[Expr] = []) -> List[Expr]:
        """
        Return list of dependencies needed to evaluate the given target exprs (expressed as slot idxs).
        The exprs given in 'exclude' are excluded.
        Returns:
            list of Exprs from unique_exprs (= with slot_idx set)
        """
        if len(targets) == 0:
            return []
        # make sure we only refer to recorded exprs
        targets = [self.unique_exprs[e] for e in targets]
        exclude = [self.unique_exprs[e] for e in exclude]
        target_slot_idxs = [e.slot_idx for e in targets]
        excluded_slot_idxs = [e.slot_idx for e in exclude]
        all_dependencies = set(self._compute_dependencies(target_slot_idxs, excluded_slot_idxs))
        all_dependencies.update(target_slot_idxs)
        result_ids = list(all_dependencies)
        result_ids.sort()
        return [self.unique_exprs[id] for id in result_ids]

    def create_eval_ctx(self, targets: List[Expr], exclude: List[Expr] = []) -> EvalCtx:
        """Return EvalCtx for targets"""
        if len(targets) == 0:
            return self.EvalCtx([], [], [], [])
        dependencies = self.get_dependencies(targets, exclude)
        targets = [self.unique_exprs[e] for e in targets]
        target_slot_idxs = [e.slot_idx for e in targets]
        ctx_slot_idxs = [e.slot_idx for e in dependencies]
        return self.EvalCtx(
            slot_idxs=ctx_slot_idxs, exprs=[self.unique_exprs[slot_idx] for slot_idx in ctx_slot_idxs],
            target_slot_idxs=target_slot_idxs, target_exprs=targets)

    def eval(
            self, data_row: DataRow, ctx: EvalCtx, profile: Optional[ExecProfile] = None, ignore_errors: bool = False
    ) -> None:
        """
        Populates the slots in data_row given in ctx.
        If an expr.eval() raises an exception, records the exception in the corresponding slot of data_row
        and omits any of that expr's dependents's eval().
        profile: if present, populated with execution time of each expr.eval() call; indexed by expr.slot_idx
        ignore_errors: if False, raises ExprEvalError if any expr.eval() raises an exception
        """
        for expr in ctx.exprs:
            assert expr.slot_idx >= 0
            if data_row.has_val[expr.slot_idx] or data_row.has_exc(expr.slot_idx):
                continue
            try:
                start_time = time.perf_counter()
                expr.eval(data_row, self)
                if profile is not None:
                    profile.eval_time[expr.slot_idx] += time.perf_counter() - start_time
                    profile.eval_count[expr.slot_idx] += 1
            except Exception as exc:
                _, _, exc_tb = sys.exc_info()
                # propagate exception to dependents
                data_row.set_exc(expr.slot_idx, exc)
                for slot_idx in self.dependents[expr.slot_idx]:
                    data_row.set_exc(slot_idx, exc)
                if not ignore_errors:
                    input_vals = [data_row[d.slot_idx] for d in expr.dependencies()]
                    raise ExprEvalError(
                        expr, f'expression {expr}', data_row.get_exc(expr.slot_idx), exc_tb, input_vals, 0)

    def create_table_row(self, data_row: DataRow, exc_col_ids: Set[int]) -> Tuple[Dict[str, Any], int]:
        """Create a table row from the slots that have an output column assigned"""
        """Return Tuple[dict that represents a stored row (can be passed to sql.insert()), # of exceptions]
            This excludes system columns.
        """
        num_excs = 0
        table_row: Dict[str, Any] = {}
        for info in self.table_columns:
            col, slot_idx = info.col, info.slot_idx
            if data_row.has_exc(slot_idx):
                # exceptions get stored in the errortype/-msg columns
                exc = data_row.get_exc(slot_idx)
                num_excs += 1
                exc_col_ids.add(col.id)
                table_row[col.storage_name()] = None
                table_row[col.errortype_storage_name()] = type(exc).__name__
                table_row[col.errormsg_storage_name()] = str(exc)
            else:
                val = data_row.get_stored_val(slot_idx)
                table_row[col.storage_name()] = val
                # we unfortunately need to set these, even if there are no errors
                table_row[col.errortype_storage_name()] = None
                table_row[col.errormsg_storage_name()] = None

        for slot_idx, col in enumerate(self.index_columns):
            if col is None:
                continue
            # don't use get_stored_val() here, we need to pass in the ndarray
            val = data_row[slot_idx]
            table_row[col.index_storage_name()] = val

        return table_row, num_excs


class UniqueExprList:
    """
    A List[Expr] which ignores duplicates and which supports [] access by Expr.equals().
    We can't use set() because Expr doesn't have a __hash__() and Expr.__eq__() has been repurposed.

    TODO: now that we have Expr.id, replace with UniqueExprs, implemented as a Dict[Expr.id, Expr]
    """
    def __init__(self, elements: Optional[Iterable[Expr]] = None):
        self.exprs: List[Expr] = []
        if elements is not None:
            for e in elements:
                self.append(e)

    def append(self, expr: Expr) -> None:
        try:
            _ = next(e for e in self.exprs if e.equals(expr))
        except StopIteration:
            self.exprs.append(expr)

    def extend(self, elements: Iterable[Expr]) -> None:
        for e in elements:
            self.append(e)

    def __contains__(self, item: Expr) -> bool:
        assert isinstance(item, Expr)
        try:
            _ = next(e for e in self.exprs if e.equals(item))
            return True
        except StopIteration:
            return False

    def contains(self, cls: Type[Expr]) -> bool:
        try:
            _ = next(e for e in self.exprs if isinstance(e, cls))
            return True
        except StopIteration:
            return False

    def __len__(self) -> int:
        return len(self.exprs)

    def __iter__(self) -> Iterator[Expr]:
        return iter(self.exprs)

    def __getitem__(self, index: object) -> Optional[Expr]:
        assert isinstance(index, int) or isinstance(index, Expr)
        if isinstance(index, int):
            # return expr with matching slot_idx
            return [e for e in self.exprs if e.slot_idx == index][0]
        else:
            try:
                return next(e for e in self.exprs if e.equals(index))
            except StopIteration:
                return None
