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

import PIL.Image
import jmespath
import numpy as np
import sqlalchemy as sql
import nos

from pixeltable import catalog
from pixeltable.type_system import \
    ColumnType, InvalidType, StringType, IntType, FloatType, BoolType, JsonType, ArrayType
from pixeltable.function import Function, FunctionRegistry
from pixeltable.exceptions import Error, ExprEvalError
from pixeltable.utils.video import FrameIterator
from pixeltable.utils import print_perf_counter_delta
from pixeltable.utils.clip import embed_image, embed_text

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
        # index of the expr's value in the data row; set for all materialized exprs; -1: invalid
        # not set for subexprs that don't need to be materialized because the parent can be materialized via SQL
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

    def display_name(self) -> str:
        """
        Displayed column name in DataFrame. '': assigned by DataFrame
        """
        return ''

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
        """
        Produce subexprs for all exprs in list.
        """
        for e in expr_list:
            yield from e.subexprs(expr_class=expr_class, filter=filter, traverse_matches=traverse_matches)

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
    def sql_expr(self) -> Optional[sql.sql.expression.ClauseElement]:
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
    def eval(self, data_row: DataRow, evaluator: Evaluator) -> None:
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
    """
    A reference to a table column that is materialized in the store (ie, corresponds to a column in the store).
    """
    class Property(enum.Enum):
        VALUE = 0
        ERRORTYPE = 1
        ERRORMSG = 2

    def __init__(self, col: catalog.Column, prop: Property = Property.VALUE):
        assert col.tbl is not None
        self.col = col
        self.prop = prop
        if prop == self.Property.VALUE:
            col_type = col.col_type
        else:
            col_type = StringType(nullable=True)
        super().__init__(col_type)

    def __getattr__(self, name: str) -> Expr:
        if name == self.Property.ERRORTYPE.name.lower():
            if not self.col.is_computed:
                raise Error(f'{name} not valid for a non-computed column: {self}')
            return ColumnRef(self.col, self.Property.ERRORTYPE)
        if name == self.Property.ERRORMSG.name.lower():
            if not self.col.is_computed:
                raise Error(f'{name} not valid for a non-computed column: {self}')
            return ColumnRef(self.col, self.Property.ERRORMSG)
        if self.col_type.is_json_type():
            return JsonPath(self).__getattr__(name)
        return super().__getattr__(name)

    def display_name(self) -> str:
        return str(self)

    def _equals(self, other: ColumnRef) -> bool:
        return self.col == other.col and self.prop == other.prop

    def __str__(self) -> str:
        if self.prop == self.Property.VALUE:
            return self.col.name
        return f'{self.col.name}.{self.prop.name.lower()}'

    def sql_expr(self) -> Optional[sql.sql.expression.ClauseElement]:
        if not self.col.is_stored:
            return None
        if self.prop == self.Property.VALUE:
            assert self.col.sa_col is not None
            return self.col.sa_col
        if self.prop == self.Property.ERRORTYPE:
            assert self.col.sa_errortype_col is not None
            return self.col.sa_errortype_col
        if self.prop == self.Property.ERRORMSG:
            assert self.col.sa_errormsg_col is not None
            return self.col.sa_errormsg_col

    def eval(self, data_row: DataRow, evaluator: Evaluator) -> None:
        # we get called while materializing computed cols
        pass

    def _as_dict(self) -> Dict:
        return {'tbl_id': str(self.col.tbl.id), 'col_id': self.col.id, 'prop': self.prop.value}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr], t: catalog.TableVersion) -> Expr:
        assert 'col_id' in d
        assert t.id is not None
        origin_id = UUID(d['tbl_id'])
        origin = t
        while origin_id != origin.id:
            # only views can reference Tables other than themselves
            assert t.is_view() and t.base is not None
            origin = t.base

        col_id = d['col_id']
        assert col_id in origin.cols_by_id
        result = cls(origin.cols_by_id[col_id])
        if d['prop'] != cls.Property.VALUE.value:
            result = getattr(result, cls.Property(d['prop']).name.lower())
        return result


class FrameColumnRef(ColumnRef):
    """
    Reference to an extracted frame column.
    TODO: create window function that can do frame extraction and remove this class.
    """
    def __init__(self, col: catalog.Column):
        assert col.col_type.is_image_type()
        assert col.tbl.parameters.frame_col_id == col.id
        super().__init__(col)
        # we need to reference the video and frame idx cols in eval()
        self.tbl = col.tbl
        video_col = self.tbl.cols_by_id[self.tbl.parameters.frame_src_col_id]
        frame_idx_col = self.tbl.cols_by_id[self.tbl.parameters.frame_idx_col_id]
        self.components = [ColumnRef(video_col), ColumnRef(frame_idx_col)]
        # execution state
        self.current_video: Optional[str] = None
        self.frames: Optional[FrameIterator] = None

    @property
    def _video_ref(self) -> ColumnRef:
        return self.components[0]

    @property
    def _frame_idx_ref(self) -> ColumnRef:
        return self.components[1]

    def sql_expr(self) -> Optional[sql.sql.expression.ClauseElement]:
        return None

    def eval(self, data_row: DataRow, evaluator: Evaluator) -> None:
        # extract frame
        video_path = data_row[self._video_ref.slot_idx]
        if self.frames is None or self.current_video != video_path:
            self.current_video = video_path
            if self.frames is not None:
                self.frames.close()
            self.frames = FrameIterator(self.current_video, fps=self.tbl.parameters.extraction_fps)
        frame_idx = data_row[self._frame_idx_ref.slot_idx]
        self.frames.seek(frame_idx)
        _, frame = next(self.frames, None)
        data_row[self.slot_idx] = frame

    def release(self) -> None:
        if self.frames is not None:
            self.frames.close()
            self.frames = None


class FunctionCall(Expr):
    def __init__(
            self, fn: Function, bound_args: Dict[str, Any], order_by_exprs: List[Expr] = [],
            group_by_exprs: List[Expr] = [], is_method_call: bool = False):
        signature = fn.md.signature
        super().__init__(signature.get_return_type(bound_args))
        self.fn = fn
        self.is_method_call = is_method_call

        # check argument types and values
        for param_name, arg in bound_args.items():
            if not isinstance(arg, Expr):
                # make sure that non-Expr args are json-serializable
                try:
                    _ = json.dumps(arg)
                    continue
                except TypeError:
                    raise Error(f'Argument for parameter {param_name} is not json-serializable: {arg}')

            param_type = signature.parameters[param_name]
            if not param_type.is_supertype_of(arg.col_type):
                raise Error((
                    f'Parameter {param_name}: argument type {arg.col_type} does not match parameter type '
                    f'{param_type}'))

        # construct components, args, kwargs
        self.components: List[Expr] = []
        # Tuple[int, Any]:
        # - for Exprs: (index into components, None)
        # - otherwise: (-1, val)
        self.args: List[Tuple[int, Any]] = []
        self.arg_types: List[ColumnType] = []  # needed for runtime type checks
        self.kwargs: Dict[str, Tuple[int, Any]] = {}
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
        if len(group_by_exprs) > 0:
            # record grouping exprs in self.components, we need to evaluate them to get partition vals
            self.group_by_start_idx = len(self.components)
            self.group_by_stop_idx = len(self.components) + len(group_by_exprs)
            self.components.extend(group_by_exprs)
        # we want to make sure that order_by_exprs get assigned slot_idxs, even though we won't need to evaluate them
        # (that's done in SQL)
        self.order_by_start_idx = len(self.components)
        self.components.extend(order_by_exprs)

        self.nos_info = FunctionRegistry.get().get_nos_info(self.fn)
        self.constant_args = {param_name for param_name, arg in bound_args.items() if not isinstance(arg, Expr)}

        # execution state for aggregate functions
        self.aggregator: Optional[Any] = None
        self.current_partition_vals: Optional[List[Any]] = None

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

    def sql_expr(self) -> Optional[sql.sql.expression.ClauseElement]:
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

    def eval(self, data_row: DataRow, evaluator: Evaluator) -> None:
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
            Function.from_dict(d['fn']), bound_args, group_by_exprs=group_by_exprs, order_by_exprs=order_by_exprs)
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

    def display_name(self) -> str:
        return self.member_name

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

    def sql_expr(self) -> Optional[sql.sql.expression.ClauseElement]:
        return None

    def eval(self, data_row: DataRow, evaluator: Evaluator) -> None:
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

    def display_name(self) -> str:
        anchor_name = self._anchor.display_name() if self._anchor is not None else ''
        return f'{anchor_name}.{self._json_path()}'

    def _equals(self, other: JsonPath) -> bool:
        return self.path_elements == other.path_elements

    def sql_expr(self) -> Optional[sql.sql.expression.ClauseElement]:
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

    def eval(self, data_row: DataRow, evaluator: Evaluator) -> None:
        val = data_row[self._anchor.slot_idx]
        if self.compiled_path is not None:
            val = self.compiled_path.search(val)
        data_row[self.slot_idx] = val


RELATIVE_PATH_ROOT = JsonPath(None)


class Literal(Expr):
    def __init__(self, val: Any, col_type: Optional[ColumnType] = None):
        if col_type is not None:
            col_type.validate_literal(val)
        else:
            # try to determine a type for val
            col_type = ColumnType.infer_literal_type(val)
            if col_type is None:
                raise TypeError(f'Not a valid literal: {val}')
        super().__init__(col_type)
        self.val = val

    def display_name(self) -> str:
        return 'Literal'

    def __str__(self) -> str:
        if self.col_type.is_string_type() or self.col_type.is_timestamp_type():
            return f"'{self.val}'"
        return str(self.val)

    def _equals(self, other: Literal) -> bool:
        return self.val == other.val

    def sql_expr(self) -> Optional[sql.sql.expression.ClauseElement]:
        # we need to return something here so that we can generate a Where clause for predicates
        # that involve literals (like Where c > 0)
        return sql.sql.expression.literal(self.val)

    def eval(self, data_row: DataRow, evaluator: Evaluator) -> None:
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

    def sql_expr(self) -> Optional[sql.sql.expression.ClauseElement]:
        return None

    def eval(self, data_row: DataRow, evaluator: Evaluator) -> None:
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

    def __str__(self) -> str:
        elem_strs = [str(val) if val is not None else str(self.components[idx]) for idx, val in self.elements]
        return f'[{", ".join(elem_strs)}]'

    def _equals(self, other: InlineDict) -> bool:
        return self.elements == other.elements

    def sql_expr(self) -> Optional[sql.sql.expression.ClauseElement]:
        return None

    def eval(self, data_row: DataRow, evaluator: Evaluator) -> None:
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

    def sql_expr(self) -> Optional[sql.sql.expression.ClauseElement]:
        return None

    def eval(self, data_row: DataRow, evaluator: Evaluator) -> None:
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

    def extract_sql_predicate(self) -> Tuple[Optional[sql.sql.expression.ClauseElement], Optional[Predicate]]:
        """
        Return ClauseElement for what can be evaluated in SQL and a predicate for the remainder that needs to be
        evaluated in Python.
        Needed to for predicate push-down into SQL.
        """
        e = self.sql_expr()
        return (None, self) if e is None else (e, None)

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

    def extract_sql_predicate(self) -> Tuple[Optional[sql.sql.expression.ClauseElement], Optional[Predicate]]:
        if self.operator == LogicalOperator.NOT:
            e = self.sql_expr()
            return (None, self) if e is None else (e, None)

        sql_exprs = [op.sql_expr() for op in self.components]
        if self.operator == LogicalOperator.OR and any(e is None for e in sql_exprs):
            # if any clause of a | can't be evaluated in SQL, we need to evaluate everything in Python
            return None, self
        if not(any(e is None for e in sql_exprs)):
            # we can do everything in SQL
            return self.sql_expr(), None

        assert self.operator == LogicalOperator.AND
        if not any(e is not None for e in sql_exprs):
            # there's nothing that can be done in SQL
            return None, self

        sql_preds = [e for e in sql_exprs if e is not None]
        other_preds = [self.components[i] for i, e in enumerate(sql_exprs) if e is None]
        assert len(sql_preds) > 0
        combined_sql_pred = sql.and_(*sql_preds)
        combined_other = self.make_conjunction(other_preds)
        return combined_sql_pred, combined_other

    def split_conjuncts(
            self, condition: Callable[[Predicate], bool]) -> Tuple[List[Predicate], Optional[Predicate]]:
        if self.operator == LogicalOperator.OR or self.operator == LogicalOperator.NOT:
            return super().split_conjuncts(condition)
        matches = [op for op in self.components if condition(op)]
        non_matches = [op for op in self.components if not condition(op)]
        return (matches, self.make_conjunction(non_matches))

    def sql_expr(self) -> Optional[sql.sql.expression.ClauseElement]:
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

    def eval(self, data_row: DataRow, evaluator: Evaluator) -> None:
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

    def __str__(self) -> str:
        return f'{self._op1} {self.operator} {self._op2}'

    def _equals(self, other: Comparison) -> bool:
        return self.operator == other.operator

    @property
    def _op1(self) -> Expr:
        return self.components[0]

    @property
    def _op2(self) -> Expr:
        return self.components[1]

    def sql_expr(self) -> Optional[sql.sql.expression.ClauseElement]:
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

    def eval(self, data_row: DataRow, evaluator: Evaluator) -> None:
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

    def embedding(self) -> np.ndarray:
        if self.text is not None:
            return embed_text(self.text)
        else:
            return embed_image(self.img)

    def __str__(self) -> str:
        return f'{str(self.img_col_ref)}.nearest({"<img>" if self.img is not None else self.text})'

    def _equals(self, other: ImageSimilarityPredicate) -> bool:
        return False

    def sql_expr(self) -> Optional[sql.sql.expression.ClauseElement]:
        return None

    def eval(self, data_row: DataRow, evaluator: Evaluator) -> None:
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

    def __str__(self) -> str:
        return f'{str(self.components[0])} == None'

    def _equals(self, other: ImageSimilarityPredicate) -> bool:
        return True

    def sql_expr(self) -> Optional[sql.sql.expression.ClauseElement]:
        e = self.components[0].sql_expr()
        if e is None:
            return None
        return e == None

    def eval(self, data_row: DataRow, evaluator: Evaluator) -> None:
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

    def __str__(self) -> str:
        # add parentheses around operands that are ArithmeticExprs to express precedence
        op1_str = f'({self._op1})' if isinstance(self._op1, ArithmeticExpr) else str(self._op1)
        op2_str = f'({self._op2})' if isinstance(self._op2, ArithmeticExpr) else str(self._op2)
        return f'{op1_str} {str(self.operator)} {op2_str}'

    def _equals(self, other: ArithmeticExpr) -> bool:
        return self.operator == other.operator

    @property
    def _op1(self) -> Expr:
        return self.components[0]

    @property
    def _op2(self) -> Expr:
        return self.components[1]

    def sql_expr(self) -> Optional[sql.sql.expression.ClauseElement]:
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

    def eval(self, data_row: DataRow, evaluator: Evaluator) -> None:
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

    def scope(self) -> ExprScope:
        return self._scope

    def __str__(self) -> str:
        assert False

    def _equals(self, other: ObjectRef) -> bool:
        return self.owner is other.owner

    def sql_expr(self) -> Optional[sql.sql.expression.ClauseElement]:
        return None

    def eval(self, data_row: DataRow, evaluator: Evaluator) -> None:
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
        self.target_expr_eval_ctx: List[Expr] = []

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

    def sql_expr(self) -> Optional[sql.sql.expression.ClauseElement]:
        return None

    def eval(self, data_row: DataRow, evaluator: Evaluator) -> None:
        # this will be called, but the value has already been materialized elsewhere
        src = data_row[self._src_expr.slot_idx]
        if not isinstance(src, list):
            # invalid/non-list src path
            data_row[self.slot_idx] = None
            return

        result = [None] * len(src)
        if len(self.target_expr_eval_ctx) == 0:
            self.target_expr_eval_ctx = evaluator.get_eval_ctx([self._target_expr])
        for i, val in enumerate(src):
            data_row[self.scope_anchor.slot_idx] = val
            # stored target_expr
            exc_tb = evaluator.eval(data_row, self.target_expr_eval_ctx)
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
    Encapsulates data and execution state needed by Evaluator and DataRowBatch.
    """
    def __init__(self, size: int):
        self.vals: List[Any] = [None] * size  # either cell values or exceptions
        self.has_val = [False] * size
        self.excs: List[Optional[Exception]] = [None] * size
        # the primary key of a row is a sequence of ints (the number is different for table vs view)
        self.pk: Optional[Tuple[int, ...]] = None

        # img_files:
        # - path of file for image in vals[i]
        # - None if vals[i] is not an image or if the image hasn't been written to a file yet
        self.img_files: Optional[str] = [None] * size

    def clear(self) -> None:
        size = len(self.vals)
        self.vals = [None] * size
        self.has_val = [False] * size
        self.excs = [None] * size
        self.pk = None
        self.img_files = [None] * size

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
        if not self.has_val[index]:
            pass
        assert self.has_val[index]
        if self.img_files[index] is not None and self.vals[index] is None:
            self.vals[index] = PIL.Image.open(self.img_files[index])
        return self.vals[index]

    def get_stored_val(self, index: object) -> Any:
        """Return the value that gets stored in the db"""
        assert self.excs[index] is None
        if not self.has_val[index]:
            pass
        assert self.has_val[index]
        if self.img_files[index] is not None:
            return str(self.img_files[index])
        if isinstance(self.vals[index], np.ndarray):
            np_array = self.vals[index]
            buffer = io.BytesIO()
            np.save(buffer, np_array)
            return buffer.getvalue()
        return self.vals[index]

    def __setitem__(self, index: object, val: Any) -> None:
        assert self.excs[index] is None
        # we allow overwriting
        self.vals[index] = val
        self.has_val[index] = True

    def flush_img(self, index: object, filepath: Optional[str] = None) -> None:
        if self.vals[index] is None:
                return
        assert self.excs[index] is None
        if self.img_files[index] is None:
            if filepath is not None:
                # we want to save this to a file
                self.img_files[index] = filepath
                self.vals[index].save(filepath, format='JPEG')
            else:
                # we discard the content of this cell
                self.has_val[index] = False
        else:
            # we already have a file for this image, nothing left to do
            pass
        self.vals[index] = None

    def __len__(self) -> int:
        return len(self.vals)

    def __copy__(self):
        result = DataRow(len(self.vals))
        result.vals = copy.copy(self.vals)
        result.has_val = copy.copy(self.has_val)
        return result


class ExecProfile:
    def __init__(self, evaluator: Evaluator):
        self.eval_time = [0.0] * evaluator.num_materialized
        self.eval_count = [0] * evaluator.num_materialized
        self.evaluator = evaluator

    def print(self, num_rows: int) -> str:
        for i in range(self.evaluator.num_materialized):
            if self.eval_count[i] == 0:
                continue
            per_call_time = self.eval_time[i] / self.eval_count[i]
            calls_per_row = self.eval_count[i] / num_rows
            multiple_str = f'({calls_per_row}x)' if calls_per_row > 1 else ''
            print(f'{self.evaluator.unique_exprs[i]}: {print_perf_counter_delta(per_call_time)} {multiple_str}')


class Evaluator:
    """
    Evaluates a list of Exprs against a data row.
    """

    def __init__(self, output_exprs: List[Expr], input_exprs: List[Expr] = []):
        """
        Set up Evaluator to evaluate any Expr in expr_list.
        If with_sql == True, assumes that exprs that have a SQL equivalent (to_sql() != None) will be materialized
        via SQL, and that none of its dependencies will need to be materialized.

        Args:
            output_exprs: list of Exprs to be evaluated
            input_exprs: list of Exprs that are assigned a slot_idx but aren't evaluated (expected to be input)
        """
        # all following list are indexed with slot_idx
        self.unique_exprs = UniqueExprList()  # dependencies precede their dependents
        self.next_slot_idx = 0

        # we start by assigning slot_idxs to input exprs
        for e in input_exprs:
            self._assign_idxs(e, recursive=False)
        self.input_expr_slot_idxs = [e.slot_idx for e in input_exprs]
        for e in output_exprs:
            self._assign_idxs(e, recursive=True)

        for i in range(len(self.unique_exprs)):
            assert self.unique_exprs[i].slot_idx == i

        # record transitive dependencies
        self.dependencies: Set[int] = [set() for _ in range(self.num_materialized)]
        for i in range(self.num_materialized):
            expr = self.unique_exprs[i]
            if expr.slot_idx in self.input_expr_slot_idxs:
                # this is input and therefore doesn't depend on other exprs
                continue
            for d in self.unique_exprs[i].dependencies():
                self.dependencies[i].add(d.slot_idx)
                self.dependencies[i].update(self.dependencies[d.slot_idx])

        # derive transitive dependents
        self.dependents: Set[int] = [set() for _ in range(self.num_materialized)]
        for i in range(self.num_materialized):
            for j in self.dependencies[i]:
                self.dependents[j].add(i)

        # records the Exprs in parameter 'expr_list' that a subexpr belongs to
        # (a subexpr can be shared across multiple output exprs)
        self.output_expr_ids = [set() for _ in range(self.num_materialized)]
        for e in output_exprs:
            self._record_output_expr_id(e, e.slot_idx)

    def _compute_dependencies(self, target_slot_idxs: List[int], excluded_slot_idxs: List[int]) -> List[int]:
        """Compute exprs needed to materialize the given target slots, but excluding 'excluded_slot_idxs'"""
        dependencies = [set() for _ in range(self.num_materialized)]
        # doing this front-to-back ensures that we capture transitive dependencies
        for i in range(max(target_slot_idxs) + 1):
            expr = self.unique_exprs[i]
            if expr.slot_idx in excluded_slot_idxs:
                continue
            if expr.slot_idx in self.input_expr_slot_idxs:
                # this is input and therefore doesn't depend on other exprs
                continue
            for d in expr.dependencies():
                if d.slot_idx in excluded_slot_idxs:
                    continue
                dependencies[i].add(d.slot_idx)
                dependencies[i].update(dependencies[d.slot_idx])
        # merge dependencies and convert to list
        return sorted(set().union(*[dependencies[i] for i in target_slot_idxs]))

    def _assign_idxs(self, expr: Expr, recursive: bool) -> None:
        if expr in self.unique_exprs:
            # expr is a duplicate: we copy slot_idx for itself and its components
            original = self.unique_exprs[expr]
            expr.slot_idx = original.slot_idx
            if recursive and expr.slot_idx not in self.input_expr_slot_idxs:
                for c in expr.components:
                    self._assign_idxs(c, True)
            # nothing left to do
            return

        # expr value needs to be computed via Expr.eval()
        if recursive:
            for c in expr.components:
                self._assign_idxs(c, True)
        assert expr.slot_idx < 0
        expr.slot_idx = self.next_slot_idx
        self.next_slot_idx += 1
        self.unique_exprs.append(expr)

    def _record_output_expr_id(self, e: Expr, output_expr_id: int) -> None:
        self.output_expr_ids[e.slot_idx].add(output_expr_id)
        for d in e.dependencies():
            self._record_output_expr_id(d, output_expr_id)

    @property
    def num_materialized(self) -> int:
        return self.next_slot_idx

    def prepare(self) -> DataRow:
        """
        Prepare for evaluation by initializing data_row.
        object is reused across multiple executions of the same query.
        Returns data row.
        """
        return DataRow(self.num_materialized)

    def get_eval_ctx(self, targets: List[Expr], exclude: List[Expr] = []) -> List[Expr]:
        """
        Return list of dependencies needed to evaluate the given target exprs (expressed as slot idxs).
        The exprs given in 'exclude' are also excluded.
        """
        if len(targets) == 0:
            return []
        target_slot_idxs = [e.slot_idx for e in targets]
        excluded_slot_idxs = [e.slot_idx for e in exclude]
        all_dependencies = set(self._compute_dependencies(target_slot_idxs, excluded_slot_idxs))
        all_dependencies.update(target_slot_idxs)
        result_ids = list(all_dependencies)
        result_ids.sort()
        return [self.unique_exprs[id] for id in result_ids]

    def get_dependency_exc(self, data_row: DataRow, e: Expr) -> Optional[Exception]:
        """Return the first exception in our immediate dependencies, or None if our dependencies don't have any"""
        for d in e.dependencies():
            if data_row.has_exc(d.slot_idx):
                return data_row.get_exc(d.slot_idx)
        return None

    def eval(
            self, data_row: DataRow, ctx: List[Expr], profile: Optional[ExecProfile] = None, ignore_errors: bool = False
    ) -> None:
        """
        Populates the slots in data_row given in ctx.
        If an expr.eval() raises an exception, records the exception in the corresponding slot of data_row
        and omits any of that expr's dependents's eval().
        profile: if present, populated with execution time of each expr.eval() call; indexed by expr.slot_idx
        ignore_errors: if False, raises ExprEvalError if any expr.eval() raises an exception
        """
        for expr in ctx:
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


class UniqueExprList:
    """
    A List[Expr] which ignores duplicates and which supports [] access by Expr.equals().
    We can't use set() because Expr doesn't have a __hash__() and Expr.__eq__() has been repurposed.
    """
    def __init__(self, elements: Optional[Iterable[Expr]] = None):
        self.unique_exprs: List[Expr] = []
        if elements is not None:
            for e in elements:
                self.append(e)

    def append(self, expr: Expr) -> None:
        try:
            _ = next(e for e in self.unique_exprs if e.equals(expr))
        except StopIteration:
            self.unique_exprs.append(expr)

    def extend(self, elements: Iterable[Expr]) -> None:
        for e in elements:
            self.append(e)

    def __contains__(self, item: Expr) -> bool:
        assert isinstance(item, Expr)
        try:
            _ = next(e for e in self.unique_exprs if e.equals(item))
            return True
        except StopIteration:
            return False

    def contains(self, cls: Type[Expr]) -> bool:
        try:
            _ = next(e for e in self.unique_exprs if isinstance(e, cls))
            return True
        except StopIteration:
            return False

    def __len__(self) -> int:
        return len(self.unique_exprs)

    def __iter__(self) -> Iterator[Expr]:
        return iter(self.unique_exprs)

    def __getitem__(self, index: object) -> Optional[Expr]:
        assert isinstance(index, int) or isinstance(index, Expr)
        if isinstance(index, int):
            return self.unique_exprs[index]
        else:
            try:
                return next(e for e in self.unique_exprs if e.equals(index))
            except StopIteration:
                return None
