from __future__ import annotations
import abc
import copy
import datetime
import enum
import sys
import typing
from typing import Union, Optional, List, Callable, Any, Dict, Tuple, Set, Generator, Iterator
from types import TracebackType
import operator
import json
import io
from collections.abc import Iterable

import PIL.Image
import jmespath
import numpy as np
import sqlalchemy as sql

from pixeltable import catalog
from pixeltable.type_system import \
    ColumnType, InvalidType, StringType, IntType, FloatType, BoolType, TimestampType, ImageType, JsonType, ArrayType
from pixeltable.function import Function
from pixeltable import exceptions as exc
from pixeltable.utils import clip
from pixeltable.utils.video import FrameIterator

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
    - all state except for components and data/sql_row_idx is shared between copies of an Expr
    - data/sql_row_idx is set during analysis (DataFrame.show())
    - during eval(), components can only be accessed via self.components; any Exprs outside of that won't
      have data_row_idx set
    """
    def __init__(self, col_type: ColumnType):
        self.col_type = col_type
        # index of the expr's value in the data row; set for all materialized exprs; -1: invalid
        # not set for subexprs that don't need to be materialized because the parent can be materialized via SQL
        self.data_row_idx = -1
        # index of the expr's value in the SQL row; only set for exprs that can be materialized in SQL; -1: invalid
        self.sql_row_idx = -1
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
        Creates a copy that can be evaluated separately: it doesn't share any eval context (data/sql_row_idx)
        but shares everything else (catalog objects, etc.)
        """
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        result.data_row_idx = -1
        result.sql_row_idx = -1
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
        Replace 'old' with 'new recursively.
        """
        if self.equals(old):
            return new.copy()
        for i in range(len(self.components)):
            self.components[i] = self.components[i].substitute(old, new)
        return self

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

    def subexprs(self, filter: Optional[Callable[[Expr], bool]] = None) -> Generator[Expr, None, None]:
        """
        Iterate over all subexprs, including self.
        """
        for c in self.components:
            yield from c.subexprs(filter)
        if filter is None or filter(self):
            yield self

    @classmethod
    def list_subexprs(
            cls, expr_list: List[Expr], filter: Optional[Callable[[Expr], bool]] = None
    ) -> Generator[Expr, None, None]:
        """
        Produce subexprs for all exprs in list.
        """
        for e in expr_list:
            yield from e.subexprs(filter)

    @classmethod
    def from_object(cls, o: object) -> Optional[Expr]:
        """
        Try to turn a literal object into an Expr.
        """
        if isinstance(o, Expr):
            return o
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
        Compute the expr value for data_row and store the result in data_row[data_row_idx].
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
    def deserialize(cls, dict_str: str, t: catalog.Table) -> Expr:
        return cls.from_dict(json.loads(dict_str), t)

    @classmethod
    def from_dict(cls, d: Dict, t: catalog.Table) -> Expr:
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
    def from_dict_list(cls, dict_list: List[Dict], t: catalog.Table) -> List[Expr]:
        return [cls.from_dict(d, t) for d in dict_list]

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr], t: catalog.Table) -> Expr:
        assert False, 'not implemented'

    def __getitem__(self, index: object) -> Expr:
        if self.col_type.is_json_type():
            return JsonPath(self).__getitem__(index)
        if self.col_type.is_array_type():
            return ArraySlice(self, index)
        raise exc.Error(f'Type {self.col_type} is not subscriptable')

    def __getattr__(self, name: str) -> ImageMemberAccess:
        """
        ex.: <img col>.rotate(60)
        """
        if self.col_type.is_image_type():
            return ImageMemberAccess(name, self)
        if self.col_type.is_json_type():
            return JsonPath(self).__getattr__(name)
        raise exc.Error(f'Member access not supported on type {self.col_type}: {name}')

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

    def __init__(self, col: catalog.Column):
        super().__init__(col.col_type)
        self.col = col
        # a plain reference of a column accesses its value
        self.prop = self.Property.VALUE

    def __getattr__(self, name: str) -> Expr:
        if name == self.Property.ERRORTYPE.name.lower():
            if not self.col.is_computed:
                raise exc.Error(f'{name} not valid for a non-computed column: {self}')
            self.prop = self.Property.ERRORTYPE
            return self
        if name == self.Property.ERRORMSG.name.lower():
            if not self.col.is_computed:
                raise exc.Error(f'{name} not valid for a non-computed column: {self}')
            self.prop = self.Property.ERRORMSG
            return self
        if self.col_type.is_json_type():
            return JsonPath(self).__getattr__(name)
        return super().__getattr__(name)

    def display_name(self) -> str:
        return str(self)
        if self.prop == self.Property.VALUE:
            return self.col.name
        return f'{self.col.name}.{self.prop.name.lower()}'

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
        return {'col_id': self.col.id, 'prop': self.prop.value}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr], t: catalog.Table) -> Expr:
        assert 'col_id' in d
        result = cls(t.cols_by_id[d['col_id']])
        if d['prop'] != cls.Property.VALUE.value:
            result = getattr(result, cls.Property(d['prop']).name.lower())
        return result


class FrameColumnRef(ColumnRef):
    """
    Reference to an unstored extracted frame column.
    Frames are extracted from videos if they're not in the cache.
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
        video_path = data_row[self._video_ref.data_row_idx]
        if self.frames is None or self.current_video != video_path:
            self.current_video = video_path
            if self.frames is not None:
                self.frames.close()
            self.frames = FrameIterator(self.current_video, fps=self.tbl.parameters.extraction_fps)
        frame_idx = data_row[self._frame_idx_ref.data_row_idx]
        self.frames.seek(frame_idx)
        _, frame = next(self.frames, None)
        data_row[self.data_row_idx] = frame

    def release(self) -> None:
        if self.frames is not None:
            self.frames.close()
            self.frames = None


class FunctionCall(Expr):
    def __init__(
            self, fn: Function, args: Tuple[Any], order_by_exprs: List[Expr] = [], group_by_exprs: List[Expr] = []):
        signature = fn.md.signature
        super().__init__(signature.return_type)
        self.fn = fn

        if signature.parameters is not None:
            # check if arg types match param types and convert values, if necessary
            if len(args) != len(signature.parameters):
                raise exc.Error(
                    f"Number of arguments doesn't match signature: {args} vs {signature}")
            args = list(args)
            for i in range(len(args)):
                if not isinstance(args[i], Expr):
                    # TODO: check non-Expr args
                    continue
                param_type = signature.parameters[i][1]
                if args[i].col_type == param_type:
                    # nothing to do
                    continue
                converter = args[i].col_type.conversion_fn(param_type)
                if converter is None:
                    raise exc.Error(f'Cannot convert {args[i].col_type} to {param_type}')
                if converter == ColumnType.no_conversion:
                    # nothing to do
                    continue
                convert_fn = Function.make_function(param_type, [args[i].col_type], converter)
                args[i] = FunctionCall(convert_fn, (args[i],))

        self.components = [arg for arg in args if isinstance(arg, Expr)]
        self.args = [arg if not isinstance(arg, Expr) else None for arg in args]

        # window function state
        self.group_by_idx = -1  # self.components[self.group_by_index:] contains group_by exprs
        if len(group_by_exprs) > 0:
            # record grouping exprs in self.components, we need to evaluate them to get partition vals
            self.group_by_idx = len(self.components)
            self.components.extend(group_by_exprs)
        # we don't need to evaluate order_by_exprs, so they stay out of self.components
        self.order_by = order_by_exprs

        # execution state for aggregate functions
        self.aggregator: Optional[Any] = None
        self.current_partition_vals: Optional[List[Any]] = None

    @property
    def _eval_fn(self) -> Optional[Callable]:
        return self.fn.eval_fn

    def _equals(self, other: FunctionCall) -> bool:
        if self.fn != other.fn:
            return False
        if len(self.args) != len(other.args):
            return False
        for i in range(len(self.args)):
            if self.args[i] != other.args[i]:
                return False
        if self.group_by_idx != other.group_by_idx:
            return False
        if not self.list_equals(self.order_by, other.order_by):
            return False
        return True

    def __str__(self) -> str:
        return self.display_str()

    def display_str(self, inline: bool = True) -> str:
        fn_name = self.fn.display_name if self.fn.display_name != '' else 'anonymous_fn'
        return f'{fn_name}({self._print_args()})'

    def _print_args(self, start_idx: int = 0, inline: bool = True) -> str:
        arg_strs = [str(arg) if arg is not None else '' for arg in self.args[start_idx:]]
        # fill in missing expr args
        i = 0
        for j in range(start_idx, len(self.args)):
            if self.args[j] is None:
                arg_strs[j] = str(self.components[i])
                i += 1
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

    @property
    def group_by(self) -> List[Expr]:
        if self.group_by_idx == -1:
            return []
        return self.components[self.group_by_idx:]

    @property
    def is_window_fn_call(self) -> bool:
        return self.fn.is_aggregate and self.fn.allows_window and \
            (not self.fn.allows_std_agg \
             or self.group_by_idx != -1 \
             or (len(self.order_by) > 0 and not self.fn.requires_order_by))

    def get_window_sort_exprs(self) -> List[Expr]:
        return [*self.group_by, *self.order_by]

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
        args = self._make_args(data_row)
        self.fn.update_fn(self.aggregator, *args)

    def _make_args(self, data_row: DataRow) -> List[Any]:
        args = copy.copy(self.args)
        # fill in missing child values
        i = 0
        for j in range(len(args)):
            if args[j] is None:
                args[j] = data_row[self.components[i].data_row_idx]
                i += 1
        return args

    def eval(self, data_row: DataRow, evaluator: Evaluator) -> None:
        args = self._make_args(data_row)
        if not self.fn.is_aggregate:
            data_row[self.data_row_idx] = self.fn.eval_fn(*args)
        elif self.is_window_fn_call:
            if self.group_by_idx != -1:
                if self.current_partition_vals is None:
                    self.current_partition_vals = [None] * len(self.group_by)
                partition_vals = [data_row[e.data_row_idx] for e in self.group_by]
                if partition_vals != self.current_partition_vals:
                    # new partition
                    self.aggregator = self.fn.init_fn()
                    self.current_partition_vals = partition_vals
            elif self.aggregator is None:
                self.aggregator = self.fn.init_fn()
            self.fn.update_fn(self.aggregator, *args)
            data_row[self.data_row_idx] = self.fn.value_fn(self.aggregator)
        else:
            assert self.is_agg_fn_call
            data_row[self.data_row_idx] = self.fn.value_fn(self.aggregator)

    def _as_dict(self) -> Dict:
        result = {'fn': self.fn.as_dict(), 'args': self.args, **super()._as_dict()}
        if self.fn.is_aggregate:
            result.update({'group_by_idx': self.group_by_idx, 'order_by': Expr.as_dict_list(self.order_by)})
        return result

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr], t: catalog.Table) -> Expr:
        assert 'fn' in d
        assert 'args' in d
        # reassemble args
        args = [arg if arg is not None else components[i] for i, arg in enumerate(d['args'])]
        fn_call = cls(Function.from_dict(d['fn']), args)
        if fn_call.fn.is_aggregate:
            fn_call.group_by_idx = d['group_by_idx']
            fn_call.components.extend(components[fn_call.group_by_idx:])
            fn_call.order_by = Expr.from_dict_list(d['order_by'], t)
        return fn_call


def _caller_return_type(caller: Expr, *args: object, **kwargs: object) -> ColumnType:
    return caller.col_type

def _convert_return_type(caller: Expr, *args: object, **kwargs: object) -> ColumnType:
    mode_str = args[0]
    assert isinstance(mode_str, str)
    assert isinstance(caller.col_type, ImageType)
    return ImageType(
        width=caller.col_type.width, height=caller.col_type.height, mode=ImageType.Mode.from_pil(mode_str))

def _crop_return_type(caller: Expr, *args: object, **kwargs: object) -> ColumnType:
    left, upper, right, lower = args[0]
    assert isinstance(caller.col_type, ImageType)
    return ImageType(width=(right - left), height=(lower - upper), mode=caller.col_type.mode)

def _resize_return_type(caller: Expr, *args: object, **kwargs: object) -> ColumnType:
    w, h = args[0]
    assert isinstance(caller.col_type, ImageType)
    return ImageType(width=w, height=h, mode=caller.col_type.mode)

# This only includes methods that return something that can be displayed in pixeltable
# and that make sense to call (counterexample: copy() doesn't make sense to call)
# This is hardcoded here instead of being dynamically extracted from the PIL type stubs because
# doing that is messy and it's unclear whether it has any advantages.
# TODO: how to capture return values like List[Tuple[int, int]]?
# dict from method name to (function to compute value, function to compute return type)
# TODO: JsonTypes() where it should be ArrayType(): need to determine the shape and base type
_PIL_METHOD_INFO: Dict[str, Union[ColumnType, Callable]] = {
    'convert': _convert_return_type,
    'crop': _crop_return_type,
    'effect_spread': _caller_return_type,
    'entropy': FloatType(),
    'filter': _caller_return_type,
    'getbands': ArrayType((None,), StringType()),
    'getbbox': ArrayType((4,), IntType()),
    'getchannel': _caller_return_type,
    'getcolors': JsonType(),
    'getextrema': JsonType(),
    'getpalette': JsonType(),
    'getpixel': JsonType(),
    'getprojection': JsonType(),
    'histogram': JsonType(),
# TODO: what to do with this? it modifies the img in-place
#    paste: <ast.Constant object at 0x7f9e9a9be3a0>
    'point': _caller_return_type,
    'quantize': _caller_return_type,
    'reduce': _caller_return_type,  # TODO: this is incorrect
    'remap_palette': _caller_return_type,
    'resize': _resize_return_type,
    'rotate': _caller_return_type,  # TODO: this is incorrect
# TODO: this returns a Tuple[Image], which we can't express
#    split: <ast.Subscript object at 0x7f9e9a9cc9d0>
    'transform': _caller_return_type,  # TODO: this is incorrect
    'transpose': _caller_return_type,  # TODO: this is incorrect
}


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
    """
    attr_info = _create_pil_attr_info()
    special_img_predicates = ['nearest', 'matches']

    def __init__(self, member_name: str, caller: Expr):
        if member_name in self.special_img_predicates:
            super().__init__(InvalidType())  # requires FunctionCall to return value
        elif member_name in _PIL_METHOD_INFO:
            super().__init__(InvalidType())  # requires FunctionCall to return value
        elif member_name in self.attr_info:
            super().__init__(self.attr_info[member_name])
        else:
            raise exc.Error(f'Unknown Image member: {member_name}')
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
    def _from_dict(cls, d: Dict, components: List[Expr], t: catalog.Table) -> Expr:
        assert 'member_name' in d
        assert len(components) == 1
        return cls(d['member_name'], components[0])

    # TODO: correct signature?
    def __call__(self, *args, **kwargs) -> Union['ImageMethodCall', 'ImageSimilarityPredicate']:
        caller = self._caller
        call_signature = f'({",".join([type(arg).__name__ for arg in args])})'
        if self.member_name == 'nearest':
            # - caller must be ColumnRef
            # - signature is (PIL.Image.Image)
            if not isinstance(caller, ColumnRef):
                raise exc.Error(f'nearest(): caller must be an IMAGE column')
            if len(args) != 1 or not isinstance(args[0], PIL.Image.Image):
                raise exc.Error(
                    f'nearest(): required signature is (PIL.Image.Image) (passed: {call_signature})')
            return ImageSimilarityPredicate(caller, img=args[0])

        if self.member_name == 'matches':
            # - caller must be ColumnRef
            # - signature is (str)
            if not isinstance(caller, ColumnRef):
                raise exc.Error(f'matches(): caller must be an IMAGE column')
            if len(args) != 1 or not isinstance(args[0], str):
                raise exc.Error(f"matches(): required signature is (str) (passed: {call_signature})")
            return ImageSimilarityPredicate(caller, text=args[0])

        # TODO: verify signature
        return ImageMethodCall(self.member_name, caller, *args, **kwargs)

    def _equals(self, other: ImageMemberAccess) -> bool:
        return self.member_name == other.member_name

    def sql_expr(self) -> Optional[sql.sql.expression.ClauseElement]:
        return None

    def eval(self, data_row: DataRow, evaluator: Evaluator) -> None:
        caller_val = data_row[self._caller.data_row_idx]
        try:
            data_row[self.data_row_idx] = getattr(caller_val, self.member_name)
        except AttributeError:
            data_row[self.data_row_idx] = None


class ImageMethodCall(FunctionCall):
    """
    Ex.: tbl.img_col_ref.rotate(90)
    """
    def __init__(self, method_name: str, caller: Expr, *args: object, **kwargs: object):
        assert method_name in _PIL_METHOD_INFO
        self.method_name = method_name
        method_info = _PIL_METHOD_INFO[self.method_name]
        if isinstance(method_info, ColumnType):
            return_type = method_info
        else:
            return_type = method_info(caller, *args, **kwargs)
        # TODO: register correct parameters
        fn = Function.make_library_function(
            return_type, None, module_name='PIL.Image', eval_symbol=f'Image.{method_name}')
        super().__init__(fn, (caller, *args))
        # TODO: deal with kwargs

    def display_name(self) -> str:
        return self.method_name

    def __str__(self) -> str:
        return f'{self.components[0]}.{self.method_name}({self._print_args(1)})'

    def _as_dict(self) -> Dict:
        return {'method_name': self.method_name, 'args': self.args, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr], t: catalog.Table) -> Expr:
        """
        We're implementing this, instead of letting FunctionCall handle it, in order to return an
        ImageMethodCall instead of a FunctionCall, which is useful for testing that a serialize()/deserialize()
        roundtrip ends up with the same Expr.
        """
        assert 'method_name' in d
        assert 'args' in d
        # reassemble args, but skip args[0], which is our caller
        args = [arg if arg is not None else components[i+1] for i, arg in enumerate(d['args'][1:])]
        return cls(d['method_name'], components[0], *args)


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
    def _from_dict(cls, d: Dict, components: List[Expr], t: catalog.Table) -> Expr:
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
            raise exc.Error(f'() for an absolute path is invalid')
        if len(args) != 1 or not isinstance(args[0], int) or args[0] >= 0:
            raise exc.Error(f'R() requires a negative index')
        return JsonPath(None, [], args[0])

    def __getattr__(self, name: str) -> 'JsonPath':
        assert isinstance(name, str)
        return JsonPath(self._anchor, self.path_elements + [name])

    def __getitem__(self, index: object) -> 'JsonPath':
        if isinstance(index, str):
            if index != '*':
                raise exc.Error(f'Invalid json list index: {index}')
        else:
            if not isinstance(index, slice) and not isinstance(index, int):
                raise exc.Error(f'Invalid json list index: {index}')
        return JsonPath(self._anchor, self.path_elements + [index])

    def __rshift__(self, other: object) -> 'JsonMapper':
        rhs_expr = Expr.from_object(other)
        if rhs_expr is None:
            raise exc.Error(f'>> requires an expression on the right-hand side, found {type(other)}')
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
        val = data_row[self._anchor.data_row_idx]
        if self.compiled_path is not None:
            val = self.compiled_path.search(val)
        data_row[self.data_row_idx] = val


RELATIVE_PATH_ROOT = JsonPath(None)


class Literal(Expr):
    def __init__(self, val: LiteralPythonTypes):
        if isinstance(val, str):
            super().__init__(StringType())
        if isinstance(val, int):
            super().__init__(IntType())
        if isinstance(val, float):
            super().__init__(FloatType())
        if isinstance(val, bool):
            super().__init__(BoolType())
        if isinstance(val, datetime.datetime) or isinstance(val, datetime.date):
            super().__init__(TimestampType())
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
        data_row[self.data_row_idx] = self.val

    def _as_dict(self) -> Dict:
        return {'val': self.val, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr], t: catalog.Table) -> Expr:
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
                raise exc.Error(f'Dictionary requires string keys, {key} has type {type(key)}')
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
                result[key] = data_row[self.components[idx].data_row_idx]
            else:
                result[key] = copy.deepcopy(val)
        data_row[self.data_row_idx] = result

    def _as_dict(self) -> Dict:
        return {'dict_items': self.dict_items, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr], t: catalog.Table) -> Expr:
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
                element_type = ColumnType.supertype(element_type, ColumnType.get_value_type(val))
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
                result[i] = data_row[self.components[child_idx].data_row_idx]
            else:
                result[i] = copy.deepcopy(val)
        data_row[self.data_row_idx] = np.array(result)

    def _as_dict(self) -> Dict:
        return {'elements': self.elements, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr], t: catalog.Table) -> Expr:
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
        val = data_row[self._array.data_row_idx]
        data_row[self.data_row_idx] = val[self.index]

    def _as_dict(self) -> Dict:
        index = []
        for el in self.index:
            if isinstance(el, slice):
                index.append([el.start, el.stop, el.step])
            else:
                index.append(el)
        return {'index': index, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr], t: catalog.Table) -> Expr:
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

    def extract_sql_predicate(self) -> Tuple[Optional[sql.sql.expression.ClauseElement], Optional['Predicate']]:
        """
        Return ClauseElement for what can be evaluated in SQL and a predicate for the remainder that needs to be
        evaluated in Python.
        Needed to for predicate push-down into SQL.
        """
        e = self.sql_expr()
        return (None, self) if e is None else (e, None)

    def split_conjuncts(
            self, condition: Callable[['Predicate'], bool]) -> Tuple[List['Predicate'], Optional['Predicate']]:
        """
        Returns clauses of a conjunction that meet condition in the first element.
        The second element contains remaining clauses, rolled into a conjunction.
        """
        if condition(self):
            return [self], None
        else:
            return [], self

    def __and__(self, other: object) -> 'CompoundPredicate':
        if not isinstance(other, Expr):
            raise TypeError(f'Other needs to be an expression: {type(other)}')
        if not other.col_type.is_bool_type():
            raise TypeError(f'Other needs to be an expression that returns a boolean: {other.col_type}')
        return CompoundPredicate(LogicalOperator.AND, [self, other])

    def __or__(self, other: object) -> 'CompoundPredicate':
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
            self, condition: Callable[['Predicate'], bool]) -> Tuple[List['Predicate'], Optional['Predicate']]:
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
            data_row[self.data_row_idx] = not data_row[self.components[0].data_row_idx]
        else:
            val = True if self.operator == LogicalOperator.AND else False
            op_function = operator.and_ if self.operator == LogicalOperator.AND else operator.or_
            for op in self.components:
                val = op_function(val, data_row[op.data_row_idx])
            data_row[self.data_row_idx] = val

    def _as_dict(self) -> Dict:
        return {'operator': self.operator.value, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr], t: catalog.Table) -> Expr:
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
            data_row[self.data_row_idx] = data_row[self._op1.data_row_idx] < data_row[self._op2.data_row_idx]
        elif self.operator == ComparisonOperator.LE:
            data_row[self.data_row_idx] = data_row[self._op1.data_row_idx] <= data_row[self._op2.data_row_idx]
        elif self.operator == ComparisonOperator.EQ:
            data_row[self.data_row_idx] = data_row[self._op1.data_row_idx] == data_row[self._op2.data_row_idx]
        elif self.operator == ComparisonOperator.NE:
            data_row[self.data_row_idx] = data_row[self._op1.data_row_idx] != data_row[self._op2.data_row_idx]
        elif self.operator == ComparisonOperator.GT:
            data_row[self.data_row_idx] = data_row[self._op1.data_row_idx] > data_row[self._op2.data_row_idx]
        elif self.operator == ComparisonOperator.GE:
            data_row[self.data_row_idx] = data_row[self._op1.data_row_idx] >= data_row[self._op2.data_row_idx]

    def _as_dict(self) -> Dict:
        return {'operator': self.operator.value, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr], t: catalog.Table) -> Expr:
        assert 'operator' in d
        return cls(ComparisonOperator(d['operator']), components[0], components[1])


class ImageSimilarityPredicate(Predicate):
    def __init__(self, img_col: ColumnRef, img: Optional[PIL.Image.Image] = None, text: Optional[str] = None):
        assert (img is None) != (text is None)
        super().__init__()
        self.img_col_ref = img_col
        self.components = [img_col]
        self.img = img
        self.text = text

    def embedding(self) -> np.ndarray:
        if self.text is not None:
            return clip.encode_text(self.text)
        else:
            return clip.encode_image(self.img)

    def __str__(self) -> str:
        op_str = 'nearest' if self.img is not None else 'matches'
        return f'{str(self.img_col_ref)}.{op_str}({"<img>" if self.img is not None else self.text})'

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
    def _from_dict(cls, d: Dict, components: List[Expr], t: catalog.Table) -> Expr:
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
        data_row[self.data_row_idx] = data_row[self.components[0].data_row_idx] is None

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr], t: catalog.Table) -> Expr:
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
            raise exc.Error(f'{self}: {operator} requires numeric types, but {op1} has type {op1.col_type}')
        if not op2.col_type.is_numeric_type() and not op2.col_type.is_json_type():
            raise exc.Error(f'{self}: {operator} requires numeric types, but {op2} has type {op2.col_type}')

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
        op1_val = data_row[self._op1.data_row_idx]
        op2_val = data_row[self._op2.data_row_idx]
        # check types if we couldn't do that prior to execution
        if self._op1.col_type.is_json_type() and not isinstance(op1_val, int) and not isinstance(op1_val, float):
            raise exc.Error(
                f'{self.operator} requires numeric type, but {self._op1} has type {type(op1_val).__name__}')
        if self._op2.col_type.is_json_type() and not isinstance(op2_val, int) and not isinstance(op2_val, float):
            raise exc.Error(
                f'{self.operator} requires numeric type, but {self._op2} has type {type(op2_val).__name__}')

        if self.operator == ArithmeticOperator.ADD:
            data_row[self.data_row_idx] = op1_val + op2_val
        elif self.operator == ArithmeticOperator.SUB:
            data_row[self.data_row_idx] = op1_val - op2_val
        elif self.operator == ArithmeticOperator.MUL:
            data_row[self.data_row_idx] = op1_val * op2_val
        elif self.operator == ArithmeticOperator.DIV:
            data_row[self.data_row_idx] = op1_val / op2_val
        elif self.operator == ArithmeticOperator.MOD:
            data_row[self.data_row_idx] = op1_val % op2_val

    def _as_dict(self) -> Dict:
        return {'operator': self.operator.value, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr], t: catalog.Table) -> Expr:
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
        src = data_row[self._src_expr.data_row_idx]
        if not isinstance(src, list):
            # invalid/non-list src path
            data_row[self.data_row_idx] = None
            return

        result = [None] * len(src)
        if len(self.target_expr_eval_ctx) == 0:
            self.target_expr_eval_ctx = evaluator.get_eval_ctx([self._target_expr.data_row_idx])
        for i, val in enumerate(src):
            data_row[self.scope_anchor.data_row_idx] = val
            # stored target_expr
            exc_tb = evaluator.eval(data_row, self.target_expr_eval_ctx)
            assert exc_tb is None
            result[i] = data_row[self._target_expr.data_row_idx]
        data_row[self.data_row_idx] = result

    def _as_dict(self) -> Dict:
        """
        We need to avoid serializing component[2], which is an ObjectRef.
        """
        return {'components': [c.as_dict() for c in self.components[0:2]]}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr], t: catalog.Table) -> Expr:
        assert len(components) == 2
        return cls(components[0], components[1])


class DataRow:
    """
    Encapsulates data and execution state needed by Evaluator.
    """
    def __init__(self, size: int):
        self.vals: List[Any] = [None] * size
        self.has_val = [False] * size

    def __getitem__(self, index: object) -> Any:
        assert self.has_val[index]
        return self.vals[index]

    def __setitem__(self, index: object, val: Any) -> None:
        # we allow overwriting
        self.vals[index] = val
        self.has_val[index] = True

    def __len__(self) -> int:
        return len(self.vals)


class Evaluator:
    """
    Evaluates a list of Exprs against a data row.
    """

    def __init__(self, expr_list: List[Expr], with_sql: bool = True):
        """
        Set up Evaluator to evaluate any Expr in expr_list.
        If with_sql == True, assumes that exprs that have a SQL equivalent (to_sql() != None) will be materialized
        via SQL, and that none of its dependencies will need to be materialized.
        """
        self.sql_exprs: List[Expr] = []  # contains Exprs with sql_row_idx != -1

        # all following list are indexed with data_row_idx
        self.unique_exprs = UniqueExprList()  # dependencies precede their dependents
        self.next_data_row_idx = 0
        # select list providing the input to the SQL scan stage
        self.sql_select_list: List[sql.sql.expression.ClauseElement] = []

        for e in expr_list:
            self._assign_idxs(e, with_sql)

        for i in range(len(self.unique_exprs)):
            assert self.unique_exprs[i].data_row_idx == i

        # record transitive dependencies
        self.dependencies = [set() for _ in range(self.num_materialized)]
        for i in range(self.num_materialized):
            expr = self.unique_exprs[i]
            if expr.sql_row_idx != -1:
                # it is materialized directly by SQL and therefore doesn't depend on other exprs
                continue
            for d in self.unique_exprs[i].dependencies():
                self.dependencies[i].add(d.data_row_idx)
                self.dependencies[i].update(self.dependencies[d.data_row_idx])

        # derive transitive dependents
        self.dependents = [set() for _ in range(self.num_materialized)]
        for i in range(self.num_materialized):
            for j in self.dependencies[i]:
                self.dependents[j].add(i)

        # records the Exprs in parameter 'expr_list' that a subexpr belongs to
        # (a subexpr can be shared across multiple output exprs)
        self.output_expr_ids = [set() for _ in range(self.num_materialized)]
        for e in expr_list:
            self._record_output_expr_id(e, e.data_row_idx)


    def _assign_idxs(self, expr: Expr, with_sql: bool) -> None:
        if expr in self.unique_exprs:
            # expr is a duplicate: we copy sql_/data_row_idx for itself and its components
            original = self.unique_exprs[expr]
            expr.data_row_idx = original.data_row_idx
            expr.sql_row_idx = original.sql_row_idx
            if expr.sql_row_idx == -1:
                for c in expr.components:
                    self._assign_idxs(c, with_sql)
            # nothing left to do
            return

        if with_sql:
            sql_expr = expr.sql_expr()
            # if this can be materialized via SQL we don't need to look at its components;
            # we special-case Literals because we don't want to have to materialize them via SQL
            if sql_expr is not None and not isinstance(expr, Literal):
                assert expr.data_row_idx < 0
                expr.data_row_idx = self.next_data_row_idx
                self.next_data_row_idx += 1
                expr.sql_row_idx = len(self.sql_select_list)
                self.sql_select_list.append(sql_expr)
                self.sql_exprs.append(expr)
                self.unique_exprs.append(expr)
                return

        # expr value needs to be computed via Expr.eval()
        for c in expr.components:
            self._assign_idxs(c, with_sql)
        assert expr.data_row_idx < 0
        expr.data_row_idx = self.next_data_row_idx
        self.next_data_row_idx += 1
        self.unique_exprs.append(expr)

    def _record_output_expr_id(self, e: Expr, output_expr_id: int) -> None:
        self.output_expr_ids[e.data_row_idx].add(output_expr_id)
        for d in e.dependencies():
            self._record_output_expr_id(d, output_expr_id)

    @property
    def num_materialized(self) -> int:
        return self.next_data_row_idx

    def prepare(self, sql_row: List[Any], with_pk: bool = False) -> DataRow:
        """
        Prepare for evaluation by initializing data_row from given sql_row.
        with_pk == True: sql_row's last element is PK tuple, which we append to the data_row.
        object is reused across multiple executions of the same query.
        Returns data row.
        """
        data_row = DataRow(self.num_materialized + with_pk)
        # sql_row might have the PK added to the end
        assert len(sql_row) >= len(self.sql_exprs)
        if with_pk:
            data_row[-1] = (sql_row[-2], sql_row[-1])

        for i in range(len(self.sql_exprs)):
            data_row[self.sql_exprs[i].data_row_idx] = sql_row[i]
            expr = self.sql_exprs[i]
            if expr.col_type.is_image_type():
                # column value is a file path that we need to open
                file_path = sql_row[i]
                try:
                    img = PIL.Image.open(file_path)
                    data_row[expr.data_row_idx] = img
                except Exception:
                    raise exc.Error(f'Error reading image file: {file_path}')
            elif expr.col_type.is_array_type():
                # column value is a saved numpy array
                array_data = sql_row[i]
                data_row[expr.data_row_idx] = np.load(io.BytesIO(array_data))
            else:
                data_row[expr.data_row_idx] = sql_row[i]
        return data_row

    def get_eval_ctx(self, targets: List[int]) -> List[Expr]:
        """
        Return list of dependencies needed to evaluate the given target exprs. This excludes exprs that
        are materialized in SQL (sql_row_idx != -1)
        """
        all_dependencies: Set[int] = set()
        for id in targets:
            assert id < self.num_materialized
            all_dependencies.update(self.dependencies[id])
        all_dependencies.update(targets)
        result_ids = list(all_dependencies)
        result_ids = [id for id in result_ids if self.unique_exprs[id].sql_row_idx == -1]
        result_ids.sort()
        return [self.unique_exprs[id] for id in result_ids]

    def eval(self, data_row: DataRow, ctx: List[Expr]) -> Optional[TracebackType]:
        """
        Populates the slots in data_row given in ctx.
        If an expr.eval() raises an exception, records the exception in the corresponding slot of data_row
        and omits any of that expr's dependents's eval().
        Returns exception traceback for the first exception that was recorded
        """
        exc_tb: Optional[TracebackType] = None
        skip_exprs: Set[int] = set()  # skip dependents of exprs that had exception
        for expr in ctx:
            if expr.data_row_idx in skip_exprs or data_row.has_val[expr.data_row_idx]:
                continue

            try:
                expr.eval(data_row, self)
            except Exception as exc:
                _, _, exc_tb = sys.exc_info()
                data_row[expr.data_row_idx] = exc
                skip_exprs.update(self.dependents[expr.data_row_idx])

        if exc_tb is not None:
            # propagate exceptions in data_row to their respective output_expr slots
            for i, exc_val in [(i, val) for i, val in enumerate(data_row.vals) if isinstance(val, Exception)]:
                for j in self.output_expr_ids[i]:
                    data_row[j] = exc_val
        return exc_tb


class ExprDict:
    """
    Implements semantics of dict from Expr.data_row_idx (= id) to unique exprs. Exprs that test True for Expr.equals()
    are updated to share the same data_row_idx.
    """
    def __init__(self):
        self.unique_exprs: List[Expr] = []
        self.expr_dict: Dict[int, Expr] = {}

    def add(self, expr: Expr) -> bool:
        """
        If expr is not unique, sets expr.data/sql_row_idx to that of the already-recorded duplicate and returns
        False, otherwise returns True.
        """
        try:
            existing = next(e for e in self.unique_exprs if e.equals(expr))
            expr.data_row_idx = existing.data_row_idx
            expr.sql_row_idx = existing.sql_row_idx
            return False
        except StopIteration:
            self.unique_exprs.append(expr)
            return True

    def get(self, id: int) -> Expr:
        """
        Return expr with given id (= data_row_idx)
        """
        if len(self.expr_dict) == 0:
            self.expr_dict = {e.data_row_idx: e for e in self.unique_exprs}
            assert -1 not in self.expr_dict
        return self.expr_dict[id]

    def __iter__(self) -> Iterator[Expr]:
        return iter(self.unique_exprs)


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

    def __contains__(self, item: Expr) -> bool:
        assert isinstance(item, Expr)
        try:
            _ = next(e for e in self.unique_exprs if e.equals(item))
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
