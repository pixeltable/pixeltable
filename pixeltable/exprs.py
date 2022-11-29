import abc
import datetime
import enum
import inspect
import typing
from typing import Union, Optional, List, Callable, Any, Dict, Tuple
import operator

import PIL.Image
import numpy as np
import sqlalchemy as sql

from pixeltable import catalog
from pixeltable.type_system import \
    ColumnType, InvalidType, StringType, IntType, FloatType, BoolType, TimestampType, ImageType, DictType, ArrayType
from pixeltable import exceptions as exc
from pixeltable.utils import clip

# Python types corresponding to our literal types
LiteralPythonTypes = Union[str, int, float, bool, datetime.datetime, datetime.date]


class ComparisonOperator(enum.Enum):
    LT = 1
    LE = 2
    EQ = 3
    NE = 4
    GT = 5
    GE = 6


class LogicalOperator(enum.Enum):
    AND = 1
    OR = 2
    NOT = 3


class Expr(abc.ABC):
    def __init__(self, col_type: ColumnType):
        self.col_type = col_type
        # index of the expr's value in the data row; set for all materialized exprs; -1: invalid
        self.data_row_idx = -1
        # index of the expr's value in the SQL row; only set for exprs that can be materialized in SQL; -1: invalid
        self.sql_row_idx = -1

    def display_name(self) -> str:
        """
        Displayed column name in DataFrame. '': assigned by DataFrame
        """
        return ''

    def equals(self, other: 'Expr') -> bool:
        """
        Subclass-specific comparison. Implemented as a function because __eq__() is needed to construct Comparisons.
        """
        if type(self) != type(other):
            return False
        return self._equals(other)

    @abc.abstractmethod
    def _equals(self, other: 'Expr') -> bool:
        pass

    @abc.abstractmethod
    def sql_expr(self) -> Optional[sql.sql.expression.ClauseElement]:
        """
        If this expr can be materialized directly in SQL:
        - returns a ClauseElement
        - eval() will not be called
        Otherwise
        - returns None
        - eval() will be called
        """
        pass

    @abc.abstractmethod
    def dependencies(self) -> List['Expr']:
        """
        Returns all exprs whose results are needed for eval().
        Not called if sql_expr() != None
        """
        pass

    @abc.abstractmethod
    def eval(self, data_row: List[Any]) -> None:
        """
        Compute the expr value for data_row and store the result in data_row[data_row_id].
        Not called if sql_expr() != None.
        """
        pass

    def __getattr__(self, name: str) -> 'ImageMemberAccess':
        """
        ex.: <img col>.rotate(60)
        """
        if not self.col_type.is_image_type():
            raise exc.OperationalError(f'Member access not supported on type {self.col_type}: {name}')
        return ImageMemberAccess(name, self)

    def __lt__(self, other: object) -> 'Comparison':
        return self._make_comparison(ComparisonOperator.LT, other)

    def __le__(self, other: object) -> 'Comparison':
        return self._make_comparison(ComparisonOperator.LE, other)

    def __eq__(self, other: object) -> 'Comparison':
        return self._make_comparison(ComparisonOperator.EQ, other)

    def __ne__(self, other: object) -> 'Comparison':
        return self._make_comparison(ComparisonOperator.NE, other)

    def __gt__(self, other: object) -> 'Comparison':
        return self._make_comparison(ComparisonOperator.GT, other)

    def __ge__(self, other: object) -> 'Comparison':
        return self._make_comparison(ComparisonOperator.GE, other)

    def _make_comparison(self, op: ComparisonOperator, other: object) -> 'Comparison':
        """
        other: Union[Expr, LiteralPythonTypes]
        """
        # TODO: check for compatibility
        if isinstance(other, Expr):
            return Comparison(op, self, other)
        if isinstance(other, typing.get_args(LiteralPythonTypes)):
            return Comparison(op, self, Literal(other))  # type: ignore[arg-type]
        raise TypeError(f'Other must be Expr or literal: {type(other)}')


class ColumnRef(Expr):
    def __init__(self, col: catalog.Column):
        super().__init__(col.col_type)
        self.col = col

    def display_name(self) -> str:
        return self.col.name

    def _equals(self, other: 'ColumnRef') -> bool:
        return self.col == other.col

    def sql_expr(self) -> Optional[sql.sql.expression.ClauseElement]:
        return self.col.sa_col

    def dependencies(self) -> List['Expr']:
        return []

    def eval(self, data_row: List[Any]) -> None:
        assert False


class FunctionCall(Expr):
    def __init__(
            self, fn: Callable,  return_type: ColumnType,
            args: Optional[List[Any]] = None, tbl: Optional[catalog.Table] = None):
        """
        If args is None, interprets fn's arguments to be column references in 'tbl'.
        """
        super().__init__(return_type)
        self.fn = fn
        params = inspect.signature(self.fn).parameters
        if args is not None:
            if len(args) != len(params):
                raise exc.OperationalError(
                    f"FunctionCall: number of arguments ({len(args)}) doesn't match the number of expected parameters "
                    f"({len(params)}")
            self.args = args
            return

        self.args: List[ColumnRef] = []
        # we're constructing ColumnRefs for the function parameters;
        # make sure that fn's params are valid col names in tbl
        if len(params) > 0 and tbl is None:
            raise exc.OperationalError(f'FunctionCall is missing tbl parameter')
        for param_name in params:
            if param_name not in tbl.cols_by_name:
                raise exc.OperationalError(
                    (f'FunctionCall: lambda argument names need to be valid column names in table {tbl.name}: '
                     'column {param_name} unknown'))
            self.args.append(ColumnRef(tbl.cols_by_name[param_name]))

    def _equals(self, other: 'FunctionCall') -> bool:
        # we don't know whether self.fn and other.fn compute the same thing
        return False

    def sql_expr(self) -> Optional[sql.sql.expression.ClauseElement]:
        return None

    def dependencies(self) -> List['Expr']:
        expr_children = [arg for arg in self.args if isinstance(arg, Expr)]
        return expr_children

    def eval(self, data_row: List[Any]) -> None:
        arg_vals = [data_row[arg.data_row_idx] if isinstance(arg, Expr) else arg for arg in self.args]
        data_row[self.data_row_idx] = self.fn(*arg_vals)


def _caller_return_type(caller: Expr, *args: object) -> ColumnType:
    return caller.col_type

def _float_return_type(_: Expr, *args: object) -> ColumnType:
    return FloatType()

def _dict_return_type(_: Expr, *args: object) -> ColumnType:
    return DictType()

def _array_return_type(_: Expr, *args: object) -> ColumnType:
    return ArrayType()

def _crop_return_type(_: Expr, *args: object) -> ColumnType:
    left, upper, right, lower = args[0]
    return ImageType(width=(right - left), height=(lower - upper))

def _resize_return_type(_: Expr, *args: object) -> ColumnType:
    w, h = args[0]
    return ImageType(width=w, height=h)

# This only includes methods that return something that can be displayed in pixeltable
# and that make sense to call (counterexample: copy() doesn't make sense to call)
# This is hardcoded here instead of being dynamically extracted from the PIL type stubs because
# doing that is messy and it's unclear whether it has any advantages.
# TODO: how to capture return values like List[Tuple[int, int]]?
# dict from method name to (function to compute value, function to compute return type)
_PIL_METHOD_INFO: Dict[str, Tuple[Callable, ColumnType]] = {
    'convert': (PIL.Image.Image.convert, _caller_return_type),
    'crop': (PIL.Image.Image.crop, _crop_return_type),
    'effect_spread': (PIL.Image.Image.effect_spread, _caller_return_type),
    'entropy': (PIL.Image.Image.entropy, _float_return_type),
    'filter': (PIL.Image.Image.filter, _caller_return_type),
    'getbands': (PIL.Image.Image.getbands, _array_return_type),
    'getbbox': (PIL.Image.Image.getbbox, _array_return_type),
    'getchannel': (PIL.Image.Image.getchannel, _caller_return_type),
    'getcolors': (PIL.Image.Image.getcolors, _array_return_type),
    'getextrema': (PIL.Image.Image.getextrema, _array_return_type),
    'getpalette': (PIL.Image.Image.getpalette, _array_return_type),
    'getpixel': (PIL.Image.Image.getpixel, _array_return_type),
    'getprojection': (PIL.Image.Image.getprojection, _array_return_type),
    'histogram': (PIL.Image.Image.histogram, _array_return_type),
# TODO: what to do with this? it modifies the img in-place
#    paste: <ast.Constant object at 0x7f9e9a9be3a0>
    'point': (PIL.Image.Image.point, _caller_return_type),
    'quantize': (PIL.Image.Image.quantize, _caller_return_type),
    'reduce': (PIL.Image.Image.reduce, _caller_return_type),  # TODO: this is incorrect
    'remap_palette': (PIL.Image.Image.remap_palette, _caller_return_type),
    'resize': (PIL.Image.Image.resize, _resize_return_type),
    'rotate': (PIL.Image.Image.rotate, _caller_return_type),  # TODO: this is incorrect
# TODO: this returns a Tuple[Image], which we can't express
#    split: <ast.Subscript object at 0x7f9e9a9cc9d0>
    'transform': (PIL.Image.Image.transform, _caller_return_type),  # TODO: this is incorrect
    'transpose': (PIL.Image.Image.transpose, _caller_return_type),  # TODO: this is incorrect
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
            result[name] = DictType()
    return result


class ImageMemberAccess(Expr):
    """
    Access of either an attribute or function member of PIL.Image.Image.
    Ex.: tbl.img_col.rotate(90), tbl.img_col.width
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
            raise exc.OperationalError(f'Unknown Image member: {member_name}')
        self.member_name = member_name
        self.caller = caller

    def display_name(self) -> str:
        return self.member_name

    # TODO: correct signature?
    def __call__(self, *args, **kwargs) -> Union['ImageMethodCall', 'ImageSimilarityPredicate']:
        call_signature = f'({",".join([type(arg).__name__ for arg in args])})'
        if self.member_name == 'nearest':
            # - caller must be ColumnRef
            # - signature is (PIL.Image.Image)
            if not isinstance(self.caller, ColumnRef):
                raise exc.OperationalError(f'nearest(): caller must be an IMAGE column')
            if len(args) != 1 or not isinstance(args[0], PIL.Image.Image):
                raise exc.OperationalError(
                    f'nearest(): required signature is (PIL.Image.Image) (passed: {call_signature})')
            return ImageSimilarityPredicate(self.caller, img=args[0])

        if self.member_name == 'matches':
            # - caller must be ColumnRef
            # - signature is (str)
            if not isinstance(self.caller, ColumnRef):
                raise exc.OperationalError(f'matches(): caller must be an IMAGE column')
            if len(args) != 1 or not isinstance(args[0], str):
                raise exc.OperationalError(f"matches(): required signature is (str) (passed: {call_signature})")
            return ImageSimilarityPredicate(self.caller, text=args[0])

        # TODO: verify signature
        return ImageMethodCall(self.member_name, self.caller, *args, **kwargs)

    def _equals(self, other: 'ImageMemberAccess') -> bool:
        return self.caller.equals(other.caller) and self.member_name == other.member_name

    def sql_expr(self) -> Optional[sql.sql.expression.ClauseElement]:
        return None

    def dependencies(self) -> List['Expr']:
        return [self.caller]

    def eval(self, data_row: List[Any]) -> None:
        caller_val = data_row[self.caller.data_row_idx]
        try:
            data_row[self.data_row_idx] = getattr(caller_val, self.member_name)
        except AttributeError:
            data_row[self.data_row_idx] = None


class ImageMethodCall(Expr):
    """
    Ex.: tbl.img_col.rotate(90)
    TODO:
    - check arg types
    - resolve Expr args in eval()
    """
    def __init__(self, method_name: str, caller: Expr, *args, **kwargs):
        assert method_name in _PIL_METHOD_INFO
        self.method_name = method_name
        method_info = _PIL_METHOD_INFO[self.method_name]
        self.fn = method_info[0]
        return_type = method_info[1](caller, *args)
        super().__init__(return_type)
        self.caller = caller
        self.args = args
        self.kw_args = kwargs

    def display_name(self) -> str:
        return self.method_name

    def _equals(self, other: 'ImageMethodCall') -> bool:
        return self.caller.equals(other.caller) and self.method_name == other.method_name \
           and self.args == other.args and self.kw_args == other.kw_args

    def sql_expr(self) -> Optional[sql.sql.expression.ClauseElement]:
        return None

    def dependencies(self) -> List['Expr']:
        return [self.caller]

    def eval(self, data_row: List[Any]) -> None:
        # first argument is the value of the caller
        args = [data_row[self.caller.data_row_idx]]
        if args[0] is None:
            data_row[self.data_row_idx] = None
            return
        args.extend(self.args)
        data_row[self.data_row_idx] = self.fn(*args, **self.kw_args)


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

    def _equals(self, other: 'Literal') -> bool:
        return self.val == other.eval

    def sql_expr(self) -> Optional[sql.sql.expression.ClauseElement]:
        return sql.sql.expression.literal(self.val)

    def dependencies(self) -> List['Expr']:
        return []

    def eval(self, data_row: List[Any]) -> None:
        data_row[self.data_row_idx] = self.val


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
            return ([self], None)
        else:
            return ([], self)

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

    def __invert__(self) -> 'CompoundPredicate':
        return CompoundPredicate(LogicalOperator.NOT, [self])


class CompoundPredicate(Predicate):
    def __init__(self, operator: LogicalOperator, operands: List[Predicate]):
        super().__init__()
        self.operator = operator
        if self.operator == LogicalOperator.NOT:
            assert len(operands) == 1
            self.operands = operands
        else:
            assert len(operands) > 1
            self.operands: List[Predicate] = []
            for op in operands:
                self._merge_operand(op)

    @classmethod
    def make_conjunction(cls, operands: List[Predicate]) -> Optional[Predicate]:
        if len(operands) == 0:
            return None
        if len(operands) == 1:
            return operands[0]
        return CompoundPredicate(LogicalOperator.AND, operands)

    def _merge_operand(self, op: Predicate) -> None:
        """
        Merge this operand, if possible, otherwise simply record it.
        """
        if isinstance(op, CompoundPredicate) and op.operator == self.operator:
            # this can be merged
            for child_op in op.operands:
                self._merge_operand(child_op)
        else:
            self.operands.append(op)

    def _equals(self, other: 'CompoundPredicate') -> bool:
        if self.operator != other.operator or len(self.operands) != len(other.operands):
            return False
        for i in range(len(self.operands)):
            if not self.operands[i].equals(other.operands[i]):
                return False
        return True

    def extract_sql_predicate(self) -> Tuple[Optional[sql.sql.expression.ClauseElement], Optional[Predicate]]:
        if self.operator == LogicalOperator.NOT:
            e = self.operands[0].sql_expr()
            return (None, self) if e is None else (e, None)

        sql_exprs = [op.sql_expr() for op in self.operands]
        if self.operator == LogicalOperator.OR and any(e is None for e in sql_exprs):
            # if any clause of a | can't be evaluated in SQL, we need to evaluate everything in Python
            return (None, self)
        if not(any(e is None for e in sql_exprs)):
            # we can do everything in SQL
            return (self.sql_expr(), None)

        assert self.operator == LogicalOperator.AND
        if not any(e is not None for e in sql_exprs):
            # there's nothing that can be done in SQL
            return (None, self)

        sql_preds = [e for e in sql_exprs if e is not None]
        other_preds = [self.operands[i] for i, e in enumerate(sql_exprs) if e is None]
        assert len(sql_preds) > 0
        combined_sql_pred = sql.and_(*sql_preds)
        combined_other = self.make_conjunction(other_preds)
        return (combined_sql_pred, combined_other)

    def split_conjuncts(
            self, condition: Callable[['Predicate'], bool]) -> Tuple[List['Predicate'], Optional['Predicate']]:
        if self.operator == LogicalOperator.OR or self.operator == LogicalOperator.NOT:
            return super().split_conjuncts(condition)
        matches = [op for op in self.operands if condition(op)]
        non_matches = [op for op in self.operands if not condition(op)]
        return (matches, self.make_conjunction(non_matches))

    def sql_expr(self) -> Optional[sql.sql.expression.ClauseElement]:
        sql_exprs = [op.sql_expr() for op in self.operands]
        if any(e is None for e in sql_exprs):
            return None
        if self.operator == LogicalOperator.NOT:
            assert len(sql_exprs) == 1
            return sql.not_(sql_exprs[0])
        assert len(sql_exprs) > 1
        operator = sql.and_ if self.operator == LogicalOperator.AND else sql.or_
        combined = operator(*sql_exprs)
        return combined

    def dependencies(self) -> List['Expr']:
        return self.operands

    def eval(self, data_row: List[Any]) -> None:
        if self.operator == LogicalOperator.NOT:
            data_row[self.data_row_idx] = not data_row[self.operands[0].data_row_idx]
        else:
            val = True if self.operator == LogicalOperator.AND else False
            op_function = operator.and_ if self.operator == LogicalOperator.AND else operator.or_
            for op in self.operands:
                val = op_function(val, data_row[op.data_row_idx])
            data_row[self.data_row_idx] = val


class Comparison(Predicate):
    def __init__(self, operator: ComparisonOperator, op1: Expr, op2: Expr):
        super().__init__()
        self.operator = operator
        self.op1 = op1
        self.op2 = op2

    def _equals(self, other: 'Comparison') -> bool:
        return self.operator == other.operator and self.op1.equals(other.op1) \
            and ((self.op2 is None and other.op2 is None) or self.op2.equals(other.op2))

    def sql_expr(self) -> Optional[sql.sql.expression.ClauseElement]:
        left = self.op1.sql_expr()
        right = self.op2.sql_expr()
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

    def dependencies(self) -> List['Expr']:
        return [self.op1, self.op2]

    def eval(self, data_row: List[Any]) -> None:
        if self.operator == ComparisonOperator.LT:
            data_row[self.data_row_idx] = data_row[self.op1.data_row_idx] < data_row[self.op2.data_row_idx]
        elif self.operator == ComparisonOperator.LE:
            data_row[self.data_row_idx] = data_row[self.op1.data_row_idx] <= data_row[self.op2.data_row_idx]
        elif self.operator == ComparisonOperator.EQ:
            data_row[self.data_row_idx] = data_row[self.op1.data_row_idx] == data_row[self.op2.data_row_idx]
        elif self.operator == ComparisonOperator.NE:
            data_row[self.data_row_idx] = data_row[self.op1.data_row_idx] != data_row[self.op2.data_row_idx]
        elif self.operator == ComparisonOperator.GT:
            data_row[self.data_row_idx] = data_row[self.op1.data_row_idx] > data_row[self.op2.data_row_idx]
        elif self.operator == ComparisonOperator.GE:
            data_row[self.data_row_idx] = data_row[self.op1.data_row_idx] >= data_row[self.op2.data_row_idx]


class ImageSimilarityPredicate(Predicate):
    def __init__(self, img_col: ColumnRef, img: Optional[PIL.Image.Image] = None, text: Optional[str] = None):
        assert (img is None) != (text is None)
        super().__init__()
        self.img_col = img_col
        self.img = img
        self.text = text

    def embedding(self) -> np.ndarray:
        if self.text is not None:
            return clip.encode_text(self.text)
        else:
            return clip.encode_image(self.img)

    def dependencies(self) -> List['Expr']:
        return [self.img_col]

    def _equals(self, other: 'ImageSimilarityPredicate') -> bool:
        return False

    def sql_expr(self) -> Optional[sql.sql.expression.ClauseElement]:
        return None

    def eval(self, data_row: List[Any]) -> None:
        assert False
