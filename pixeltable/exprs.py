import abc
import copy
import datetime
import enum
import inspect
import typing
from typing import Union, Optional, List, Callable, Any, Dict, Tuple
import operator

import PIL.Image
import jmespath
import numpy as np
import sqlalchemy as sql

from pixeltable import catalog
from pixeltable.type_system import \
    ColumnType, InvalidType, StringType, IntType, FloatType, BoolType, TimestampType, ImageType, DictType, ArrayType, \
    Function, UnknownType
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
        self.children: List[Expr] = []  # all exprs that this one depends on for the purpose of eval()

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
        if len(self.children) != len(other.children):
            return False
        for i in range(len(self.children)):
            if not self.children[i].equals(other.children[i]):
                return False
        return self._equals(other)

    def copy(self) -> 'Expr':
        """
        Creates a copy that can be evaluated separately: it doesn't share any eval context (data/sql_row_idx)
        but shares everything else (catalog objects, etc.)
        """
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        result.data_row_idx = -1
        result.sql_row_idx = -1
        for i in range(len(self.children)):
            self.children[i] = self.children[i].copy()
        return result

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

    def __getattr__(self, name: str) -> Expr:
        if self.col_type.is_dict_type():
            return DictPath(self).__getattr__(name)
        return super().__getattr__(name)

    def display_name(self) -> str:
        return self.col.name

    def _equals(self, other: 'ColumnRef') -> bool:
        return self.col == other.col

    def sql_expr(self) -> Optional[sql.sql.expression.ClauseElement]:
        return self.col.sa_col

    def eval(self, data_row: List[Any]) -> None:
        assert False


class FunctionCall(Expr):
    def __init__(self, fn: Function, args: Tuple[Any] = None):
        super().__init__(fn.return_type)
        self.eval_fn = fn.eval_fn
        params = inspect.signature(self.eval_fn).parameters
        required_params = [p for p in params.values() if p.default == inspect.Parameter.empty]
        if len(args) < len(required_params):
            raise exc.OperationalError(
                f"FunctionCall: number of arguments ({len(args)}) doesn't match the number of expected parameters "
                f"({len(params)})")

        if fn.param_types is not None:
            # check if arg types match param types and convert values, if necessary
            if len(args) != len(fn.param_types):
                raise exc.OperationalError(
                    f"Number of arguments doesn't match parameter list: {args} vs {fn.param_types}")
            args = list(args)
            for i in range(len(args)):
                if not isinstance(args[i], Expr):
                    # TODO: check non-Expr args
                    continue
                if args[i].col_type == fn.param_types[i]:
                    # nothing to do
                    continue
                converter = args[i].col_type.conversion_fn(fn.param_types[i])
                if converter is None:
                    raise exc.OperationalError(f'Cannot convert {args[i]} to {fn.param_types[i]}')
                if converter == ColumnType.no_conversion:
                    # nothing to do
                    continue
                convert_fn = Function(converter, fn.param_types[i], [args[i].col_type])
                args[i] = FunctionCall(convert_fn, (args[i],))

        self.children = [arg for arg in args if isinstance(arg, Expr)]
        self.args = [arg if not isinstance(arg, Expr) else None for arg in args]

    def _equals(self, other: 'FunctionCall') -> bool:
        if self.eval_fn != other.eval_fn:
            return False
        if len(self.args) != len(other.args):
            return False
        for i in range(len(self.args)):
            if (self.args[i] is None) != (other.args[i] is None):
                return False
            if self.args[i] != other.args[i]:
                return False
        return True

    def sql_expr(self) -> Optional[sql.sql.expression.ClauseElement]:
        return None

    def eval(self, data_row: List[Any]) -> None:
        args = copy.copy(self.args)
        # fill in missing child values
        i = 0
        for j in range(len(args)):
            if args[j] is None:
                args[j] = data_row[self.children[i].data_row_idx]
                i += 1
        data_row[self.data_row_idx] = self.eval_fn(*args)


def _caller_return_type(caller: Expr, *args: object, **kwargs: object) -> ColumnType:
    return caller.col_type

def _float_return_type(_: Expr, *args: object, **kwargs: object) -> ColumnType:
    return FloatType()

def _dict_return_type(_: Expr, *args: object, **kwargs: object) -> ColumnType:
    return DictType()

def _array_return_type(_: Expr, *args: object, **kwargs: object) -> ColumnType:
    return ArrayType()

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
_PIL_METHOD_INFO: Dict[str, Tuple[Callable, Callable]] = {
    'convert': (PIL.Image.Image.convert, _convert_return_type),
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
        self.children = [caller]

    def display_name(self) -> str:
        return self.member_name

    def _caller(self) -> Expr:
        return self.children[0]

    # TODO: correct signature?
    def __call__(self, *args, **kwargs) -> Union['ImageMethodCall', 'ImageSimilarityPredicate']:
        caller = self._caller()
        call_signature = f'({",".join([type(arg).__name__ for arg in args])})'
        if self.member_name == 'nearest':
            # - caller must be ColumnRef
            # - signature is (PIL.Image.Image)
            if not isinstance(caller, ColumnRef):
                raise exc.OperationalError(f'nearest(): caller must be an IMAGE column')
            if len(args) != 1 or not isinstance(args[0], PIL.Image.Image):
                raise exc.OperationalError(
                    f'nearest(): required signature is (PIL.Image.Image) (passed: {call_signature})')
            return ImageSimilarityPredicate(caller, img=args[0])

        if self.member_name == 'matches':
            # - caller must be ColumnRef
            # - signature is (str)
            if not isinstance(caller, ColumnRef):
                raise exc.OperationalError(f'matches(): caller must be an IMAGE column')
            if len(args) != 1 or not isinstance(args[0], str):
                raise exc.OperationalError(f"matches(): required signature is (str) (passed: {call_signature})")
            return ImageSimilarityPredicate(caller, text=args[0])

        # TODO: verify signature
        return ImageMethodCall(self.member_name, caller, *args, **kwargs)

    def _equals(self, other: 'ImageMemberAccess') -> bool:
        return self.member_name == other.member_name

    def sql_expr(self) -> Optional[sql.sql.expression.ClauseElement]:
        return None

    def eval(self, data_row: List[Any]) -> None:
        caller_val = data_row[self._caller().data_row_idx]
        try:
            data_row[self.data_row_idx] = getattr(caller_val, self.member_name)
        except AttributeError:
            data_row[self.data_row_idx] = None


class ImageMethodCall(FunctionCall):
    """
    Ex.: tbl.img_col.rotate(90)
    """
    def __init__(self, method_name: str, caller: Expr, *args: object, **kwargs: object):
        assert method_name in _PIL_METHOD_INFO
        self.method_name = method_name
        method_info = _PIL_METHOD_INFO[self.method_name]
        return_type = method_info[1](caller, *args, **kwargs)
        fn = Function(method_info[0], return_type, None)
        super().__init__(fn, (caller, *args))
        # TODO: deal with kwargs

    def display_name(self) -> str:
        return self.method_name


class DictPath(Expr):
    def __init__(self, anchor: ColumnRef, path_elements: List[str] = []):
        super().__init__(UnknownType())
        self.children = [anchor]
        self.path_elements: List[Union[str, int]] = path_elements
        self.compiled_path = jmespath.compile(self._json_path()) if len(path_elements) > 1 else None

    def _anchor(self) -> Expr:
        return self.children[0]

    def __getattr__(self, name: str) -> 'DictPath':
        assert isinstance(name, str)
        return DictPath(self._anchor(), self.path_elements + [name])

    def __getitem__(self, index: object) -> 'DictPath':
        if isinstance(index, str) and index != '*':
            raise exc.OperationalError(f'A DictType path index ')
        return DictPath(self._anchor(), self.path_elements + [index])

    def display_name(self) -> str:
        return f'{self._anchor().col.name}.{self._json_path()}'

    def _equals(self, other: 'DictPath') -> bool:
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
        first_element = self.path_elements[0]
        assert isinstance(first_element, str) and first_element != '*'
        result: List[str] = [first_element]
        for element in self.path_elements[1:]:
            if element == '*':
                result.append('[*]')
            elif isinstance(element, str):
                result.append(f'.{element}')
            elif isinstance(element, int):
                result.append(f'[{element}]')
        return ''.join(result)

    def eval(self, data_row: List[Any]) -> None:
        assert self.compiled_path is not None  # there should always be at least one path element
        _ = self._json_path()
        dict_val = data_row[self._anchor().data_row_idx]
        val = self.compiled_path.search(dict_val)
        data_row[self.data_row_idx] = val


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
        # operands are stored in self.children
        if self.operator == LogicalOperator.NOT:
            assert len(operands) == 1
            self.children = operands
        else:
            assert len(operands) > 1
            self.operands: List[Predicate] = []
            for operand in operands:
                self._merge_operand(operand)

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
            for child_op in operand.children:
                self._merge_operand(child_op)
        else:
            self.children.append(operand)

    def _equals(self, other: 'CompoundPredicate') -> bool:
        return self.operator == other.operator

    def extract_sql_predicate(self) -> Tuple[Optional[sql.sql.expression.ClauseElement], Optional[Predicate]]:
        if self.operator == LogicalOperator.NOT:
            e = self.children[0].sql_expr()
            return (None, self) if e is None else (e, None)

        sql_exprs = [op.sql_expr() for op in self.children]
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
        other_preds = [self.children[i] for i, e in enumerate(sql_exprs) if e is None]
        assert len(sql_preds) > 0
        combined_sql_pred = sql.and_(*sql_preds)
        combined_other = self.make_conjunction(other_preds)
        return (combined_sql_pred, combined_other)

    def split_conjuncts(
            self, condition: Callable[['Predicate'], bool]) -> Tuple[List['Predicate'], Optional['Predicate']]:
        if self.operator == LogicalOperator.OR or self.operator == LogicalOperator.NOT:
            return super().split_conjuncts(condition)
        matches = [op for op in self.children if condition(op)]
        non_matches = [op for op in self.children if not condition(op)]
        return (matches, self.make_conjunction(non_matches))

    def sql_expr(self) -> Optional[sql.sql.expression.ClauseElement]:
        sql_exprs = [op.sql_expr() for op in self.children]
        if any(e is None for e in sql_exprs):
            return None
        if self.operator == LogicalOperator.NOT:
            assert len(sql_exprs) == 1
            return sql.not_(sql_exprs[0])
        assert len(sql_exprs) > 1
        operator = sql.and_ if self.operator == LogicalOperator.AND else sql.or_
        combined = operator(*sql_exprs)
        return combined

    def eval(self, data_row: List[Any]) -> None:
        if self.operator == LogicalOperator.NOT:
            data_row[self.data_row_idx] = not data_row[self.children[0].data_row_idx]
        else:
            val = True if self.operator == LogicalOperator.AND else False
            op_function = operator.and_ if self.operator == LogicalOperator.AND else operator.or_
            for op in self.children:
                val = op_function(val, data_row[op.data_row_idx])
            data_row[self.data_row_idx] = val


class Comparison(Predicate):
    def __init__(self, operator: ComparisonOperator, op1: Expr, op2: Expr):
        super().__init__()
        self.operator = operator
        self.children = [op1, op2]

    def _equals(self, other: 'Comparison') -> bool:
        return self.operator == other.operator

    @property
    def _op1(self) -> Expr:
        return self.children[0]

    @property
    def _op2(self) -> Expr:
        return self.children[1]

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

    def eval(self, data_row: List[Any]) -> None:
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


class ImageSimilarityPredicate(Predicate):
    def __init__(self, img_col: ColumnRef, img: Optional[PIL.Image.Image] = None, text: Optional[str] = None):
        assert (img is None) != (text is None)
        super().__init__()
        self.img_col = img_col
        self.children = [img_col]
        self.img = img
        self.text = text

    def embedding(self) -> np.ndarray:
        if self.text is not None:
            return clip.encode_text(self.text)
        else:
            return clip.encode_image(self.img)

    def _equals(self, other: 'ImageSimilarityPredicate') -> bool:
        return False

    def sql_expr(self) -> Optional[sql.sql.expression.ClauseElement]:
        return None

    def eval(self, data_row: List[Any]) -> None:
        assert False
