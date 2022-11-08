import abc
import datetime
import enum
import inspect
import typing
from typing import Union, Optional, List, Callable, Any, Dict, Tuple

import PIL.Image
import sqlalchemy as sql

from pixeltable import catalog
from pixeltable.type_system import ColumnType
from pixeltable import exceptions as exc

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

    @abc.abstractmethod
    def display_name(self) -> str:
        """
        Displayed column name in DataFrame. '': assigned by DataFrame
        """
        pass

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
    def child_exprs(self) -> List['Expr']:
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
        if self.col_type != ColumnType.IMAGE:
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

    def child_exprs(self) -> List['Expr']:
        assert False
        return []

    def eval(self, data_row: List[Any]) -> None:
        assert False


class FunctionCall(Expr):
    def __init__(self, fn: Callable, tbl: Optional[catalog.Table], col_type: ColumnType):
        super().__init__(col_type)
        self.fn = fn
        self.col_args: List[ColumnRef] = []
        # make sure that fn's params are valid col names in tbl
        params = inspect.signature(self.fn).parameters
        assert len(params) == 0 or tbl is not None
        for param_name in params:
            if param_name not in tbl.cols_by_name:
                raise exc.OperationalError(
                    (f'FunctionCall: lambda argument names need to be valid column names in table {tbl.name}: '
                     'column {param_name} unknown'))
            self.col_args.append(ColumnRef(tbl.cols_by_name[param_name]))

    def display_name(self) -> str:
        return ''

    def _equals(self, other: 'Expr') -> bool:
        # we don't know whether self.fn and other.fn compute the same thing
        return False

    def sql_expr(self) -> Optional[sql.sql.expression.ClauseElement]:
        return None

    def child_exprs(self) -> List['Expr']:
        return self.col_args

    def eval(self, data_row: List[Any]) -> None:
        param_vals = [data_row[col_ref.data_row_idx] for col_ref in self.col_args]
        data_row[self.data_row_idx] = self.fn(*param_vals)


def _create_pil_method_info() -> Dict[str, Tuple[Callable, inspect.Signature]]:
    fn_members = inspect.getmembers(PIL.Image.Image, inspect.isfunction)
    public_fns = [t for t in fn_members if not(t[0].startswith('_'))]
    return {t[0]: (t[1], inspect.signature(t[1])) for t in public_fns}

def _create_pil_attr_info() -> Dict[str, ColumnType]:
    # create random Image to inspect for attrs
    img = PIL.Image.new('RGB', (100, 100))
    # we're only interested in public attrs (including properties)
    result: Dict[str, ColumnType] = {}
    for name in [name for name in dir(img) if not callable(getattr(img, name)) and not name.startswith('_')]:
        if getattr(img, name) is None:
            continue
        if isinstance(getattr(img, name), str):
            result[name] = ColumnType.STRING
        if isinstance(getattr(img, name), int):
            result[name] = ColumnType.INT
        if getattr(img, name) is dict:
            result[name] = ColumnType.DICT
    return result


class ImageMemberAccess(Expr):
    """
    Access of either an attribute or function member of PIL.Image.Image.
    Ex.: tbl.img_col.rotate(90), tbl.img_col.width
    """
    method_info = _create_pil_method_info()
    attr_info = _create_pil_attr_info()

    def __init__(self, member_name: str, caller: Expr):
        if member_name in self.method_info:
            super().__init__(ColumnType.IMAGE)  # TODO: should be INVALID; requires a __call__() invocation
        elif member_name in self.attr_info:
            super().__init__(self.attr_info[member_name])
        else:
            raise exc.OperationalError(f'Unknown Image member: {member_name}')
        self.member_name = member_name
        self.caller = caller

    def display_name(self) -> str:
        return self.member_name

    # TODO: correct signature?
    def __call__(self, *args, **kwargs) -> 'ImageMethodCall':
        # TODO: verify signature
        return ImageMethodCall(self.member_name, self.caller, *args, **kwargs)

    def _equals(self, other: 'ImageMemberAccess') -> bool:
        return self.caller.equals(other.caller) and self.member_name == other.member_name

    def sql_expr(self) -> Optional[sql.sql.expression.ClauseElement]:
        return None

    def child_exprs(self) -> List['Expr']:
        return [self.caller]

    def eval(self, data_row: List[Any]) -> None:
        caller_val = data_row[self.caller.data_row_idx]
        data_row[self.data_row_idx] = getattr(caller_val, self.member_name)


class ImageMethodCall(Expr):
    """
    Ex.: tbl.img_col.rotate(90)
    """
    method_info = _create_pil_method_info()

    def __init__(self, method_name: str, caller: Expr, *args, **kwargs):
        super().__init__(ColumnType.IMAGE)  # TODO: set to INVALID
        assert method_name in self.method_info
        self.method_name = method_name
        self.fn = self.method_info[self.method_name][0]
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

    def child_exprs(self) -> List['Expr']:
        return [self.caller]

    def eval(self, data_row: List[Any]) -> None:
        args = [data_row[self.caller.data_row_idx]]
        args.extend(self.args)
        data_row[self.data_row_idx] = self.fn(*args, **self.kw_args)


class Literal(Expr):
    def __init__(self, val: LiteralPythonTypes):
        if isinstance(val, str):
            super().__init__(ColumnType.STRING)
        if isinstance(val, int):
            super().__init__(ColumnType.INT)
        if isinstance(val, float):
            super().__init__(ColumnType.FLOAT)
        if isinstance(val, bool):
            super().__init__(ColumnType.BOOL)
        if isinstance(val, datetime.datetime) or isinstance(val, datetime.date):
            super().__init__(ColumnType.TIMESTAMP)
        self.val = val

    def display_name(self) -> str:
        return 'Literal'

    def _equals(self, other: 'Literal') -> bool:
        return self.val == other.eval

    def sql_expr(self) -> Optional[sql.sql.expression.ClauseElement]:
        return sql.sql.expression.literal(self.val)

    def child_exprs(self) -> List['Expr']:
        assert False

    def eval(self, data_row: List[Any]) -> None:
        assert False


class Predicate(Expr):
    def __init__(self) -> None:
        super().__init__(ColumnType.BOOL)

    def __and__(self, other: object) -> 'CompoundPredicate':
        if not isinstance(other, Predicate):
            raise TypeError(f'Other needs to be a predicate: {type(other)}')
        assert isinstance(other, Predicate)
        return CompoundPredicate(LogicalOperator.AND, self, other)

    def __or__(self, other: object) -> 'CompoundPredicate':
        if not isinstance(other, Predicate):
            raise TypeError(f'Other needs to be a predicate: {type(other)}')
        assert isinstance(other, Predicate)
        return CompoundPredicate(LogicalOperator.OR, self, other)

    def __invert__(self) -> 'CompoundPredicate':
        return CompoundPredicate(LogicalOperator.NOT, self)


class CompoundPredicate(Predicate):
    def __init__(self, operator: LogicalOperator, op1: Predicate, op2: Optional[Predicate] = None):
        super().__init__()
        self.operator = operator
        self.op1 = op1
        self.op2 = op2

    def display_name(self) -> str:
        return ''

    def _equals(self, other: 'CompoundPredicate') -> bool:
        return self.operator == other.operator and self.op1.equals(other.op1) \
            and ((self.op2 is None and other.op2 is None) or self.op2.equals(other.op2))

    def sql_expr(self) -> Optional[sql.sql.expression.ClauseElement]:
        left = self.op1.sql_expr()
        assert left is not None  # TODO: implement mixed-mode predicates
        right = None if self.op2 is None else self.op2.sql_expr()
        if self.operator == LogicalOperator.AND:
            assert right is not None
            return sql.and_(left, right)
        if self.operator == LogicalOperator.OR:
            assert right is not None
            return sql.or_(left, right)
        if self.operator == LogicalOperator.NOT:
            assert right is None
            return sql.not_(left)

    def child_exprs(self) -> List['Expr']:
        assert False

    def eval(self, data_row: List[Any]) -> None:
        assert False


class Comparison(Predicate):
    def __init__(self, operator: ComparisonOperator, op1: Expr, op2: Expr):
        super().__init__()
        self.operator = operator
        self.op1 = op1
        self.op2 = op2

    def display_name(self) -> str:
        return ''

    def _equals(self, other: 'CompoundPredicate') -> bool:
        return self.operator == other.operator and self.op1.equals(other.op1) \
            and ((self.op2 is None and other.op2 is None) or self.op2.equals(other.op2))

    def sql_expr(self) -> Optional[sql.sql.expression.ClauseElement]:
        left = self.op1.sql_expr()
        right = self.op2.sql_expr()
        assert left is not None and right is not None
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

    def child_exprs(self) -> List['Expr']:
        assert False

    def eval(self, data_row: List[Any]) -> None:
        assert False
