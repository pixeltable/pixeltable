from __future__ import annotations

import abc
import hashlib
import importlib
import inspect
import json
import sys
import typing
from itertools import islice
from typing import Union, Optional, List, Callable, Any, Dict, Tuple, Set, Generator, Type
from uuid import UUID

import sqlalchemy as sql

import pixeltable
import pixeltable.catalog as catalog
import pixeltable.exceptions as excs
import pixeltable.type_system as ts
import pixeltable.func as func
from .data_row import DataRow
from .globals import ComparisonOperator, LogicalOperator, LiteralPythonTypes, ArithmeticOperator


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
    def __init__(self, col_type: ts.ColumnType):
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

    def bind_rel_paths(self, mapper: Optional['pixeltable.exprs.JsonMapper'] = None) -> None:
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
        """Returns attribute name/value pairs that are used to construct the instance id.

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

    def __hash__(self) -> int:
        assert self.id is not None
        return self.id

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

    def __deepcopy__(self, memo=None) -> Expr:
        # we don't need to create an actual deep copy because all state other than execution state is read-only
        if memo is None:
            memo = {}
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

    def resolve_computed_cols(self, resolve_cols: Optional[Set[catalog.Column]] = None) -> Expr:
        """
        Recursively replace ColRefs to unstored computed columns with their value exprs.
        Also replaces references to stored computed columns in resolve_cols.
        """
        from .expr_set import ExprSet
        from .column_ref import ColumnRef
        if resolve_cols is None:
            resolve_cols = set()
        result = self
        while True:
            target_col_refs = ExprSet([
                e for e in result.subexprs()
                if isinstance(e, ColumnRef) and e.col.is_computed and (not e.col.is_stored or e.col in resolve_cols)
            ])
            if len(target_col_refs) == 0:
                return result
            for ref in target_col_refs:
                assert ref.col.value_expr is not None
                result = result.substitute(ref, ref.col.value_expr)

    def is_bound_by(self, tbl: catalog.TableVersionPath) -> bool:
        """Returns True if this expr can be evaluated in the context of tbl."""
        from .column_ref import ColumnRef
        col_refs = self.subexprs(ColumnRef)
        for col_ref in col_refs:
            if not tbl.has_column(col_ref.col):
                return False
        return True

    def retarget(self, tbl: catalog.TableVersionPath) -> Expr:
        """Retarget ColumnRefs in this expr to the specific TableVersions in tbl."""
        tbl_versions = {tbl_version.id: tbl_version for tbl_version in tbl.get_tbl_versions()}
        return self._retarget(tbl_versions)

    def _retarget(self, tbl_versions: Dict[UUID, catalog.TableVersion]) -> Expr:
        from .column_ref import ColumnRef
        if isinstance(self, ColumnRef):
            target = tbl_versions[self.col.tbl.id]
            assert self.col.id in target.cols_by_id
            col = target.cols_by_id[self.col.id]
            return ColumnRef(col)
        for i in range (len(self.components)):
            self.components[i] = self.components[i]._retarget(tbl_versions)
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
        from .column_ref import ColumnRef
        from .rowid_ref import RowidRef
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
        obj_type = ts.ColumnType.infer_literal_type(o)
        if obj_type is not None:
            from .literal import Literal
            return Literal(o, col_type=obj_type)
        if isinstance(o, dict):
            from .inline_dict import InlineDict
            return InlineDict(o)
        elif isinstance(o, list):
            from .inline_array import InlineArray
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
    def eval(self, data_row: DataRow, row_builder: 'pixeltable.exprs.RowBuilder') -> None:
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
    def deserialize(cls, dict_str: str) -> Expr:
        return cls.from_dict(json.loads(dict_str))

    @classmethod
    def from_dict(cls, d: Dict) -> Expr:
        """
        Turn dict that was produced by calling Expr.as_dict() into an instance of the correct Expr subclass.
        """
        assert '_classname' in d
        exprs_module = importlib.import_module(cls.__module__.rsplit('.', 1)[0])
        type_class = getattr(exprs_module, d['_classname'])
        components: List[Expr] = []
        if 'components' in d:
            components = [cls.from_dict(component_dict) for component_dict in d['components']]
        return type_class._from_dict(d, components)

    @classmethod
    def from_dict_list(cls, dict_list: List[Dict]) -> List[Expr]:
        return [cls.from_dict(d) for d in dict_list]

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr]) -> Expr:
        assert False, 'not implemented'

    def astype(self, new_type: ts.ColumnType) -> 'pixeltable.exprs.TypeCast':
        from pixeltable.exprs import TypeCast
        return TypeCast(self, new_type)

    def apply(self, fn: Callable, *, col_type: Optional[ts.ColumnType] = None) -> 'pixeltable.exprs.FunctionCall':
        function = self._make_applicator_function(fn, col_type)
        # Return a `FunctionCall` obtained by passing this `Expr` to the new `function`.
        return function(self)

    def __getitem__(self, index: object) -> Expr:
        if self.col_type.is_json_type():
            from .json_path import JsonPath
            return JsonPath(self).__getitem__(index)
        if self.col_type.is_array_type():
            from .array_slice import ArraySlice
            return ArraySlice(self, index)
        raise excs.Error(f'Type {self.col_type} is not subscriptable')

    def __getattr__(self, name: str) -> Union['pixeltable.exprs.ImageMemberAccess', 'pixeltable.exprs.JsonPath']:
        """
        ex.: <img col>.rotate(60)
        """
        if self.col_type.is_image_type():
            from .image_member_access import ImageMemberAccess
            return ImageMemberAccess(name, self)
        if self.col_type.is_json_type():
            from .json_path import JsonPath
            return JsonPath(self).__getattr__(name)
        raise excs.Error(f'Member access not supported on type {self.col_type}: {name}')

    def __bool__(self) -> bool:
        raise TypeError(
            'Pixeltable expressions cannot be used in conjunction with Python boolean operators (and/or/not)')

    def __lt__(self, other: object) -> 'pixeltable.exprs.Comparison':
        return self._make_comparison(ComparisonOperator.LT, other)

    def __le__(self, other: object) -> 'pixeltable.exprs.Comparison':
        return self._make_comparison(ComparisonOperator.LE, other)

    def __eq__(self, other: object) -> 'pixeltable.exprs.Comparison':
        if other is None:
            from .is_null import IsNull
            return IsNull(self)
        return self._make_comparison(ComparisonOperator.EQ, other)

    def __ne__(self, other: object) -> 'pixeltable.exprs.Comparison':
        if other is None:
            from .compound_predicate import CompoundPredicate
            from .is_null import IsNull
            return CompoundPredicate(LogicalOperator.NOT, [IsNull(self)])
        return self._make_comparison(ComparisonOperator.NE, other)

    def __gt__(self, other: object) -> 'pixeltable.exprs.Comparison':
        return self._make_comparison(ComparisonOperator.GT, other)

    def __ge__(self, other: object) -> 'pixeltable.exprs.Comparison':
        return self._make_comparison(ComparisonOperator.GE, other)

    def _make_comparison(self, op: ComparisonOperator, other: object) -> 'pixeltable.exprs.Comparison':
        """
        other: Union[Expr, LiteralPythonTypes]
        """
        # TODO: check for compatibility
        from .comparison import Comparison
        from .literal import Literal
        if isinstance(other, Expr):
            return Comparison(op, self, other)
        if isinstance(other, typing.get_args(LiteralPythonTypes)):
            return Comparison(op, self, Literal(other))  # type: ignore[arg-type]
        raise TypeError(f'Other must be Expr or literal: {type(other)}')

    def __add__(self, other: object) -> 'pixeltable.exprs.ArithmeticExpr':
        return self._make_arithmetic_expr(ArithmeticOperator.ADD, other)

    def __sub__(self, other: object) -> 'pixeltable.exprs.ArithmeticExpr':
        return self._make_arithmetic_expr(ArithmeticOperator.SUB, other)

    def __mul__(self, other: object) -> 'pixeltable.exprs.ArithmeticExpr':
        return self._make_arithmetic_expr(ArithmeticOperator.MUL, other)

    def __truediv__(self, other: object) -> 'pixeltable.exprs.ArithmeticExpr':
        return self._make_arithmetic_expr(ArithmeticOperator.DIV, other)

    def __mod__(self, other: object) -> 'pixeltable.exprs.ArithmeticExpr':
        return self._make_arithmetic_expr(ArithmeticOperator.MOD, other)

    def _make_arithmetic_expr(self, op: ArithmeticOperator, other: object) -> 'pixeltable.exprs.ArithmeticExpr':
        """
        other: Union[Expr, LiteralPythonTypes]
        """
        # TODO: check for compatibility
        from .arithmetic_expr import ArithmeticExpr
        from .literal import Literal
        if isinstance(other, Expr):
            return ArithmeticExpr(op, self, other)
        if isinstance(other, typing.get_args(LiteralPythonTypes)):
            return ArithmeticExpr(op, self, Literal(other))  # type: ignore[arg-type]
        raise TypeError(f'Other must be Expr or literal: {type(other)}')

    def _make_applicator_function(self, fn: Callable, col_type: Optional[ts.ColumnType]) -> 'pixeltable.func.Function':
        """
        Creates a unary pixeltable `Function` that encapsulates a python `Callable`. The result type of
        the new `Function` is given by `col_type`, and its parameter type will be `self.col_type`.

        Args:
            fn: The `Callable` to encapsulate. Must have at least one parameter, and at most one required
                parameter.
            col_type: The pixeltable result type of the new `Function`.
        """
        if col_type is not None:
            # col_type is specified explicitly
            fn_type = col_type
        elif fn in _known_applicator_types:
            # For convenience, various built-ins and other Python functions that don't
            # have type hints are hardcoded
            fn_type = _known_applicator_types[fn]
        elif 'return' in typing.get_type_hints(fn):
            # Attempt to infer the column type from the return type of the callable;
            # this will set fn_type to None if it cannot be inferred
            return_type = typing.get_type_hints(fn)['return']
            fn_type = ts.ColumnType.from_python_type(return_type)
        else:
            # No type hint
            fn_type = None

        if fn_type is None:
            raise excs.Error(
                f'Column type of `{fn.__name__}` cannot be inferred. '
                f'Use `.apply({fn.__name__}, col_type=...)` to specify.')

        # TODO(aaron-siegel) Currently we assume that `fn` has exactly one required parameter
        # and all optional parameters take their default values. Should we provide a more
        # flexible API? For example, by defining
        # expr.apply(fn, my_kw=my_arg)
        # to mean: transform each x by calling
        # fn(x, my_kw=my_arg)
        # In the current implementation, a lambda is needed in order to specify this pattern:
        # expr.apply(lambda x: fn(x, my_kw=my_arg))

        try:
            # If `fn` is not a builtin, we can do some basic validation to ensure it's
            # compatible with `apply`.
            params = inspect.signature(fn).parameters
            params_iter = iter(params.values())
            first_param = next(params_iter) if len(params) >= 1 else None
            second_param = next(params_iter) if len(params) >= 2 else None
            # Check that fn has at least one positional parameter
            if len(params) == 0 or first_param.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD):
                raise excs.Error(
                    f'Function `{fn.__name__}` has no positional parameters.'
                )
            # Check that fn has at most one required parameter, i.e., its second parameter
            # has no default and is not a varargs
            if len(params) >= 2 and \
                    second_param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD) and \
                    second_param.default == inspect.Parameter.empty:
                raise excs.Error(
                    f'Function `{fn.__name__}` has multiple required parameters.'
                )
        except ValueError:
            # inspect.signature(fn) will raise a `ValueError` if `fn` is a builtin; I don't
            # know of any way to get the signature of a builtin, nor to check for this in
            # advance (without the try/except pattern). For now, builtins will not be
            # validated.
            pass

        # Since `fn` might have optional parameters, we wrap it in a lambda to get a unary
        # equivalent, so that its signature is understood by `make_function`. This also
        # ensures that `decorated_fn` is never a builtin.
        # We also set the display_name explicitly, so that the `FunctionCall` gets the
        # name of `decorated_fn`, not the lambda.
        return func.make_function(
            decorated_fn=lambda x: fn(x), return_type=fn_type, param_types=[self.col_type], function_name=fn.__name__)


# A dictionary of result types of various stdlib functions that are
# commonly used in computed columns. stdlib does not have type hints, so these
# are used to infer their result types (as pixeltable types) to avoid having
# to specify them explicitly in Expr.apply().
# This is purely for convenience and does not impact the supported functionality
# (it's always possible to specify a result type explicitly for a function
# that does not have type hints and is not present in this dict).
_known_applicator_types: dict[Callable, ts.ColumnType] = {
    str: ts.StringType(),
    json.dumps: ts.StringType(),
    json.loads: ts.JsonType(),
}
