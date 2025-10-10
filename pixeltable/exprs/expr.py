from __future__ import annotations

import abc
import hashlib
import importlib
import inspect
import json
import sys
import typing
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Optional, TypeVar, Union, overload
from uuid import UUID

import numpy as np
import sqlalchemy as sql
from typing_extensions import Self, _AnnotatedAlias

from pixeltable import catalog, exceptions as excs, func, type_system as ts

from .data_row import DataRow
from .globals import ArithmeticOperator, ComparisonOperator, LiteralPythonTypes, LogicalOperator, StringOperator

if TYPE_CHECKING:
    from pixeltable import exprs


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

    col_type: ts.ColumnType

    # the subexprs are needed to construct this expr
    components: list[Expr]

    # each instance has an id that is used for equality comparisons
    # - set by the subclass's __init__()
    # - produced by _create_id()
    # - not expected to survive a serialize()/deserialize() roundtrip
    id: Optional[int]

    # index of the expr's value in the data row:
    # - set for all materialized exprs
    # - None: not executable
    # - not set for subexprs that don't need to be materialized because the parent can be materialized via SQL
    slot_idx: Optional[int]

    T = TypeVar('T', bound='Expr')

    def __init__(self, col_type: ts.ColumnType):
        self.col_type = col_type
        self.components = []
        self.id = None
        self.slot_idx = None

    def dependencies(self) -> list[Expr]:
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

    def bind_rel_paths(self) -> None:
        """
        Binds relative JsonPaths to mapper.
        This needs to be done in a separate phase after __init__(), because RelativeJsonPath()(-1) cannot be resolved
        by the immediately containing JsonMapper during initialization.
        """
        self._bind_rel_paths()
        has_rel_path = self._has_relative_path()
        assert not has_rel_path, self._expr_tree()
        assert not self._has_relative_path(), self._expr_tree()

    def _bind_rel_paths(self, mapper: Optional['exprs.JsonMapperDispatch'] = None) -> None:
        for c in self.components:
            c._bind_rel_paths(mapper)

    def _expr_tree(self) -> str:
        """Returns a string representation of this expression as a multi-line tree. Useful for debugging."""
        buf: list[str] = []
        self._expr_tree_r(0, buf)
        return '\n'.join(buf)

    def _expr_tree_r(self, indent: int, buf: list[str]) -> None:
        buf.append(f'{" " * indent}{type(self).__name__}: {self}'.replace('\n', '\\n'))
        for c in self.components:
            c._expr_tree_r(indent + 2, buf)

    def default_column_name(self) -> Optional[str]:
        """
        Returns:
            None if this expression lacks a default name,
            or a valid identifier (according to catalog.is_valid_identifer) otherwise.
        """
        return None

    @property
    def validation_error(self) -> Optional[str]:
        """
        Subclasses can override this to indicate that validation has failed after a catalog load.

        If an Expr (or any of its transitive components) is invalid, then it cannot be evaluated, but its metadata
        will still be preserved in the catalog (so that the user can take appropriate corrective action).
        """
        for c in self.components:
            error = c.validation_error
            if error is not None:
                return error
        return None

    @property
    def is_valid(self) -> bool:
        return self.validation_error is None

    def equals(self, other: Expr) -> bool:
        """
        Subclass-specific comparison. Implemented as a function because __eq__() is needed to construct Comparisons.
        """
        if type(self) is not type(other):
            return False
        if len(self.components) != len(other.components):
            return False
        for i in range(len(self.components)):
            if not self.components[i].equals(other.components[i]):
                return False
        return self._equals(other)

    def _equals(self, other: Self) -> bool:
        # we already compared the type and components in equals(); subclasses that require additional comparisons
        # override this
        return True

    def _id_attrs(self) -> list[tuple[str, Any]]:
        """Returns attribute name/value pairs that are used to construct the instance id.

        Attribute values must be immutable and have str() defined.
        """
        return [('classname', self.__class__.__name__)]

    def _create_id(self) -> int:
        hasher = hashlib.sha256()
        for attr, value in self._id_attrs():
            hasher.update(attr.encode('utf-8'))
            hasher.update(str(value).encode('utf-8'))
        # Include the col_type of the expression to avoid expressions with identical str() representations
        # but different types being considered the same expression, e.g. str(int(4)) == "4"
        hasher.update(repr(self.col_type).encode('utf-8'))
        for expr in self.components:
            hasher.update(str(expr.id).encode('utf-8'))
        # truncate to machine's word size
        return int(hasher.hexdigest(), 16) & sys.maxsize

    def __hash__(self) -> int:
        assert self.id is not None
        return self.id

    @classmethod
    def list_equals(cls, a: list[Expr], b: list[Expr]) -> bool:
        if len(a) != len(b):
            return False
        return all(a[i].equals(b[i]) for i in range(len(a)))

    def copy(self: T) -> T:
        """
        Creates a copy that can be evaluated separately: it doesn't share any eval context (slot_idx)
        but shares everything else (catalog objects, etc.)
        """
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        result.slot_idx = None
        result.components = [c.copy() for c in self.components]
        return result

    @classmethod
    def copy_list(cls, expr_list: Optional[list[Expr]]) -> Optional[list[Expr]]:
        if expr_list is None:
            return None
        return [e.copy() for e in expr_list]

    def __deepcopy__(self, memo: Optional[dict[int, Any]] = None) -> Expr:
        # we don't need to create an actual deep copy because all state other than execution state is read-only
        if memo is None:
            memo = {}
        result = self.copy()
        memo[id(self)] = result
        return result

    def substitute(self, spec: dict[Expr, Expr]) -> Expr:
        """
        Replace 'old' with 'new' recursively, and return a new version of the expression
        This method must be used in the form: expr = expr.substitute(spec)
        """
        from .literal import Literal

        if isinstance(self, Literal):
            return self
        for old, new in spec.items():
            if self.equals(old):
                return new.copy()
        for i in range(len(self.components)):
            self.components[i] = self.components[i].substitute(spec)
        result = self.maybe_literal()
        result.id = result._create_id()
        return result

    @classmethod
    def list_substitute(cls, expr_list: list[Expr], spec: dict[Expr, Expr]) -> None:
        for i in range(len(expr_list)):
            expr_list[i] = expr_list[i].substitute(spec)

    def resolve_computed_cols(self, resolve_cols: Optional[set[catalog.Column]] = None) -> Expr:
        """
        Recursively replace ColRefs to unstored computed columns with their value exprs.
        Also replaces references to stored computed columns in resolve_cols.
        """
        from .column_ref import ColumnRef
        from .expr_set import ExprSet

        if resolve_cols is None:
            resolve_cols = set()
        result = self
        while True:
            target_col_refs = ExprSet(
                [
                    e
                    for e in result.subexprs()
                    if isinstance(e, ColumnRef) and e.col.is_computed and (not e.col.is_stored or e.col in resolve_cols)
                ]
            )
            if len(target_col_refs) == 0:
                return result
            result = result.substitute({ref: ref.col.value_expr for ref in target_col_refs})

    def is_bound_by(self, tbls: list[catalog.TableVersionPath]) -> bool:
        """Returns True if this expr can be evaluated in the context of tbls."""
        from .column_ref import ColumnRef

        col_refs = self.subexprs(ColumnRef)
        return all(any(tbl.has_column(col_ref.col) for tbl in tbls) for col_ref in col_refs)

    def retarget(self, tbl: catalog.TableVersionPath) -> Self:
        """Retarget ColumnRefs in this expr to the specific TableVersions in tbl."""
        tbl_versions = {tbl_version.id: tbl_version.get() for tbl_version in tbl.get_tbl_versions()}
        return self._retarget(tbl_versions)

    @classmethod
    def retarget_list(cls, expr_list: list[Expr], tbl: catalog.TableVersionPath) -> None:
        """Retarget ColumnRefs in expr_list to the specific TableVersions in tbl."""
        tbl_versions = {tbl_version.id: tbl_version.get() for tbl_version in tbl.get_tbl_versions()}
        for i in range(len(expr_list)):
            expr_list[i] = expr_list[i]._retarget(tbl_versions)

    def _retarget(self, tbl_versions: dict[UUID, catalog.TableVersion]) -> Self:
        for i in range(len(self.components)):
            self.components[i] = self.components[i]._retarget(tbl_versions)
        return self

    def __repr__(self) -> str:
        return f'<Expression of type {type(self)}>'

    def display_str(self, inline: bool = True) -> str:
        """
        inline: if False, use line breaks where appropriate; otherwise don't use linebreaks
        """
        return str(self)

    @classmethod
    def print_list(cls, expr_list: list[Any]) -> str:
        if len(expr_list) == 1:
            return str(expr_list[0])
        return f'({", ".join(str(e) for e in expr_list)})'

    # `subexprs` has two forms: one that takes an explicit subclass of `Expr` as an argument and returns only
    # instances of that subclass; and another that returns all subexpressions that match the given filter.
    # In order for type checking to behave correctly on both forms, we provide two overloaded signatures.

    @overload
    def subexprs(
        self, *, filter: Optional[Callable[[Expr], bool]] = None, traverse_matches: bool = True
    ) -> Iterator[Expr]: ...

    @overload
    def subexprs(
        self, expr_class: type[T], filter: Optional[Callable[[Expr], bool]] = None, traverse_matches: bool = True
    ) -> Iterator[T]: ...

    def subexprs(
        self,
        expr_class: Optional[type[T]] = None,
        filter: Optional[Callable[[Expr], bool]] = None,
        traverse_matches: bool = True,
    ) -> Iterator[T]:
        """
        Iterate over all subexprs, including self.
        """
        is_match = isinstance(self, expr_class) if expr_class is not None else True
        # apply filter after checking for expr_class
        if filter is not None and is_match:
            is_match = filter(self)
        if not is_match or traverse_matches:
            for c in self.components:
                yield from c.subexprs(expr_class=expr_class, filter=filter, traverse_matches=traverse_matches)
        if is_match:
            yield self  # type: ignore[misc]

    @overload
    @classmethod
    def list_subexprs(
        cls,
        expr_list: Iterable[Expr],
        *,
        filter: Optional[Callable[[Expr], bool]] = None,
        traverse_matches: bool = True,
    ) -> Iterator[Expr]: ...

    @overload
    @classmethod
    def list_subexprs(
        cls,
        expr_list: Iterable[Expr],
        expr_class: type[T],
        filter: Optional[Callable[[Expr], bool]] = None,
        traverse_matches: bool = True,
    ) -> Iterator[T]: ...

    @classmethod
    def list_subexprs(
        cls,
        expr_list: Iterable[Expr],
        expr_class: Optional[type[T]] = None,
        filter: Optional[Callable[[Expr], bool]] = None,
        traverse_matches: bool = True,
    ) -> Iterator[T]:
        """Produce subexprs for all exprs in list. Can contain duplicates."""
        for e in expr_list:
            yield from e.subexprs(expr_class=expr_class, filter=filter, traverse_matches=traverse_matches)

    def _contains(self, cls: Optional[type[Expr]] = None, filter: Optional[Callable[[Expr], bool]] = None) -> bool:
        """
        Returns True if any subexpr is an instance of cls and/or matches filter.
        """
        assert cls is not None or filter is not None
        try:
            _ = next(self.subexprs(expr_class=cls, filter=filter, traverse_matches=False))
            return True
        except StopIteration:
            return False

    def _has_relative_path(self) -> bool:
        return any(c._has_relative_path() for c in self.components)

    def tbl_ids(self) -> set[UUID]:
        """Returns table ids referenced by this expr."""
        from .column_ref import ColumnRef
        from .rowid_ref import RowidRef

        return {ref.col.tbl.id for ref in self.subexprs(ColumnRef)} | {ref.tbl.id for ref in self.subexprs(RowidRef)}

    @classmethod
    def all_tbl_ids(cls, exprs_: Iterable[Expr]) -> set[UUID]:
        return {tbl_id for e in exprs_ for tbl_id in e.tbl_ids()}

    @classmethod
    def get_refd_column_ids(cls, expr_dict: dict[str, Any]) -> set[catalog.QColumnId]:
        """Return Columns referenced by expr_dict."""
        result: set[catalog.QColumnId] = set()
        assert '_classname' in expr_dict
        from .column_ref import ColumnRef

        if expr_dict['_classname'] == 'ColumnRef':
            result.add(ColumnRef.get_column_id(expr_dict))
        if 'components' in expr_dict:
            for component_dict in expr_dict['components']:
                result.update(cls.get_refd_column_ids(component_dict))
        return result

    def as_literal(self) -> Optional[Expr]:
        """
        Return a Literal expression if this expression can be evaluated to a constant value, otherwise return None.
        """
        return None

    @classmethod
    def from_array(cls, elements: Iterable) -> Optional[Expr]:
        from .inline_expr import InlineArray
        from .literal import Literal

        if isinstance(elements, np.ndarray):
            pxttype = ts.ArrayType.from_literal(elements)
            if pxttype is not None:
                return Literal(elements, col_type=pxttype)

        inline_array = InlineArray(elements)
        return inline_array.maybe_literal()

    def maybe_literal(self: Expr) -> Expr:
        """
        Return a Literal if this expression can be evaluated to a constant value, otherwise return the expression.
        """
        lit_expr = self.as_literal()
        if lit_expr is not None:
            return lit_expr
        else:
            return self

    @classmethod
    def from_object(cls, o: object) -> Optional[Expr]:
        """
        Try to turn a literal object into an Expr.
        """
        from .inline_expr import InlineDict, InlineList
        from .literal import Literal

        # Try to create a literal. We need to check for InlineList/InlineDict
        # first, to prevent them from inappropriately being interpreted as JsonType
        # literals.
        if isinstance(o, Literal):
            return o

        if isinstance(o, (list, tuple, dict, Expr)):
            expr: Expr
            if isinstance(o, (list, tuple)):
                expr = InlineList(o)
            elif isinstance(o, dict):
                expr = InlineDict(o)
            else:
                expr = o

            return expr.maybe_literal()
        else:
            # convert scalar to a literal
            obj_type = ts.ColumnType.infer_literal_type(o)
            if obj_type is not None:
                return Literal(o, col_type=obj_type)
        return None

    def sql_expr(self, sql_elements: 'exprs.SqlElementCache') -> Optional[sql.ColumnElement]:
        """
        If this expr can be materialized directly in SQL:
        - returns a ColumnElement
        - eval() will not be called (exception: Literal)
        Otherwise
        - returns None
        - eval() will be called
        """
        return None

    @abc.abstractmethod
    def eval(self, data_row: DataRow, row_builder: 'exprs.RowBuilder') -> None:
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
    def release_list(cls, expr_list: list[Expr]) -> None:
        for e in expr_list:
            e.release()

    def serialize(self) -> str:
        return json.dumps(self.as_dict())

    def as_dict(self) -> dict:
        """
        Turn Expr object into a dict that can be passed to json.dumps().
        Subclasses override _as_dict().
        """
        return {'_classname': self.__class__.__name__, **self._as_dict()}

    @classmethod
    def as_dict_list(cls, expr_list: list[Expr]) -> list[dict]:
        return [e.as_dict() for e in expr_list]

    def _as_dict(self) -> dict:
        if len(self.components) > 0:
            return {'components': [c.as_dict() for c in self.components]}
        return {}

    @classmethod
    def deserialize(cls, dict_str: str) -> Expr:
        return cls.from_dict(json.loads(dict_str))

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        """
        Turn dict that was produced by calling Expr.as_dict() into an instance of the correct Expr subclass.
        """
        assert '_classname' in d
        exprs_module = importlib.import_module(cls.__module__.rsplit('.', 1)[0])
        type_class = getattr(exprs_module, d['_classname'])
        components: list[Expr] = []
        if 'components' in d:
            components = [cls.from_dict(component_dict) for component_dict in d['components']]
        return type_class._from_dict(d, components)

    @classmethod
    def from_dict_list(cls, dict_list: list[dict]) -> list[Expr]:
        return [cls.from_dict(d) for d in dict_list]

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr]) -> Self:
        raise AssertionError(f'not implemented: {cls.__name__}')

    def isin(self, value_set: Any) -> 'exprs.InPredicate':
        from .in_predicate import InPredicate

        if isinstance(value_set, Expr):
            return InPredicate(self, value_set_expr=value_set)
        else:
            return InPredicate(self, value_set_literal=value_set)

    def astype(self, new_type: Union[ts.ColumnType, type, _AnnotatedAlias]) -> 'exprs.TypeCast':
        from pixeltable.exprs import TypeCast

        # Interpret the type argument the same way we would if given in a schema
        col_type = ts.ColumnType.normalize_type(new_type, nullable_default=True, allow_builtin_types=False)
        if not self.col_type.nullable:
            # This expression is non-nullable; we can prove that the output is non-nullable, regardless of
            # whether new_type is given as nullable.
            col_type = col_type.copy(nullable=False)
        return TypeCast(self, col_type)

    def apply(
        self, fn: Callable, *, col_type: Union[ts.ColumnType, type, _AnnotatedAlias, None] = None
    ) -> 'exprs.FunctionCall':
        if col_type is not None:
            col_type = ts.ColumnType.normalize_type(col_type)
        function = self._make_applicator_function(fn, col_type)
        # Return a `FunctionCall` obtained by passing this `Expr` to the new `function`.
        return function(self)

    def __dir__(self) -> list[str]:
        attrs = ['isin', 'astype', 'apply']
        attrs += [f.name for f in func.FunctionRegistry.get().get_type_methods(self.col_type.type_enum)]
        return attrs

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(f'Expression of type `{type(self)}` is not callable')

    def __getitem__(self, index: object) -> Expr:
        if self.col_type.is_json_type():
            from .json_path import JsonPath

            return JsonPath(self)[index]
        if self.col_type.is_array_type():
            from .array_slice import ArraySlice

            if not isinstance(index, tuple):
                index = (index,)
            if any(not isinstance(i, (int, slice)) for i in index):
                raise AttributeError(f'Invalid array indices: {index}')
            return ArraySlice(self, index)
        raise AttributeError(f'Type {self.col_type} is not subscriptable')

    def __getattr__(self, name: str) -> 'exprs.Expr':
        """
        ex.: <img col>.rotate(60)
        """
        from .json_path import JsonPath
        from .method_ref import MethodRef

        if self.col_type.is_json_type():
            return JsonPath(self).__getattr__(name)
        else:
            method_ref = MethodRef(self, name)
            if method_ref.fn.is_property:
                # Marked as a property, so autoinvoke the method to obtain a `FunctionCall`
                assert method_ref.fn.arity == 1
                return method_ref.fn(method_ref.base_expr)
            else:
                # Return the `MethodRef` object itself; it requires arguments to become a `FunctionCall`
                return method_ref

    def __bool__(self) -> bool:
        raise TypeError(
            f'Pixeltable expressions cannot be used in conjunction with Python boolean operators (and/or/not)\n{self!r}'
        )

    def __lt__(self, other: object) -> 'exprs.Comparison':
        return self._make_comparison(ComparisonOperator.LT, other)

    def __le__(self, other: object) -> 'exprs.Comparison':
        return self._make_comparison(ComparisonOperator.LE, other)

    def __eq__(self, other: object) -> 'exprs.Expr':  # type: ignore[override]
        if other is None:
            from .is_null import IsNull

            return IsNull(self)
        return self._make_comparison(ComparisonOperator.EQ, other)

    def __ne__(self, other: object) -> 'exprs.Expr':  # type: ignore[override]
        if other is None:
            from .compound_predicate import CompoundPredicate
            from .is_null import IsNull

            return CompoundPredicate(LogicalOperator.NOT, [IsNull(self)])
        return self._make_comparison(ComparisonOperator.NE, other)

    def __gt__(self, other: object) -> 'exprs.Comparison':
        return self._make_comparison(ComparisonOperator.GT, other)

    def __ge__(self, other: object) -> 'exprs.Comparison':
        return self._make_comparison(ComparisonOperator.GE, other)

    def _make_comparison(self, op: ComparisonOperator, other: object) -> 'exprs.Comparison':
        """
        other: Union[Expr, LiteralPythonTypes]
        """
        # TODO: check for compatibility
        from .comparison import Comparison
        from .literal import Literal

        if isinstance(other, Expr):
            return Comparison(op, self, other)
        if isinstance(other, typing.get_args(LiteralPythonTypes)):
            return Comparison(op, self, Literal(other))
        raise TypeError(f'Other must be Expr or literal: {type(other)}')

    def __neg__(self) -> 'exprs.ArithmeticExpr':
        return self._make_arithmetic_expr(ArithmeticOperator.MUL, -1)

    def __add__(self, other: object) -> Union[exprs.ArithmeticExpr, exprs.StringOp]:
        if isinstance(self, str) or (isinstance(self, Expr) and self.col_type.is_string_type()):
            return self._make_string_expr(StringOperator.CONCAT, other)
        return self._make_arithmetic_expr(ArithmeticOperator.ADD, other)

    def __sub__(self, other: object) -> 'exprs.ArithmeticExpr':
        return self._make_arithmetic_expr(ArithmeticOperator.SUB, other)

    def __mul__(self, other: object) -> Union['exprs.ArithmeticExpr', 'exprs.StringOp']:
        if isinstance(self, str) or (isinstance(self, Expr) and self.col_type.is_string_type()):
            return self._make_string_expr(StringOperator.REPEAT, other)
        return self._make_arithmetic_expr(ArithmeticOperator.MUL, other)

    def __truediv__(self, other: object) -> 'exprs.ArithmeticExpr':
        return self._make_arithmetic_expr(ArithmeticOperator.DIV, other)

    def __mod__(self, other: object) -> 'exprs.ArithmeticExpr':
        return self._make_arithmetic_expr(ArithmeticOperator.MOD, other)

    def __floordiv__(self, other: object) -> 'exprs.ArithmeticExpr':
        return self._make_arithmetic_expr(ArithmeticOperator.FLOORDIV, other)

    def __radd__(self, other: object) -> Union['exprs.ArithmeticExpr', 'exprs.StringOp']:
        if isinstance(other, str) or (isinstance(other, Expr) and other.col_type.is_string_type()):
            return self._rmake_string_expr(StringOperator.CONCAT, other)
        return self._rmake_arithmetic_expr(ArithmeticOperator.ADD, other)

    def __rsub__(self, other: object) -> 'exprs.ArithmeticExpr':
        return self._rmake_arithmetic_expr(ArithmeticOperator.SUB, other)

    def __rmul__(self, other: object) -> Union['exprs.ArithmeticExpr', 'exprs.StringOp']:
        if isinstance(other, str) or (isinstance(other, Expr) and other.col_type.is_string_type()):
            return self._rmake_string_expr(StringOperator.REPEAT, other)
        return self._rmake_arithmetic_expr(ArithmeticOperator.MUL, other)

    def __rtruediv__(self, other: object) -> 'exprs.ArithmeticExpr':
        return self._rmake_arithmetic_expr(ArithmeticOperator.DIV, other)

    def __rmod__(self, other: object) -> 'exprs.ArithmeticExpr':
        return self._rmake_arithmetic_expr(ArithmeticOperator.MOD, other)

    def __rfloordiv__(self, other: object) -> 'exprs.ArithmeticExpr':
        return self._rmake_arithmetic_expr(ArithmeticOperator.FLOORDIV, other)

    def _make_string_expr(self, op: StringOperator, other: object) -> 'exprs.StringOp':
        """
        Make left-handed version of string expression.
        """
        from .literal import Literal
        from .string_op import StringOp

        if isinstance(other, Expr):
            return StringOp(op, self, other)
        if isinstance(other, typing.get_args(LiteralPythonTypes)):
            return StringOp(op, self, Literal(other))
        raise TypeError(f'Other must be Expr or literal: {type(other)}')

    def _rmake_string_expr(self, op: StringOperator, other: object) -> 'exprs.StringOp':
        """
        Right-handed version of _make_string_expr. other must be a literal; if it were an Expr,
        the operation would have already been evaluated in its left-handed form.
        """
        from .literal import Literal
        from .string_op import StringOp

        assert not isinstance(other, Expr)  # Else the left-handed form would have evaluated first
        if isinstance(other, typing.get_args(LiteralPythonTypes)):
            return StringOp(op, Literal(other), self)
        raise TypeError(f'Other must be Expr or literal: {type(other)}')

    def _make_arithmetic_expr(self, op: ArithmeticOperator, other: object) -> 'exprs.ArithmeticExpr':
        """
        other: Union[Expr, LiteralPythonTypes]
        """
        # TODO: check for compatibility
        from .arithmetic_expr import ArithmeticExpr
        from .literal import Literal

        if isinstance(other, Expr):
            return ArithmeticExpr(op, self, other)
        if isinstance(other, typing.get_args(LiteralPythonTypes)):
            return ArithmeticExpr(op, self, Literal(other))
        raise TypeError(f'Other must be Expr or literal: {type(other)}')

    def _rmake_arithmetic_expr(self, op: ArithmeticOperator, other: object) -> 'exprs.ArithmeticExpr':
        """
        Right-handed version of _make_arithmetic_expr. other must be a literal; if it were an Expr,
        the operation would have already been evaluated in its left-handed form.
        """
        # TODO: check for compatibility
        from .arithmetic_expr import ArithmeticExpr
        from .literal import Literal

        assert not isinstance(other, Expr)  # Else the left-handed form would have evaluated first
        if isinstance(other, typing.get_args(LiteralPythonTypes)):
            return ArithmeticExpr(op, Literal(other), self)
        raise TypeError(f'Other must be Expr or literal: {type(other)}')

    def __and__(self, other: object) -> Expr:
        if not isinstance(other, Expr):
            raise TypeError(f'Other needs to be an expression: {type(other)}')
        if not other.col_type.is_bool_type():
            raise TypeError(f'Other needs to be an expression that returns a boolean: {other.col_type}')
        from .compound_predicate import CompoundPredicate

        return CompoundPredicate(LogicalOperator.AND, [self, other])

    def __or__(self, other: object) -> Expr:
        if not isinstance(other, Expr):
            raise TypeError(f'Other needs to be an expression: {type(other)}')
        if not other.col_type.is_bool_type():
            raise TypeError(f'Other needs to be an expression that returns a boolean: {other.col_type}')
        from .compound_predicate import CompoundPredicate

        return CompoundPredicate(LogicalOperator.OR, [self, other])

    def __invert__(self) -> Expr:
        from .compound_predicate import CompoundPredicate

        return CompoundPredicate(LogicalOperator.NOT, [self])

    def split_conjuncts(self, condition: Callable[[Expr], bool]) -> tuple[list[Expr], Optional[Expr]]:
        """
        Returns clauses of a conjunction that meet condition in the first element.
        The second element contains remaining clauses, rolled into a conjunction.
        """
        assert self.col_type.is_bool_type()  # only valid for predicates
        if condition(self):
            return [self], None
        else:
            return [], self

    def _make_applicator_function(self, fn: Callable, col_type: Optional[ts.ColumnType]) -> 'func.Function':
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
                f'Use `.apply({fn.__name__}, col_type=...)` to specify.'
            )

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
                raise excs.Error(f'Function `{fn.__name__}` has no positional parameters.')
            # Check that fn has at most one required parameter, i.e., its second parameter
            # has no default and is not a varargs
            if (
                len(params) >= 2
                and second_param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                and second_param.default is inspect.Parameter.empty
            ):
                raise excs.Error(f'Function `{fn.__name__}` has multiple required parameters.')
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
            decorated_fn=lambda x: fn(x), return_type=fn_type, param_types=[self.col_type], function_name=fn.__name__
        )


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
