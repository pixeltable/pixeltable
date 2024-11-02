from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import sqlalchemy as sql

import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable import exprs
from .globals import is_valid_identifier, MediaValidation

if TYPE_CHECKING:
    from .table_version import TableVersion

_logger = logging.getLogger('pixeltable')


class Column:
    """Representation of a column in the schema of a Table/DataFrame.

    A Column contains all the metadata necessary for executing queries and updates against a particular version of a
    table/view.
    """
    name: str
    id: Optional[int]
    col_type: ts.ColumnType
    stored: bool
    is_pk: bool
    _media_validation: Optional[MediaValidation]  # if not set, TableVersion.media_validation applies
    schema_version_add: Optional[int]
    schema_version_drop: Optional[int]
    _records_errors: Optional[bool]
    sa_col: Optional[sql.schema.Column]
    sa_col_type: Optional[sql.sqltypes.TypeEngine]
    sa_errormsg_col: Optional[sql.schema.Column]
    sa_errortype_col: Optional[sql.schema.Column]
    compute_func: Optional[Callable]
    _value_expr: Optional[exprs.Expr]
    value_expr_dict: Optional[dict[str, Any]]
    dependent_cols: set[Column]
    tbl: Optional[TableVersion]

    def __init__(
            self, name: Optional[str], col_type: Optional[ts.ColumnType] = None,
            computed_with: Optional[Union[exprs.Expr, Callable]] = None,
            is_pk: bool = False, stored: bool = True, media_validation: Optional[MediaValidation] = None,
            col_id: Optional[int] = None, schema_version_add: Optional[int] = None,
            schema_version_drop: Optional[int] = None, sa_col_type: Optional[sql.sqltypes.TypeEngine] = None,
            records_errors: Optional[bool] = None, value_expr_dict: Optional[dict[str, Any]] = None,
    ):
        """Column constructor.

        Args:
            name: column name; None for system columns (eg, index columns)
            col_type: column type; can be None if the type can be derived from ``computed_with``
            computed_with: a callable or an Expr object that computes the column value
            is_pk: if True, this column is part of the primary key
            stored: determines whether a computed column is present in the stored table or recomputed on demand
            col_id: column ID (only used internally)

        Computed columns: those have a non-None ``computed_with`` argument
        - when constructed by the user: ``computed_with`` was constructed explicitly and is passed in;
          col_type is None
        - when loaded from md store: ``computed_with`` is set and col_type is set

        ``computed_with`` is a Callable:
        - the callable's parameter names must correspond to existing columns in the table for which this Column
          is being used
        - ``col_type`` needs to be set to the callable's return type

        ``stored`` (only valid for computed image columns):
        - if True: the column is present in the stored table
        - if False: the column is not present in the stored table and recomputed during a query
        - if None: the system chooses for you (at present, this is always False, but this may change in the future)
        """
        if name is not None and not is_valid_identifier(name):
            raise excs.Error(f"Invalid column name: '{name}'")
        self.name = name
        if col_type is None and computed_with is None:
            raise excs.Error(f'Column `{name}`: col_type is required if computed_with is not specified')

        self._value_expr = None
        self.compute_func = None
        self.value_expr_dict = value_expr_dict
        if computed_with is not None:
            value_expr = exprs.Expr.from_object(computed_with)
            if value_expr is None:
                # computed_with needs to be a Callable
                if not callable(computed_with):
                    raise excs.Error(
                        f'Column {name}: computed_with needs to be either a Pixeltable expression or a Callable, '
                        f'but it is a {type(computed_with)}')
                if col_type is None:
                    raise excs.Error(f'Column {name}: col_type is required if computed_with is a Callable')
                # we need to turn the computed_with function into an Expr, but this requires resolving
                # column name references and for that we need to wait until we're assigned to a Table
                self.compute_func = computed_with
            else:
                self._value_expr = value_expr.copy()
                self.col_type = self._value_expr.col_type

        if col_type is not None:
            self.col_type = col_type
        assert self.col_type is not None

        self.stored = stored
        self.dependent_cols = set()  # cols with value_exprs that reference us; set by TableVersion
        self.id = col_id
        self.is_pk = is_pk
        self._media_validation = media_validation
        self.schema_version_add = schema_version_add
        self.schema_version_drop = schema_version_drop

        self._records_errors = records_errors

        # column in the stored table for the values of this Column
        self.sa_col = None
        self.sa_col_type = sa_col_type

        # computed cols also have storage columns for the exception string and type
        self.sa_errormsg_col = None
        self.sa_errortype_col = None

        self.tbl = None  # set by owning TableVersion

    @property
    def value_expr(self) -> Optional[exprs.Expr]:
        """Instantiate value_expr on-demand"""
        # TODO: instantiate expr in the c'tor and add an Expr.prepare() that can create additional state after the
        # catalog has been fully loaded; that way, we encounter bugs in the serialization/deserialization logic earlier
        if self.value_expr_dict is not None and self._value_expr is None:
            from pixeltable import exprs
            self._value_expr = exprs.Expr.from_dict(self.value_expr_dict)
        return self._value_expr

    def set_value_expr(self, value_expr: exprs.Expr) -> None:
        self._value_expr = value_expr
        self.value_expr_dict = None

    def check_value_expr(self) -> None:
        assert self._value_expr is not None
        if self.stored == False and self.is_computed and self.has_window_fn_call():
            raise excs.Error(
                f'Column {self.name}: stored={self.stored} not supported for columns computed with window functions:'
                f'\n{self.value_expr}')

    def has_window_fn_call(self) -> bool:
        if self.value_expr is None:
            return False
        from pixeltable import exprs
        l = list(self.value_expr.subexprs(filter=lambda e: isinstance(e, exprs.FunctionCall) and e.is_window_fn_call))
        return len(l) > 0

    def get_idx_info(self) -> dict[str, 'TableVersion.IndexInfo']:
        assert self.tbl is not None
        return {name: info for name, info in self.tbl.idxs_by_name.items() if info.col == self}

    @property
    def is_computed(self) -> bool:
        return self.compute_func is not None or self._value_expr is not None or self.value_expr_dict is not None

    @property
    def is_stored(self) -> bool:
        """Returns True if column is materialized in the stored table."""
        assert self.stored is not None
        return self.stored

    @property
    def records_errors(self) -> bool:
        """True if this column also stores error information."""
        # default: record errors for computed and media columns
        if self._records_errors is not None:
            return self._records_errors
        return self.is_stored and (self.is_computed or self.col_type.is_media_type())

    @property
    def qualified_name(self) -> str:
        assert self.tbl is not None
        return f'{self.tbl.name}.{self.name}'

    @property
    def media_validation(self) -> MediaValidation:
        if self._media_validation is not None:
            return self._media_validation
        assert self.tbl is not None
        return self.tbl.media_validation

    def source(self) -> None:
        """
        If this is a computed col and the top-level expr is a function call, print the source, if possible.
        """
        from pixeltable import exprs
        if self.value_expr is None or not isinstance(self.value_expr, exprs.FunctionCall):
            return
        self.value_expr.fn.source()

    def create_sa_cols(self) -> None:
        """
        These need to be recreated for every new table schema version.
        """
        assert self.is_stored
        # all storage columns are nullable (we deal with null errors in Pixeltable directly)
        self.sa_col = sql.Column(
            self.store_name(), self.col_type.to_sa_type() if self.sa_col_type is None else self.sa_col_type,
            nullable=True)
        if self.is_computed or self.col_type.is_media_type():
            self.sa_errormsg_col = sql.Column(self.errormsg_store_name(), ts.StringType().to_sa_type(), nullable=True)
            self.sa_errortype_col = sql.Column(self.errortype_store_name(), ts.StringType().to_sa_type(), nullable=True)

    def get_sa_col_type(self) -> sql.sqltypes.TypeEngine:
        return self.col_type.to_sa_type() if self.sa_col_type is None else self.sa_col_type

    def store_name(self) -> str:
        assert self.id is not None
        assert self.is_stored
        return f'col_{self.id}'

    def errormsg_store_name(self) -> str:
        return f'{self.store_name()}_errormsg'

    def errortype_store_name(self) -> str:
        return f'{self.store_name()}_errortype'

    def __str__(self) -> str:
        return f'{self.name}: {self.col_type}'

    def __repr__(self) -> str:
        return f'Column({self.id!r}, {self.name!r}, tbl={self.tbl.name!r})'

    def __hash__(self) -> int:
        # TODO(aaron-siegel): This and __eq__ do not capture the table version. We need to rethink the Column
        #     abstraction (perhaps separating out the version-dependent properties into a different abstraction).
        assert self.tbl is not None
        return hash((self.tbl.id, self.id))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Column):
            return False
        assert self.tbl is not None
        assert other.tbl is not None
        return self.tbl.id == other.tbl.id and self.id == other.id
