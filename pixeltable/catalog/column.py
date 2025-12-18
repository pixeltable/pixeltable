from __future__ import annotations

import logging
import warnings
from textwrap import dedent
from typing import TYPE_CHECKING, Any

import pgvector.sqlalchemy  # type: ignore[import-untyped]
import sqlalchemy as sql

import pixeltable.exceptions as excs
import pixeltable.exprs as exprs
import pixeltable.type_system as ts
from pixeltable.env import Env
from pixeltable.metadata import schema

from .globals import MediaValidation, QColumnId, is_valid_identifier

if TYPE_CHECKING:
    from .table_version import TableVersion
    from .table_version_handle import ColumnHandle, TableVersionHandle
    from .table_version_path import TableVersionPath

_logger = logging.getLogger('pixeltable')


class Column:
    """Representation of a column in the schema of a Table/Query.

    A Column contains all the metadata necessary for executing queries and updates against a particular version of a
    table/view.

    Args:
        name: column name; None for system columns (eg, index columns)
        col_type: column type; can be None if the type can be derived from ``computed_with``
        computed_with: an Expr that computes the column value
        is_pk: if True, this column is part of the primary key
        stored: determines whether a computed column is present in the stored table or recomputed on demand
        destination: An object store reference for persisting computed files
        col_id: column ID (only used internally)

    Computed columns: those have a non-None ``computed_with`` argument
    - when constructed by the user: ``computed_with`` was constructed explicitly and is passed in;
        col_type is None
    - when loaded from md store: ``computed_with`` is set and col_type is set

    ``stored`` (only valid for computed columns):
    - if True: the column is present in the stored table
    - if False: the column is not present in the stored table and recomputed during a query
    - if None: the system chooses for you (at present, this is always False, but this may change in the future)
    """

    name: str | None
    id: int | None
    col_type: ts.ColumnType
    stored: bool
    is_pk: bool
    is_iterator_col: bool
    _explicit_destination: str | None  # An object store reference for computed files
    _media_validation: MediaValidation | None  # if not set, TableVersion.media_validation applies
    schema_version_add: int | None
    schema_version_drop: int | None
    stores_cellmd: bool
    sa_col: sql.schema.Column | None
    sa_col_type: sql.types.TypeEngine
    sa_cellmd_col: sql.schema.Column | None  # JSON metadata for the cell, e.g. errortype, errormsg for media columns
    _value_expr: exprs.Expr | None
    value_expr_dict: dict[str, Any] | None
    # we store a handle here in order to allow Column construction before there is a corresponding TableVersion
    tbl_handle: 'TableVersionHandle' | None

    def __init__(
        self,
        name: str | None,
        col_type: ts.ColumnType | None = None,
        computed_with: exprs.Expr | None = None,
        is_pk: bool = False,
        is_iterator_col: bool = False,
        stored: bool = True,
        media_validation: MediaValidation | None = None,
        col_id: int | None = None,
        schema_version_add: int | None = None,
        schema_version_drop: int | None = None,
        sa_col_type: sql.types.TypeEngine | None = None,
        stores_cellmd: bool | None = None,
        value_expr_dict: dict[str, Any] | None = None,
        tbl_handle: 'TableVersionHandle' | None = None,
        destination: str | None = None,
    ):
        if name is not None and not is_valid_identifier(name):
            raise excs.Error(f'Invalid column name: {name}')
        self.name = name
        self.tbl_handle = tbl_handle
        if col_type is None and computed_with is None:
            raise excs.Error(f'Column {name!r}: `col_type` is required if `computed_with` is not specified')

        self._value_expr = None
        self.value_expr_dict = value_expr_dict
        if computed_with is not None:
            value_expr = exprs.Expr.from_object(computed_with)
            if value_expr is None:
                # TODO: this shouldn't be a user-facing error
                raise excs.Error(
                    f'Column {name!r}: `computed_with` needs to be a valid Pixeltable expression, '
                    f'but it is a {type(computed_with)}'
                )
            else:
                self._value_expr = value_expr.copy()
                self.col_type = self._value_expr.col_type
        if self._value_expr is not None and self.value_expr_dict is None:
            self.value_expr_dict = self._value_expr.as_dict()

        if col_type is not None:
            self.col_type = col_type
        assert self.col_type is not None

        self.stored = stored
        # self.dependent_cols = set()  # cols with value_exprs that reference us; set by TableVersion
        self.id = col_id
        self.is_pk = is_pk
        self.is_iterator_col = is_iterator_col
        self._media_validation = media_validation
        self.schema_version_add = schema_version_add
        self.schema_version_drop = schema_version_drop

        if stores_cellmd is not None:
            self.stores_cellmd = stores_cellmd
        else:
            self.stores_cellmd = stored and (
                self.is_computed or self.col_type.is_media_type() or self.col_type.supports_file_offloading()
            )

        # column in the stored table for the values of this Column
        self.sa_col = None
        self.sa_col_type = self.col_type.to_sa_type() if sa_col_type is None else sa_col_type

        # computed cols also have storage columns for the exception string and type
        self.sa_cellmd_col = None
        self._explicit_destination = destination

    def to_md(self, pos: int | None = None) -> tuple[schema.ColumnMd, schema.SchemaColumn | None]:
        """Returns the Column and optional SchemaColumn metadata for this Column."""
        assert self.is_pk is not None
        col_md = schema.ColumnMd(
            id=self.id,
            col_type=self.col_type.as_dict(),
            is_pk=self.is_pk,
            schema_version_add=self.schema_version_add,
            schema_version_drop=self.schema_version_drop,
            value_expr=self.value_expr.as_dict() if self.value_expr is not None else None,
            stored=self.stored,
            destination=self._explicit_destination,
        )
        if pos is None:
            return col_md, None
        assert self.name is not None, 'Column name must be set for user-facing columns'
        sch_md = schema.SchemaColumn(
            name=self.name,
            pos=pos,
            media_validation=self._media_validation.name.lower() if self._media_validation is not None else None,
        )
        return col_md, sch_md

    def init_value_expr(self, tvp: 'TableVersionPath' | None) -> None:
        """
        Initialize the value_expr from its dict representation, if necessary.

        If `tvp` is not None, retarget the value_expr to the given TableVersionPath.
        """
        from pixeltable import exprs

        if self._value_expr is None and self.value_expr_dict is None:
            return

        if self._value_expr is None:
            # Instantiate the Expr from its dict
            self._value_expr = exprs.Expr.from_dict(self.value_expr_dict)
            self._value_expr.bind_rel_paths()
            if not self._value_expr.is_valid:
                message = (
                    dedent(
                        f"""
                        The computed column {self.name!r} in table {self.get_tbl().name!r} is no longer valid.
                        {{validation_error}}
                        You can continue to query existing data from this column, but evaluating it on new data will raise an error.
                        """  # noqa: E501
                    )
                    .strip()
                    .format(validation_error=self._value_expr.validation_error)
                )
                warnings.warn(message, category=excs.PixeltableWarning, stacklevel=2)

        if tvp is not None:
            # Retarget the Expr
            self._value_expr = self._value_expr.retarget(tvp)

    def get_tbl(self) -> TableVersion:
        tv = self.tbl_handle.get()
        return tv

    @property
    def destination(self) -> str | None:
        if self._explicit_destination is not None:
            # An expilicit destination was set as part of the column definition
            return self._explicit_destination

        # Otherwise, if this is a stored media column, use the default destination if one is configured (input
        #     destination or output destination, depending on whether this is a computed column)
        # TODO: The `self.name is not None` clause is necessary because index columns currently follow the type of
        #     the underlying media column. We should move to using pxt.String as the col_type of index columns; this
        #     would be a more robust solution, and then `self.name is not None` could be removed.
        if self.is_stored and self.col_type.is_media_type() and self.name is not None:
            if self.is_computed:
                return Env.get().default_output_media_dest
            else:
                return Env.get().default_input_media_dest

        return None

    @property
    def handle(self) -> 'ColumnHandle':
        """Returns a ColumnHandle for this Column."""
        from .table_version_handle import ColumnHandle

        assert self.tbl_handle is not None
        assert self.id is not None
        return ColumnHandle(self.tbl_handle, self.id)

    @property
    def qid(self) -> QColumnId:
        assert self.tbl_handle is not None
        assert self.id is not None
        return QColumnId(self.tbl_handle.id, self.id)

    @property
    def value_expr(self) -> exprs.Expr | None:
        assert self.value_expr_dict is None or self._value_expr is not None
        return self._value_expr

    def set_value_expr(self, value_expr: exprs.Expr) -> None:
        self._value_expr = value_expr
        self.value_expr_dict = self._value_expr.as_dict()

    def check_value_expr(self) -> None:
        assert self._value_expr is not None
        if not self.stored and self.is_computed and self.has_window_fn_call():
            raise excs.Error(
                f'Column {self.name!r}: `stored={self.stored}` not supported for columns '
                f'computed with window functions:\n{self.value_expr}'
            )

    def has_window_fn_call(self) -> bool:
        from pixeltable import exprs

        if self.value_expr is None:
            return False
        window_fn_calls = list(
            self.value_expr.subexprs(filter=lambda e: isinstance(e, exprs.FunctionCall) and e.is_window_fn_call)
        )
        return len(window_fn_calls) > 0

    def has_sa_vector_type(self) -> bool:
        """Returns True if this column is a pgvector Vector or Halfvec."""
        return isinstance(self.sa_col_type, (pgvector.sqlalchemy.Vector, pgvector.sqlalchemy.HALFVEC))

    def stores_external_array(self) -> bool:
        """Returns True if this is an Array column that might store its values externally."""
        # Vector: if this is a vector column (ie, used for a vector index), it stores the array itself
        return self.col_type.is_array_type() and not self.has_sa_vector_type()

    @property
    def is_computed(self) -> bool:
        return self._value_expr is not None or self.value_expr_dict is not None

    @property
    def is_stored(self) -> bool:
        """Returns True if column is materialized in the stored table."""
        assert self.stored is not None
        return self.stored

    @property
    def qualified_name(self) -> str:
        assert self.get_tbl() is not None
        return f'{self.get_tbl().name}.{self.name}'

    @property
    def media_validation(self) -> MediaValidation:
        if self._media_validation is not None:
            return self._media_validation
        assert self.get_tbl() is not None
        return self.get_tbl().media_validation

    @property
    def is_required_for_insert(self) -> bool:
        """Returns True if column is required when inserting rows."""
        return not self.col_type.nullable and not self.is_computed

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
        These need to be recreated for every sql.Table instance
        """
        assert self.is_stored
        assert self.stores_cellmd is not None
        # all storage columns are nullable (we deal with null errors in Pixeltable directly)
        self.sa_col = sql.Column(self.store_name(), self.sa_col_type, nullable=True)
        if self.stores_cellmd:
            self.sa_cellmd_col = sql.Column(self.cellmd_store_name(), self.sa_cellmd_type(), nullable=True)

    @classmethod
    def cellmd_type(cls) -> ts.ColumnType:
        return ts.JsonType(nullable=True)

    @classmethod
    def sa_cellmd_type(cls) -> sql.types.TypeEngine:
        return cls.cellmd_type().to_sa_type()

    def store_name(self) -> str:
        assert self.id is not None
        assert self.is_stored
        return f'col_{self.id}'

    def cellmd_store_name(self) -> str:
        return f'{self.store_name()}_cellmd'

    def __str__(self) -> str:
        return f'{self.name}: {self.col_type}'

    def __repr__(self) -> str:
        return f'Column({self.id!r}, {self.name!r}, tbl={self.get_tbl().name!r})'

    def __hash__(self) -> int:
        # TODO(aaron-siegel): This and __eq__ do not capture the table version. We need to rethink the Column
        #     abstraction (perhaps separating out the version-dependent properties into a different abstraction).
        assert self.tbl_handle is not None
        return hash((self.tbl_handle.id, self.id))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Column):
            return False
        assert self.tbl_handle is not None
        assert other.tbl_handle is not None
        return self.tbl_handle.id == other.tbl_handle.id and self.id == other.id
