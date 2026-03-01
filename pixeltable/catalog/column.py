from __future__ import annotations

import json
import warnings
from keyword import iskeyword as is_python_keyword
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any

from pixeltable import catalog, exceptions as excs
from pixeltable.types import ColumnSpec
from pixeltable.utils.object_stores import ObjectOps

from typing import _GenericAlias  # type: ignore[attr-defined]  # isort: skip

import pgvector.sqlalchemy  # type: ignore[import-untyped]
import sqlalchemy as sql

import pixeltable.exprs as exprs
import pixeltable.index as index
import pixeltable.type_system as ts
from pixeltable.env import Env
from pixeltable.metadata import schema

from .globals import MediaValidation, QColumnId, is_system_column_name, is_valid_identifier

if TYPE_CHECKING:
    from .table_version import TableVersion
    from .table_version_handle import ColumnHandle, TableVersionHandle
    from .table_version_path import TableVersionPath


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
    _custom_metadata: Any  # user-defined metadata; must be a valid JSON-serializable object
    _comment: str | None
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
        stores_cellmd: bool = False,
        value_expr_dict: dict[str, Any] | None = None,
        tbl_handle: 'TableVersionHandle' | None = None,
        destination: str | Path | None = None,
        comment: str | None = None,
        custom_metadata: Any = None,
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
        self.stores_cellmd = stores_cellmd

        # column in the stored table for the values of this Column
        self.sa_col = None
        self.sa_col_type = self.col_type.to_sa_type() if sa_col_type is None else sa_col_type

        # computed cols also have storage columns for the exception string and type
        self.sa_cellmd_col = None

        if isinstance(destination, Path):
            destination = str(destination)

        self._explicit_destination = destination

        # user-defined metadata - stored but not used by Pixeltable itself
        self._custom_metadata = custom_metadata
        self._comment = comment

    @classmethod
    def create_index_columns(
        cls,
        tbl_handle: TableVersionHandle,
        col: Column,
        idx: index.IndexBase,
        val_col_id: int,
        undo_col_id: int,
        schema_version: int,
    ) -> tuple[Column, Column]:
        """Create value and undo columns for an index."""
        value_expr = idx.create_value_expr(col)
        val_col = cls(
            col_id=val_col_id,
            name=None,
            computed_with=value_expr,
            sa_col_type=idx.get_index_sa_type(value_expr.col_type),
            stored=True,
            stores_cellmd=idx.records_value_errors(),
            schema_version_add=schema_version,
            schema_version_drop=None,
            tbl_handle=tbl_handle,
        )
        val_col.col_type = val_col.col_type.copy(nullable=True)

        undo_col = cls(
            col_id=undo_col_id,
            name=None,
            col_type=val_col.col_type,
            sa_col_type=val_col.sa_col_type,
            stored=True,
            stores_cellmd=False,
            schema_version_add=schema_version,
            schema_version_drop=None,
            tbl_handle=tbl_handle,
        )
        undo_col.col_type = undo_col.col_type.copy(nullable=True)
        return val_col, undo_col

    @classmethod
    def create(cls, name: str, spec: type | ColumnSpec | exprs.Expr) -> Column:
        col_type: ts.ColumnType | None = None
        value_expr: exprs.Expr | None = None
        primary_key: bool = False
        media_validation: catalog.MediaValidation | None = None
        stored: bool = True
        destination: str | Path | None = None
        custom_metadata: Any = None
        comment: str | None = None

        # TODO: Should we fully deprecate passing ts.ColumnType here?
        if isinstance(spec, (ts.ColumnType, type, _GenericAlias)):
            col_type = ts.ColumnType.normalize_type(spec, nullable_default=True, allow_builtin_types=False)
        elif isinstance(spec, exprs.Expr):
            # create copy so we can modify it
            value_expr = spec.copy()
            value_expr.bind_rel_paths()
        elif isinstance(spec, dict):
            cls._validate_column_spec(name, spec)
            if 'type' in spec:
                col_type = ts.ColumnType.normalize_type(spec['type'], nullable_default=True, allow_builtin_types=False)
            value_expr = spec.get('value')
            if value_expr is not None and isinstance(value_expr, exprs.Expr):
                # create copy so we can modify it
                value_expr = value_expr.copy()
                value_expr.bind_rel_paths()
            stored = spec.get('stored', True)
            primary_key = spec.get('primary_key', False)
            media_validation_str = spec.get('media_validation')
            media_validation = (
                catalog.MediaValidation[media_validation_str.upper()] if media_validation_str is not None else None
            )
            destination = spec.get('destination')
            custom_metadata = spec.get('custom_metadata')
            comment = spec.get('comment')
            if comment == '':
                comment = None
        else:
            raise excs.Error(f'Invalid spec for column {name!r}: {type(spec)}')

        stores_cellmd = stored and (
            value_expr is not None or col_type.is_media_type() or col_type.supports_file_offloading()
        )
        column = cls(
            name,
            col_type=col_type,
            computed_with=value_expr,
            stored=stored,
            is_pk=primary_key,
            media_validation=media_validation,
            destination=destination,
            custom_metadata=custom_metadata,
            comment=comment,
            stores_cellmd=stores_cellmd,
        )
        ObjectOps.validate_destination(column.destination, column.name)
        return column

    @classmethod
    def create_iterator_column(cls, name: str, col_type: ts.ColumnType, is_stored: bool) -> Column:
        stores_cellmd = is_stored and (col_type.is_media_type() or col_type.supports_file_offloading())
        return Column(name, col_type=col_type, is_iterator_col=True, stored=is_stored, stores_cellmd=stores_cellmd)

    @classmethod
    def _validate_column_spec(cls, name: str, spec: ColumnSpec) -> None:
        """Check integrity of user-supplied Column spec"""
        assert isinstance(spec, dict)
        # We cannot use get_type_hints() here since ColumnSpec doesn't import exprs outside of a TYPE_CHECKING block
        valid_keys = ColumnSpec.__annotations__.keys()
        for k in spec:
            if k not in valid_keys:
                raise excs.Error(f'Column {name!r}: invalid key {k!r}')

        if 'type' not in spec and 'value' not in spec:
            raise excs.Error(f"Column {name!r}: 'type' or 'value' must be specified")

        if 'type' in spec and not isinstance(spec['type'], (ts.ColumnType, type, _GenericAlias)):
            raise excs.Error(f"Column {name!r}: 'type' must be a type; got {spec['type']}")

        if 'value' in spec:
            value_expr = exprs.Expr.from_object(spec['value'])
            if value_expr is None:
                raise excs.Error(f"Column {name!r}: 'value' must be a Pixeltable expression.")
            if 'type' in spec:
                raise excs.Error(f"Column {name!r}: 'type' is redundant if 'value' is specified")

        if 'media_validation' in spec:
            _ = catalog.MediaValidation.validated(spec['media_validation'], f'Column {name!r}: media_validation')

        if 'stored' in spec and not isinstance(spec['stored'], bool):
            raise excs.Error(f"Column {name!r}: 'stored' must be a bool; got {spec['stored']}")

        if 'comment' in spec and not isinstance(spec['comment'], str):
            raise excs.Error(f"Column {name!r}: 'comment' must be a string; got {spec['comment']}")

        d = spec.get('destination')
        if d is not None and not isinstance(d, (str, Path)):
            raise excs.Error(f'Column {name!r}: `destination` must be a string or path; got {d}')

        if 'custom_metadata' in spec:
            # we require custom_metadata to be JSON-serializable
            try:
                json.dumps(spec['custom_metadata'])
            except (TypeError, ValueError) as err:
                raise excs.Error(
                    f'Column {name!r}: `custom_metadata` must be JSON-serializable; got Error: {err}'
                ) from err

    @classmethod
    def create_stored_proxy_column(cls, col: Column) -> Column:
        """Creates a proxy column for the specified column."""
        from pixeltable import exprs

        assert col.col_type.is_media_type() and not (col.is_stored and col.is_computed)
        proxy_col = cls(
            name=None,
            # Force images in the proxy column to be materialized inside the media store, in a normalized format.
            # TODO(aaron-siegel): This is a temporary solution and it will be replaced by a proper `destination`
            #   parameter for computed columns. Among other things, this solution does not work for video or audio.
            #   Once `destination` is implemented, it can be replaced with a simple `ColumnRef`.
            computed_with=exprs.ColumnRef(col).apply(lambda x: x, col_type=col.col_type),
            stored=True,
            stores_cellmd=True,
        )
        return proxy_col

    @classmethod
    def validate_name(cls, name: str) -> None:
        """Check that a name is usable as a pixeltable column name"""
        if is_system_column_name(name) or is_python_keyword(name):
            raise excs.Error(f'{name!r} is a reserved name in Pixeltable; please choose a different column name.')
        if not is_valid_identifier(name):
            raise excs.Error(f'Invalid column name: {name}')

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
            stores_cellmd=self.stores_cellmd,
            destination=self._explicit_destination,
        )
        if pos is None:
            return col_md, None
        assert self.name is not None, 'Column name must be set for user-facing columns'
        sch_md = schema.SchemaColumn(
            name=self.name,
            pos=pos,
            media_validation=self._media_validation.name.lower() if self._media_validation is not None else None,
            custom_metadata=self._custom_metadata,
            comment=self._comment,
        )
        return col_md, sch_md

    def verify(self) -> None:
        """Self-validation of a user column.

        Verifies the column name and the combination of its user-configured properties.

        Does nothing if it's a non-user column, i.e. a column without a name."""
        # TODO this verification should be done before or during the Column construction, not after it
        if self.name is None:
            # Non-user columns are created internally by Pixeltable, thus need no validation
            return
        Column.validate_name(self.name)
        if self.stored is False and not self.is_computed:
            raise excs.Error(f'Column {self.name!r}: `stored={self.stored}` only applies to computed columns')
        if self.stored is False and self.has_window_fn_call():
            raise excs.Error(
                f'Column {self.name!r}: `stored={self.stored}` is not valid for image columns computed with a'
                f' streaming function'
            )
        if self._explicit_destination is not None and not (self.stored and self.is_computed):
            raise excs.Error(f'Column {self.name!r}: `destination` property only applies to stored computed columns')

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
    def custom_metadata(self) -> Any:
        return self._custom_metadata

    @property
    def comment(self) -> str | None:
        return self._comment

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
