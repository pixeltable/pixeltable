from __future__ import annotations

import enum
import logging
import time
from typing import TYPE_CHECKING, Any, Literal, Sequence, cast, overload
from uuid import UUID

import pydantic
import pydantic_core

import pixeltable as pxt
from pixeltable import exceptions as excs, type_system as ts
from pixeltable.env import Env
from pixeltable.utils.filecache import FileCache
from pixeltable.utils.pydantic import is_json_convertible

from .globals import MediaValidation
from .table import Table
from .table_version import TableVersion, TableVersionMd
from .table_version_handle import TableVersionHandle
from .table_version_path import TableVersionPath
from .tbl_ops import CreateStoreTableOp, TableOp
from .update_status import UpdateStatus

if TYPE_CHECKING:
    from pixeltable import exprs
    from pixeltable.globals import TableDataSource
    from pixeltable.io.table_data_conduit import TableDataConduit

_logger = logging.getLogger('pixeltable')


class OnErrorParameter(enum.Enum):
    """Supported values for the on_error parameter"""

    ABORT = 'abort'
    IGNORE = 'ignore'

    @classmethod
    def is_valid(cls, v: Any) -> bool:
        if isinstance(v, str):
            return v.lower() in [c.value for c in cls]
        return False

    @classmethod
    def fail_on_exception(cls, v: Any) -> bool:
        if not cls.is_valid(v):
            raise ValueError(f'Invalid value for on_error: {v}')
        if isinstance(v, str):
            return v.lower() != cls.IGNORE.value
        return True


class InsertableTable(Table):
    """A `Table` that allows inserting and deleting rows."""

    def __init__(self, dir_id: UUID, tbl_version: TableVersionHandle):
        tbl_version_path = TableVersionPath(tbl_version)
        super().__init__(tbl_version.id, dir_id, tbl_version.get().name, tbl_version_path)
        self._tbl_version = tbl_version

    def _display_name(self) -> str:
        assert not self._tbl_version_path.is_replica()
        return 'table'

    @classmethod
    def _create(
        cls,
        name: str,
        schema: dict[str, ts.ColumnType],
        primary_key: list[str],
        num_retained_versions: int,
        comment: str,
        media_validation: MediaValidation,
        create_default_idxs: bool,
    ) -> tuple[TableVersionMd, list[TableOp]]:
        columns = cls._create_columns(schema)
        cls._verify_schema(columns)
        column_names = [col.name for col in columns]
        for pk_col in primary_key:
            if pk_col not in column_names:
                raise excs.Error(f'Primary key column {pk_col!r} not found in table schema.')
            col = columns[column_names.index(pk_col)]
            if col.col_type.nullable:
                raise excs.Error(
                    f'Primary key column {pk_col!r} cannot be nullable. '
                    f'Declare it as `Required` instead: `pxt.Required[pxt.{col.col_type._to_base_str()}]`'
                )
            col.is_pk = True

        md = TableVersion.create_initial_md(
            name,
            columns,
            num_retained_versions,
            comment,
            media_validation,
            create_default_idxs=create_default_idxs,
            view_md=None,
        )

        ops = [
            TableOp(
                tbl_id=md.tbl_md.tbl_id,
                op_sn=0,
                num_ops=1,
                needs_xact=False,
                create_store_table_op=CreateStoreTableOp(),
            )
        ]
        return md, ops

    @overload
    def insert(
        self,
        source: TableDataSource | None = None,
        /,
        *,
        source_format: Literal['csv', 'excel', 'parquet', 'json'] | None = None,
        schema_overrides: dict[str, ts.ColumnType] | None = None,
        on_error: Literal['abort', 'ignore'] = 'abort',
        print_stats: bool = False,
        **kwargs: Any,
    ) -> UpdateStatus: ...

    @overload
    def insert(
        self, /, *, on_error: Literal['abort', 'ignore'] = 'abort', print_stats: bool = False, **kwargs: Any
    ) -> UpdateStatus: ...

    def insert(
        self,
        source: TableDataSource | None = None,
        /,
        *,
        source_format: Literal['csv', 'excel', 'parquet', 'json'] | None = None,
        schema_overrides: dict[str, ts.ColumnType] | None = None,
        on_error: Literal['abort', 'ignore'] = 'abort',
        print_stats: bool = False,
        **kwargs: Any,
    ) -> UpdateStatus:
        from pixeltable.catalog import Catalog
        from pixeltable.io.table_data_conduit import UnkTableDataConduit

        if source is not None and isinstance(source, Sequence) and len(source) == 0:
            raise excs.Error('Cannot insert an empty sequence.')
        fail_on_exception = OnErrorParameter.fail_on_exception(on_error)

        with Catalog.get().begin_xact(tbl=self._tbl_version_path, for_write=True, lock_mutable_tree=True):
            table = self
            start_ts = time.monotonic()

            # TODO: unify with TableDataConduit
            if source is not None and isinstance(source, Sequence) and isinstance(source[0], pydantic.BaseModel):
                status = self._insert_pydantic(
                    cast(Sequence[pydantic.BaseModel], source),  # needed for mypy
                    print_stats=print_stats,
                    fail_on_exception=fail_on_exception,
                )
                Env.get().console_logger.info(status.insert_msg(start_ts))
                FileCache.get().emit_eviction_warnings()
                return status

            if source is None:
                source = [kwargs]
                kwargs = None

            tds = UnkTableDataConduit(
                source, source_format=source_format, src_schema_overrides=schema_overrides, extra_fields=kwargs
            )
            data_source = tds.specialize()
            if data_source.source_column_map is None:
                data_source.src_pk = []

            assert isinstance(table, Table)
            data_source.add_table_info(table)
            data_source.prepare_for_insert_into_table()

            return table.insert_table_data_source(
                data_source=data_source, fail_on_exception=fail_on_exception, print_stats=print_stats
            )

    def insert_table_data_source(
        self, data_source: TableDataConduit, fail_on_exception: bool, print_stats: bool = False
    ) -> pxt.UpdateStatus:
        """Insert row batches into this table from a `TableDataConduit`."""
        from pixeltable.catalog import Catalog
        from pixeltable.io.table_data_conduit import QueryTableDataConduit

        with Catalog.get().begin_xact(tbl=self._tbl_version_path, for_write=True, lock_mutable_tree=True):
            start_ts = time.perf_counter()
            if isinstance(data_source, QueryTableDataConduit):
                status = pxt.UpdateStatus()
                status += self._tbl_version.get().insert(
                    rows=None, query=data_source.pxt_query, print_stats=print_stats, fail_on_exception=fail_on_exception
                )
            else:
                status = pxt.UpdateStatus()
                for row_batch in data_source.valid_row_batch():
                    status += self._tbl_version.get().insert(
                        rows=row_batch, query=None, print_stats=print_stats, fail_on_exception=fail_on_exception
                    )

        Env.get().console_logger.info(status.insert_msg(start_ts))

        FileCache.get().emit_eviction_warnings()
        return status

    def _insert_pydantic(
        self, rows: Sequence[pydantic.BaseModel], print_stats: bool = False, fail_on_exception: bool = True
    ) -> UpdateStatus:
        model_class = type(rows[0])
        self._validate_pydantic_model(model_class)
        # convert rows one-by-one in order to be able to print meaningful error messages
        pxt_rows: list[dict[str, Any]] = []
        for i, row in enumerate(rows):
            try:
                pxt_rows.append(row.model_dump(mode='json'))
            except pydantic_core.PydanticSerializationError as e:
                raise excs.Error(f'Row {i}: error serializing pydantic model to JSON:\n{e}') from e

        # explicitly check that all required columns are present and non-None in the rows,
        # because we ignore nullability when validating the pydantic model
        reqd_col_names = [col.name for col in self._tbl_version_path.columns() if col.is_required_for_insert]
        for i, pxt_row in enumerate(pxt_rows):
            if type(rows[i]) is not model_class:
                raise excs.Error(
                    f'Expected an instance of `{model_class.__name__}`; got `{type(rows[i]).__name__}` (in row {i})'
                )
            for col_name in reqd_col_names:
                if pxt_row.get(col_name) is None:
                    raise excs.Error(f'Missing required column {col_name!r} in row {i}')

        status = self._tbl_version.get().insert(
            rows=pxt_rows, query=None, print_stats=print_stats, fail_on_exception=fail_on_exception
        )
        return status

    def _validate_pydantic_model(self, model: type[pydantic.BaseModel]) -> None:
        """
        Check if a Pydantic model is compatible with this table for insert operations.

        A model is compatible if:
        - All required table columns have corresponding model fields with compatible types
        - Model does not define fields for computed columns
        - Model field types are compatible with table column types
        """
        assert isinstance(model, type) and issubclass(model, pydantic.BaseModel)

        schema = self._get_schema()
        required_cols = set(self._tbl_version.get().get_required_col_names())
        computed_cols = set(self._tbl_version.get().get_computed_col_names())
        model_fields = model.model_fields
        model_field_names = set(model_fields.keys())

        missing_required = required_cols - model_field_names
        if missing_required:
            raise excs.Error(
                f'Pydantic model `{model.__name__}` is missing required columns: ' + ', '.join(missing_required)
            )

        computed_in_model = computed_cols & model_field_names
        if computed_in_model:
            raise excs.Error(
                f'Pydantic model `{model.__name__}` has fields for computed columns: ' + ', '.join(computed_in_model)
            )

        # validate type compatibility
        common_fields = model_field_names & set(schema.keys())
        if len(common_fields) == 0:
            raise excs.Error(
                f'Pydantic model `{model.__name__}` has no fields that map to columns in table {self._name!r}'
            )
        for field_name in common_fields:
            pxt_col_type = schema[field_name]
            model_field = model_fields[field_name]
            model_type = model_field.annotation

            # we ignore nullability: we want to accept optional model fields for required table columns, as long as
            # the model instances provide a non-null value
            # allow_enum=True: model_dump(mode='json') converts enums to their values
            inferred_pxt_type = ts.ColumnType.from_python_type(model_type, infer_pydantic_json=True)
            if inferred_pxt_type is None:
                raise excs.Error(
                    f'Pydantic model `{model.__name__}`: cannot infer Pixeltable type for column {field_name!r}'
                )

            if pxt_col_type.is_media_type():
                # media types require file paths, either as str or Path
                if not inferred_pxt_type.is_string_type():
                    raise excs.Error(
                        f'Column {field_name!r} requires a `str` or `Path` field in `{model.__name__}`, but it is '
                        f'`{model_type.__name__}`'
                    )
            else:
                if not pxt_col_type.is_supertype_of(inferred_pxt_type, ignore_nullable=True):
                    raise excs.Error(
                        f'Pydantic model `{model.__name__}` has incompatible type `{model_type.__name__}` '
                        f'for column {field_name!r} (of Pixeltable type `{pxt_col_type}`)'
                    )

                if (
                    isinstance(model_type, type)
                    and issubclass(model_type, pydantic.BaseModel)
                    and not is_json_convertible(model_type)
                ):
                    raise excs.Error(
                        f'Pydantic model `{model.__name__}` has field {field_name!r} with nested model '
                        f'`{model_type.__name__}`, which is not JSON-convertible'
                    )

    def delete(self, where: 'exprs.Expr' | None = None) -> UpdateStatus:
        """Delete rows in this table.

        Args:
            where: a predicate to filter rows to delete.

        Examples:
            Delete all rows in a table:

            >>> tbl.delete()

            Delete all rows in a table where column `a` is greater than 5:

            >>> tbl.delete(tbl.a > 5)
        """
        from pixeltable.catalog import Catalog

        with Catalog.get().begin_xact(tbl=self._tbl_version_path, for_write=True, lock_mutable_tree=True):
            return self._tbl_version.get().delete(where=where)

    def _get_base_table(self) -> 'Table' | None:
        return None

    @property
    def _effective_base_versions(self) -> list[int | None]:
        return []

    def _table_descriptor(self) -> str:
        return self._display_str()
