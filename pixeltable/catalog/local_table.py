from __future__ import annotations

import abc
import builtins
import datetime
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping
from uuid import UUID

import pandas as pd
from typing_extensions import overload

import pixeltable as pxt
from pixeltable import exceptions as excs, exprs, index, type_system as ts
from pixeltable.catalog.table_metadata import (
    ColumnMetadata,
    EmbeddingIndexParams,
    IndexMetadata,
    TableMetadata,
    VersionMetadata,
)
from pixeltable.metadata.utils import MetadataUtils
from pixeltable.runtime import get_runtime
from pixeltable.types import ColumnSpec
from pixeltable.utils.formatter import Formatter

from ..exprs import ColumnRef
from ..utils.description_helper import DescriptionHelper
from ..utils.filecache import FileCache
from .column import Column
from .globals import (
    _ROWID_COLUMN_NAME,
    IfExistsParam,
    IfNotExistsParam,
    MediaValidation,
    QColumnId,
    is_valid_identifier,
)
from .table import Table
from .table_path import TableVersionPath
from .table_version_handle import TableVersionHandle
from .update_status import UpdateStatus

from typing import _GenericAlias  # type: ignore[attr-defined]  # isort: skip


if TYPE_CHECKING:
    import torch.utils.data

    import pixeltable.plan
    from pixeltable.globals import TableDataSource

    from .table_version import TableVersion


_logger = logging.getLogger(__name__)


class LocalTable(Table):
    """
    A handle to a local table, view, or snapshot, executed directly against the local catalog.

    Implements the Table public API for tables that live in the local Postgres.

    Thread-safe.
    """

    # Every user-invoked operation that runs an ExecNode tree (directly or indirectly) needs to call
    # FileCache.emit_eviction_warnings() at the end of the operation.

    # the chain of TableVersions needed to run queries and supply metadata (eg, schema)
    _tbl_version_path: TableVersionPath

    # the physical TableVersion backing this Table; None for pure snapshots
    _tbl_version: TableVersionHandle | None

    def __init__(self, id: UUID, tbl_version_path: TableVersionPath):
        super().__init__(id)
        self._tbl_version_path = tbl_version_path
        self._tbl_version = None

    @property
    def _tbl_path(self) -> TableVersionPath:
        return self._tbl_version_path

    def __deepcopy__(self, memo: dict[int, Any]) -> 'Table':
        return self

    def _name(self) -> str:
        cat = get_runtime().catalog
        with cat.begin_xact(for_write=False):
            return cat.read_tbl_record(self._id).md['name']

    def _dir_id(self) -> UUID | None:
        cat = get_runtime().catalog
        with cat.begin_xact(for_write=False):
            return cat.read_tbl_record(self._id).dir_id

    def get_metadata(self) -> 'TableMetadata':
        from pixeltable.catalog import retry_loop

        @retry_loop(for_write=False)
        def op() -> 'TableMetadata':
            return self._get_metadata()

        return op()

    def _get_metadata(self) -> TableMetadata:
        tvp = self._tbl_version_path
        tv = tvp.tbl_version.get()
        columns = tvp.columns()
        column_info: dict[str, ColumnMetadata] = {}
        for col in columns:
            dependencies: list[tuple[str, str]] | None = None
            if col.is_computed:
                value_expr = col.value_expr
                assert value_expr is not None
                dependencies = sorted(
                    {
                        (col_ref.col.tbl_handle.get().name, col_ref.col.name)
                        for col_ref in value_expr.subexprs(expr_class=exprs.ColumnRef, traverse_matches=False)
                    }
                )
            column_info[col.name] = ColumnMetadata(
                name=col.name,
                type_=col.col_type._to_str(as_schema=True),
                version_added=col.schema_version_add,
                is_stored=col.is_stored,
                is_primary_key=col.is_pk,
                media_validation=col.media_validation.name.lower() if col.media_validation is not None else None,  # type: ignore[typeddict-item]
                is_computed=col.is_computed,
                computed_with=col.value_expr.display_str(inline=False) if col.value_expr is not None else None,
                is_builtin=(not col.calls_custom_udf) if col.value_expr is not None else None,
                depends_on=dependencies,
                defined_in=col.get_tbl().name,
                comment=col.comment,
                custom_metadata=col.custom_metadata,
                is_iterator_col=False,
                destination=col._explicit_destination,
            )

        indices = tv.idxs_by_name.values()
        index_info: dict[str, IndexMetadata] = {}
        for info in indices:
            # Only surface indexes whose underlying column is user-visible.
            if info.col.name not in column_info:
                continue
            if isinstance(info.idx, index.EmbeddingIndex):
                indexed_col_md = self._tbl_version_path.get_column_md(QColumnId(info.col.tbl_handle.id, info.col.id))
                col_ref = ColumnRef(indexed_col_md)
                embedding_fncall = info.idx.embeddings[info.col.col_type._type](col_ref)
                index_info[info.name] = IndexMetadata(
                    name=info.name,
                    columns=[info.col.name],
                    index_type='embedding',
                    parameters=EmbeddingIndexParams(
                        metric=info.idx.metric.name.lower(),  # type: ignore[typeddict-item]
                        embedding=str(embedding_fncall),
                        embedding_functions=[str(fn) for fn in info.idx.embeddings.values()],
                    ),
                )
            elif isinstance(info.idx, index.BtreeIndex):
                index_info[info.name] = IndexMetadata(
                    name=info.name, columns=[info.col.name], index_type='btree', parameters=None
                )

        primary_key: list[str] | None = None
        if any(col.is_pk for col in columns):
            primary_key = [col.name for col in columns if col.is_pk]

        return TableMetadata(
            id=self._id,
            name=self._name(),
            path=str(self._path()),
            columns=column_info,
            indices=index_info,
            is_versioned=tv.is_versioned,
            is_view=False,
            is_snapshot=False,
            version=self._get_version(),
            version_created=datetime.datetime.fromtimestamp(tv.created_at, tz=datetime.timezone.utc),
            schema_version=tvp.schema_version(),
            comment=self._get_comment(),
            custom_metadata=self._get_custom_metadata(),
            media_validation=self._get_media_validation().name.lower(),  # type: ignore[typeddict-item]
            primary_key=primary_key,
            kind=self._display_name(),  # type: ignore[typeddict-item]
            base=None,
            view_filter=None,
            view_sample=None,
            iterator_call=None,
        )

    def _get_version(self) -> int | None:
        """Return the version of this table or None if not versioned. Used by tests to ascertain version changes."""
        return self._tbl_version_path.version()

    def __hash__(self) -> int:
        return hash(self._tbl_version_path.tbl_id)

    def __getattr__(self, name: str) -> 'exprs.ColumnRef':
        col_md = self._tbl_version_path.get_column_md_by_name(name)
        if col_md is None:
            raise AttributeError(f'Unknown column: {name}')
        return ColumnRef(col_md, self._tbl_version_path.is_validate_on_read(col_md))

    def __getitem__(self, name: str) -> 'exprs.ColumnRef':
        return getattr(self, name)

    def list_views(self, *, recursive: bool = True) -> list[str]:
        from pixeltable.catalog import retry_loop

        # we need retry_loop() here, because we end up loading Tables for the views
        @retry_loop(read_tvps=[self._tbl_version_path])
        def op() -> list[str]:
            paths: list[str] = []
            for t in self._get_views(recursive=recursive):
                try:
                    paths.append(str(t._path()))
                except excs.NotFoundError as e:
                    # view was dropped concurrently between enumeration and _path() call; skip it
                    if not excs.is_table_not_found_error(e):
                        raise
            return paths

        return op()

    def _get_views(self, *, recursive: bool = True, mutable_only: bool = False) -> list['LocalTable']:
        cat = get_runtime().catalog
        view_ids = cat.get_view_ids(self._id)
        views = [t for id in view_ids if (t := cat.get_table_by_id(id, ignore_if_dropped=True)) is not None]
        if mutable_only:
            views = [t for t in views if t._tbl_version_path.is_mutable()]
        if recursive:
            views.extend(t for view in views for t in view._get_views(recursive=True, mutable_only=mutable_only))
        return views

    def columns(self) -> list[str]:
        cols = self._tbl_version_path.columns()
        return [c.name for c in cols]

    def _get_base_tables(self) -> list['Table']:
        """The ancestor list of bases of this table, starting with its immediate base. Requires a transaction context"""
        bases: list[Table] = []
        base = self._get_base_table()
        while base is not None:
            bases.append(base)
            base = base._get_base_table()
        return bases

    @property
    @abc.abstractmethod
    def _effective_base_versions(self) -> list[int | None]:
        """The effective versions of the ancestor bases, starting with its immediate base."""

    def _is_versioned(self) -> bool:
        return self._tbl_version_path.is_versioned()

    def _get_comment(self) -> str:
        return self._tbl_version_path.comment()

    def _get_custom_metadata(self) -> Any:
        return self._tbl_version_path.custom_metadata()

    def _get_media_validation(self) -> MediaValidation:
        return self._tbl_version_path.media_validation()

    def __repr__(self) -> str:
        return self._descriptors().to_string()

    def _repr_html_(self) -> str:
        return self._descriptors().to_html()

    def _descriptors(self, path: 'pxt.catalog.Path | None' = None) -> DescriptionHelper:
        """
        Constructs a list of descriptors for this table that can be pretty-printed.

        path overrides the displayed path in the title (used by delegated execution to show the full
        pxt:// path of a hosted table); when None the local in-catalog path is shown.
        """

        with get_runtime().catalog.begin_xact(read_tvps=[self._tbl_version_path]):
            helper = DescriptionHelper()
            helper.append(self._table_descriptor(path))
            col_df, separator_idxs = self._col_descriptor()
            helper.append(col_df, separator_idxs=separator_idxs)
            idxs = self._index_descriptor()
            if not idxs.empty:
                helper.append(idxs)
            if self._get_comment():
                helper.append(f'Comment: {self._get_comment()}')
            if self._get_custom_metadata():
                helper.append(f'Custom Metadata: {Formatter.summarize_json(self._get_custom_metadata())}')
            return helper

    def _col_descriptor(self, columns: list[str] | None = None) -> tuple[pd.DataFrame, list[int] | None]:
        """Generates column descriptor DataFrame and a list of vertical separators.

        The DataFrame contains the following columns, in addition to Column Name and Type:
        - Source: the table from which the column is inherited, or this table's name if the column originates here.
        - Computed With: The expression that Pixeltable evaluates to fill in this column's values. This could be a
          Python expression, a UDF call, or an iterator name. Blank if the data in the row is not computed.

        The separators are used to visually group columns by their Source when the table description is rendered.

        Args:
            columns: List of columns to include, or all columns if None.

        Returns:
            A tuple of the column descriptor DataFrame, and a list of row indexes after which a vertical separator
            should be placed.
        """
        cols = [col for col in self._tbl_version_path.columns() if columns is None or col.name in columns]
        col_descriptors: list[dict[str, str]] = []
        separator_idxs: list[int] = []
        prev_source: str | None = None
        for i, col in enumerate(cols):
            computed_with = col.value_expr.display_str(inline=False) if col.value_expr is not None else ''
            source_tv = col.get_tbl()
            if source_tv.is_iterator_column(col):
                assert source_tv.iterator_call is not None
                computed_with = source_tv.iterator_call.it.name

            col_descriptors.append(
                {
                    'Column Name': col.name,
                    'Type': col.col_type._to_str(as_schema=True),
                    'Source': source_tv.name,
                    'Computed With': computed_with,
                    'Comment': col.comment if col.comment is not None else '',
                }
            )
            # Insert a separator if this column's source is different from the last one.
            if prev_source is not None and source_tv.name != prev_source:
                separator_idxs.append(i - 1)
            prev_source = source_tv.name
        return pd.DataFrame(col_descriptors), separator_idxs

    def _index_descriptor(self, columns: list[str] | None = None) -> pd.DataFrame:
        from pixeltable import index

        if self._tbl_version is None:
            return pd.DataFrame([])
        pd_rows = []
        for name, info in self._tbl_version.get().idxs_by_name.items():
            if isinstance(info.idx, index.EmbeddingIndex) and (columns is None or info.col.name in columns):
                col_md = self._tbl_version_path.get_column_md(QColumnId(info.col.tbl_handle.id, info.col.id))
                col_ref = ColumnRef(col_md)
                embedding = info.idx.embeddings[info.col.col_type._type](col_ref)
                row = {
                    'Index Name': name,
                    'Column': info.col.name,
                    'Metric': str(info.idx.metric.name.lower()),
                    'Embedding': str(embedding),
                }
                pd_rows.append(row)
        return pd.DataFrame(pd_rows)

    def describe(self) -> None:
        if getattr(builtins, '__IPYTHON__', False):
            from IPython.display import Markdown, display

            display(Markdown(self._repr_html_()))
        else:
            print(repr(self))

    # TODO Factor this out into a separate module.
    # The return type is unresolvable, but torch can't be imported since it's an optional dependency.
    def to_pytorch_dataset(self, image_format: str = 'pt') -> 'torch.utils.data.IterableDataset':
        return self.select().to_pytorch_dataset(image_format=image_format)

    def to_coco_dataset(self) -> Path:
        return self.select().to_coco_dataset()

    def _get_dependent_user_cols(self, col: Column) -> list[Column]:
        """Returns the named (user-visible) columns that depend on `col`."""
        cat = get_runtime().catalog
        return [c for c in cat.get_column_dependents(col.get_tbl().id, col.id) if c.name is not None]

    def _ignore_or_drop_existing_columns(self, new_col_names: list[str], if_exists: IfExistsParam) -> list[str]:
        """Check and handle existing columns in the new column specification based on the if_exists parameter.

        If `if_exists='ignore'`, returns a list of existing columns, if any, in `new_col_names`.
        """
        assert self._tbl_version is not None
        existing_col_names = set(self._get_schema().keys())
        cols_to_ignore = []
        for new_col_name in new_col_names:
            if new_col_name in existing_col_names:
                if if_exists == IfExistsParam.ERROR:
                    raise excs.AlreadyExistsError(
                        excs.ErrorCode.COLUMN_ALREADY_EXISTS, f'Duplicate column name: {new_col_name}'
                    )
                elif if_exists == IfExistsParam.IGNORE:
                    cols_to_ignore.append(new_col_name)
                elif if_exists in (IfExistsParam.REPLACE, IfExistsParam.REPLACE_FORCE):
                    if new_col_name not in self._tbl_version.get().cols_by_name:
                        # for views, it is possible that the existing column
                        # is a base table column; in that case, we should not
                        # drop/replace that column. Continue to raise error.
                        raise excs.RequestError(
                            excs.ErrorCode.UNSUPPORTED_OPERATION,
                            f'Column {new_col_name!r} is a base table column. Cannot replace it.',
                        )
                    col = self._tbl_version.get().cols_by_name[new_col_name]
                    # cannot drop a column with dependents; so reject
                    # replace directive if column has dependents.
                    if len(self._get_dependent_user_cols(col)) > 0:
                        raise excs.AlreadyExistsError(
                            excs.ErrorCode.COLUMN_ALREADY_EXISTS,
                            f'Column {new_col_name!r} already exists and has dependents. '
                            f'Cannot {if_exists.name.lower()} it.',
                        )
                    self.drop_column(new_col_name)
                    assert new_col_name not in self._tbl_version.get().cols_by_name
        return cols_to_ignore

    def add_columns(
        self,
        schema: Mapping[str, type | ColumnSpec],
        if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error',
    ) -> UpdateStatus:
        from pixeltable.catalog import retry_loop

        self._validate_column_schema(schema)

        # a retry loop is necessary because drop column needs it
        # lock_mutable_tree=True: we might end up having to drop existing columns, which requires locking the tree
        @retry_loop(for_write=True, write_tvps=[self._tbl_version_path], lock_mutable_tree=True)
        def do_add_columns() -> list[Column] | None:
            self._check_mutable('add columns to')

            # make a copy of schema so del operations below don't modify the caller's dict
            schema_copy = dict(schema)

            # handle existing columns based on if_exists parameter
            cols_to_ignore = self._ignore_or_drop_existing_columns(
                list(schema_copy.keys()), IfExistsParam.validated(if_exists, 'if_exists')
            )
            # if all columns to be added already exist and user asked to ignore
            # existing columns, there's nothing to do.
            for cname in cols_to_ignore:
                assert cname in schema_copy
                del schema_copy[cname]
            if len(schema_copy) == 0:
                return None
            new_cols = [Column.create(name, spec) for name, spec in schema_copy.items()]
            for new_col in new_cols:
                self._verify_column(new_col)
            return new_cols

        new_cols = do_add_columns()
        if new_cols is None:
            return UpdateStatus()

        assert self._tbl_version is not None
        get_runtime().catalog.add_columns(self._tbl_version_path, new_cols)
        FileCache.get().emit_eviction_warnings()
        # TODO: return the row count here?
        return UpdateStatus()

    def add_column(
        self,
        *,
        if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error',
        **kwargs: type | ColumnSpec,
    ) -> UpdateStatus:
        # verify kwargs and construct column schema dict
        self._check_single_column_kwarg('add_column', '`col_name=col_type`', kwargs)
        col_type = next(iter(kwargs.values()))
        if not isinstance(col_type, (ts.ColumnType, type, _GenericAlias, dict)):
            raise excs.RequestError(
                excs.ErrorCode.INVALID_ARGUMENT,
                'The argument to add_column() must be a type; did you intend to use add_computed_column() instead?',
            )
        return self.add_columns(kwargs, if_exists=if_exists)

    def add_computed_column(
        self,
        *,
        stored: bool | None = None,
        destination: str | Path | None = None,
        custom_metadata: Any = None,
        comment: str = '',
        print_stats: bool = False,
        on_error: Literal['abort', 'ignore'] = 'abort',
        if_exists: Literal['error', 'ignore', 'replace'] = 'error',
        **kwargs: exprs.Expr,
    ) -> UpdateStatus:
        from pixeltable.catalog import retry_loop

        # a retry loop is necessary because drop column needs it.
        @retry_loop(for_write=True, write_tvps=[self._tbl_version_path], lock_mutable_tree=True)
        def do_add_computed_column() -> UpdateStatus:
            self._check_mutable('add columns to')
            self._check_single_column_kwarg(
                'add_computed_column', '`col_name=col_type` or `col_name=expression`', kwargs
            )
            col_name, spec = next(iter(kwargs.items()))
            if not is_valid_identifier(col_name):
                raise excs.RequestError(excs.ErrorCode.INVALID_COLUMN_NAME, f'Invalid column name: {col_name}')

            col_schema: ColumnSpec = {'value': spec}
            if stored is not None:
                col_schema['stored'] = stored

            col_schema['destination'] = destination
            col_schema['custom_metadata'] = custom_metadata
            col_schema['comment'] = comment

            # Raise an error if the column expression refers to a column error property
            if isinstance(spec, exprs.Expr):
                for e in spec.subexprs(expr_class=exprs.ColumnPropertyRef, traverse_matches=False):
                    if e.is_cellmd_prop():
                        raise excs.RequestError(
                            excs.ErrorCode.UNSUPPORTED_OPERATION,
                            f'Use of a reference to the {e.prop.name.lower()!r} property of another column '
                            f'is not allowed in a computed column.',
                        )

            # handle existing columns based on if_exists parameter
            cols_to_ignore = self._ignore_or_drop_existing_columns(
                [col_name], IfExistsParam.validated(if_exists, 'if_exists')
            )
            # if the column to add already exists and user asked to ignore
            # existing column, there's nothing to do.
            result = UpdateStatus()
            if len(cols_to_ignore) != 0:
                assert cols_to_ignore[0] == col_name
                return result

            new_col = Column.create(col_name, col_schema)
            self._verify_column(new_col)
            assert self._tbl_version is not None
            result += self._tbl_version.get().add_columns([new_col], print_stats=print_stats, on_error=on_error)
            FileCache.get().emit_eviction_warnings()
            return result

        return do_add_computed_column()

    @classmethod
    def _verify_column(cls, col: Column) -> None:
        """Check integrity of user-supplied Column and supply defaults"""
        col.verify()

    @classmethod
    def _verify_schema(cls, schema: list[Column]) -> None:
        """Check integrity of user-supplied schema and set defaults"""
        for col in schema:
            cls._verify_column(col)

    def drop_column(self, column: str | ColumnRef, if_not_exists: Literal['error', 'ignore'] = 'error') -> None:
        from pixeltable.catalog import retry_loop

        # Retry loop is necessary because table metadata is loaded inside.
        # Note: the provided ColumnRef may belong to a different table.
        # lock_mutable_tree=True: we need to be able to see whether any transitive view has column dependents
        @retry_loop(for_write=True, write_tvps=[self._tbl_version_path], lock_mutable_tree=True)
        def do_drop_column() -> None:
            self._check_mutable('drop columns from')
            col: Column = None
            if_not_exists_ = IfNotExistsParam.validated(if_not_exists, 'if_not_exists')

            if isinstance(column, str):
                col = self._tbl_version_path.get_column(column)
                if col is None:
                    if if_not_exists_ == IfNotExistsParam.ERROR:
                        raise excs.NotFoundError(excs.ErrorCode.COLUMN_NOT_FOUND, f'Unknown column: {column}')
                    assert if_not_exists_ == IfNotExistsParam.IGNORE
                    return
                if col.get_tbl().id != self._tbl_version_path.tbl_id:
                    raise excs.RequestError(
                        excs.ErrorCode.UNSUPPORTED_OPERATION, f'Cannot drop base table column {col.name!r}'
                    )
                col = self._tbl_version.get().cols_by_name[column]
            else:
                exists = self._tbl_version_path.has_column(column.col_md.qcolid)
                if not exists:
                    if if_not_exists_ == IfNotExistsParam.ERROR:
                        raise excs.NotFoundError(
                            excs.ErrorCode.COLUMN_NOT_FOUND, f'Unknown column: {column.col.qualified_name}'
                        )
                    assert if_not_exists_ == IfNotExistsParam.IGNORE
                    return
                col = column.col
                if col.get_tbl().id != self._tbl_version_path.tbl_id:
                    raise excs.RequestError(
                        excs.ErrorCode.UNSUPPORTED_OPERATION, f'Cannot drop base table column {col.name!r}'
                    )

            dependent_user_cols = self._get_dependent_user_cols(col)
            if len(dependent_user_cols) > 0:
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'Cannot drop column {col.name!r} because the following columns depend on it:\n'
                    f'{", ".join(c.name for c in dependent_user_cols)}',
                )

            views = self._get_views(recursive=True, mutable_only=True)

            # See if any view predicates depend on this column
            dependent_views: list[tuple[Table, exprs.Expr]] = []
            for view in views:
                if view._tbl_version is not None:
                    predicate = view._tbl_version.get().predicate
                    if predicate is not None:
                        for predicate_col in exprs.Expr.get_refd_column_ids(predicate.as_dict()):
                            if predicate_col.tbl_id == col.get_tbl().id and predicate_col.col_id == col.id:
                                dependent_views.append((view, predicate))

            if len(dependent_views) > 0:
                dependent_views_str = '\n'.join(
                    f'view: {view._path()}, predicate: {predicate}' for view, predicate in dependent_views
                )
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'Cannot drop column {col.name!r} because the following views depend on it:\n{dependent_views_str}',
                )

            all_columns = self.columns()
            if len(all_columns) == 1 and col.name == all_columns[0]:
                raise excs.RequestError(
                    excs.ErrorCode.MISSING_REQUIRED,
                    f'Cannot drop column {col.name!r} because it is the last remaining column in this table.'
                    f' Tables must have at least one column.',
                )

            self._tbl_version.get().drop_column(col)

        do_drop_column()

    def rename_column(self, old_name: str, new_name: str) -> None:
        with get_runtime().catalog.begin_xact(
            for_write=True, write_tvps=[self._tbl_version_path], lock_mutable_tree=False
        ):
            self._check_mutable('rename columns of')
            self._tbl_version.get().rename_column(old_name, new_name)

    def alter_column(self, column: str | ColumnRef, *, type_: type) -> None:
        from pixeltable.catalog import retry_loop

        new_col_type = ts.ColumnType.normalize_type(type_, nullable_default=True, allow_builtin_types=False)

        @retry_loop(for_write=True, write_tvps=[self._tbl_version_path], lock_mutable_tree=True)
        def do_alter_column() -> None:
            self._check_mutable('alter columns of')

            if isinstance(column, str):
                col = self._tbl_version_path.get_column(column)
                if col is None:
                    raise excs.NotFoundError(excs.ErrorCode.COLUMN_NOT_FOUND, f'Unknown column: {column}')
            else:
                if not self._tbl_version_path.has_column(column.col_md.qcolid):
                    raise excs.NotFoundError(
                        excs.ErrorCode.COLUMN_NOT_FOUND, f'Unknown column: {column.col.qualified_name}'
                    )
                col = column.col
            if col.get_tbl().id != self._tbl_version_path.tbl_id:
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION, f'Cannot alter base table column {col.name!r}'
                )
            if col.is_computed:
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION, f'Cannot alter the type of computed column {col.name!r}'
                )
            if col.is_pk:
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION, f'Cannot alter the type of primary key column {col.name!r}'
                )

            # TODO(PXT-960): follow up: allow alteration if it doesn't invalidate any dependents, and doesn't change
            # their column types.
            dependent_user_cols = self._get_dependent_user_cols(col)
            if len(dependent_user_cols) > 0:
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'Cannot alter column {col.name!r} because the following columns depend on it: '
                    f'{", ".join(c.qualified_name for c in dependent_user_cols)}',
                )

            self._tbl_version.get().alter_column(col, new_col_type)

        do_alter_column()

    def add_embedding_index(
        self,
        column: str | ColumnRef,
        *,
        idx_name: str | None = None,
        embedding: pxt.Function | None = None,
        string_embed: pxt.Function | None = None,
        image_embed: pxt.Function | None = None,
        audio_embed: pxt.Function | None = None,
        video_embed: pxt.Function | None = None,
        document_embed: pxt.Function | None = None,
        metric: Literal['cosine', 'ip', 'l2'] = 'cosine',
        precision: Literal['fp16', 'fp32'] = 'fp16',
        if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error',
    ) -> None:
        self._validate_embedding_args(embedding, string_embed, image_embed)
        assert self._tbl_version is None or self._tbl_version.get().is_versioned, (
            'TODO: implement for unversioned tables [PXT-1101]'
        )

        with get_runtime().catalog.begin_xact(
            for_write=True, write_tvps=[self._tbl_version_path], lock_mutable_tree=True
        ):
            self._check_mutable('add an index to')
            col = self._resolve_column_parameter(column)

            # idx_name must be a valid pixeltable column name
            if idx_name is not None:
                Column.validate_name(idx_name)
                # Named index: duplicate detection is by name. Handle a name collision before constructing the new
                # index, so that if_exists='ignore' remains a true no-op and never surfaces validation errors.
                if idx_name in self._tbl_version.get().idxs_by_name:
                    if_exists_ = IfExistsParam.validated(if_exists, 'if_exists')
                    # An index with the same name already exists. Handle it according to if_exists.
                    if if_exists_ == IfExistsParam.ERROR:
                        raise excs.AlreadyExistsError(
                            excs.ErrorCode.INDEX_ALREADY_EXISTS, f'Duplicate index name: {idx_name}'
                        )
                    if not isinstance(self._tbl_version.get().idxs_by_name[idx_name].idx, index.EmbeddingIndex):
                        raise excs.RequestError(
                            excs.ErrorCode.UNSUPPORTED_OPERATION,
                            f'Index {idx_name!r} is not an embedding index. Cannot {if_exists_.name.lower()} it.',
                        )
                    if if_exists_ == IfExistsParam.IGNORE:
                        return
                    assert if_exists_ in (IfExistsParam.REPLACE, IfExistsParam.REPLACE_FORCE)
                    self.drop_index(idx_name=idx_name)
                    assert idx_name not in self._tbl_version.get().idxs_by_name

            from pixeltable.index import EmbeddingIndex

            # validate EmbeddingIndex args; the resulting index is also used for duplicate detection
            idx = EmbeddingIndex(
                metric=metric,
                precision=precision,
                embed=embedding,
                string_embed=string_embed,
                image_embed=image_embed,
                audio_embed=audio_embed,
                video_embed=video_embed,
                document_embed=document_embed,
                column=col,  # Pass column for shape validation
            )
            _ = idx.create_value_expr(col)  # validation only; result discarded

            if idx_name is None:
                # Unnamed index: duplicate detection is by index definition on this column.
                matches = self._find_matching_embedding_idxs(col, idx)
                if len(matches) > 0:
                    if_exists_ = IfExistsParam.validated(if_exists, 'if_exists')
                    if if_exists_ == IfExistsParam.ERROR:
                        raise excs.AlreadyExistsError(
                            excs.ErrorCode.INDEX_ALREADY_EXISTS,
                            f'An identical embedding index already exists on column {col.name!r} '
                            f'(index {matches[0].name!r}). Pass a distinct idx_name, '
                            f"or if_exists='ignore' / 'replace'.",
                        )
                    if if_exists_ == IfExistsParam.IGNORE:
                        return
                    assert if_exists_ in (IfExistsParam.REPLACE, IfExistsParam.REPLACE_FORCE)
                    for info in matches:
                        self.drop_index(idx_name=info.name)

            _ = self._tbl_version.get().add_index(col, idx_name=idx_name, idx=idx)
            # TODO: how to deal with exceptions here? drop the index and raise?
            FileCache.get().emit_eviction_warnings()

    def _find_matching_embedding_idxs(self, col: Column, idx: index.EmbeddingIndex) -> list[TableVersion.IndexInfo]:
        """Return existing embedding indices on col that are defined identically to idx."""
        # the serialized dict contains everything we care about
        target = idx.as_dict()
        return [
            info
            for info in self._tbl_version.get().idxs_by_col.get(col.qid, [])
            if isinstance(info.idx, index.EmbeddingIndex) and info.idx.as_dict() == target
        ]

    def drop_embedding_index(
        self,
        *,
        column: str | ColumnRef | None = None,
        idx_name: str | None = None,
        if_not_exists: Literal['error', 'ignore'] = 'error',
    ) -> None:
        if (column is None) == (idx_name is None):
            raise excs.RequestError(
                excs.ErrorCode.MISSING_REQUIRED, "Exactly one of 'column' or 'idx_name' must be provided"
            )

        with get_runtime().catalog.begin_xact(
            for_write=True, write_tvps=[self._tbl_version_path], lock_mutable_tree=True
        ):
            col: Column = None
            if idx_name is None:
                col = self._resolve_column_parameter(column)
                assert col is not None

            self._drop_index(col=col, idx_name=idx_name, _idx_class=index.EmbeddingIndex, if_not_exists=if_not_exists)

    def _resolve_column_parameter(self, column: str | ColumnRef) -> Column:
        """Resolve a column parameter to a Column object"""
        col: Column = None
        if isinstance(column, str):
            col = self._tbl_version_path.get_column(column)
            if col is None:
                raise excs.NotFoundError(excs.ErrorCode.COLUMN_NOT_FOUND, f'Unknown column: {column}')
        elif isinstance(column, ColumnRef):
            exists = self._tbl_version_path.has_column(column.col.qid)
            if not exists:
                raise excs.NotFoundError(
                    excs.ErrorCode.COLUMN_NOT_FOUND, f'Unknown column: {column.col.qualified_name}'
                )
            col = column.col
        else:
            raise excs.RequestError(excs.ErrorCode.TYPE_MISMATCH, f'Invalid column parameter type: {type(column)}')
        return col

    def drop_index(
        self,
        *,
        column: str | ColumnRef | None = None,
        idx_name: str | None = None,
        if_not_exists: Literal['error', 'ignore'] = 'error',
    ) -> None:
        if (column is None) == (idx_name is None):
            raise excs.RequestError(
                excs.ErrorCode.MISSING_REQUIRED, "Exactly one of 'column' or 'idx_name' must be provided"
            )

        with get_runtime().catalog.begin_xact(
            for_write=True, write_tvps=[self._tbl_version_path], lock_mutable_tree=True
        ):
            col: Column = None
            if idx_name is None:
                col = self._resolve_column_parameter(column)
                assert col is not None

            self._drop_index(col=col, idx_name=idx_name, if_not_exists=if_not_exists)

    def _drop_index(
        self,
        *,
        col: Column | None = None,
        idx_name: str | None = None,
        _idx_class: type[index.IndexBase] | None = None,
        if_not_exists: Literal['error', 'ignore'] = 'error',
    ) -> None:
        self._check_mutable('drop an index from')
        assert (col is None) != (idx_name is None)

        if idx_name is not None:
            if_not_exists_ = IfNotExistsParam.validated(if_not_exists, 'if_not_exists')
            if idx_name not in self._tbl_version.get().idxs_by_name:
                if if_not_exists_ == IfNotExistsParam.ERROR:
                    raise excs.NotFoundError(excs.ErrorCode.INDEX_NOT_FOUND, f'Index {idx_name!r} does not exist')
                assert if_not_exists_ == IfNotExistsParam.IGNORE
                return
            idx_info = self._tbl_version.get().idxs_by_name[idx_name]
        else:
            if col.get_tbl().id != self._tbl_version.id:
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'Column {col.name!r}: '
                    f'cannot drop index from column that belongs to base table {col.get_tbl().name!r}',
                )
            idx_info_list = [info for info in self._tbl_version.get().idxs_by_name.values() if info.col.id == col.id]
            if _idx_class is not None:
                idx_info_list = [info for info in idx_info_list if isinstance(info.idx, _idx_class)]
            if len(idx_info_list) == 0:
                if_not_exists_ = IfNotExistsParam.validated(if_not_exists, 'if_not_exists')
                if if_not_exists_ == IfNotExistsParam.ERROR:
                    raise excs.NotFoundError(
                        excs.ErrorCode.INDEX_NOT_FOUND, f'Column {col.name!r} does not have an index'
                    )
                assert if_not_exists_ == IfNotExistsParam.IGNORE
                return
            if len(idx_info_list) > 1:
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'Column {col.name!r} has multiple indices; specify `idx_name` explicitly to drop one',
                )
            idx_info = idx_info_list[0]

        # Find out if anything depends on this index
        dependent_user_cols = self._get_dependent_user_cols(idx_info.val_col)
        if len(dependent_user_cols) > 0:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                f'Cannot drop index {idx_info.name!r} because the following columns depend on it:\n'
                f'{", ".join(c.name for c in dependent_user_cols)}',
            )
        self._tbl_version.get().drop_index(idx_info.id)

    @overload
    def insert(
        self,
        source: TableDataSource,
        /,
        *,
        source_format: Literal['csv', 'excel', 'parquet', 'json'] | None = None,
        schema_overrides: dict[str, ts.ColumnType] | None = None,
        on_error: Literal['abort', 'ignore'] = 'abort',
        print_stats: bool = False,
        return_rows: bool = False,
        **kwargs: Any,
    ) -> UpdateStatus: ...

    @overload
    def insert(
        self,
        /,
        *,
        on_error: Literal['abort', 'ignore'] = 'abort',
        print_stats: bool = False,
        return_rows: bool = False,
        **kwargs: Any,
    ) -> UpdateStatus: ...

    @abc.abstractmethod
    def insert(
        self,
        source: TableDataSource | None = None,
        /,
        *,
        source_format: Literal['csv', 'excel', 'parquet', 'json'] | None = None,
        schema_overrides: dict[str, ts.ColumnType] | None = None,
        on_error: Literal['abort', 'ignore'] = 'abort',
        print_stats: bool = False,
        return_rows: bool = False,
        **kwargs: Any,
    ) -> UpdateStatus:
        raise NotImplementedError

    def update(
        self,
        value_spec: dict[str, Any],
        where: 'exprs.Expr' | None = None,
        cascade: bool = True,
        return_rows: bool = False,
    ) -> UpdateStatus:
        self._validate_update_value_spec(value_spec)
        self._validate_where(where)
        with get_runtime().catalog.begin_xact(
            for_write=True, write_tvps=[self._tbl_version_path], lock_mutable_tree=True
        ):
            self._check_mutable('update')
            result = self._tbl_version.get().update(value_spec, where, cascade, return_rows=return_rows)
            FileCache.get().emit_eviction_warnings()
            return result

    def batch_update(
        self,
        rows: Iterable[dict[str, Any]],
        cascade: bool = True,
        if_not_exists: Literal['error', 'ignore', 'insert'] = 'error',
        return_rows: bool = False,
    ) -> UpdateStatus:
        with get_runtime().catalog.begin_xact(
            for_write=True, write_tvps=[self._tbl_version_path], lock_mutable_tree=True
        ):
            self._check_mutable('update')
            rows = list(rows)

            row_updates: list[dict[Column, exprs.Expr]] = []
            pk_col_names = {c.name for c in self._tbl_version.get().primary_key_columns()}

            # pseudo-column _rowid: contains the rowid of the row to update and can be used instead of the primary key
            has_rowid = _ROWID_COLUMN_NAME in rows[0]
            rowids: list[tuple[int, ...]] = []
            if len(pk_col_names) == 0 and not has_rowid:
                raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, 'Table must have primary key for batch update')

            for row_spec in rows:
                col_vals = self._tbl_version.get()._validate_update_spec(
                    row_spec, allow_pk=not has_rowid, allow_exprs=False, allow_media=False
                )
                if has_rowid:
                    # every row must specify _rowid if any does
                    if _ROWID_COLUMN_NAME not in row_spec:
                        raise excs.Error(
                            excs.ErrorCode.INTERNAL_ERROR,
                            f'Malformed batch update: row is missing {_ROWID_COLUMN_NAME}',
                        )
                    rowids.append(row_spec[_ROWID_COLUMN_NAME])
                else:
                    col_names = {col.name for col in col_vals}
                    if any(pk_col_name not in col_names for pk_col_name in pk_col_names):
                        missing_cols = pk_col_names - {col.name for col in col_vals}
                        raise excs.RequestError(
                            excs.ErrorCode.UNSUPPORTED_OPERATION,
                            f'Primary key column(s) {", ".join(repr(c) for c in missing_cols)} missing in {row_spec}',
                        )
                row_updates.append(col_vals)

            result = self._tbl_version.get().batch_update(
                row_updates,
                rowids,
                error_if_not_exists=if_not_exists == 'error',
                insert_if_not_exists=if_not_exists == 'insert',
                cascade=cascade,
                return_rows=return_rows,
            )
            FileCache.get().emit_eviction_warnings()
            return result

    def recompute_columns(
        self,
        *columns: str | ColumnRef,
        where: 'exprs.Expr' | None = None,
        errors_only: bool = False,
        cascade: bool = True,
    ) -> UpdateStatus:
        cat = get_runtime().catalog
        # lock_mutable_tree=True: we need to be able to see whether any transitive view has column dependents
        with cat.begin_xact(for_write=True, write_tvps=[self._tbl_version_path], lock_mutable_tree=True):
            self._check_mutable('recompute columns of')
            if len(columns) == 0:
                raise excs.RequestError(
                    excs.ErrorCode.MISSING_REQUIRED, 'At least one column must be specified to recompute'
                )
            if errors_only and len(columns) > 1:
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION, 'Cannot use errors_only=True with multiple columns'
                )

            col_names: list[str] = []
            for column in columns:
                col_name: str
                col: Column
                if isinstance(column, str):
                    col = self._tbl_version_path.get_column(column)
                    if col is None:
                        raise excs.NotFoundError(excs.ErrorCode.COLUMN_NOT_FOUND, f'Unknown column: {column}')
                    col_name = column
                else:
                    assert isinstance(column, ColumnRef)
                    col = column.col
                    if not self._tbl_version_path.has_column(col.qid):
                        raise excs.NotFoundError(excs.ErrorCode.COLUMN_NOT_FOUND, f'Unknown column: {col.name}')
                    col_name = col.name
                if not col.is_computed:
                    raise excs.RequestError(
                        excs.ErrorCode.UNSUPPORTED_OPERATION, f'Column {col_name!r} is not a computed column'
                    )
                if col.get_tbl().id != self._tbl_version_path.tbl_id:
                    raise excs.RequestError(
                        excs.ErrorCode.UNSUPPORTED_OPERATION, f'Cannot recompute column of a base: {col_name}'
                    )
                col_names.append(col_name)

            if where is not None and not where.is_bound_by([self._tbl_version_path]):
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'`where` predicate ({where}) is not bound by {self._display_str()}',
                )

            result = self._tbl_version.get().recompute_columns(
                col_names, where=where, errors_only=errors_only, cascade=cascade
            )
            FileCache.get().emit_eviction_warnings()
            return result

    def delete(self, where: 'exprs.Expr' | None = None) -> UpdateStatus:
        raise NotImplementedError

    def revert(self) -> None:
        with get_runtime().catalog.begin_xact(
            for_write=True, write_tvps=[self._tbl_version_path], lock_mutable_tree=True
        ):
            self._check_mutable('revert')
            tv = self._tbl_version.get()
            if not tv.is_versioned:
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION, 'Revert is supported on versioned tables only'
                )
            tv.revert()
            # remove cached md in order to force a reload on the next operation
            self._tbl_version_path.clear_cached_md()

    def __dir__(self) -> list[str]:
        return list(super().__dir__()) + list(self._get_schema().keys())

    def _ipython_key_completions_(self) -> list[str]:
        return list(self._get_schema().keys())

    def get_versions(self, n: int | None = None) -> list[VersionMetadata]:
        if n is None:
            n = 1_000_000_000
        if not isinstance(n, int) or n < 1:
            raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f'Invalid value for `n`: {n}')

        # Retrieve the table history components from the catalog
        tbl_id = self._id
        # Collect an extra version, if available, to allow for computation of the first version's schema change
        vers_list = get_runtime().catalog.collect_tbl_history(tbl_id, n + 1)
        assert vers_list[0].tbl_md.is_versioned, 'TODO: implement for unversioned tables [PXT-1101]'

        # Construct the metadata change description dictionary
        md_list = [(vers_md.version_md.version, vers_md.schema_version_md.columns) for vers_md in vers_list]
        md_dict = MetadataUtils.create_md_change_dict(md_list)

        # Construct report lines
        if len(vers_list) > n:
            assert len(vers_list) == n + 1
            over_count = 1
        else:
            over_count = 0

        metadata_dicts: list[VersionMetadata] = []
        for vers_md in vers_list[0 : len(vers_list) - over_count]:
            version = vers_md.version_md.version
            schema_change = md_dict.get(version, None)
            update_status = vers_md.version_md.update_status
            if update_status is None:
                update_status = UpdateStatus()
            change_type: Literal['schema', 'data'] = 'schema' if schema_change is not None else 'data'
            rcs = update_status.row_count_stats + update_status.cascade_row_count_stats
            metadata_dicts.append(
                VersionMetadata(
                    version=version,
                    created_at=datetime.datetime.fromtimestamp(vers_md.version_md.created_at, tz=datetime.timezone.utc),
                    user=vers_md.version_md.user,
                    change_type=change_type,
                    inserts=rcs.ins_rows,
                    updates=rcs.upd_rows,
                    deletes=rcs.del_rows,
                    errors=rcs.num_excs,
                    schema_change=schema_change,
                )
            )

        return metadata_dicts
