from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Any, List, Literal, Optional
from uuid import UUID

import pixeltable.exceptions as excs
import pixeltable.metadata.schema as md_schema
import pixeltable.type_system as ts
from pixeltable import catalog, exprs, func
from pixeltable.env import Env
from pixeltable.iterators import ComponentIterator

if TYPE_CHECKING:
    from pixeltable.plan import SampleClause


from .column import Column
from .globals import _POS_COLUMN_NAME, MediaValidation
from .table import Table
from .table_version import TableVersion
from .table_version_handle import TableVersionHandle
from .table_version_path import TableVersionPath
from .update_status import UpdateStatus

if TYPE_CHECKING:
    from pixeltable.globals import TableDataSource

_logger = logging.getLogger('pixeltable')


class View(Table):
    """A `Table` that presents a virtual view of another table (or view).

    A view is typically backed by a store table, which records the view's columns and is joined back to the bases
    at query execution time.
    The exception is a snapshot view without a predicate and without additional columns: in that case, the view
    is simply a reference to a specific set of base versions.
    """

    def __init__(self, id: UUID, dir_id: UUID, name: str, tbl_version_path: TableVersionPath, snapshot_only: bool):
        super().__init__(id, dir_id, name, tbl_version_path)
        self._snapshot_only = snapshot_only
        if not snapshot_only:
            self._tbl_version = tbl_version_path.tbl_version

    @classmethod
    def _display_name(cls) -> str:
        return 'view'

    @classmethod
    def select_list_to_additional_columns(cls, select_list: list[tuple[exprs.Expr, Optional[str]]]) -> dict[str, dict]:
        """Returns a list of columns in the same format as the additional_columns parameter of View.create.
        The source is the list of expressions from a select() statement on a DataFrame.
        If the column is a ColumnRef, to a base table column, it is marked to not be stored.sy
        """
        from pixeltable.dataframe import DataFrame

        r: dict[str, dict] = {}
        exps, names = DataFrame._normalize_select_list([], select_list)
        for expr, name in zip(exps, names):
            stored = not isinstance(expr, exprs.ColumnRef)
            r[name] = {'value': expr, 'stored': stored}
        return r

    @classmethod
    def _create(
        cls,
        dir_id: UUID,
        name: str,
        base: TableVersionPath,
        select_list: Optional[list[tuple[exprs.Expr, Optional[str]]]],
        additional_columns: dict[str, Any],
        predicate: Optional['exprs.Expr'],
        sample_clause: Optional['SampleClause'],
        is_snapshot: bool,
        num_retained_versions: int,
        comment: str,
        media_validation: MediaValidation,
        iterator_cls: Optional[type[ComponentIterator]],
        iterator_args: Optional[dict],
    ) -> View:
        from pixeltable.plan import SampleClause

        # Convert select_list to more additional_columns if present
        include_base_columns: bool = select_list is None
        select_list_columns: List[Column] = []
        if not include_base_columns:
            r = cls.select_list_to_additional_columns(select_list)
            select_list_columns = cls._create_columns(r)

        columns_from_additional_columns = cls._create_columns(additional_columns)
        columns = select_list_columns + columns_from_additional_columns
        cls._verify_schema(columns)

        # verify that filters can be evaluated in the context of the base
        if predicate is not None:
            if not predicate.is_bound_by([base]):
                raise excs.Error(f'Filter cannot be computed in the context of the base {base.tbl_name()}')
            # create a copy that we can modify and store
            predicate = predicate.copy()
        if sample_clause is not None:
            # make sure that the sample clause can be computed in the context of the base
            if sample_clause.stratify_exprs is not None and not all(
                stratify_expr.is_bound_by([base]) for stratify_expr in sample_clause.stratify_exprs
            ):
                raise excs.Error(f'Sample clause cannot be computed in the context of the base {base.tbl_name()}')
            # create a copy that we can modify and store
            sc = sample_clause
            sample_clause = SampleClause(
                sc.version, sc.n, sc.n_per_stratum, sc.fraction, sc.seed, sc.stratify_exprs.copy()
            )

        # same for value exprs
        for col in columns:
            if not col.is_computed:
                continue
            # make sure that the value can be computed in the context of the base
            if col.value_expr is not None and not col.value_expr.is_bound_by([base]):
                raise excs.Error(
                    f'Column {col.name}: value expression cannot be computed in the context of the '
                    f'base {base.tbl_name()}'
                )

        if iterator_cls is not None:
            assert iterator_args is not None

            # validate iterator_args
            py_signature = inspect.signature(iterator_cls.__init__)

            # make sure iterator_args can be used to instantiate iterator_cls
            bound_args: dict[str, Any]
            try:
                bound_args = py_signature.bind(None, **iterator_args).arguments  # None: arg for self
            except TypeError as exc:
                raise excs.Error(f'Invalid iterator arguments: {exc}') from exc
            # we ignore 'self'
            first_param_name = next(iter(py_signature.parameters))  # can't guarantee it's actually 'self'
            del bound_args[first_param_name]

            # construct Signature and type-check bound_args
            params = [
                func.Parameter(param_name, param_type, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for param_name, param_type in iterator_cls.input_schema().items()
            ]
            sig = func.Signature(ts.InvalidType(), params)

            expr_args = {k: exprs.Expr.from_object(v) for k, v in bound_args.items()}
            sig.validate_args(expr_args, context=f'in iterator {iterator_cls.__name__!r}')
            literal_args = {k: v.val if isinstance(v, exprs.Literal) else v for k, v in expr_args.items()}

            # prepend pos and output_schema columns to cols:
            # a component view exposes the pos column of its rowid;
            # we create that column here, so it gets assigned a column id;
            # stored=False: it is not stored separately (it's already stored as part of the rowid)
            iterator_cols = [Column(_POS_COLUMN_NAME, ts.IntType(), stored=False)]
            output_dict, unstored_cols = iterator_cls.output_schema(**literal_args)
            iterator_cols.extend(
                [
                    Column(col_name, col_type, stored=col_name not in unstored_cols)
                    for col_name, col_type in output_dict.items()
                ]
            )

            iterator_col_names = {col.name for col in iterator_cols}
            for col in columns:
                if col.name in iterator_col_names:
                    raise excs.Error(
                        f'Duplicate name: column {col.name} is already present in the iterator output schema'
                    )
            columns = iterator_cols + columns

        session = Env.get().session
        from pixeltable.exprs import InlineDict

        iterator_args_expr: exprs.Expr = InlineDict(iterator_args) if iterator_args is not None else None
        iterator_class_fqn = f'{iterator_cls.__module__}.{iterator_cls.__name__}' if iterator_cls is not None else None
        base_version_path = cls._get_snapshot_path(base) if is_snapshot else base

        # if this is a snapshot, we need to retarget all exprs to the snapshot tbl versions
        if is_snapshot:
            predicate = predicate.retarget(base_version_path) if predicate is not None else None
            if sample_clause is not None:
                exprs.Expr.retarget_list(sample_clause.stratify_exprs, base_version_path)
            iterator_args_expr = (
                iterator_args_expr.retarget(base_version_path) if iterator_args_expr is not None else None
            )
            for col in columns:
                if col.value_expr is not None:
                    col.set_value_expr(col.value_expr.retarget(base_version_path))

        view_md = md_schema.ViewMd(
            is_snapshot=is_snapshot,
            include_base_columns=include_base_columns,
            predicate=predicate.as_dict() if predicate is not None else None,
            sample_clause=sample_clause.as_dict() if sample_clause is not None else None,
            base_versions=base_version_path.as_md(),
            iterator_class_fqn=iterator_class_fqn,
            iterator_args=iterator_args_expr.as_dict() if iterator_args_expr is not None else None,
        )

        id, tbl_version = TableVersion.create(
            dir_id,
            name,
            columns,
            num_retained_versions,
            comment,
            media_validation=media_validation,
            # base_path=base_version_path,
            view_md=view_md,
        )
        if tbl_version is None:
            # this is purely a snapshot: we use the base's tbl version path
            view = cls(id, dir_id, name, base_version_path, snapshot_only=True)
            _logger.info(f'created snapshot {name}')
        else:
            view = cls(
                id,
                dir_id,
                name,
                TableVersionPath(
                    TableVersionHandle(tbl_version.id, tbl_version.effective_version), base=base_version_path
                ),
                snapshot_only=False,
            )
            _logger.info(f'Created view `{name}`, id={tbl_version.id}')

            from pixeltable.plan import Planner

            try:
                plan, _ = Planner.create_view_load_plan(view._tbl_version_path)
                _, row_counts = tbl_version.store_tbl.insert_rows(plan, v_min=tbl_version.version)
                status = UpdateStatus(row_count_stats=row_counts)
                tbl_version._write_md_update_status(0, update_status=status)

            except:
                # we need to remove the orphaned TableVersion instance
                del catalog.Catalog.get()._tbl_versions[tbl_version.id, tbl_version.effective_version]
                base_tbl_version = base.tbl_version.get()
                if tbl_version.effective_version is None and not base_tbl_version.is_snapshot:
                    # also remove tbl_version from the base
                    base_tbl_version.mutable_views.remove(TableVersionHandle.create(tbl_version))
                raise
            Env.get().console_logger.info(
                f'Created view `{name}` with {status.num_rows} rows, {status.num_excs} exceptions.'
            )

        session.commit()
        return view

    @classmethod
    def _verify_column(cls, col: Column) -> None:
        # make sure that columns are nullable or have a default
        if not col.col_type.nullable and not col.is_computed:
            raise excs.Error(f'Column {col.name}: non-computed columns in views must be nullable')
        super()._verify_column(col)

    @classmethod
    def _get_snapshot_path(cls, tbl_version_path: TableVersionPath) -> TableVersionPath:
        """Returns snapshot of the given table version path.
        All TableVersions of that path will be snapshot versions. Creates new versions from mutable versions,
        if necessary.
        """
        if tbl_version_path.is_snapshot():
            return tbl_version_path
        tbl_version = tbl_version_path.tbl_version.get()
        if not tbl_version.is_snapshot:
            # create and register snapshot version
            tbl_version = tbl_version.create_snapshot_copy()
            assert tbl_version.is_snapshot

        return TableVersionPath(
            TableVersionHandle(tbl_version.id, tbl_version.effective_version),
            base=cls._get_snapshot_path(tbl_version_path.base) if tbl_version_path.base is not None else None,
        )

    def _get_metadata(self) -> dict[str, Any]:
        md = super()._get_metadata()
        md['is_view'] = True
        md['is_snapshot'] = self._tbl_version_path.is_snapshot()
        base_tbl = self._get_base_table()
        base_version = self._effective_base_versions[0]
        md['base'] = base_tbl._path() if base_version is None else f'{base_tbl._path()}:{base_version}'
        return md

    def insert(
        self,
        source: Optional[TableDataSource] = None,
        /,
        *,
        source_format: Optional[Literal['csv', 'excel', 'parquet', 'json']] = None,
        schema_overrides: Optional[dict[str, ts.ColumnType]] = None,
        on_error: Literal['abort', 'ignore'] = 'abort',
        print_stats: bool = False,
        **kwargs: Any,
    ) -> UpdateStatus:
        raise excs.Error(f'{self._display_name()} {self._name!r}: cannot insert into view')

    def delete(self, where: Optional[exprs.Expr] = None) -> UpdateStatus:
        raise excs.Error(f'{self._display_name()} {self._name!r}: cannot delete from view')

    def _get_base_table(self) -> Optional['Table']:
        # if this is a pure snapshot, our tbl_version_path only reflects the base (there is no TableVersion instance
        # for the snapshot itself)
        base_id = self._tbl_version_path.tbl_id if self._snapshot_only else self._tbl_version_path.base.tbl_id
        return catalog.Catalog.get().get_table_by_id(base_id)

    @property
    def _effective_base_versions(self) -> list[Optional[int]]:
        effective_versions = [tv.effective_version for tv in self._tbl_version_path.get_tbl_versions()]
        if self._snapshot_only:
            return effective_versions
        else:
            return effective_versions[1:]

    def _table_descriptor(self) -> str:
        display_name = 'Snapshot' if self._snapshot_only else 'View'
        result = [f'{display_name} {self._path()!r}']
        bases_descrs: list[str] = []
        for base, effective_version in zip(self._get_base_tables(), self._effective_base_versions):
            if effective_version is None:
                bases_descrs.append(f'{base._path()!r}')
            else:
                base_descr = f'{base._path()}:{effective_version}'
                bases_descrs.append(f'{base_descr!r}')
        result.append(f' (of {", ".join(bases_descrs)})')

        if self._tbl_version_path.tbl_version.get().predicate is not None:
            result.append(f'\nWhere: {self._tbl_version_path.tbl_version.get().predicate!s}')
        if self._tbl_version_path.tbl_version.get().sample_clause is not None:
            result.append(f'\nSample: {self._tbl_version.get().sample_clause!s}')
        return ''.join(result)
