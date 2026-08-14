"""Resolution of a model's declarations into the catalog objects a table is created and altered from."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypedDict
from uuid import UUID

from pixeltable import catalog, exceptions as excs, exprs, func, index
from pixeltable.types import ColumnSpec

from ..globals import MediaValidation
from ..table_version_handle import TableVersionHandle

if TYPE_CHECKING:
    import pixeltable as pxt

if TYPE_CHECKING:
    from .declaration import IndexDeclaration


def prepare_model(
    tbl_handle: TableVersionHandle,
    columns: dict[str, ColumnSpec],
    display_name: str,
    iterator: func.GeneratingFunctionCall | None,
    base: 'pxt.Query | None',
    idxs: list[IndexDeclaration],
) -> tuple[func.GeneratingFunctionCall | None, list[catalog.Column], list[catalog.IndexSpec]]:
    """
    Given model declarations in the form of columns, base, iterator, and index specifications, along with
    the relevant metadata, assembles lists of additional columns and additional indices to be created in the table.
    The outputs will be fully resolved (ColumnRefByNames replaced with actual ColumnRefs and the index-spec
    dataclass instances replaced with actual instances of index.IndexBase).

    Returns: a tuple of (rebound iterator, additional columns, additional idxs).
    """

    # imported here rather than at module scope: declaration imports this module
    from .declaration import BtreeIndex, EmbeddingIndex

    # View columns always go in a specific order:
    # - iterator columns first
    # - then columns from the base query's select_list
    #     (but not if it's a select(*): then just inherit the base table's columns)
    # - finally, the view's additional_columns.

    # A registry of visible columns of the table (base table/query columns, iterator columns,
    # and additional columns).
    # A column of the table being created is the Column instance that will be given an id; an inherited one
    # already has an id, and is identified by its metadata.
    user_cols: dict[str, catalog.Column | catalog.ColumnVersionMd] = {}

    # A substitution dictionary resolving ColumnRefByNames to actual ColumnRefs. It holds only the base tables'
    # columns; the columns of the table being created have no ids yet, so references to those stay ColumnRefByName.
    subst_dict: exprs.ExprDict[exprs.Expr] = exprs.ExprDict()

    # Names of the columns of the table being created, accumulated as they are processed.
    preceding_names: set[str] = set()

    # First the iterator columns, if present.
    if iterator is not None:
        # Rebind the iterator, resolving its argument references against the base table.
        assert base is not None
        base_tbl_subst_dict = exprs.ExprDict[exprs.Expr](
            (exprs.ColumnRefByName(col_md.name), exprs.ColumnRef(col_md))
            for col_md in base._from_clause._first_tbl.column_md()
        )
        subst_args = [arg.substitute(base_tbl_subst_dict) for arg in iterator.args]
        subst_kwargs = {k: v.substitute(base_tbl_subst_dict) for k, v in iterator.kwargs.items()}
        subst_bound_args = {k: v.substitute(base_tbl_subst_dict) for k, v in iterator.bound_args.items()}
        iterator = func.GeneratingFunctionCall(
            iterator.it, subst_args, subst_kwargs, subst_bound_args, iterator.outputs, iterator.validation_error
        )
        # Build substitutions for the iterator's output columns.
        for name, output in iterator.outputs.items():
            catalog_col = catalog.Column.create(name, {'type': output.col_type, 'stored': output.is_stored})  # type: ignore[arg-type]
            catalog_col.tbl_handle = tbl_handle
            user_cols[name] = catalog_col
            preceding_names.add(name)

    if base is not None:
        # Build substitutions for the base table/query's columns.
        if base.select_list is None:
            # select(*): all visible columns from the base table
            for col_md in base._from_clause._first_tbl.column_md():
                # Iterator column names take precedence over base table column names in the model namespace, so
                # only update the substitution dicts if the name isn't already present.
                if col_md.name not in user_cols:
                    user_cols[col_md.name] = col_md
                    subst_dict[exprs.ColumnRefByName(col_md.name)] = exprs.ColumnRef(col_md)
        else:
            # explicit select list: new columns will be created that represent the selected expressions.
            for expr, select_name in base.select_list:
                col_name: str | None
                if select_name is not None:
                    # The select list has an explicit name for this expression as a kwarg; use it.
                    col_name = select_name
                elif isinstance(expr, exprs.ColumnRef):
                    # It's an unnamed column reference; use the name of the referenced column as a fallback.
                    col_name = expr.column_md.name
                else:
                    # It's a compound expression with no explicit name. A name will be assigned when the table
                    # is created, but it's anonymous to the TableModel.
                    # TODO: Revisit this behavior. Should we be allowing unnamed compound expressions in the
                    #     first place?
                    col_name = None

                if col_name is not None:
                    # Column names that arrived via an explicit select list take precedence over iterator column
                    # names in the model namespace, so here we always update the dicts.
                    catalog_col = catalog.Column.create(col_name, expr.col_type)
                    catalog_col.tbl_handle = tbl_handle
                    user_cols[col_name] = catalog_col
                    preceding_names.add(col_name)

    # Process any additional columns specified in the view model body.
    additional_cols: list[catalog.Column] = []
    for name, spec in columns.items():
        subst_spec = spec.copy()
        if 'value' in subst_spec:
            subst_spec['value'] = subst_spec['value'].substitute(subst_dict)
            # whatever is still a ColumnRefByName has to name a column of this table that precedes this one, so that
            # it has a value by the time this column is computed
            unresolved = [
                ref for ref in subst_spec['value'].subexprs(exprs.ColumnRefByName) if ref.name not in preceding_names
            ]
            if len(unresolved) > 0:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_SCHEMA,
                    f'Column {name!r} in {display_name} references columns that are not in '
                    f"the model's scope: {[ref.name for ref in unresolved]}",
                )

        # subst_dict holds exactly the inherited base columns, we don't allow an explicitly named column to shadow one
        if exprs.ColumnRefByName(name) in subst_dict:
            assert base is not None
            raise excs.AlreadyExistsError(
                excs.ErrorCode.COLUMN_ALREADY_EXISTS,
                f'Column {name!r} already exists in the base table {base._from_clause._first_tbl.tbl_name()!r}.',
            )
        catalog_col = catalog.Column.create(name, subst_spec)
        catalog_col.tbl_handle = tbl_handle
        additional_cols.append(catalog_col)
        user_cols[name] = catalog_col
        preceding_names.add(name)

    # Resolve each declared index against the model's visible columns.
    resolved_idxs: list[catalog.IndexSpec] = []
    for idx_spec in idxs:
        if not isinstance(idx_spec.column, exprs.ColumnRefByName):
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA, f'Index in {display_name} has an invalid column reference.'
            )
        col_name = idx_spec.column.name
        if col_name not in user_cols:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA, f'Index in {display_name} references unknown column {col_name!r}.'
            )
        idx: index.IndexBase
        idx_name: str | None
        if isinstance(idx_spec, EmbeddingIndex):
            idx = index.EmbeddingIndex(
                metric=idx_spec.metric,
                precision=idx_spec.precision,
                embed=idx_spec.embedding,
                string_embed=idx_spec.string_embed,
                image_embed=idx_spec.image_embed,
                audio_embed=idx_spec.audio_embed,
                video_embed=idx_spec.video_embed,
                document_embed=idx_spec.document_embed,
                column=user_cols[col_name],
            )
            idx_name = idx_spec.name
        else:
            assert isinstance(idx_spec, BtreeIndex)
            idx = index.BtreeIndex()
            idx_name = None
        resolved_idxs.append(catalog.IndexSpec(col_name, idx_name, idx))

    return iterator, additional_cols, resolved_idxs


class TableSchemaChangeSet(TypedDict):
    """
    Schema change operations applied to a single table.

    Used for proxy communication of changes that need to be applied to a catalog table during update_all().
    """

    path: catalog.Path

    # name -> (spec, origin). A 'base_query' column comes from the view's base query select() list and resolves
    # against the base table's columns; a 'model_body' column resolves against the view's own visible columns.
    new_columns: dict[str, tuple[ColumnSpec, Literal['base_query', 'model_body']]]
    dropped_columns: list[str]
    new_idxs: list[IndexDeclaration]
    dropped_idxs: list[str]

    # tbl_id of the table to update, and {tbl_id: schema_version} for its version path, captured when the diff was
    # computed.
    tbl_id: UUID
    schema_versions: dict[UUID, int]


def prepare_model_updates(
    tvp: catalog.TableVersionPath,
    display_name: str,
    new_columns: dict[str, tuple[ColumnSpec, Literal['base_query', 'model_body']]],
    new_idxs: list[IndexDeclaration],
) -> tuple[list[catalog.Column], list[catalog.IndexSpec]]:
    """
    Given `new_columns` and `new_idxs` as declared by a model, resolves them into proper catalog abstractions
    in preparation for catalog changes. This is the analog of `prepare_model()` for `update_all()`.

    Each column in `new_columns` is a (spec, origin) pair. A 'base_query' column comes from the view's base query
    `select()` list and is resolved against the base table's columns; a 'model_body' column is resolved against the
    view's own visible columns.
    """

    # imported here rather than at module scope: declaration imports this module
    from .declaration import BtreeIndex, EmbeddingIndex

    user_cols: dict[str, catalog.Column] = {}
    subst_dict: exprs.ExprDict[exprs.Expr] = exprs.ExprDict()  # ColumnRefByName -> ColumnRef

    # Pre-populate the user columns and substitution dict with the existing table's user columns.
    # This includes iterator columns and base table columns.
    for col in tvp.columns():
        user_cols[col.name] = col
        subst_dict[exprs.ColumnRefByName(col.name)] = exprs.ColumnRef(
            col.column_version_md(), perform_validation=(col.media_validation == MediaValidation.ON_READ)
        )

    # Base-query columns are projections of the base query and resolve against the base table's columns (which,
    # for a select() view, are not among the view's own visible columns above).
    has_base_query_cols = any(origin == 'base_query' for _, origin in new_columns.values())
    base_subst_dict: exprs.ExprDict[exprs.Expr] = exprs.ExprDict()
    if has_base_query_cols:
        assert tvp.base is not None
        for col in tvp.base.columns():
            base_subst_dict[exprs.ColumnRefByName(col.name)] = exprs.ColumnRef(
                col.column_version_md(), perform_validation=(col.media_validation == MediaValidation.ON_READ)
            )

    tbl_handle = tvp.tbl_version

    # Process base-query columns first, so a model-body column may reference a newly-projected base-query column.
    ordered_names = [n for n, (_, origin) in new_columns.items() if origin == 'base_query'] + [
        n for n, (_, origin) in new_columns.items() if origin != 'base_query'
    ]

    # substitute ColumnRefByName with ColumnRef for existing columns; references to added columns are left as
    # ColumnRefByName, to be resolved once ids have been assigned
    resolved_cols: list[catalog.Column] = []
    preceding_names: set[str] = set()
    for name in ordered_names:
        spec, origin = new_columns[name]
        resolved_spec = spec.copy()
        if 'value' in resolved_spec:
            resolve_against = base_subst_dict if origin == 'base_query' else subst_dict
            resolved_spec['value'] = resolved_spec['value'].substitute(resolve_against)
            # whatever is still a ColumnRefByName has to name a column that precedes this one in resolved_cols:
            # anything else is either outside the model's scope or would be evaluated before it has a value
            unresolved = [
                ref for ref in resolved_spec['value'].subexprs(exprs.ColumnRefByName) if ref.name not in preceding_names
            ]
            if len(unresolved) > 0:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_SCHEMA,
                    f'Column {name!r} in {display_name} references columns that are not in '
                    f"the model's scope: {[ref.name for ref in unresolved]}",
                )

        # a new column can only collide with an inherited one: a name already among this table's own columns
        # wouldn't have been diffed as new
        if exprs.ColumnRefByName(name) in subst_dict:
            assert tvp.base is not None
            raise excs.AlreadyExistsError(
                excs.ErrorCode.COLUMN_ALREADY_EXISTS,
                f'Column {name!r} already exists in the base table {tvp.base.tbl_name()!r}.',
            )

        catalog_col = catalog.Column.create(name, resolved_spec)
        catalog_col.tbl_handle = tbl_handle
        resolved_cols.append(catalog_col)
        preceding_names.add(name)
        user_cols[name] = catalog_col

    # Resolve each declared index against the model's visible columns.
    resolved_idxs: list[catalog.IndexSpec] = []
    for idx_spec in new_idxs:
        idx_display_name = (
            f'Index {idx_spec.name!r}'
            if isinstance(idx_spec, EmbeddingIndex) and idx_spec.name is not None
            else f'Index on column {idx_spec.column.name!r}'
        )
        if not isinstance(idx_spec.column, exprs.ColumnRefByName):
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA, f'{idx_display_name} in {display_name} has an invalid column reference.'
            )
        col_name = idx_spec.column.name
        if col_name not in user_cols:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_SCHEMA,
                f'{idx_display_name} in {display_name} references unknown column {col_name!r}.',
            )
        idx: index.IndexBase
        if isinstance(idx_spec, EmbeddingIndex):
            idx = index.EmbeddingIndex(
                metric=idx_spec.metric,
                precision=idx_spec.precision,
                embed=idx_spec.embedding,
                string_embed=idx_spec.string_embed,
                image_embed=idx_spec.image_embed,
                audio_embed=idx_spec.audio_embed,
                video_embed=idx_spec.video_embed,
                document_embed=idx_spec.document_embed,
                column=user_cols[col_name],
            )
        else:
            assert isinstance(idx_spec, BtreeIndex)
            idx = index.BtreeIndex()
        resolved_idxs.append(
            catalog.IndexSpec(user_cols[col_name], idx_spec.name if isinstance(idx_spec, EmbeddingIndex) else None, idx)
        )

    return resolved_cols, resolved_idxs
