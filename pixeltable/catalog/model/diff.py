"""How a model's declared schema differs from the table it is bound to, and how that difference reads."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Literal, TypedDict
from uuid import UUID

from pixeltable import catalog, exprs
from pixeltable.types import ColumnSpec

from ..globals import col_type_from_spec
from ..table_metadata import ColumnMetadata, TableMetadata

if TYPE_CHECKING:
    from .declaration import IndexDeclaration, TableModelMeta


class SchemaChangeOp(TypedDict):
    """
    A single schema change operation (eg, add column, drop column, etc).

    Mirrored by pixeltable_cli.schema_types.SchemaChangeOp; adding, removing or retyping a field here means
    doing the same there.
    """

    target: Literal['column', 'index', 'table']

    # column name, index name, or for 'table', the differing attribute:
    # 'kind' | 'iterator' | 'view_filter' | 'view_sample' | 'media_validation' | 'comment' | 'custom_metadata'
    name: str

    op: Literal['add', 'drop', 'alter']
    severity: Literal['additive', 'destructive', 'unsupported']
    model: Any | None  # model-side value; None for drops
    existing: Any | None  # catalog-side value; None for adds
    description: str

    # the change's operands, rendered as strings so they survive serialization: 'type' or 'value' for a column add,
    # 'on' for an index add. Empty when the change has no operand beyond name.
    details: dict[str, str]


# Mirrored by pixeltable_cli.schema_types.DiffResolution; a value added here has to be added there too
DiffResolution = Literal['up_to_date', 'create', 'update_additive', 'update_destructive', 'unsupported']


class TableDiff(TypedDict):
    """How one model differs from its catalog table.

    Mirrored by pixeltable_cli.schema_types.TableDiff; adding, removing or retyping a field here means doing
    the same there.
    """

    path: str  # catalog path of the table
    model_cls: str  # model class name, so an agent can map back to code
    kind: Literal['table', 'view']
    exists: bool
    resolution: DiffResolution
    ops: list[SchemaChangeOp]

    # identity of the existing table, as of the read this diff was computed from; None if it doesn't exist yet
    tbl_id: UUID | None

    # schema versions of the TableVersionPath
    schema_versions: dict[UUID, int] | None


# Table-level attribute names that are reported as a single grouped diff (as opposed to kind/iterator/filter/
# sample, which each get their own diff line).
_TABLE_PROP_NAMES: tuple[str, ...] = ('media_validation', 'comment', 'custom_metadata', 'has_default_idxs')


def _resolution(exists: bool, ops: list[SchemaChangeOp]) -> DiffResolution:
    """Reduce a table's list of operations to the single action `update_all()` would take."""
    if not exists:
        return 'create'
    if len(ops) == 0:
        return 'up_to_date'
    severities = {op['severity'] for op in ops}
    if 'unsupported' in severities:
        return 'unsupported'
    if 'destructive' in severities:
        return 'update_destructive'
    return 'update_additive'


@dataclasses.dataclass
class _ColumnProperties:
    """The comparable properties of a column, either from a model or from an existing table."""

    type: str
    value: str | None
    primary_key: bool
    stored: bool
    media_validation: str | None  # None for non-media columns
    comment: str | None
    custom_metadata: Any
    destination: str | None

    @classmethod
    def from_spec(cls, spec: ColumnSpec, default_media_validation: str) -> _ColumnProperties:
        """The comparable properties of a column declared by spec, resolved to match a stored column's metadata.

        A computed column's value expression carries ColumnRefByName placeholders, but those render identically to
        the ColumnRefs in the stored expression, so the display strings are directly comparable. Defaults mirror
        Column.create (stored=True, primary_key=False) and a media column's media_validation falls back to
        the table default, as it does on the stored column.
        """
        col_type = col_type_from_spec(spec)
        value = spec.get('value')
        comment = spec.get('comment')
        dest = spec.get('destination')
        dest_str = str(dest) if dest is not None else None
        return cls(
            type=col_type._to_str(as_schema=True),
            value=exprs.Expr.from_object(value).display_str(inline=False) if value is not None else None,
            primary_key=spec.get('primary_key', False),
            stored=spec.get('stored', True),
            media_validation=(spec.get('media_validation') or default_media_validation)
            if col_type.is_media_type()
            else None,
            comment=comment if comment else None,
            custom_metadata=spec.get('custom_metadata'),
            destination=dest_str,
        )

    @classmethod
    def from_metadata(cls, col_md: ColumnMetadata) -> _ColumnProperties:
        """The comparable properties of an existing column, drawn from its `ColumnMetadata`."""
        return cls(
            type=col_md['type_'],
            value=col_md['computed_with'],
            primary_key=col_md['is_primary_key'],
            stored=col_md['is_stored'],
            media_validation=col_md['media_validation'],
            comment=col_md['comment'],
            custom_metadata=col_md['custom_metadata'],
            destination=col_md['destination'],
        )


@dataclasses.dataclass
class _TableProperties:
    """The comparable properties of a table, either from a model or from an existing table."""

    media_validation: str
    comment: str | None
    custom_metadata: Any

    @classmethod
    def from_model(cls, model: TableModelMeta) -> _TableProperties:
        """The comparable table-level properties declared by a model."""
        spec = model.__table_spec__
        return cls(
            media_validation=spec['media_validation'].name.lower(),
            comment=spec['comment'],
            custom_metadata=spec['custom_metadata'],
        )

    @classmethod
    def from_metadata(cls, md: TableMetadata) -> _TableProperties:
        """The comparable table-level properties of an existing table, drawn from its `TableMetadata`."""
        return cls(
            media_validation=md['media_validation'], comment=md['comment'], custom_metadata=md['custom_metadata']
        )


def user_columns(model: TableModelMeta) -> dict[str, ColumnSpec]:
    """The model's declared columns, plus any its base query projects via a `select()` clause."""
    specs: dict[str, ColumnSpec] = dict(model.__columns__)
    base = model.__table_spec__['base']
    if base is not None and base.select_list is not None:
        for expr, col_name in base.select_list:
            if col_name is None:
                # "anonymous" compound expressions are not allowed here
                assert expr.is_column_ref, expr
                specs[expr.default_column_name()] = {'value': expr, 'stored': False}
            else:
                specs[col_name] = {'value': expr, 'stored': not expr.is_column_ref}
    return specs


def base_query_columns(model: TableModelMeta) -> set[str]:
    """Names of the columns a model's base query projects via its `select()` clause (empty if there is none)."""
    base = model.__table_spec__['base']
    if base is None or base.select_list is None:
        return set()
    # "anonymous" compound expressions are not allowed here, so every unnamed item names a column
    assert all(expr.is_column_ref for expr, name in base.select_list if name is None)
    return {expr.default_column_name() if name is None else name for expr, name in base.select_list}


def _format_column_spec(spec: ColumnSpec) -> str:
    """A display string for a column spec. The value expression is rendered via str() (not repr()), so a bare
    ColumnRefByName placeholder shows as its column name (e.g. extra1) rather than ColumnRefByName('extra1'),
    matching how it renders inside a compound expression and how the stored value expression renders."""
    parts = []
    for key, val in spec.items():
        rendered = str(val) if key == 'value' else repr(val)
        parts.append(f'{key!r}: {rendered}')
    return '{' + ', '.join(parts) + '}'


def _add_column_change(col_name: str, spec: ColumnSpec) -> SchemaChangeOp:
    details = {'type': col_type_from_spec(spec)._to_str(as_schema=True)}
    value = spec.get('value')
    if value is not None:
        details['value'] = exprs.Expr.from_object(value).display_str(inline=False)
    return SchemaChangeOp(
        target='column',
        name=col_name,
        op='add',
        severity='additive',
        model=_format_column_spec(spec),
        existing=None,
        description=f'column {col_name!r} will be added',
        details=details,
    )


def _add_index_change(idx_name: str, idx: IndexDeclaration) -> SchemaChangeOp:
    # str(), not .name: a ModelColumnRef renders as its bare column name, and a spec holding anything else
    # is reported as it stands rather than dropped from the plan
    details = {'on': str(idx.column)}
    return SchemaChangeOp(
        target='index',
        name=idx_name,
        op='add',
        severity='additive',
        model=str(idx),
        existing=None,
        description=f'index {idx_name!r} will be added',
        details=details,
    )


def validate_models(registered_models: dict[str, TableModelMeta], catalog_dir: str) -> dict[str, TableDiff]:
    """
    Analyze each registered model against the current catalog state, summarizing the schema changes that creating
    the models would entail, along with any incompatibilities with an already-existing table of the same name.
    This is purely informational: it neither modifies the catalog nor raises on incompatibilities.
    All metadata reads happen in a single transaction, and each diff's tbl_id and schema_versions record the catalog
    state it was computed against.
    """
    from ..catalog import retry_loop
    from .declaration import BtreeIndex

    catalog_dir = catalog.Path.dir_prefix(catalog_dir)

    @retry_loop(for_write=False)
    def op() -> dict[str, TableDiff]:
        results: dict[str, TableDiff] = {}

        for name, model in registered_models.items():
            user_cols = user_columns(model)
            model_cols = set(user_cols.keys())
            model_idxs = set(model.__indexes__.keys())
            base = model.__table_spec__['base']
            model_kind: Literal['table', 'view'] = 'table' if base is None else 'view'
            iterator = model.__table_spec__['iterator']
            model_iterator = None if iterator is None else iterator.display_str()
            model_filter = None if base is None or base.where_clause is None else str(base.where_clause)
            model_sample = None if base is None or base.sample_clause is None else str(base.sample_clause)

            bound_path = f'{catalog_dir}{name}'
            existing = model._resolve_tbl(catalog_dir, if_not_exists='ignore')

            ops: list[SchemaChangeOp]

            if existing is None:
                # The table does not yet exist; every column and index is an addition.
                ops = [_add_column_change(col_name, user_cols[col_name]) for col_name in sorted(model_cols)]
                ops += [_add_index_change(idx_name, model.__indexes__[idx_name]) for idx_name in sorted(model_idxs)]
                results[name] = TableDiff(
                    path=bound_path,
                    model_cls=model.__name__,
                    kind=model_kind,
                    exists=False,
                    resolution=_resolution(False, ops),
                    ops=ops,
                    tbl_id=None,
                    schema_versions=None,
                )
                continue

            existing_md = existing.get_metadata()
            tbl_path = existing._tbl_path

            # Restrict the existing columns to those defined in this table (i.e. not inherited from a base) and not
            # produced by an iterator, so that they line up with the model's own declared columns.
            existing_cols = {
                col_name
                for col_name, col_md in existing_md['columns'].items()
                if col_md['defined_in'] == existing_md['name'] and not col_md['is_iterator_col']
            }
            ops = []

            # has_default_idxs mismatch is unsupported.
            model_default_idxs = model.__table_spec__['has_default_idxs']
            existing_default_idxs = existing_md['has_default_idxs']
            if model_default_idxs != existing_default_idxs:
                ops.append(
                    SchemaChangeOp(
                        target='table',
                        name='has_default_idxs',
                        op='alter',
                        severity='unsupported',
                        model=model_default_idxs,
                        existing=existing_default_idxs,
                        description=f'`{model.__name__}` specifies has_default_idxs={model_default_idxs}, '
                        f'but {name!r} was created with has_default_idxs={existing_default_idxs}',
                        details={},
                    )
                )

            # Default indexes have no counterpart in __indexes__, and has_default_idxs=True is incompatible with
            # explicitly declared B-tree indexes. So, if the existing table has default indexes enabled, all of its
            # B-tree indexes are default indexes, and they can all be ignored, because resolving the column diff takes
            # care of the indexes too. Otherwise all B-tree indexes are explicitly declared, so they are diffed and
            # compared like embedding indexes. If the two sides disagree on has_default_idxs, no meaningful diff of
            # B-tree indexes can be computed, so they are left out.
            include_btree_idxs = not model_default_idxs and not existing_default_idxs
            if not include_btree_idxs:
                model_idxs = {
                    idx_name for idx_name, idx in model.__indexes__.items() if not isinstance(idx, BtreeIndex)
                }
            existing_idxs = {
                idx_name
                for idx_name, info in existing_md['indices'].items()
                if info['index_type'] == 'embedding' or (include_btree_idxs and info['index_type'] == 'btree')
            }

            # Structural mismatches (kind/iterator/filter/sample); each is unsupported (requires a manual migration).
            if model_kind != existing_md['kind']:
                ops.append(
                    SchemaChangeOp(
                        target='table',
                        name='kind',
                        op='alter',
                        severity='unsupported',
                        model=model_kind,
                        existing=existing_md['kind'],
                        description=(
                            f'`{model.__name__}` specifies a {model_kind}, but {name!r} is a {existing_md["kind"]}'
                        ),
                        details={},
                    )
                )
            for attr, model_val, existing_val in (
                ('iterator', model_iterator, existing_md['iterator_call']),
                ('view_filter', model_filter, existing_md['view_filter']),
                ('view_sample', model_sample, existing_md['view_sample']),
            ):
                if model_val != existing_val:
                    ops.append(
                        SchemaChangeOp(
                            target='table',
                            name=attr,
                            op='alter',
                            severity='unsupported',
                            model=model_val,
                            existing=existing_val,
                            description=f'{attr} mismatch: model={model_val!r}, existing={existing_val!r}',
                            details={},
                        )
                    )

            # Table-level properties that differ (media_validation/comment/custom_metadata); unsupported for now.
            model_table_props = _TableProperties.from_model(model)
            existing_table_props = _TableProperties.from_metadata(existing_md)
            for prop in model_table_props.__dataclass_fields__:
                model_val = getattr(model_table_props, prop)
                existing_val = getattr(existing_table_props, prop)
                if model_val != existing_val:
                    ops.append(
                        SchemaChangeOp(
                            target='table',
                            name=prop,
                            op='alter',
                            severity='unsupported',
                            model=model_val,
                            existing=existing_val,
                            description=f'table property {prop!r}: model={model_val!r}, existing={existing_val!r}',
                            details={},
                        )
                    )

            # Columns present in both, whose properties differ; unsupported for now (some alterations will later be
            # applicable via allow_destructive=True).
            default_media_validation = model.__table_spec__['media_validation'].name.lower()
            for col_name in sorted(model_cols & existing_cols):
                model_props = _ColumnProperties.from_spec(user_cols[col_name], default_media_validation)
                existing_props = _ColumnProperties.from_metadata(existing_md['columns'][col_name])
                altered = [
                    prop
                    for prop in model_props.__dataclass_fields__
                    if getattr(model_props, prop) != getattr(existing_props, prop)
                ]
                if len(altered) > 0:
                    ops.append(
                        SchemaChangeOp(
                            target='column',
                            name=col_name,
                            op='alter',
                            severity='unsupported',
                            model={prop: getattr(model_props, prop) for prop in altered},
                            existing={prop: getattr(existing_props, prop) for prop in altered},
                            description=f'column {col_name!r} has altered properties: {", ".join(altered)}',
                            details={},
                        )
                    )

            # Additive/destructive column and index changes.
            for col_name in sorted(model_cols - existing_cols):
                ops.append(_add_column_change(col_name, user_cols[col_name]))
            for col_name in sorted(existing_cols - model_cols):
                ops.append(
                    SchemaChangeOp(
                        target='column',
                        name=col_name,
                        op='drop',
                        severity='destructive',
                        model=None,
                        existing=None,
                        description=f'column {col_name!r} will be dropped',
                        details={},
                    )
                )
            # TODO(PXT-1258): compare index parameters, not just names
            for idx_name in sorted(model_idxs - existing_idxs):
                ops.append(_add_index_change(idx_name, model.__indexes__[idx_name]))
            for idx_name in sorted(existing_idxs - model_idxs):
                ops.append(
                    SchemaChangeOp(
                        target='index',
                        name=idx_name,
                        op='drop',
                        severity='destructive',
                        model=None,
                        existing=None,
                        description=f'index {idx_name!r} will be dropped',
                        details={},
                    )
                )

            results[name] = TableDiff(
                path=bound_path,
                model_cls=model.__name__,
                kind=model_kind,
                exists=True,
                resolution=_resolution(True, ops),
                ops=ops,
                tbl_id=tbl_path.tbl_id,
                schema_versions=tbl_path.schema_versions(),
            )

        return results

    return op()


def _format_diff(name: str, diff: TableDiff) -> list[str]:
    """Human-readable lines describing how the model named `name` differs from the current catalog state."""
    kind = diff['kind']
    if not diff['exists']:
        return [
            f'{kind.capitalize()} {name!r} (from model `{diff["model_cls"]}`) does not yet exist, and will be CREATED.'
        ]

    ops = diff['ops']
    if len(ops) == 0:
        return []

    def by(target: str, op: str | None = None, names: tuple[str, ...] | None = None) -> list[SchemaChangeOp]:
        return [
            c
            for c in ops
            if c['target'] == target and (op is None or c['op'] == op) and (names is None or c['name'] in names)
        ]

    detail: list[str] = []

    for c in by('table', names=('kind',)):
        detail.append(f'  kind mismatch (FATAL): {c["description"]}')
    for attr, label in (('iterator', 'iterator'), ('view_filter', 'filter'), ('view_sample', 'sample')):
        for c in by('table', names=(attr,)):
            detail.append(f'  {label} mismatch (FATAL):')
            detail.append(f'    model {label}   : {c["model"]}')
            detail.append(f'    existing {label}: {c["existing"]}')

    table_props = by('table', names=_TABLE_PROP_NAMES)
    if len(table_props) > 0:
        detail.append('  the following table properties have changed (FATAL):')
        for c in table_props:
            detail.append(f'    {c["name"]}: model={c["model"]!r}, existing={c["existing"]!r}')

    altered_cols = by('column', op='alter')
    if len(altered_cols) > 0:
        detail.append('  the following columns have altered properties (FATAL):')
        for c in altered_cols:
            for prop, model_val in c['model'].items():
                detail.append(f'    {c["name"]!r} {prop}: model={model_val!r}, existing={c["existing"][prop]!r}')

    new_cols = by('column', op='add')
    if len(new_cols) > 0:
        detail.append('  the following columns are new to the model, and will be ADDED:')
        for c in new_cols:
            detail.append(f'    {c["name"]!r} = {c["model"]}')

    dropped_cols = by('column', op='drop')
    if len(dropped_cols) > 0:
        detail.append('  the following columns are no longer in the model, and will be DROPPED:')
        for c in dropped_cols:
            detail.append(f'    {c["name"]!r}')

    new_idxs = by('index', op='add')
    if len(new_idxs) > 0:
        detail.append('  the following indexes are new to the model, and will be ADDED:')
        for c in new_idxs:
            detail.append(f'    {c["name"]!r} = {c["model"]}')

    dropped_idxs = by('index', op='drop')
    if len(dropped_idxs) > 0:
        detail.append('  the following indexes are no longer in the model, and will be DROPPED:')
        for c in dropped_idxs:
            detail.append(f'    {c["name"]!r}')

    return [f'{kind.capitalize()} {name!r} (from model `{diff["model_cls"]}`) has differences:', *detail]


# closing lines of the refusals raised by create_all()/update_all(), phrased for the Python API
_PY_MISMATCH_HINT = 'Call `update_all()` instead if you intended to also modify existing tables.'
PY_DESTRUCTIVE_HINT = (
    'If you wish to apply these changes, re-run `update_all()` with `allow_destructive=True`.\n'
    'If you intended to rename columns or indexes instead of dropping them, apply those changes '
    'directly with `pxt.move()`.'
)
