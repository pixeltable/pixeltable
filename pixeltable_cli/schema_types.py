from typing import Literal

from typing_extensions import TypedDict  # pydantic requires importing this from typing_extensions on Python < 3.12

from pixeltable_cli.utils import PxtPath

OpStatus = Literal['applied', 'skipped', 'refused', 'failed']

# Mirror of pixeltable.catalog.model.DiffResolution
DiffResolution = Literal['up_to_date', 'create', 'update_additive', 'update_destructive', 'unsupported']


class _Status(TypedDict, total=False):
    """Present only once the plan has been carried out or refused."""

    status: OpStatus


class SchemaChangeIndexRef(TypedDict):
    index_type: Literal['btree', 'embedding']
    columns: list[str]
    name: str | None


class SchemaChangeOpDetails(TypedDict, total=False):
    type: str
    value: str
    index_ref: SchemaChangeIndexRef


class SchemaChangeOp(_Status):
    """Mirror of pixeltable.catalog.model.SchemaChangeOp: one operation reconciling a table with its model."""

    target: Literal['column', 'index', 'table']

    # column name, index name, the differing attribute for an alter of a table, or the path for a drop of one.
    # None for an index that carries no name.
    name: str | None

    op: Literal['add', 'drop', 'alter']
    severity: Literal['additive', 'destructive', 'unsupported']
    description: str  # one sentence, ready to print
    details: SchemaChangeOpDetails
    destructive: bool  # the boolean form of severity


class TableDiff(_Status):
    """Mirror of pixeltable.catalog.model.TableDiff: how one table differs from the model that declares it."""

    path: str
    model_cls: str
    kind: Literal['table', 'view']
    exists: bool
    resolution: DiffResolution

    # empty for a create, which subsumes the additions that constitute it
    ops: list[SchemaChangeOp]

    destructive: bool  # whether any of the operations is


class SchemaPlanSummary(TypedDict):
    up_to_date: int
    create: int
    update_additive: int
    update_destructive: int
    unsupported: int
    extras: int
    destructive: int  # operations, not tables


class _PlanOps(TypedDict, total=False):
    ops: list[SchemaChangeOp]  # on whole tables, unlike TableDiff.ops


class SchemaPlan(_PlanOps):
    """Set of changes needed to reconcile a target directory with a schema model."""

    schema_file: str
    catalog_dir: PxtPath
    in_agreement: bool  # True if no table needs a create or an update; extras don't count
    tables: list[TableDiff]
    extras: list[PxtPath]  # tables under catalog_dir that no model declares
    summary: SchemaPlanSummary


def drop_table_op(pxt_path: PxtPath, status: OpStatus) -> SchemaChangeOp:
    """The operation for dropping the table at the given path, in the given status."""
    return {
        'target': 'table',
        'name': pxt_path,
        'op': 'drop',
        'severity': 'destructive',
        'description': f'table {pxt_path!r} will be dropped',
        'details': {},
        'destructive': True,
        'status': status,
    }
