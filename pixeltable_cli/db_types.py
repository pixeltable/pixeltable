from typing import Literal

from typing_extensions import TypedDict

# a db plan, a service plan and a schema plan share these definitions; moving them to a common module is a
# follow-up
from pixeltable_cli.schema_types import OpStatus, _Status

# Mirror of pixeltable.service.db_diff.Severity.
Severity = Literal['additive', 'destructive', 'unsupported']

# Mirror of pixeltable.service.db_diff.Target.
Target = Literal['image', 'project', 'capacity', 'secret', 'placement']

DbResolution = Literal['up_to_date', 'create', 'update_additive', 'update_destructive', 'unsupported']


class DbChangeOp(_Status):
    """Mirror of pixeltable.service.db_diff.DbChangeOp: one operation reconciling a hosted database."""

    target: Target
    name: str
    op: Literal['add', 'drop', 'alter']
    severity: Severity
    description: str  # one sentence, ready to print
    details: dict[str, str]  # 'from' and 'to' for an alter, 'changes' for the image
    requires_restart: bool  # whether applying this interrupts what the database is serving

def image_built_op() -> DbChangeOp:
    """The operation for an image build the caller asked for rather than one a difference calls for."""
    return {
        'target': 'image',
        'name': 'image',
        'op': 'alter',
        'severity': 'additive',
        'description': "the image will be rebuilt from the project's environment",
        'details': {},
        'requires_restart': True,
    }


def project_shipped_op() -> DbChangeOp:
    """The operation for shipping the project the caller named rather than one a difference calls for."""
    return {
        'target': 'project',
        'name': 'project',
        'op': 'alter',
        'severity': 'additive',
        'description': 'the project will be shipped',
        'details': {},
        'requires_restart': True,
    }



class DbPlanSummary(TypedDict):
    ops: int
    destructive: int
    unsupported: int
    rebuild: bool  # whether the plan rebuilds the image, which is the one step that takes minutes
    restarts: bool  # whether applying the plan interrupts what the database is serving


class DbPlan(_Status):
    """Set of changes needed to reconcile a hosted database with the entry in a project that declares it."""

    config_file: str
    target: str  # the pxt://org:db uri
    exists: bool
    state: str | None  # the database's state, None when it does not exist
    resolution: DbResolution
    in_agreement: bool  # True if the database matches what the entry declares
    ops: list[DbChangeOp]

    # the classes with no reported state, which the plan leaves uncompared
    not_compared: list[str]

    destructive: bool  # whether any of the operations is
    summary: DbPlanSummary


__all__ = ['DbChangeOp', 'DbPlan', 'DbPlanSummary', 'DbResolution', 'OpStatus', 'Severity', 'Target']
