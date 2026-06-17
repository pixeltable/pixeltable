from typing import Any

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md


@register_converter(version=51)
def _(engine: sql.engine.Engine) -> None:
    convert_table_md(engine, substitution_fn=__substitute_md)


def __substitute_md(k: str | None, v: Any) -> tuple[str | None, Any] | None:
    # Migrate the ColumnRef serialization format.
    #
    # Old (v51): the column's physical owner (tbl_id/tbl_version) plus a separate reference_tbl path that carried
    #   the access context (the view/snapshot the column was reached through).
    # New (v52): the path-context table (tbl_id/effective_version) and the column's physical owner
    #   (col_tbl_id/col_tbl_effective_version), both stored directly; reference_tbl is gone.
    if not (isinstance(v, dict) and v.get('_classname') == 'ColumnRef'):
        return None

    owner_tbl_id = v['tbl_id']
    owner_effective_version = v['tbl_version']
    reference_tbl = v.get('reference_tbl')
    if reference_tbl is not None:
        # the context is the reference table's (leaf) version
        context_key = reference_tbl['tbl_version']
        context_tbl_id = context_key['id']
        context_effective_version = context_key['effective_version']
    else:
        # no reference table: the column was accessed directly on its owner, so context == owner
        context_tbl_id = owner_tbl_id
        context_effective_version = owner_effective_version

    v['tbl_id'] = context_tbl_id
    v['effective_version'] = context_effective_version
    v['col_tbl_id'] = owner_tbl_id
    v['col_tbl_effective_version'] = owner_effective_version
    del v['tbl_version']
    v.pop('reference_tbl', None)
    return k, v
