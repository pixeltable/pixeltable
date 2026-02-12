"""Convert SimilarityExpr from legacy 'indexed_col' (serialized ColumnRef) to 'tbl_version_key' + 'idx_name'."""

from typing import Any

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md


def _substitute_md(k: str | None, v: Any) -> tuple[str | None, Any] | None:
    if not isinstance(v, dict):
        return None
    if v.get('_classname') != 'SimilarityExpr':
        return None
    if 'tbl_version_key' in v: # already migrated
        return None
    assert 'components' in v
    components = v['components']
    assert len(components) == 2
    col_ref_dict = components[0]
    tbl_id = col_ref_dict['tbl_id']
    tbl_version = col_ref_dict['tbl_version']
    tbl_version_key = {'id': tbl_id, 'effective_version': tbl_version, 'anchor_tbl_id': None}
    # copy index name, class name etc
    new_d: dict[str, Any] = {kk: vv for kk, vv in v.items() if kk != 'components'}
    # Skip column ref from components
    new_d['components'] = [components[1]]
    new_d['tbl_version_key'] = tbl_version_key
    new_d['idx_name'] = v.get('idx_name')  # index name can be none or missing
    new_d['_classname'] = 'SimilarityExpr'
    return (k, new_d)


@register_converter(version=45)
def _(engine: sql.engine.Engine) -> None:
    """Migrate SimilarityExpr from serialized col_ref (indexed_col) to tbl_version_key + idx_name."""
    convert_table_md(engine, substitution_fn=_substitute_md)
