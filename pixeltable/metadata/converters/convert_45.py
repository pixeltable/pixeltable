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
    assert 'indexed_col' in v, 'SimilarityExpr must have indexed_col'
    indexed_col = v['indexed_col']
    assert isinstance(indexed_col, dict), 'SimilarityExpr migration: indexed_col must be serialized ColumnRef dict'
    tbl_id = indexed_col['tbl_id']
    tbl_version = indexed_col['tbl_version']
    idx_name = v.get('idx_name')
    assert idx_name is not None, (
        'SimilarityExpr migration: dict with indexed_col must contain idx_name (format not corrupted)'
    )
    tbl_version_key = {'id': tbl_id, 'effective_version': tbl_version, 'anchor_tbl_id': None}
    new_d: dict[str, Any] = {kk: vv for kk, vv in v.items() if kk != 'indexed_col'}
    new_d['tbl_version_key'] = tbl_version_key
    new_d['idx_name'] = idx_name
    return (k, new_d)


@register_converter(version=45)
def _(engine: sql.engine.Engine) -> None:
    """Migrate SimilarityExpr from serialized col_ref (indexed_col) to tbl_version_key + idx_name."""
    convert_table_md(engine, substitution_fn=_substitute_md)
