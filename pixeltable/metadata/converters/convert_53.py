import copy
from typing import Any

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.schema import Table, TableSchemaVersion


@register_converter(version=53)
def _(engine: sql.engine.Engine) -> None:
    """
    Changes in version 54:
    - Function serialization is normalized to always use a list form. Function._as_dict() now emits 'signatures'
      (a list) instead of a singular 'signature' for the non-polymorphic case, and the by-value form of
      ExprTemplateFunction._as_dict() now emits 'templates' instead of a singular 'expr' + 'signature'.

    This converter rewrites all stored function serializations to the new form. Function dicts can appear in
    Table.md (embedding-index init_args, view predicates/iterator args) and in TableSchemaVersion.md
    (computed-column value_exprs, which moved out of Table.md in version 53).
    """
    with engine.begin() as conn:
        for row in conn.execute(sql.select(Table.id, Table.md)):
            tbl_id, table_md = row[0], row[1]
            updated_md = copy.deepcopy(table_md)
            _normalize(updated_md)
            if updated_md != table_md:
                conn.execute(sql.update(Table).where(Table.id == tbl_id).values(md=updated_md))

        stmt = sql.select(TableSchemaVersion.tbl_id, TableSchemaVersion.schema_version, TableSchemaVersion.md)
        for row in conn.execute(stmt):
            tbl_id, schema_version, sv_md = row[0], row[1], row[2]
            updated_md = copy.deepcopy(sv_md)
            _normalize(updated_md)
            if updated_md != sv_md:
                conn.execute(
                    sql.update(TableSchemaVersion)
                    .where(TableSchemaVersion.tbl_id == tbl_id)
                    .where(TableSchemaVersion.schema_version == schema_version)
                    .values(md=updated_md)
                )


def _normalize(md: Any) -> None:
    """Recursively rewrite singular function serializations to the list form, in place.

    A serialized Function is a dict with a '_classpath' key. The two singular forms produced by the old code are:
      - {'_classpath', 'path', 'signature'}            (Function._as_dict, non-polymorphic)
      - {'_classpath', 'expr', 'signature', 'name'}     (ExprTemplateFunction._as_dict, by-value, non-polymorphic)
    The '_classpath' requirement avoids touching look-alikes that carry their own 'signature' but are read back
    unchanged (the {'signature', 'batch_size'} store_md of pickled UDFs, QueryTemplateFunction's dict, and the
    inner {'expr', 'signature'} entries of an already-converted 'templates' list).
    """
    if isinstance(md, dict):
        if '_classpath' in md and 'path' in md and 'signature' in md and 'signatures' not in md:
            md['signatures'] = [md.pop('signature')]
        elif '_classpath' in md and 'expr' in md and 'signature' in md:
            md['templates'] = [{'expr': md.pop('expr'), 'signature': md.pop('signature')}]
        for v in md.values():
            _normalize(v)
    elif isinstance(md, list):
        for v in md:
            _normalize(v)
