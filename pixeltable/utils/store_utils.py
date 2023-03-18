from typing import Tuple

import sqlalchemy as sql

from pixeltable import store
from pixeltable.env import Env


def column_names() -> Tuple[int, int, str, str, str]:
    """
    Returns list of [tbl id, col id, db name, tbl name, col name] for all columns.
    """
    with Env.get().engine.begin() as conn:
        # we want to know the most recent name for any column
        max_col_versions = sql.select(
                store.SchemaColumn.tbl_id, store.SchemaColumn.col_id,
                sql.func.max(store.SchemaColumn.schema_version).label('max_schema_version'))\
            .group_by(store.SchemaColumn.tbl_id, store.SchemaColumn.col_id)\
            .cte('max_col_versions')
        stmt = sql.select(
                store.Table.id, store.SchemaColumn.col_id, store.Db.name, store.Table.name, store.SchemaColumn.name) \
            .where(store.Db.id == store.Table.db_id) \
            .where(store.Table.id == store.SchemaColumn.tbl_id) \
            .where(
                store.SchemaColumn.schema_version.in_(
                    sql.select(max_col_versions.c.max_schema_version)\
                        .where(max_col_versions.c.tbl_id == store.SchemaColumn.tbl_id)\
                        .where(max_col_versions.c.col_id == store.SchemaColumn.col_id)))
        return [row._data for row in conn.execute(stmt)]
