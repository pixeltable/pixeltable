from uuid import UUID

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.schema import Table, TableSchemaVersion


@register_converter(version=45)
def _(engine: sql.engine.Engine) -> None:
    """
    TODO
    """
    with engine.begin() as conn:
        # Read the table and table schema version metadata from the store
        # table id -> table md
        table_mds: dict[UUID, dict] = {}
        # table id -> schema version -> schema version md
        table_schema_versions: dict[UUID, dict[int, dict]] = {}
        for row in conn.execute(sql.select(Table.id, Table.md)):
            assert row.id not in table_mds
            table_mds[row.id] = row.md
        for row in conn.execute(
            sql.select(TableSchemaVersion.tbl_id, TableSchemaVersion.schema_version, TableSchemaVersion.md)
        ):
            if row.tbl_id not in table_schema_versions:
                table_schema_versions[row.tbl_id] = {}
            assert row.schema_version not in table_schema_versions[row.tbl_id]
            table_schema_versions[row.tbl_id][row.schema_version] = row.md

        # Convert and write back the updated metadata
        for tbl_id, tbl_md in table_mds.items():
            _convert_table_and_versions(tbl_md, table_schema_versions[tbl_id])
            conn.execute(sql.update(Table).where(Table.id == tbl_id).values(md=tbl_md))
            for schema_version, schema_version_md in table_schema_versions[tbl_id].items():
                conn.execute(
                    sql.update(TableSchemaVersion)
                    .where(TableSchemaVersion.tbl_id == tbl_id)
                    .where(TableSchemaVersion.schema_version == schema_version)
                    .values(md=schema_version_md)
                )


_COL_FIELDS_TO_MOVE = ['col_type', 'is_pk', 'value_expr', 'destination']


def _convert_table_and_versions(table_md: dict, schema_versions: dict[int, dict]) -> None:
    assert len(schema_versions) > 0
    for col_id, table_col_md in table_md['column_md'].items():
        # For each column in table md, find the relevant schema versions in schema versions md and add/update columns
        # there accordingly
        # lowest and highest versions are inclusive
        lowest_schema_ver = table_col_md['schema_version_add']
        highest_schema_ver = table_col_md.get('schema_version_drop', None)
        highest_schema_ver = Table.MAX_VERSION if highest_schema_ver is None else highest_schema_ver - 1
        for schema_ver, schema_version_md in schema_versions.items():
            if schema_ver < lowest_schema_ver or schema_ver > highest_schema_ver:
                continue
            # Update user-visible columns and add system columns
            col = schema_version_md['columns'].setdefault(col_id, {'pos': None, 'name': None, 'media_validation': None})
            for field in _COL_FIELDS_TO_MOVE:
                col[field] = table_col_md[field]
        # Finally, remove the moved fields from the source table md
        for field in _COL_FIELDS_TO_MOVE:
            del table_col_md[field]
