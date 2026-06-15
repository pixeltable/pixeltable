from uuid import UUID

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.schema import Table, TableSchemaVersion
from pixeltable.utils.dbms import CockroachDbms, Dbms, PostgresqlDbms


@register_converter(version=51)
def _(engine: sql.engine.Engine, dbms: Dbms) -> None:
    """
    Changes in version 52:
    - some column metadata (col_type, is_pk, value_expr) moved from the table level (ColumnMd) to
    SchemaVersionMd in order to allow for future versioning of these metadata.
    - Only user columns used to be represented by SchemaVersionMd, now system columns are also there.
    - New attrs in ColumnMd: is_media_type and sa_col_type.

    This converter reads all table and schema metadata in memory, updates them, and writes them back.
    """
    assert isinstance(dbms, (CockroachDbms, PostgresqlDbms)), dbms
    is_cockroachdb = isinstance(dbms, CockroachDbms)

    with engine.connect().execution_options(isolation_level='SERIALIZABLE') as conn:
        # Read the table and table schema version metadata from the store
        # table id -> TableMd
        table_mds: dict[UUID, dict] = {}
        # table id -> schema version -> SchemaVersionMd
        table_schema_versions: dict[UUID, dict[int, dict]] = {}
        for row in conn.execute(sql.select(Table.id, Table.md)):
            assert row.id not in table_mds, row.id
            table_mds[row.id] = row.md
        for row in conn.execute(
            sql.select(TableSchemaVersion.tbl_id, TableSchemaVersion.schema_version, TableSchemaVersion.md)
        ):
            versions = table_schema_versions.setdefault(row.tbl_id, {})
            assert row.schema_version not in versions, (row.tbl_id, row.schema_version)
            versions[row.schema_version] = row.md

        # Convert and write back the updated metadata
        for tbl_id, tbl_md in table_mds.items():
            _convert_table_and_versions(tbl_md, table_schema_versions[tbl_id], is_cockroachdb)
            result = conn.execute(sql.update(Table).where(Table.id == tbl_id).values(md=tbl_md))
            assert result.rowcount == 1
            for schema_version, schema_version_md in table_schema_versions[tbl_id].items():
                result = conn.execute(
                    sql.update(TableSchemaVersion)
                    .where(TableSchemaVersion.tbl_id == tbl_id)
                    .where(TableSchemaVersion.schema_version == schema_version)
                    .values(md=schema_version_md)
                )
                assert result.rowcount == 1

        conn.commit()


# field name -> is required
_COL_FIELDS_TO_MOVE = {'col_type': True, 'is_pk': True, 'value_expr': False}

_MEDIA_TYPE_CLASSNAMES = {'ImageType', 'AudioType', 'VideoType', 'DocumentType'}

# Maps Pixeltable types to serialized store type
_CLASSNAME_TO_SA_TYPE: dict[str, dict | None] = {
    'StringType': {'type': 'String'},
    'IntType': {'type': 'BigInteger'},
    'FloatType': {'type': 'Float'},
    'BoolType': {'type': 'Boolean'},
    'TimestampType': {'type': 'Timestamp'},
    'DateType': {'type': 'Date'},
    'UUIDType': {'type': 'UUID'},
    'BinaryType': {'type': 'LargeBinary'},
    'JsonType': {'type': 'JSONB'},
    'ArrayType': {'type': 'LargeBinary'},
    'ImageType': {'type': 'String'},
    'VideoType': {'type': 'String'},
    'AudioType': {'type': 'String'},
    'DocumentType': {'type': 'String'},
}


_EMBEDDING_INDEX_FQN = 'pixeltable.index.embedding_index.EmbeddingIndex'


def _get_embedding_val_col_precisions(table_md: dict) -> dict[str, str]:
    """Returns a map of col_id (as str) -> precision str for embedding index val/undo columns."""
    result: dict[str, str] = {}
    for idx_md in table_md.get('index_md', {}).values():
        if idx_md.get('class_fqn') != _EMBEDDING_INDEX_FQN:
            continue
        precision = idx_md['init_args']['precision']
        for col_id in (idx_md['index_val_col_id'], idx_md['index_val_undo_col_id']):
            result[str(col_id)] = precision
    return result


def _convert_table_and_versions(table_md: dict, schema_versions: dict[int, dict], is_cockroachdb: bool) -> None:
    assert len(schema_versions) > 0, table_md.get('tbl_id')
    # Build map of embedding index val/undo col_ids to precision, to assign correct Vector SA types. Nothing special
    # is needed for B-tree indexes as their columns simply inherit the type of the indexed column.
    embedding_val_col_precisions = _get_embedding_val_col_precisions(table_md)

    for col_id, table_col_md in table_md['column_md'].items():
        # Populate ColumnMd's is_media_type
        assert 'col_type' in table_col_md, (table_md, col_id)
        col_type = table_col_md['col_type']
        assert isinstance(col_type, dict), (table_md, col_id)
        assert '_classname' in col_type, (table_md, col_id)
        classname = col_type['_classname']
        table_col_md['is_media_type'] = classname in _MEDIA_TYPE_CLASSNAMES

        # Populate ColumnMd's sa_col_type
        stored = table_col_md.get('stored', True)
        sa_col_type = None
        if stored:
            if col_id in embedding_val_col_precisions:
                # Embedding index val/undo columns store vectors. See EmbeddingIndex for more details.
                assert classname == 'ArrayType'
                shape = col_type.get('shape')
                assert shape is not None and len(shape) == 1, f'Expected 1-D shape for embedding col, got {shape!r}'
                vector_len = shape[0]
                if is_cockroachdb:
                    sa_col_type = {'type': 'Vector', 'dim': vector_len}
                else:
                    # postgresql
                    precision = embedding_val_col_precisions[col_id]
                    sa_col_type = (
                        {'type': 'HalfVec', 'dim': vector_len}
                        if precision == 'fp16'
                        else {'type': 'Vector', 'dim': vector_len}
                    )
            else:
                sa_col_type = _CLASSNAME_TO_SA_TYPE[classname]
        table_col_md['sa_col_type'] = sa_col_type

        # For each column in table md, find the schema versions in which those columns were visible, and update
        # SchemaVersionMds to add or change those columns.
        for schema_ver, schema_version_md in schema_versions.items():
            if schema_ver < table_col_md['schema_version_add']:
                continue
            if schema_ver >= (table_col_md.get('schema_version_drop') or Table.MAX_VERSION):
                continue
            # Update user-visible columns and add system columns
            col = schema_version_md['columns'].setdefault(col_id, {'pos': None, 'name': None, 'media_validation': None})
            for field, is_required in _COL_FIELDS_TO_MOVE.items():
                assert field in table_col_md or not is_required, (field, table_col_md)
                if field in table_col_md:
                    col[field] = table_col_md[field]
        # Finally, remove the moved fields from the source table md
        for field in _COL_FIELDS_TO_MOVE:
            table_col_md.pop(field, None)
