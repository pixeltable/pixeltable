import copy

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import (
    convert_table_record,
    convert_table_schema_version_record,
    convert_table_version_record,
)
from pixeltable.metadata.schema import Table, TableSchemaVersion, TableVersion


@register_converter(version=30)
def _(engine: sql.engine.Engine) -> None:
    convert_table_record(engine, table_record_updater=__update_table_record)
    convert_table_version_record(engine, table_version_record_updater=__update_table_version_record)
    convert_table_schema_version_record(
        engine, table_schema_version_record_updater=__update_table_schema_version_record
    )


def __update_table_record(record: Table) -> None:
    """
    Update TableMd with table_id
    """
    assert isinstance(record.md, dict)
    md = copy.copy(record.md)
    md['tbl_id'] = str(record.id)
    record.md = md


def __update_table_version_record(record: TableVersion) -> None:
    """
    Update TableVersion with table_id.
    """
    assert isinstance(record.md, dict)
    md = copy.copy(record.md)
    md['tbl_id'] = str(record.tbl_id)
    record.md = md


def __update_table_schema_version_record(record: TableSchemaVersion) -> None:
    """
    Update TableSchemaVersion with table_id.
    """
    assert isinstance(record.md, dict)
    md = copy.copy(record.md)
    md['tbl_id'] = str(record.tbl_id)
    record.md = md
