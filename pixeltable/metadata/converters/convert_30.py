import copy
import uuid
from typing import Any

import sqlalchemy as sql
from sqlalchemy import orm
from sqlalchemy.dialects.postgresql import JSONB, UUID

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_sql_table_record, convert_table_md


@register_converter(version=30)
def _(engine: sql.engine.Engine) -> None:
    convert_table_md(engine, table_md_updater=__update_table_md)
    convert_sql_table_record(TableVersionAtV30, engine, record_updater=__update_table_version_record)
    convert_sql_table_record(TableSchemaVersionAtV30, engine, record_updater=__update_table_schema_version_record)


def __update_table_md(md: dict, tbl_id: uuid.UUID) -> None:
    md['tbl_id'] = str(tbl_id)


# We can't use the ORM declarations from pixeltable.metadata.schema, because those might have changed since the
# version being converted. So we include static declarations here of the tables as they existed at version 30.

Base: type = orm.declarative_base()


class TableVersionAtV30(Base):
    __tablename__ = 'tableversions'

    tbl_id: orm.Mapped[uuid.UUID] = orm.mapped_column(UUID(as_uuid=True), primary_key=True, nullable=False)
    version: orm.Mapped[int] = orm.mapped_column(sql.BigInteger, primary_key=True, nullable=False)
    md: orm.Mapped[dict[str, Any]] = orm.mapped_column(JSONB, nullable=False)


class TableSchemaVersionAtV30(Base):
    __tablename__ = 'tableschemaversions'

    tbl_id: orm.Mapped[uuid.UUID] = orm.mapped_column(UUID(as_uuid=True), primary_key=True, nullable=False)
    schema_version: orm.Mapped[int] = orm.mapped_column(sql.BigInteger, primary_key=True, nullable=False)
    md: orm.Mapped[dict[str, Any]] = orm.mapped_column(JSONB, nullable=False)  # TableSchemaVersionMd


def __update_table_version_record(record: TableVersionAtV30) -> None:
    """
    Update TableVersion with table_id.
    """
    assert isinstance(record.md, dict)
    md = copy.copy(record.md)
    md['tbl_id'] = str(record.tbl_id)
    record.md = md


def __update_table_schema_version_record(record: TableSchemaVersionAtV30) -> None:
    """
    Update TableSchemaVersion with table_id.
    """
    assert isinstance(record.md, dict)
    md = copy.copy(record.md)
    md['tbl_id'] = str(record.tbl_id)
    record.md = md
