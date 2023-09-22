from __future__ import annotations

import dataclasses
import json
import logging
from uuid import UUID
import time

import sqlalchemy as sql
import sqlalchemy.orm as orm

from .table import Table
from .table_version import TableVersion
from pixeltable.env import Env
from pixeltable.metadata import schema


_logger = logging.getLogger('pixeltable')

class TableSnapshot(Table):
    """An immutable :py:class:`Table`."""
    def __init__(self, id: UUID, dir_id: UUID, name: str, tbl_version: TableVersion):
        super().__init__(id, dir_id, name, tbl_version)

    @classmethod
    def create(cls, dir_id: UUID, name: str, tbl_version: TableVersion) -> TableSnapshot:
        with orm.Session(Env.get().engine, future=True) as session:
            snapshot_md = schema.TableSnapshotMd(name=name, created_at=time.time())
            snapshot_record = schema.TableSnapshot(
                dir_id=dir_id, tbl_id=tbl_version.id, tbl_version=tbl_version.version,
                md=dataclasses.asdict(snapshot_md))
            session.add(snapshot_record)
            session.flush()
            assert snapshot_record.id is not None
            snapshot = TableSnapshot(snapshot_record.id, dir_id, name, tbl_version)
            session.commit()
            return snapshot

    @classmethod
    def display_name(cls) -> str:
        return 'table snapshot'

    def move(self, new_name: str, new_dir_id: UUID) -> None:
        super().move(new_name, new_dir_id)
        with Env.get().engine.begin() as conn:
            stmt = sql.text((
                f"UPDATE {schema.TableSnapshot.__table__} "
                f"SET {schema.TableSnapshot.dir_id.name} = :new_dir_id, "
                f"    {schema.TableSnapshot.md.name}['name'] = :new_name "
                f"WHERE {schema.TableSnapshot.id.name} = :id"))
            conn.execute(stmt, {'new_dir_id': new_dir_id, 'new_name': json.dumps(new_name), 'id': self.id})

