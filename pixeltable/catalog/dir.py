from __future__ import annotations

import dataclasses
import logging
from uuid import UUID

import sqlalchemy as sql

from pixeltable.env import Env
from pixeltable.metadata import schema

from .schema_object import SchemaObject

_logger = logging.getLogger('pixeltable')


class Dir(SchemaObject):
    def __init__(self, id: UUID, parent_id: UUID, name: str):
        super().__init__(id, name, parent_id)

    @classmethod
    def _create(cls, parent_id: UUID, name: str) -> Dir:
        session = Env.get().session
        assert session is not None
        dir_md = schema.DirMd(name=name)
        dir_record = schema.Dir(parent_id=parent_id, md=dataclasses.asdict(dir_md))
        session.add(dir_record)
        session.flush()
        assert dir_record.id is not None
        assert isinstance(dir_record.id, UUID)
        dir = cls(dir_record.id, parent_id, name)
        return dir

    @classmethod
    def _display_name(cls) -> str:
        return 'directory'

    @property
    def _has_dependents(self) -> bool:
        """Returns True if this directory has any children."""
        from pixeltable.catalog import Catalog, Path

        return len(Catalog.get().get_dir_contents(self._id, recursive=False)) > 0

    def _move(self, new_name: str, new_dir_id: UUID) -> None:
        super()._move(new_name, new_dir_id)
        with Env.get().engine.begin() as conn:
            dir_md = schema.DirMd(name=new_name)
            conn.execute(
                sql.update(schema.Dir.__table__)
                .values({schema.Dir.parent_id: self._dir_id, schema.Dir.md: dataclasses.asdict(dir_md)})
                .where(schema.Dir.id == self._id)
            )
