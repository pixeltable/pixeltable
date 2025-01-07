from __future__ import annotations

import dataclasses
import logging
from uuid import UUID

import sqlalchemy as sql

from .schema_object import SchemaObject
from pixeltable.env import Env
from pixeltable.metadata import schema


_logger = logging.getLogger('pixeltable')


class Dir(SchemaObject):
    def __init__(self, id: UUID, parent_id: UUID, name: str):
        super().__init__(id, name, parent_id)

    @classmethod
    def _display_name(cls) -> str:
        return 'directory'

    @property
    def _has_dependents(self) -> bool:
        """Returns True if this directory has any children."""
        from pixeltable.catalog import Catalog, Path

        return len(Catalog.get().paths.get_children(Path(self._path), child_type=None, recursive=False)) > 0

    def _move(self, new_name: str, new_dir_id: UUID) -> None:
        super()._move(new_name, new_dir_id)
        with Env.get().engine.begin() as conn:
            dir_md = schema.DirMd(name=new_name)
            conn.execute(
                sql.update(schema.Dir.__table__)
                .values({schema.Dir.parent_id: self._dir_id, schema.Dir.md: dataclasses.asdict(dir_md)})
                .where(schema.Dir.id == self._id)
            )
