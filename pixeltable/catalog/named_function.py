from __future__ import annotations

import json
import logging
from uuid import UUID

import sqlalchemy as sql

from .schema_object import SchemaObject
from pixeltable.env import Env
from pixeltable.metadata import schema


_logger = logging.getLogger('pixeltable')

class NamedFunction(SchemaObject):
    """
    Contains references to functions that are named and have a path.
    The Function itself is stored in the FunctionRegistry.
    """
    def __init__(self, id: UUID, dir_id: UUID, name: str):
        super().__init__(id, name, dir_id)

    @classmethod
    def display_name(cls) -> str:
        return 'function'

    def move(self, new_name: str, new_dir_id: UUID) -> None:
        super().move(new_name, new_dir_id)
        with Env.get().engine.begin() as conn:
            stmt = sql.text((
                f"UPDATE {schema.Function.__table__} "
                f"SET {schema.Function.dir_id.name} = :new_dir_id, {schema.Function.md.name}['name'] = :new_name "
                f"WHERE {schema.Function.id.name} = :id"))
            conn.execute(stmt, {'new_dir_id': new_dir_id, 'new_name': json.dumps(new_name), 'id': self._id})

