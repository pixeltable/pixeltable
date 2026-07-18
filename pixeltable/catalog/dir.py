from __future__ import annotations

import dataclasses
from uuid import UUID

from pixeltable import telemetry
from pixeltable.env import Env
from pixeltable.metadata import schema
from pixeltable.runtime import get_runtime

from .path import ROOT_PATH, Path
from .schema_object import SchemaObject


class Dir(SchemaObject):
    def __init__(self, id: UUID):
        super().__init__(id)

    @classmethod
    @telemetry.spanned('pixeltable.catalog.create_dir', level=telemetry.DEBUG)
    def _create(cls, parent_id: UUID, name: str) -> Dir:
        session = get_runtime().session
        user = Env.get().user
        assert session is not None
        dir_md = schema.DirMd(name=name, user=user, additional_md={})
        dir_record = schema.Dir(parent_id=parent_id, md=dataclasses.asdict(dir_md))
        session.add(dir_record)
        session.flush()
        assert dir_record.id is not None
        assert isinstance(dir_record.id, UUID)
        return cls(dir_record.id)

    def _display_name(self) -> str:
        return 'directory'

    def _name(self) -> str:
        cat = get_runtime().catalog
        with cat.begin_xact(for_write=False):
            return cat.read_dir_record(self._id).md['name']

    def _dir_id(self) -> UUID | None:
        cat = get_runtime().catalog
        with cat.begin_xact(for_write=False):
            return cat.read_dir_record(self._id).parent_id

    def _path(self) -> Path:
        """Returns the path to this schema object."""
        if self._dir_id() is None:
            # we're the root dir
            return ROOT_PATH
        return super()._path()
