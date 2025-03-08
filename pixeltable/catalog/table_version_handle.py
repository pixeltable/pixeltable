from __future__ import annotations

import dataclasses
import importlib
import inspect
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Literal, Optional
from uuid import UUID

import jsonschema.exceptions
import sqlalchemy as sql
import sqlalchemy.orm as orm

import pixeltable as pxt
import pixeltable.exceptions as excs
import pixeltable.exprs as exprs
import pixeltable.func as func
import pixeltable.index as index
import pixeltable.type_system as ts
from pixeltable.env import Env
from pixeltable.iterators import ComponentIterator
from pixeltable.metadata import schema
from pixeltable.utils.filecache import FileCache
from pixeltable.utils.media_store import MediaStore

from ..func.globals import resolve_symbol
from .column import Column
from .globals import _POS_COLUMN_NAME, _ROWID_COLUMN_NAME, MediaValidation, UpdateStatus, is_valid_identifier
from .table_version import TableVersion

if TYPE_CHECKING:
    from pixeltable import exec, store

_logger = logging.getLogger('pixeltable')


class TableVersionHandle:
    id: UUID
    effective_version: Optional[int]
    _tbl_version: Optional[TableVersion]

    def __init__(self, tbl_id: UUID, effective_version: Optional[int], tbl_version: Optional[TableVersion] = None):
        self.id = tbl_id
        self.effective_version = effective_version
        self._tbl_version = tbl_version

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TableVersionHandle):
            return False
        return self.id == other.id and self.effective_version == other.effective_version

    @classmethod
    def create(cls, tbl_version: TableVersion) -> TableVersionHandle:
        return cls(tbl_version.id, tbl_version.effective_version, tbl_version)

    def get(self) -> TableVersion:
        from .catalog import Catalog

        if self._tbl_version is None:
            self._tbl_version = Catalog.get().get_tbl_version(self.id, self.effective_version)
        return self._tbl_version

    def as_dict(self) -> dict:
        return {'id': str(self.id), 'effective_version': self.effective_version}

    @classmethod
    def from_dict(cls, d: dict) -> TableVersionHandle:
        return cls(UUID(d['id']), d['effective_version'])
