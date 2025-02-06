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
    tbl_id: UUID
    version: Optional[int]
    tbl_version: Optional[TableVersion]

    def __init__(self, tbl_id: UUID, version: Optional[int] = None):
        self.tbl_id = tbl_id
        self.version = version
        self.tbl_version = None

    def get(self) -> TableVersion:
        return self.tbl_version
