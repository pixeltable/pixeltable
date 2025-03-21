from __future__ import annotations

import dataclasses
import logging
from typing import Optional, Type
from uuid import UUID

import sqlalchemy as sql

import pixeltable.env as env
import pixeltable.exceptions as excs
import pixeltable.metadata.schema as schema
from pixeltable.env import Env

from .dir import Dir
from .schema_object import SchemaObject
from .table import Table
from .table_version import TableVersion
from .table_version_handle import TableVersionHandle
from .table_version_path import TableVersionPath

# from .. import InsertableTable

_logger = logging.getLogger('pixeltable')


def _join_path(path: str, name: str) -> str:
    """Append name to path, if path is not empty."""
    return name if path == '' else f'{path}.{name}'


class Catalog:
    """The functional interface to getting access to catalog objects

    All interface functions must be called in the context of a transaction, started with Env.begin().
    """

    _instance: Optional[Catalog] = None

    # key: [id, version]
    # - mutable version of a table: version == None (even though TableVersion.version is set correctly)
    # - snapshot versions: records the version of the snapshot
    _tbl_versions: dict[tuple[UUID, Optional[int]], TableVersion]
    _tbls: dict[UUID, Table]

    @classmethod
    def get(cls) -> Catalog:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def clear(cls) -> None:
        """Remove the instance. Used for testing."""
        cls._instance = None

    def __init__(self) -> None:
        self._tbl_versions = {}
        self._tbls = {}  # don't use a defaultdict here, it doesn't cooperate with the debugger
        self._init_store()

    def get_dir_path(self, dir_id: UUID) -> str:
        """Return path for directory with given id"""
        session = env.Env.get().session
        names: list[str] = []
        while True:
            dir = session.query(schema.Dir).filter(schema.Dir.id == dir_id).one()
            if dir.md['name'] == '':
                break
            names.insert(0, dir.md['name'])
            dir_id = dir.parent_id
            assert dir_id is not None
        return '.'.join(names)

    def get_tbl_path(self, tbl_id: UUID) -> str:
        """Return path for table with given id"""
        session = env.Env.get().session
        tbl = session.query(schema.Table).filter(schema.Table.id == tbl_id).one()
        dir_path = self.get_dir_path(tbl.dir_id)
        return _join_path(dir_path, tbl.md['name'])

    @dataclasses.dataclass
    class DirEntry:
        dir: Optional[schema.Dir]
        dir_entries: dict[str, Catalog.DirEntry]
        table: Optional[schema.Table]

    def get_dir_contents(self, dir_id: UUID, recursive: bool = False) -> dict[str, DirEntry]:
        """Returns a dict mapping the entry names to DirEntry objects"""
        session = env.Env.get().session
        result: dict[str, Catalog.DirEntry] = {}

        dirs = session.query(schema.Dir).filter(schema.Dir.parent_id == dir_id).all()
        for dir in dirs:
            dir_contents: dict[str, Catalog.DirEntry] = {}
            if recursive:
                dir_contents = self.get_dir_contents(dir.id, recursive=True)
            result[dir.md['name']] = self.DirEntry(dir=dir, dir_entries=dir_contents, table=None)

        tbls = session.query(schema.Table).filter(schema.Table.dir_id == dir_id).all()
        for tbl in tbls:
            result[tbl.md['name']] = self.DirEntry(dir=None, dir_entries={}, table=tbl)

        return result

    def drop_dir(self, dir_id: UUID) -> None:
        """Delete the directory with the given id"""
        session = env.Env.get().session
        session.query(schema.Dir).filter(schema.Dir.id == dir_id).delete()

    def get_schema_object(
        self,
        path: str,
        expected: Optional[Type[SchemaObject]] = None,
        raise_if_exists: bool = False,
        raise_if_not_exists: bool = False,
        raise_if_parent_missing: bool = True,
    ) -> Optional[SchemaObject]:
        """Return the schema object at the given path, or None if it doesn't exist.

        Raises Error if
        - raise_if_parent_missing is True and the parent directory doesn't exist'
        - raise_if_exists is True and the path exists
        - raise_if_not_exists is True and the path does not exist
        - expected is not None and the existing object has a different type
        """
        session = env.Env.get().session
        if path == '':
            # the root dir
            if expected is not None and expected is not Dir:
                raise excs.Error(f'{path!r} needs to be a {expected._display_name()} but is a {Dir._display_name()}')
            dir = self._get_dir(path)
            return Dir(dir.id, dir.parent_id, dir.md['name'])

        components = path.split('.')
        parent_path = '.'.join(components[:-1])
        parent_dir = self._get_dir('.'.join(components[:-1]))
        if parent_dir is None:
            if raise_if_parent_missing:
                raise excs.Error(f'Directory {parent_path!r} does not exist')
            return None
        name = components[-1]

        # check if path points to a directory
        obj: Optional[SchemaObject] = None
        dir = (
            session.query(schema.Dir)
            .filter(schema.Dir.parent_id == parent_dir.id, schema.Dir.md['name'].astext == name)
            .one_or_none()
        )
        if dir is not None:
            obj = Dir(dir.id, dir.parent_id, dir.md['name'])
        else:
            # check if it's a table
            row = (
                session.query(schema.Table.id)
                .filter(schema.Table.dir_id == parent_dir.id, schema.Table.md['name'].astext == name)
                .one_or_none()
            )
            if row is not None:
                tbl_id = row[0]
                if not tbl_id in self._tbls:
                    self._tbls[tbl_id] = self._load_tbl(tbl_id)
                obj = self._tbls[tbl_id]

        if obj is None and raise_if_not_exists:
            raise excs.Error(f'Path {path!r} does not exist')
        elif obj is not None and raise_if_exists:
            raise excs.Error(f'Path {path!r} is an existing {type(obj)._display_name()}')
        elif obj is not None and expected is not None and not isinstance(obj, expected):
            raise excs.Error(f'{path!r} needs to be a {expected._display_name()} but is a {type(obj)._display_name()}')
        return obj

    def get_tbl(self, tbl_id: UUID) -> Optional[Table]:
        if not tbl_id in self._tbls:
            tbl = self._load_tbl(tbl_id)
            if tbl is None:
                return None
            self._tbls[tbl_id] = tbl
        return self._tbls[tbl_id]

    def add_tbl(self, tbl: Table) -> None:
        """Explicitly add a Table"""
        self._tbls[tbl._id] = tbl

    def get_views(self, tbl_id: UUID) -> list[UUID]:
        """Return the ids of views that directly reference the given table"""
        session = env.Env.get().session
        q = session.query(schema.Table.id).filter(sql.text(f"md->'view_md'->'base_versions'->0->>0 = {tbl_id.hex!r}"))
        result = [r[0] for r in q.all()]
        return result

    def remove_tbl(self, tbl_id: UUID) -> None:
        assert tbl_id in self._tbls
        del self._tbls[tbl_id]

    def get_tbl_version(self, tbl_id: UUID, effective_version: Optional[int]) -> Optional[TableVersion]:
        if (tbl_id, effective_version) not in self._tbl_versions:
            self._tbl_versions[(tbl_id, effective_version)] = self._load_tbl_version(tbl_id, effective_version)
        return self._tbl_versions[(tbl_id, effective_version)]

    def add_tbl_version(self, tbl_version: TableVersion) -> None:
        """Explicitly add a TableVersion"""
        self._tbl_versions[(tbl_version.id, tbl_version.effective_version)] = tbl_version
        # if this is a mutable view, also record it in the base
        if tbl_version.is_view and tbl_version.effective_version is None:
            base = tbl_version.base.get()
            base.mutable_views.append(TableVersionHandle(tbl_version.id, tbl_version.effective_version))

    def remove_tbl_version(self, tbl_version: TableVersion) -> None:
        assert (tbl_version.id, tbl_version.effective_version) in self._tbl_versions
        del self._tbl_versions[(tbl_version.id, tbl_version.effective_version)]

    def get_dir(self, dir_id: UUID) -> Optional[Dir]:
        """Return the Dir with the given id, or None if it doesn't exist"""
        session = env.Env.get().session
        dir_record = session.query(schema.Dir).filter(schema.Dir.id == dir_id).one_or_none()
        if dir_record is None:
            return None
        return Dir(dir_record.id, dir_record.parent_id, dir_record.md['name'])

    def _get_dir(self, path: str) -> Optional[schema.Dir]:
        session = env.Env.get().session
        assert session is not None
        if path == '':
            return session.query(schema.Dir).filter(schema.Dir.parent_id.is_(None)).one()
        else:
            components = path.split('.')
            parent_path = '.'.join(components[:-1])
            parent_dir = self._get_dir(parent_path)
            if parent_dir is None:
                return None
            name = components[-1]
            dir = (
                session.query(schema.Dir)
                .filter(schema.Dir.parent_id == parent_dir.id, schema.Dir.md['name'].astext == name)
                .one_or_none()
            )
            return dir

    def _load_tbl(self, tbl_id: UUID) -> Optional[Table]:
        _logger.info(f'Loading table {tbl_id}')
        from .insertable_table import InsertableTable
        from .view import View

        session = env.Env.get().session
        tbl_record, schema_version_record = (
            session.query(schema.Table, schema.TableSchemaVersion)
            .join(schema.TableSchemaVersion)
            .where(schema.Table.id == schema.TableSchemaVersion.tbl_id)
            # Table.md['current_schema_version'] == TableSchemaVersion.schema_version
            .where(
                sql.text(
                    f"({schema.Table.__table__}.md->>'current_schema_version')::int = "
                    f'{schema.TableSchemaVersion.__table__}.{schema.TableSchemaVersion.schema_version.name}'
                )
            )
            .where(schema.Table.id == tbl_id)
            .one_or_none()
        )
        if tbl_record is None:
            return None

        tbl_md = schema.md_from_dict(schema.TableMd, tbl_record.md)
        view_md = tbl_md.view_md
        if view_md is None:
            # this is a base table
            if (tbl_id, None) not in self._tbl_versions:
                self._tbl_versions[(tbl_id, None)] = self._load_tbl_version(tbl_id, None)
            tbl = InsertableTable(tbl_record.dir_id, TableVersionHandle(tbl_id, None))
            return tbl

        # this is a view; determine the sequence of TableVersions to load
        tbl_version_path: list[tuple[UUID, Optional[int]]] = []
        schema_version_md = schema.md_from_dict(schema.TableSchemaVersionMd, schema_version_record.md)
        pure_snapshot = view_md.is_snapshot and view_md.predicate is None and len(schema_version_md.columns) == 0
        if pure_snapshot:
            # this is a pure snapshot, without a physical table backing it; we only need the bases
            pass
        else:
            effective_version = 0 if view_md.is_snapshot else None  # snapshots only have version 0
            tbl_version_path.append((tbl_id, effective_version))
        tbl_version_path.extend((UUID(tbl_id), version) for tbl_id, version in view_md.base_versions)

        # load TableVersions, starting at the root
        base_path: Optional[TableVersionPath] = None
        view_path: Optional[TableVersionPath] = None
        for id, effective_version in tbl_version_path[::-1]:
            if (id, effective_version) not in self._tbl_versions:
                self._tbl_versions[(id, effective_version)] = self._load_tbl_version(id, effective_version)
            view_path = TableVersionPath(TableVersionHandle(id, effective_version), base=base_path)
            base_path = view_path
        view = View(tbl_id, tbl_record.dir_id, tbl_md.name, view_path, snapshot_only=pure_snapshot)
        # TODO: also load mutable views
        return view

    def _load_tbl_version(self, tbl_id: UUID, effective_version: Optional[int]) -> Optional[TableVersion]:
        _logger.info(f'Loading table version: {tbl_id}:{effective_version}')
        session = env.Env.get().session
        q = (
            session.query(schema.Table, schema.TableSchemaVersion)
            .select_from(schema.Table)
            .where(schema.Table.id == tbl_id)
            .join(schema.TableSchemaVersion)
            .where(schema.TableSchemaVersion.tbl_id == tbl_id)
        )

        if effective_version is not None:
            # we are loading a specific version
            # SELECT *
            # FROM Table t
            # JOIN TableVersion tv ON (tv.tbl_id = tbl_id AND tv.version = effective_version)
            # JOIN TableSchemaVersion tsv ON (tsv.tbl_id = tbl_id AND tv.md.schema_version = tsv.schema_version)
            # WHERE t.id = tbl_id
            q = (
                q.join(schema.TableVersion)
                .where(schema.TableVersion.tbl_id == tbl_id)
                .where(sql.text(f"({schema.TableVersion.__table__}.md->>'version')::int = {effective_version}"))
                .where(
                    sql.text(
                        (
                            f"({schema.TableVersion.__table__}.md->>'schema_version')::int = "
                            f'{schema.TableSchemaVersion.__table__}.{schema.TableSchemaVersion.schema_version.name}'
                        )
                    )
                )
            )
        else:
            # we are loading the current version
            # SELECT *
            # FROM Table t
            # JOIN TableSchemaVersion tsv ON (tsv.tbl_id = tbl_id AND t.current_schema_version = tsv.schema_version)
            # WHERE t.id = tbl_id
            q = q.where(
                sql.text(
                    (
                        f"({schema.Table.__table__}.md->>'current_schema_version')::int = "
                        f'{schema.TableSchemaVersion.__table__}.{schema.TableSchemaVersion.schema_version.name}'
                    )
                )
            )

        tbl_record, schema_version_record = q.one_or_none()
        tbl_md = schema.md_from_dict(schema.TableMd, tbl_record.md)
        schema_version_md = schema.md_from_dict(schema.TableSchemaVersionMd, schema_version_record.md)
        view_md = tbl_md.view_md

        # load mutable view ids
        q = session.query(schema.Table.id).filter(
            sql.text(
                f"md->'view_md'->'base_versions'->0->>0 = {tbl_id.hex!r} "
                "AND md->'view_md'->'base_versions'->0->1 IS NULL"
            )
        )
        mutable_view_ids = [r[0] for r in q.all()]
        mutable_views = [TableVersionHandle(id, None) for id in mutable_view_ids]

        if view_md is None:
            # this is a base table
            tbl_version = TableVersion(
                tbl_record.id, tbl_md, effective_version, schema_version_md, mutable_views=mutable_views
            )
            return tbl_version

        assert len(view_md.base_versions) > 0  # a view needs to have a base
        pure_snapshot = view_md.is_snapshot and view_md.predicate is None and len(schema_version_md.columns) == 0
        assert not pure_snapshot  # a pure snapshot doesn't have a physical table backing it, no point in loading it

        base: TableVersionHandle
        base_path: Optional[TableVersionPath] = None  # needed for live view
        if view_md.is_snapshot:
            base = TableVersionHandle(UUID(view_md.base_versions[0][0]), view_md.base_versions[0][1])
        else:
            base_path = TableVersionPath.from_md(tbl_md.view_md.base_versions)
            base = base_path.tbl_version

        tbl_version = TableVersion(
            tbl_record.id,
            tbl_md,
            effective_version,
            schema_version_md,
            base_path=base_path,
            base=base,
            mutable_views=mutable_views,
        )
        return tbl_version

    def _init_store(self) -> None:
        """One-time initialization of the stored catalog. Idempotent."""
        with env.Env.get().begin_xact():
            session = env.Env.get().session
            if session.query(sql.func.count(schema.Dir.id)).scalar() > 0:
                return
            # create a top-level directory, so that every schema object has a directory
            dir_md = schema.DirMd(name='', user=None, additional_md={})
            dir_record = schema.Dir(parent_id=None, md=dataclasses.asdict(dir_md))
            session.add(dir_record)
            session.flush()
            session.commit()
            _logger.info(f'Initialized catalog')
