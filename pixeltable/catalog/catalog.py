from __future__ import annotations
from typing import Optional, List, Any, Dict, Tuple
from uuid import UUID
import dataclasses
import logging

import sqlalchemy as sql
import sqlalchemy.orm as orm

from .table_version import TableVersion
from .table_version_path import TableVersionPath
from .table import Table
from .named_function import NamedFunction
from .path_dict import PathDict
import pixeltable.env as env
import pixeltable.metadata.schema as schema

_logger = logging.getLogger('pixeltable')

class Catalog:
    """A repository of catalog objects"""
    _instance: Optional[Catalog] = None

    @classmethod
    def get(cls) -> Catalog:
        if cls._instance is None:
            cls._instance = cls()
            with orm.Session(env.Env.get().engine, future=True) as session:
                cls._instance._load_table_versions(session)
                #cls._instance._load_functions(session)
        return cls._instance

    @classmethod
    def clear(cls) -> None:
        """Remove the instance. Used for testing."""
        cls._instance = None

    def __init__(self) -> None:
        # key: [id, version]
        # - mutable version of a table: version == None (even though TableVersion.version is set correctly)
        # - snapshot versions: records the version of the snapshot
        self.tbl_versions: Dict[Tuple[UUID, int], TableVersion] = {}

        self.tbls: Dict[UUID, Table] = {}  # don't use a defaultdict here, it doesn't cooperate with the debugger
        self.tbl_dependents: Dict[UUID, List[Table]] = {}

        self._init_store()
        self.paths = PathDict()  # do this after _init_catalog()

    def _init_store(self) -> None:
        """One-time initialization of the stored catalog. Idempotent."""
        with orm.Session(env.Env.get().engine, future=True) as session:
            if session.query(sql.func.count(schema.Dir.id)).scalar() > 0:
                return
            # create a top-level directory, so that every schema object has a directory
            dir_md = schema.DirMd(name='')
            dir_record = schema.Dir(parent_id=None, md=dataclasses.asdict(dir_md))
            session.add(dir_record)
            session.flush()
            session.commit()
            _logger.info(f'Initialized catalog')

    def _load_snapshot_version(
            self, tbl_id: UUID, version: int, base: Optional[TableVersion], session: orm.Session
    ) -> TableVersion:
        q = session.query(schema.Table, schema.TableSchemaVersion) \
            .select_from(schema.Table) \
            .join(schema.TableVersion) \
            .join(schema.TableSchemaVersion) \
            .where(schema.Table.id == tbl_id) \
            .where(sql.text(f"({schema.TableVersion.__table__}.md->>'version')::int = {version}")) \
            .where(sql.text((
                f"({schema.TableVersion.__table__}.md->>'schema_version')::int = "
                f"{schema.TableSchemaVersion.__table__}.{schema.TableSchemaVersion.schema_version.name}")))
        tbl_record, schema_version_record = q.one()
        tbl_md = schema.md_from_dict(schema.TableMd, tbl_record.md)
        schema_version_md = schema.md_from_dict(schema.TableSchemaVersionMd, schema_version_record.md)
        # we ignore tbl_record.base_tbl_id/base_snapshot_id and use 'base' instead: if the base is a snapshot
        # we'd have to look that up first
        return TableVersion(tbl_record.id, tbl_md, version, schema_version_md, is_snapshot=True, base=base)

    def _load_table_versions(self, session: orm.Session) -> None:
        from .insertable_table import InsertableTable
        from .view import View

        # load tables/views;
        # do this in ascending order of creation ts so that we can resolve base references in one pass
        q = session.query(schema.Table, schema.TableSchemaVersion) \
            .select_from(schema.Table) \
            .join(schema.TableVersion) \
            .join(schema.TableSchemaVersion) \
            .where(sql.text(f"({schema.TableVersion.__table__}.md->>'version')::int = 0")) \
            .where(sql.text((
                f"({schema.Table.__table__}.md->>'current_schema_version')::int = "
                f"{schema.TableSchemaVersion.__table__}.{schema.TableSchemaVersion.schema_version.name}"))) \
            .order_by(sql.text(f"({schema.TableVersion.__table__}.md->>'created_at')::float"))

        for tbl_record, schema_version_record in q.all():
            tbl_md = schema.md_from_dict(schema.TableMd, tbl_record.md)
            schema_version_md = schema.md_from_dict(schema.TableSchemaVersionMd, schema_version_record.md)
            view_md = tbl_md.view_md

            if view_md is not None:
                assert len(view_md.base_versions) > 0
                # construct a TableVersionPath for the view
                refd_versions = [(UUID(tbl_id), version) for tbl_id, version in view_md.base_versions]
                base_path: Optional[TableVersionPath] = None
                base: Optional[TableVersion] = None
                # go through the versions in reverse order, so we can construct TableVersionPaths
                for base_id, version in refd_versions[::-1]:
                    base_version = self.tbl_versions.get((base_id, version), None)
                    if base_version is None:
                        if version is None:
                            # debugging
                            pass
                        # if this is a reference to a mutable table, we should have loaded it already
                        assert version is not None
                        base_version = self._load_snapshot_version(base_id, version, base, session)
                    base_path = TableVersionPath(base_version, base=base_path)
                    base = base_version
                assert base_path is not None

                base_tbl = self.tbls[base_path.tbl_version.id]
                is_snapshot = view_md is not None and view_md.is_snapshot
                snapshot_only = is_snapshot and view_md.predicate is None and len(schema_version_md.columns) == 0
                if snapshot_only:
                    # this is a pure snapshot, without a physical table backing it
                    view_path = base_path
                else:
                    tbl_version = TableVersion(
                        tbl_record.id, tbl_md, tbl_md.current_version, schema_version_md, is_snapshot=is_snapshot,
                        base=base_path.tbl_version if is_snapshot else None,
                        base_path=base_path if not is_snapshot else None)
                    view_path = TableVersionPath(tbl_version, base=base_path)

                tbl = View(
                    tbl_record.id, tbl_record.dir_id, tbl_md.name, view_path, base_tbl,
                    snapshot_only=snapshot_only)
                self.tbl_dependents[base_tbl._id].append(tbl)

            else:
                tbl_version = TableVersion(tbl_record.id, tbl_md, tbl_md.current_version, schema_version_md)
                tbl = InsertableTable(tbl_record.dir_id, tbl_version)

            self.tbls[tbl._id] = tbl
            self.tbl_dependents[tbl._id] = []
            self.paths.add_schema_obj(tbl._dir_id, tbl_md.name, tbl)

    # def _load_functions(self, session: orm.Session) -> None:
    #     # load Function metadata; doesn't load the actual callable, which can be large and is only done on-demand by the
    #     # FunctionRegistry
    #     q = session.query(schema.Function.id, schema.Function.dir_id, schema.Function.md) \
    #         .where(sql.text(f"({schema.Function.__table__}.md->>'name')::text IS NOT NULL"))
    #     for id, dir_id, md in q.all():
    #         assert 'name' in md
    #         name = md['name']
    #         assert name is not None
    #         named_fn = NamedFunction(id, dir_id, name)
    #         self.paths.add_schema_obj(dir_id, name, named_fn)
