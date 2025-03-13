import dataclasses
import importlib
import os
import pkgutil
from typing import Callable

import sqlalchemy as sql
from sqlalchemy import orm

from .schema import SystemInfo, SystemInfoMd

# current version of the metadata; this is incremented whenever the metadata schema changes
VERSION = 30


def create_system_info(engine: sql.engine.Engine) -> None:
    """Create the system metadata record"""
    system_md = SystemInfoMd(schema_version=VERSION)
    record = SystemInfo(md=dataclasses.asdict(system_md))
    with orm.Session(engine, future=True) as session:
        session.add(record)
        session.flush()
        session.commit()


# conversion functions for upgrading the metadata schema from one version to the following
# key: old schema version
converter_cbs: dict[int, Callable[[sql.engine.Engine], None]] = {}


def register_converter(version: int) -> Callable[[Callable[[sql.engine.Engine], None]], None]:
    def decorator(fn: Callable[[sql.engine.Engine], None]) -> None:
        assert version not in converter_cbs
        converter_cbs[version] = fn

    return decorator


# load all converter modules
for _, modname, _ in pkgutil.iter_modules([os.path.dirname(__file__) + '/converters']):
    importlib.import_module('pixeltable.metadata.converters.' + modname)


def upgrade_md(engine: sql.engine.Engine) -> None:
    """Upgrade the metadata schema to the current version"""
    with orm.Session(engine) as session:
        system_info = session.query(SystemInfo).one().md
        md_version = system_info['schema_version']
        assert isinstance(md_version, int)
        if md_version == VERSION:
            return
        while md_version < VERSION:
            if md_version not in converter_cbs:
                raise RuntimeError(f'No metadata converter for version {md_version}')
            from pixeltable.env import Env

            Env.get().console_logger.info(f'Converting metadata from version {md_version} to {md_version + 1}')
            converter_cbs[md_version](engine)
            md_version += 1
        # update system info
        conn = session.connection()
        system_info_md = SystemInfoMd(schema_version=VERSION)
        conn.execute(SystemInfo.__table__.update().values(md=dataclasses.asdict(system_info_md)))
        session.commit()
