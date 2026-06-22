import dataclasses
import importlib
import logging
import os
import pkgutil
from typing import Callable

import sqlalchemy as sql
from sqlalchemy import orm

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable.utils.console_output import ConsoleLogger

from .schema import SystemInfo, SystemInfoMd

_logger = logging.getLogger(__name__)
_console_logger = ConsoleLogger(_logger)

# current version of the metadata; this is incremented whenever the metadata schema changes
VERSION = 53


def create_system_info(engine: sql.engine.Engine) -> None:
    """Create the system metadata record"""
    system_md = SystemInfoMd(schema_version=VERSION)
    record = SystemInfo(md=dataclasses.asdict(system_md))
    _logger.debug(f'Creating pixeltable system info record {record}')
    with orm.Session(engine, future=True) as session:
        # Write system metadata only once for idempotency
        if session.query(SystemInfo).count() == 0:
            session.add(record)
            session.flush()
            session.commit()


# conversion functions for upgrading the metadata schema from one version to the following
# key: old schema version
converter_cbs: dict[int, Callable[[sql.Connection], None]] = {}


def register_converter(version: int) -> Callable[[Callable[[sql.Connection], None]], None]:
    def decorator(fn: Callable[[sql.Connection], None]) -> None:
        assert version not in converter_cbs
        converter_cbs[version] = fn

    return decorator


# load all converter modules
for _, modname, _ in pkgutil.iter_modules([os.path.dirname(__file__) + '/converters']):
    importlib.import_module('pixeltable.metadata.converters.' + modname)


def upgrade_md(engine: sql.engine.Engine) -> None:
    """Upgrade the metadata schema to the current version.

    Each step runs its converter and the matching schema_version bump in a single transaction on one
    connection, so an interrupted or failing upgrade leaves the database at the last fully-applied version
    with no partially converted data; converters therefore need not be idempotent. An exclusive lock on the
    SystemInfo row serializes concurrent upgraders.
    """
    while True:
        with orm.Session(engine) as session:
            # exclusive lock on the SystemInfo row, held for the duration of this single step
            md_version = session.query(SystemInfo).with_for_update().one().md['schema_version']
            assert isinstance(md_version, int)
            if md_version > VERSION:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_CONFIGURATION,
                    'This Pixeltable database was created with a newer Pixeltable version '
                    f'than the one currently installed ({pxt.__version__}).\n'
                    'Please update to the latest Pixeltable version by running: pip install --upgrade pixeltable',
                )
            if md_version == VERSION:
                return
            if md_version not in converter_cbs:
                raise excs.Error(excs.ErrorCode.INTERNAL_ERROR, f'No metadata converter for version {md_version}')
            # We can't use the console logger in Env, because Env might not have been initialized yet.
            _console_logger.info(f'Converting metadata from version {md_version} to {md_version + 1}')
            # Run the converter and the version bump on the session's connection so they commit atomically.
            conn = session.connection()
            converter_cbs[md_version](conn)
            system_info_md = SystemInfoMd(schema_version=md_version + 1)
            conn.execute(SystemInfo.__table__.update().values(md=dataclasses.asdict(system_info_md)))
            session.commit()
