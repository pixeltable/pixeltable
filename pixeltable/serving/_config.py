"""The database configuration a project supplies in its pixeltable.toml."""

from __future__ import annotations

from pixeltable import catalog, config
from pixeltable.config import LOCAL_DATABASE, DatabaseConfig

__all__ = ['DatabaseConfig', 'database_config_for', 'lookup_database_config']


def lookup_database_config(name: str = LOCAL_DATABASE) -> DatabaseConfig | None:
    """The [[pixeltable.database]] entry for the named database, or None if no entry names it."""
    databases = config.Config.get().get_value('database', list)
    if databases is None:
        return None
    return next((db for db in databases if db.name == name), None)


def database_config_for(target: str) -> DatabaseConfig | None:
    """The entry that applies to target's database; None if the project has no entry for it."""
    path = catalog.Path.parse(target, allow_empty_path=True)
    return lookup_database_config(LOCAL_DATABASE if path.is_local else path.catalog_uri.uri)
