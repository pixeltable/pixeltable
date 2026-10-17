"""Utilities related to the project config"""

import pathlib
import tomllib
from typing import Any

from fasteners import InterProcessLock  # type: ignore[import-untyped]

from pixeltable_cli.utils import (
    PROJECT_CONFIG_FILE,
    PYPROJECT_FILE,
    _resolve_pixeltable_home,
    find_project_root,
    is_valid_identifier,
    split_pxt_uri,
)

# the local database's entry in a fresh project configuration
_DATABASE_ENTRY = """\
{header}
# The local database. vars.<name> sets the value of the ConfigVar called <name>, for example:
# vars.media_dest = 's3://bucket/prefix'
# A hosted database gets a second entry, with name = 'pxt://org:db'.
"""
_PROJECT_CONFIG = f"""\
# Pixeltable project configuration. This file's directory is the project root;
# every local module path is relative to it.

{_DATABASE_ENTRY.format(header='[[pixeltable.database]]')}"""
_PYPROJECT_CONFIG = f"""
{_DATABASE_ENTRY.format(header='[[tool.pixeltable.database]]')}"""


def find_project_config(start: pathlib.Path) -> pathlib.Path | None:
    """Return the configuration file of the nearest project root at or above start, or None outside a project."""
    root = find_project_root(start)
    if root is None:
        return None
    config_file = root / PROJECT_CONFIG_FILE
    return config_file if config_file.is_file() else root / PYPROJECT_FILE


def write_project_config(root: pathlib.Path) -> pathlib.Path:
    """Write the project configuration in root, and return its file."""
    pyproject = root / PYPROJECT_FILE
    try:
        if pyproject.is_file():
            try:
                with open(pyproject, 'rb') as fp:
                    tomllib.load(fp)
            except tomllib.TOMLDecodeError as e:
                raise RuntimeError(
                    f'{pyproject} is not valid TOML, so the project configuration cannot be added to it: {e}\n'
                    f'Fix the file, or remove it to configure the project with {PROJECT_CONFIG_FILE} instead.'
                ) from e
            with open(pyproject, 'a', encoding='utf-8') as fp:
                fp.write(_PYPROJECT_CONFIG)
            return pyproject
        config_file = root / PROJECT_CONFIG_FILE
        config_file.write_text(_PROJECT_CONFIG, encoding='utf-8')
        return config_file
    except OSError as e:
        raise RuntimeError(f'the project configuration in {root} could not be written: {e}') from e


def add_database_entry(db_uri: str) -> pathlib.Path | None:
    """Append a [[pixeltable.database]] entry named db_uri to the project configuration, and return its file.

    Returns None, leaving the file as it is, if an entry already matches db_uri. Entry names match in any case,
    as in pixeltable.config.DatabaseConfig. Concurrent calls in several processes append the entry once.
    """
    parts = split_pxt_uri(db_uri)
    assert parts is not None and parts.db is not None, db_uri
    for kind, name in (('organization', parts.org), ('database', parts.db)):
        if not is_valid_identifier(name, allow_hyphens=True):
            raise RuntimeError(f'invalid {kind} name: {name!r}')
    entry_name = db_uri.lower()

    config_file = find_project_config(pathlib.Path.cwd())
    if config_file is None:
        raise RuntimeError("no project configuration here; run 'pxt init' to write one")
    # serialize the read, check and append across processes; the lock file sits in the Pixeltable home
    # because a file in the project would go into its archive
    with InterProcessLock(str(pathlib.Path(_resolve_pixeltable_home()) / 'project-config.lock')):
        # find_project_root() already refused a file that does not parse
        text = config_file.read_text(encoding='utf-8')
        key, databases = _database_option(tomllib.loads(text), config_file)
        entries: list[Any] = []
        if isinstance(databases, list):
            entries = databases
        elif isinstance(databases, dict):  # a single [pixeltable.database] table is one entry
            entries = [databases]
        if any(
            isinstance(e, dict) and isinstance(e.get('name'), str) and e['name'].lower() == entry_name for e in entries
        ):
            return None

        header = f'[[{key}]]'
        separator = '' if text == '' else '\n' if text.endswith('\n') else '\n\n'
        addition = f"{separator}{header}\nname = '{entry_name}'\n"
        try:
            tomllib.loads(text + addition)
        except tomllib.TOMLDecodeError as e:
            # an appended header extends an array of tables, and conflicts with a single table or an inline array
            if isinstance(databases, dict):
                form = 'a single table'
            elif isinstance(databases, list):
                form = 'an inline array'
            else:
                raise RuntimeError(f'{config_file.name} does not parse with {header} appended: {e}') from e
            raise RuntimeError(
                f'{config_file.name} declares {key} as {form}; rewrite it as an array of {header} tables and run again'
            ) from e
        try:
            with open(config_file, 'a', encoding='utf-8') as fp:
                fp.write(addition)
        except OSError as e:
            raise RuntimeError(f'{config_file} could not be written: {e}') from e
    return config_file


def _database_option(parsed: dict[str, Any], config_file: pathlib.Path) -> tuple[str, Any]:
    """Return the dotted key and the value of the database option in parsed, the contents of config_file.

    Mirrors Config.__load_project_config() and Config.__validate_config(), which cannot be imported here.
    """
    if config_file.name != PYPROJECT_FILE:
        prefix, sections = '', parsed
    else:
        tool = parsed.get('tool', {}).get('pixeltable')
        # tool.pixeltable contains either the settings of the [pixeltable] section or the sections themselves
        if not (isinstance(tool, dict) and isinstance(tool.get('pixeltable'), dict)):
            return 'tool.pixeltable.database', tool.get('database') if isinstance(tool, dict) else None
        prefix, sections = 'tool.pixeltable.', tool
    # a database key outside any section moves into the [pixeltable] section, replacing its database option
    if 'database' in sections:
        return f'{prefix}database', sections['database']
    section = sections.get('pixeltable')
    return f'{prefix}pixeltable.database', section.get('database') if isinstance(section, dict) else None
