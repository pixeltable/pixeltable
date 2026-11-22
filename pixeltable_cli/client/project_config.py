"""Utilities related to the project config"""

import json
import pathlib
import tomllib
from typing import Any

from fasteners import InterProcessLock  # type: ignore[import-untyped]

from pixeltable_cli.utils import (
    PROJECT_CONFIG_FILE,
    PYPROJECT_FILE,
    _resolve_pixeltable_home,
    find_project_root,
    hosted_db_uri_error,
    parse_toml,
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
    uri_error = hosted_db_uri_error(parts.org, parts.db)
    if uri_error is not None:
        raise RuntimeError(uri_error)
    entry_name = db_uri.lower()

    config_file = find_project_config(pathlib.Path.cwd())
    if config_file is None:
        raise RuntimeError("no project configuration here; run 'pxt init' to write one")
    with _config_lock():
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

        header = f'[[{key}.database]]'
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
                f'{config_file.name} declares {key}.database as {form}; rewrite it as an array of {header} tables and '
                'run again'
            ) from e
        try:
            with open(config_file, 'a', encoding='utf-8') as fp:
                fp.write(addition)
        except OSError as e:
            raise RuntimeError(f'{config_file} could not be written: {e}') from e
    return config_file


def get_option(config_file: pathlib.Path, section: str, key: str) -> Any:
    """Return the value of section.key in config_file, or None if config_file does not set it."""
    return _option(parse_toml(config_file), config_file, section, key) if config_file.is_file() else None


def add_option(config_file: pathlib.Path, section: str, key: str, value: str) -> bool:
    """Append a table that sets section.key to value to config_file, unless config_file already sets it.

    Returns False if the file already sets the option, if its layout has no place for the section, or if the file
    would not parse with the table appended. Concurrent calls in several processes append the table once.
    """
    with _config_lock():
        text = config_file.read_text(encoding='utf-8')
        parsed = tomllib.loads(text)
        section_key = _section_key(parsed, config_file, section)
        if section_key is None or _option(parsed, config_file, section, key) is not None:
            return False
        separator = '' if text == '' else '\n' if text.endswith('\n') else '\n\n'
        # json.dumps() of a str is a valid TOML basic string
        addition = f'{separator}[{section_key}]\n{key} = {json.dumps(value, ensure_ascii=False)}\n'
        try:
            tomllib.loads(text + addition)
        except tomllib.TOMLDecodeError:
            return False
        with open(config_file, 'a', encoding='utf-8') as fp:
            fp.write(addition)
    return True


def _option(parsed: dict[str, Any], config_file: pathlib.Path, section: str, key: str) -> Any:
    section_key = _section_key(parsed, config_file, section)
    table = _lookup(parsed, section_key) if section_key is not None else None
    return table.get(key) if isinstance(table, dict) else None


def _config_lock() -> InterProcessLock:
    """Return the lock that serializes a read, check and append of a project configuration across processes."""
    # in the Pixeltable home, because a file in the project would go into its archive
    return InterProcessLock(str(pathlib.Path(_resolve_pixeltable_home()) / 'project-config.lock'))


def _section_key(parsed: dict[str, Any], config_file: pathlib.Path, section: str) -> str | None:
    """Return the dotted key of section's table in config_file, or None if the file's layout has no place for it.

    Mirrors pixeltable.config.Config.__load_project_config(), which cannot be imported here: a pyproject.toml has the
    configuration under tool.pixeltable, which contains the sections only if it has a tool.pixeltable.pixeltable table,
    and the settings of the [pixeltable] section otherwise.
    """
    if config_file.name != PYPROJECT_FILE:
        return section
    tool = _lookup(parsed, 'tool.pixeltable')
    if isinstance(tool, dict) and isinstance(tool.get('pixeltable'), dict):
        return f'tool.pixeltable.{section}'
    return 'tool.pixeltable' if section == 'pixeltable' else None


def _lookup(parsed: dict[str, Any], dotted_key: str) -> Any:
    value: Any = parsed
    for part in dotted_key.split('.'):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def _database_option(parsed: dict[str, Any], config_file: pathlib.Path) -> tuple[str, Any]:
    """Return the dotted key of the [pixeltable] section in config_file and the value of its database option."""
    key = _section_key(parsed, config_file, 'pixeltable')
    # every layout has a place for the [pixeltable] section
    assert key is not None
    section = _lookup(parsed, key)
    return key, section.get('database') if isinstance(section, dict) else None
