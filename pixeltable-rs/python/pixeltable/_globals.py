"""The module-level API (`pxt.create_dir()`, `pxt.create_table()`, ...): argument forwarding to the backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Literal, Mapping
from uuid import UUID

import pandas as pd

from pixeltable import _core
from pixeltable._table import Dir, Table, _make_table
from pixeltable._unimplemented import unimplemented
from pixeltable.types import DirContents, TreeNode


def init(config_overrides: dict[str, Any] | None = None) -> None:
    """Initializes the Pixeltable environment (idempotent)."""
    if config_overrides:
        raise NotImplementedError('init() with config_overrides is not implemented yet in pixeltable-rs')
    _core.init()


def home() -> Path:
    """The Pixeltable home directory (`$PIXELTABLE_HOME`, default `~/.pixeltable`)."""
    return Path(_core.home())


def create_table(
    path: str,
    schema: Mapping[str, Any] | None = None,
    *,
    source: Any = None,
    source_format: Literal['csv', 'excel', 'parquet', 'json'] | None = None,
    schema_overrides: dict[str, Any] | None = None,
    has_default_idxs: bool = False,
    on_error: Literal['abort', 'ignore'] = 'abort',
    primary_key: str | list[str] | None = None,
    comment: str | None = None,
    custom_metadata: Any = None,
    media_validation: Literal['on_read', 'on_write'] = 'on_write',
    if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error',
    extra_args: dict[str, Any] | None = None,
    _is_data_versioned: bool = True,
) -> Table:
    """Create a new base table from a `schema` (a mapping of column names to types or column specs), or from a
    `source` (rows, a table or a query) whose schema is inferred."""
    unsupported = [
        name
        for name, given in (
            ('source_format', source_format is not None),
            ('extra_args', extra_args is not None),
        )
        if given
    ]
    if len(unsupported) > 0:
        raise NotImplementedError(f'create_table() with {", ".join(unsupported)} is not implemented yet in pixeltable-rs')
    tbl_id, kind, _ = _core.create_table(
        path,
        schema,
        source,
        primary_key,
        comment,
        custom_metadata,
        media_validation,
        if_exists,
        _is_data_versioned,
        schema_overrides,
        on_error,
        has_default_idxs,
    )
    return _make_table(tbl_id, kind)


def create_view(
    path: str,
    base: Any,
    *,
    additional_columns: Mapping[str, Any] | None = None,
    is_snapshot: bool = False,
    has_default_idxs: bool = False,
    iterator: Any = None,
    comment: str | None = None,
    custom_metadata: Any = None,
    media_validation: Literal['on_read', 'on_write'] = 'on_write',
    if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error',
) -> Table | None:
    """Create a view (or, with `is_snapshot=True`, a snapshot) of a table or a query over a table."""
    if has_default_idxs:
        raise NotImplementedError('create_view() with has_default_idxs is not implemented yet in pixeltable-rs')
    tbl_id, kind, _ = _core.create_view(
        path,
        base,
        additional_columns,
        is_snapshot,
        iterator,
        comment,
        custom_metadata,
        media_validation,
        if_exists,
    )
    return _make_table(tbl_id, kind)


def create_snapshot(
    path_str: str,
    base: Any,
    *,
    additional_columns: Mapping[str, Any] | None = None,
    iterator: Any = None,
    comment: str | None = None,
    custom_metadata: Any = None,
    media_validation: Literal['on_read', 'on_write'] = 'on_write',
    if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error',
) -> Table | None:
    """Create a snapshot of a table or a query over a table."""
    return create_view(
        path_str,
        base,
        additional_columns=additional_columns,
        iterator=iterator,
        is_snapshot=True,
        comment=comment,
        custom_metadata=custom_metadata,
        media_validation=media_validation,
        if_exists=if_exists,
    )


def get_table(path: str, if_not_exists: Literal['error', 'ignore'] = 'error') -> Table | None:
    """Get a handle to an existing table, view or snapshot (`'t:3'` pins version 3)."""
    result = _core.get_table(path, if_not_exists)
    if result is None:
        return None
    tbl_id, kind, version = result
    return _make_table(tbl_id, kind, version)


def move(
    path: str,
    new_path: str,
    *,
    if_exists: Literal['error', 'ignore'] = 'error',
    if_not_exists: Literal['error', 'ignore'] = 'error',
) -> None:
    """Move and/or rename a table or directory."""
    _core.move(path, new_path, if_exists, if_not_exists)


def drop_table(table: str | Table, force: bool = False, if_not_exists: Literal['error', 'ignore'] = 'error') -> None:
    """Drop a table, view or snapshot (with `force=True`, also all views of it)."""
    if isinstance(table, Table):
        _core.drop_table(None, str(table._id), force, if_not_exists)
    else:
        _core.drop_table(table, None, force, if_not_exists)


def get_dir_contents(dir_path: str = '', recursive: bool = True) -> DirContents:
    """The directories and tables in a directory."""
    return _core.get_dir_contents(dir_path, recursive)  # type: ignore[return-value]


def get_dir_tree(path: str = '') -> list[TreeNode]:
    """A tree representation of the directory structure."""
    return _core.get_dir_tree(path)


def list_tables(dir_path: str = '', recursive: bool = True) -> list[str]:
    """The paths of the tables in a directory."""
    return _core.list_tables(dir_path, recursive)


def list_dirs(path: str = '', recursive: bool = True) -> list[str]:
    """The paths of the directories in a directory."""
    return _core.list_dirs(path, recursive)


def create_dir(
    path: str, *, if_exists: Literal['error', 'ignore', 'replace', 'replace_force'] = 'error', parents: bool = False
) -> Dir | None:
    """Create a directory (with `parents=True`, also missing parent directories)."""
    return Dir(UUID(_core.create_dir(path, if_exists, parents)))


def drop_dir(path: str, force: bool = False, if_not_exists: Literal['error', 'ignore'] = 'error') -> None:
    """Remove a directory (with `force=True`, also everything in it and every view of a dropped table)."""
    _core.drop_dir(path, force, if_not_exists)


def ls(path: str = '') -> pd.DataFrame:
    """A human-readable listing of a directory."""
    rows = _core.ls_rows(path)
    return pd.DataFrame(rows, columns=['Name', 'Kind', 'Version', 'Base'], index=[''] * len(rows))


def array(elements: Iterable) -> _core.Expr:
    """An array expression (a literal if all elements are constants)."""
    return _core.array(elements)


def list_functions() -> Any:
    """Returns information about all registered functions.

    Returns:
        Pandas DataFrame with columns 'Path', 'Function Name', 'Parameters', 'Return Type'
    """
    from pixeltable import func

    functions = func.FunctionRegistry.get().list_functions()
    paths = ['.'.join(f.self_path.split('.')[:-1]) for f in functions]
    names = [f.name for f in functions]
    params = [
        ', '.join(param_name + ': ' + str(param.col_type) for param_name, param in f.signatures[0].parameters.items())
        for f in functions
    ]
    pd_df = pd.DataFrame(
        {
            'Path': paths,
            'Function Name': names,
            'Parameters': params,
            'Return Type': [str(f.signatures[0].get_return_type()) for f in functions],
        }
    )
    styler = pd_df.style.set_properties(None, **{'text-align': 'left'}).set_table_styles(
        [{'selector': 'th', 'props': [('text-align', 'center')]}]
    )
    return styler.hide(axis='index')


def tools(*args: Any) -> Any:
    """Specifies a collection of UDFs to be used as LLM tools."""
    from pixeltable.func import Tool, Tools

    return Tools(tools=[arg if isinstance(arg, Tool) else tool(arg) for arg in args])


def tool(fn: Any, name: str | None = None, description: str | None = None) -> Any:
    """Specifies a Pixeltable UDF to be used as an LLM tool with customizable metadata."""
    from pixeltable import exceptions as excs
    from pixeltable.func import AggregateFunction, Tool

    if isinstance(fn, AggregateFunction) or getattr(fn, 'is_aggregate', False):
        raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, 'Aggregator UDFs cannot be used as tools')
    return Tool(fn=fn, name=name, description=description)
