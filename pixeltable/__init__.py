"""
Core Pixeltable API for table operations, data processing, and UDF management.
"""

# ruff: noqa: F401

from ._version import __version__
from .catalog import (
    Column,
    ColumnMetadata,
    IndexMetadata,
    InsertableTable,
    Table,
    TableMetadata,
    UpdateStatus,
    VersionMetadata,
    View,
)
from .dataframe import DataFrame
from .exceptions import Error, ExprEvalError, PixeltableWarning
from .func import Aggregator, Function, Tool, ToolChoice, Tools, expr_udf, mcp_udfs, query, retrieval_udf, uda, udf
from .globals import (
    DirContents,
    array,
    configure_logging,
    create_dir,
    create_snapshot,
    create_table,
    create_view,
    drop_dir,
    drop_table,
    get_dir_contents,
    get_table,
    init,
    list_dirs,
    list_functions,
    list_tables,
    ls,
    move,
    publish,
    replicate,
    tool,
    tools,
)
from .type_system import Array, Audio, Bool, Date, Document, Float, Image, Int, Json, Required, String, Timestamp, Video

# This import must go last to avoid circular imports.
from . import functions, io, iterators  # isort: skip

# This is the safest / most maintainable way to construct __all__: start with the default and "blacklist"
# stuff that we don't want in there. (Using a "whitelist" is considerably harder to maintain.)

__default_dir = {symbol for symbol in dir() if not symbol.startswith('_')}
__removed_symbols = {
    'catalog',
    'dataframe',
    'env',
    'exceptions',
    'exec',
    'exprs',
    'func',
    'globals',
    'index',
    'metadata',
    'plan',
    'type_system',
    'utils',
}
__all__ = sorted(__default_dir - __removed_symbols)


def __dir__() -> list[str]:
    return __all__
