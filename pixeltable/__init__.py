# ruff: noqa: F401

from .__version__ import __version__, __version_tuple__
from .catalog import Column, InsertableTable, Table, UpdateStatus, View
from .dataframe import DataFrame
from .exceptions import Error, ExprEvalError, PixeltableWarning
from .func import Aggregator, Function, expr_udf, mcp_udfs, query, retrieval_udf, uda, udf
from .globals import (
    array,
    configure_logging,
    create_dir,
    create_replica,
    create_snapshot,
    create_table,
    create_view,
    drop_dir,
    drop_table,
    get_table,
    init,
    list_dirs,
    list_functions,
    list_tables,
    ls,
    move,
    tool,
    tools,
)
from .type_system import Array, Audio, Bool, Date, Document, Float, Image, Int, Json, Required, String, Timestamp, Video

# This import must go last to avoid circular imports.
from . import ext, functions, io, iterators  # isort: skip

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
