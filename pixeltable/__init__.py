from .__version__ import __version__, __version_tuple__
from .catalog import Column, InsertableTable, Table, UpdateStatus, View
from .dataframe import DataFrame
from .exceptions import Error
from .exprs import RELATIVE_PATH_ROOT
from .func import Aggregator, Function, expr_udf, query, uda, udf
from .globals import (
    array,
    configure_logging,
    create_dir,
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
    move,
    tool,
    tools,
)
from .type_system import (
    Array,
    ArrayType,
    Audio,
    AudioType,
    Bool,
    BoolType,
    ColumnType,
    Document,
    DocumentType,
    Float,
    FloatType,
    Image,
    ImageType,
    Int,
    IntType,
    Json,
    JsonType,
    Required,
    String,
    StringType,
    Timestamp,
    TimestampType,
    Video,
    VideoType,
)

# This import must go last to avoid circular imports.
from . import ext, functions, io, iterators  # isort: skip

# This is the safest / most maintainable way to construct __all__: start with the default and "blacklist"
# stuff that we don't want in there. (Using a "whitelist" is considerably harder to maintain.)

__default_dir = set(symbol for symbol in dir() if not symbol.startswith('_'))
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
__all__ = sorted(list(__default_dir - __removed_symbols))


def __dir__():
    return __all__
