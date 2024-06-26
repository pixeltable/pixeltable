from .catalog import Column, Table, InsertableTable, View
from .dataframe import DataFrame
from .exceptions import Error
from .exprs import RELATIVE_PATH_ROOT
from .func import Function, udf, Aggregator, uda, expr_udf
from .globals import init, create_table, create_view, get_table, move, drop_table, list_tables, create_dir, rm_dir, \
    list_dirs, list_functions, get_path, configure_logging
from .type_system import (
    ColumnType,
    StringType,
    IntType,
    FloatType,
    BoolType,
    TimestampType,
    JsonType,
    ArrayType,
    ImageType,
    VideoType,
    AudioType,
    DocumentType,
)
from .utils.help import help

from . import ext, functions, io, iterators
from .__version__ import __version__, __version_tuple__

# This is the safest / most maintainable way to do this: start with the default and "blacklist" stuff that
# we don't want in there. (Using a "whitelist" is considerably harder to maintain.)

__default_dir = set(symbol for symbol in dir() if not symbol.startswith('_'))
__removed_symbols = {'catalog', 'dataframe', 'env', 'exceptions', 'exec', 'exprs', 'func', 'globals', 'index',
                     'metadata', 'plan', 'type_system', 'utils'}
__all__ = sorted(list(__default_dir - __removed_symbols))


def __dir__():
    return __all__
