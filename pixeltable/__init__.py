"""
Core Pixeltable API for table operations, data processing, and UDF management.
"""

# ruff: noqa: F401

from .__version__ import __version__, __version_tuple__
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

# Register classes as public API after all imports complete (to avoid circular imports)
from .func import public_api
from .iterators import ComponentIterator, FrameIterator
from .iterators.image import TileIterator
public_api(ComponentIterator)
public_api(FrameIterator)
public_api(TileIterator)

# Register type system classes (use already-imported module)
import pixeltable.type_system as ts  # noqa: E402
public_api(ts.Json)
public_api(ts.Video)
public_api(ts.Audio)

# Register catalog classes (use already-imported classes)
from .catalog.update_status import RowCountStats  # noqa: E402
public_api(UpdateStatus)  # Already imported above
public_api(RowCountStats)
public_api(TableMetadata)  # Already imported above
public_api(Table)  # Already imported above

# Register dataframe classes
import pixeltable.dataframe as df_module  # noqa: E402
public_api(df_module.DataFrameResultSet)

# Register expression classes
import pixeltable.exprs.expr as expr_module  # noqa: E402
public_api(expr_module.Expr)

# Register tool classes (use already-imported classes)
public_api(Tool)  # Already imported above
public_api(Tools)  # Already imported above

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
