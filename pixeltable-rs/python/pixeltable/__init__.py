"""
Core Pixeltable API for table operations, data processing, and UDF management.

pixeltable-rs: the public API is Python; everything behind it is the Rust extension `pixeltable._core`.
"""

# ruff: noqa: F401

from ._config import URI, ConfigVar
from ._globals import (
    array,
    create_dir,
    create_snapshot,
    create_table,
    create_view,
    drop_dir,
    drop_table,
    get_dir_contents,
    get_dir_tree,
    get_table,
    home,
    init,
    list_dirs,
    list_functions,
    list_tables,
    ls,
    move,
    tool,
    tools,
)
from ._metadata import ColumnMetadata, IndexMetadata, TableMetadata, VersionMetadata
from ._model import BtreeIndex, Column, EmbeddingIndex, model_base
from ._query import Query, ResultCursor, ResultSet
from ._row import CellError, Row, RowBatch
from ._table import InsertableTable, Table, View
from ._update_status import UpdateStatus
from ._version import __version__
from .exceptions import (
    AlreadyExistsError,
    AuthorizationError,
    ConcurrencyError,
    Error,
    ErrorCode,
    ExternalServiceError,
    NotFoundError,
    PixeltableWarning,
    RequestError,
    ServiceUnavailableError,
    UserError,
)
from .func import (
    Aggregator,
    Function,
    PxtIterator,
    Tool,
    ToolChoice,
    Tools,
    expr_udf,
    iterator,
    mcp_udfs,
    query,
    retrieval_udf,
    uda,
    udf,
)
from .type_system import (
    UUID,
    Array,
    Audio,
    Binary,
    Bool,
    Date,
    Document,
    Float,
    Image,
    Int,
    Json,
    Required,
    String,
    Timestamp,
    Video,
)
from .types import DirContents, DirectoryNode, TableKind, TableNode, TreeNode

# This import must go last to avoid circular imports.
from . import functions, io, iterators  # isort: skip

__default_dir = {symbol for symbol in dir() if not symbol.startswith('_')}
__removed_symbols = {'exceptions', 'func', 'type_system', 'types'}
__all__ = sorted(__default_dir - __removed_symbols)


def __dir__() -> list[str]:
    return __all__
