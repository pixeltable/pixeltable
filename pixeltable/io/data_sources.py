from __future__ import annotations

from dataclasses import dataclass

import sqlalchemy as sql
from sqlalchemy.sql.expression import SelectBase


@dataclass
class SqlDataSource:
    """A normalized SQL source for import: a SELECT, its validated output column names, and a live `Connection`.

    `col_names` are the SELECT's output column names, positionally aligned with `select_stmt.selected_columns`
    and already validated against the destination schema.

    `conn` is a handle to an external SQL database that the user wants to import from; it is never a connection
    to Pixeltable's own metadata store.
    """

    select_stmt: SelectBase
    col_names: list[str]
    conn: sql.Connection
