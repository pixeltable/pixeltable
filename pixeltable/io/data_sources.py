from __future__ import annotations

from dataclasses import dataclass

import sqlalchemy as sql
from sqlalchemy.sql.expression import SelectBase


@dataclass
class SqlDataSource:
    """A user-supplied SQL source: a normalized SELECT statement and an `Engine` or `Connection` to run it against.

    `conn` is always a handle to an external SQL database that the user wants to import from; it is never a
    connection to Pixeltable's own metadata store.
    """

    select_stmt: SelectBase
    conn: sql.Engine | sql.Connection
