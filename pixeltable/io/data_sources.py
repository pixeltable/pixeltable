from __future__ import annotations

from dataclasses import dataclass

import sqlalchemy as sql


@dataclass
class SqlDataSource:
    """A user-supplied SQL source: a SQLAlchemy `Selectable` and an `Engine` or `Connection` to run it against."""

    selectable: sql.Selectable
    conn: sql.Engine | sql.Connection
