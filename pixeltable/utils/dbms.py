import abc
from typing import Any, ClassVar

import sqlalchemy as sql


class Dbms(abc.ABC):
    """
    Provides abstractions for utilities to interact with a database system.
    """

    name: str
    transaction_isolation_level: str
    version_index_type: str
    db_url: sql.URL

    def __init__(self, name: str, transaction_isolation_level: str, version_index_type: str, db_url: sql.URL) -> None:
        self.name = name
        self.transaction_isolation_level = transaction_isolation_level
        self.version_index_type = version_index_type
        self.db_url = db_url

    @abc.abstractmethod
    def drop_db_stmt(self, database: str) -> str: ...

    @abc.abstractmethod
    def create_db_stmt(self, database: str) -> str: ...

    @abc.abstractmethod
    def default_system_db_url(self) -> str: ...

    @abc.abstractmethod
    def create_vector_index_stmt(
        self,
        store_index_name: str,
        sa_value_col: sql.Column,
        metric: str,
        index_type: str = 'hnsw',
        index_params: dict[str, Any] | None = None,
    ) -> sql.Compiled: ...


class PostgresqlDbms(Dbms):
    """
    Implements utilities to interact with Postgres database.
    """

    # Default parameters for HNSW index (pgvector)
    HNSW_DEFAULTS: ClassVar[dict[str, Any]] = {'m': 16, 'ef_construction': 64}

    # Default parameters for DiskANN index (pgvectorscale)
    DISKANN_DEFAULTS: ClassVar[dict[str, Any]] = {'num_neighbors': 50, 'search_list_size': 100}

    def __init__(self, db_url: sql.URL):
        super().__init__('postgresql', 'SERIALIZABLE', 'brin', db_url)

    def drop_db_stmt(self, database: str) -> str:
        return f'DROP DATABASE {database}'

    def create_db_stmt(self, database: str) -> str:
        return f"CREATE DATABASE {database} ENCODING 'utf-8' LC_COLLATE 'C' LC_CTYPE 'C' TEMPLATE template0"

    def default_system_db_url(self) -> str:
        a = self.db_url.set(database='postgres').render_as_string(hide_password=False)
        return a

    def create_vector_index_stmt(
        self,
        store_index_name: str,
        sa_value_col: sql.Column,
        metric: str,
        index_type: str = 'hnsw',
        index_params: dict[str, Any] | None = None,
    ) -> sql.Compiled:
        """
        Create a vector index statement.

        Args:
            store_index_name: Name of the index
            sa_value_col: SQLAlchemy column to index
            metric: Distance metric ops class (e.g., 'vector_cosine_ops')
            index_type: Type of index - 'hnsw' (pgvector) or 'diskann' (pgvectorscale)
            index_params: Optional parameters to override defaults for the index type
        """
        from sqlalchemy.dialects import postgresql

        if index_type == 'diskann':
            # StreamingDiskANN index from pgvectorscale
            params = {**self.DISKANN_DEFAULTS, **(index_params or {})}
            sa_idx = sql.Index(
                store_index_name,
                sa_value_col,
                postgresql_using='diskann',
                postgresql_with=params,
                postgresql_ops={sa_value_col.name: metric},
            )
        else:
            # Default to HNSW index from pgvector
            params = {**self.HNSW_DEFAULTS, **(index_params or {})}
        sa_idx = sql.Index(
            store_index_name,
            sa_value_col,
            postgresql_using='hnsw',
                postgresql_with=params,
            postgresql_ops={sa_value_col.name: metric},
        )
        return sql.schema.CreateIndex(sa_idx, if_not_exists=True).compile(dialect=postgresql.dialect())


class CockroachDbms(Dbms):
    """
    Implements utilities to interact with CockroachDb database.
    """

    def __init__(self, db_url: sql.URL):
        super().__init__('cockroachdb', 'SERIALIZABLE', 'btree', db_url)

    def drop_db_stmt(self, database: str) -> str:
        return f'DROP DATABASE {database} CASCADE'

    def create_db_stmt(self, database: str) -> str:
        return f"CREATE DATABASE {database} TEMPLATE template0 ENCODING 'utf-8' LC_COLLATE 'C' LC_CTYPE 'C'"

    def default_system_db_url(self) -> str:
        return self.db_url.set(database='defaultdb').render_as_string(hide_password=False)

    def sa_vector_index(self, store_index_name: str, sa_value_col: sql.schema.Column, metric: str) -> sql.Index | None:
        return None

    def create_vector_index_stmt(
        self,
        store_index_name: str,
        sa_value_col: sql.Column,
        metric: str,
        index_type: str = 'hnsw',
        index_params: dict[str, Any] | None = None,
    ) -> sql.Compiled:
        # CockroachDB has its own vector index implementation
        return sql.text(
            f'CREATE VECTOR INDEX IF NOT EXISTS {store_index_name} ON {sa_value_col.table.name}'
            f'({sa_value_col.name} {metric})'
        ).compile()
