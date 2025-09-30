import abc

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
    def create_vector_index(
        self, index_name: str, index_value_sa_col: sql.schema.Column, conn: sql.Connection, metric: str
    ) -> None: ...


class PostgresqlDbms(Dbms):
    """
    Implements utilities to interact with Postgres database.
    """

    def __init__(self, db_url: sql.URL):
        super().__init__('postgresql', 'SERIALIZABLE', 'brin', db_url)

    def drop_db_stmt(self, database: str) -> str:
        return f'DROP DATABASE {database}'

    def create_db_stmt(self, database: str) -> str:
        return f"CREATE DATABASE {database} ENCODING 'utf-8' LC_COLLATE 'C' LC_CTYPE 'C' TEMPLATE template0"

    def default_system_db_url(self) -> str:
        a = self.db_url.set(database='postgres').render_as_string(hide_password=False)
        return a

    def create_vector_index(
        self, index_name: str, index_value_sa_col: sql.schema.Column, conn: sql.Connection, metric: str
    ) -> None:
        idx = sql.Index(
            index_name,
            index_value_sa_col,
            postgresql_using='hnsw',
            postgresql_with={'m': 16, 'ef_construction': 64},
            postgresql_ops={index_value_sa_col.name: metric},
        )
        idx.create(bind=conn)


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

    def create_vector_index(
        self, index_name: str, index_value_sa_col: sql.schema.Column, conn: sql.Connection, metric: str
    ) -> None:
        create_index_sql = sql.text(
            f"""CREATE VECTOR INDEX {index_name} ON {index_value_sa_col.table.name}
             ({index_value_sa_col.name} {metric})"""
        )
        conn.execute(create_index_sql)
