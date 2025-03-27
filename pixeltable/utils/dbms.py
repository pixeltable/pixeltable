import abc

from sqlalchemy import URL


class Dbms(abc.ABC):
    def __init__(self, name: str, transaction_isolation_level: str, version_index_type: str, db_url: URL) -> None:
        self.name = name
        self.transaction_isolation_level = transaction_isolation_level
        self.version_index_type = version_index_type
        self.db_url = db_url

    @abc.abstractmethod
    def build_drop_db_stmt(self, database: str) -> str: ...

    @abc.abstractmethod
    def build_create_db_stmt(self, database: str) -> str: ...

    @abc.abstractmethod
    def default_system_db_url(self) -> str: ...


class PostgresqlDbms(Dbms):
    def __init__(self, db_url: URL):
        super().__init__('postgresql', 'REPEATABLE READ', 'brin', db_url)

    def build_drop_db_stmt(self, database: str) -> str:
        return f'DROP DATABASE {database}'

    def build_create_db_stmt(self, database: str) -> str:
        return f"CREATE DATABASE {database} ENCODING 'utf-8' LC_COLLATE 'C' LC_CTYPE 'C' TEMPLATE template0"

    def default_system_db_url(self) -> str:
        return self.db_url.set(database='postgres').render_as_string(hide_password=False)


class CockroachDbms(Dbms):
    def __init__(self, db_url: URL):
        super().__init__('cockroachdb', 'SERIALIZABLE', 'btree', db_url)

    def build_drop_db_stmt(self, database: str) -> str:
        return f'DROP DATABASE {database} CASCADE'

    def build_create_db_stmt(self, database: str) -> str:
        return f"CREATE DATABASE {database} TEMPLATE template0 ENCODING 'utf-8' LC_COLLATE 'C' LC_CTYPE 'C'"

    def default_system_db_url(self) -> str:
        return self.db_url.set(database='defaultdb').render_as_string(hide_password=False)
