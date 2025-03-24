import abc


class Dbms(abc.ABC):
    def __init__(self, name: str, transaction_isolation_level: str, version_index_type: str) -> None:
        self.name = name
        self.transaction_isolation_level = transaction_isolation_level
        self.version_index_type = version_index_type

    @abc.abstractmethod
    def build_drop_db_stmt(self, database: str) -> str: ...

    @abc.abstractmethod
    def build_create_db_stmt(self, database: str) -> str: ...


class PostgresqlDbms(Dbms):
    def __init__(self):
        super().__init__('postgresql', 'REPEATABLE READ', 'brin')

    def build_drop_db_stmt(self, database: str) -> str:
        return f'DROP DATABASE {database}'

    def build_create_db_stmt(self, database: str) -> str:
        return f"CREATE DATABASE {database} ENCODING 'utf-8' LC_COLLATE 'C' LC_CTYPE 'C' TEMPLATE template0"


class CockroachDbms(Dbms):
    def __init__(self):
        super().__init__('cockroachdb', 'SERIALIZABLE', 'btree')

    def build_drop_db_stmt(self, database: str) -> str:
        return f'DROP DATABASE {database} CASCADE'

    def build_create_db_stmt(self, database: str) -> str:
        return f"CREATE DATABASE {database} TEMPLATE template0 ENCODING 'utf-8' LC_COLLATE 'C' LC_CTYPE 'C'"
