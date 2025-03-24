from pixeltable.utils.dbms_helper import DbmsHelperBase


class PostgresqlHelper(DbmsHelperBase):
    def __init__(self):
        super().__init__('postgresql', 'REPEATABLE READ', 'brin')

    def build_drop_db_stmt(self, database: str) -> str:
        return f'DROP DATABASE {database}'

    def build_create_db_stmt(self, database: str) -> str:
        return f"CREATE DATABASE {database} ENCODING 'utf-8' LC_COLLATE 'C' LC_CTYPE 'C' TEMPLATE template0"
