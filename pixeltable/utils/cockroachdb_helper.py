from pixeltable.utils.dbms_helper import DbmsHelperBase


class CockroachDbHelper(DbmsHelperBase):
    def __init__(self):
        super().__init__('cockroachdb', 'SERIALIZABLE', 'btree')

    def build_drop_db_stmt(self, database: str) -> str:
        return f'DROP DATABASE {database} CASCADE'

    def build_create_db_stmt(self, database: str) -> str:
        return f"CREATE DATABASE {database} TEMPLATE template0 ENCODING 'utf-8' LC_COLLATE 'C' LC_CTYPE 'C'"
