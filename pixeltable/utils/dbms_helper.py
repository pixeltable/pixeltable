import abc


class DbmsHelperBase(abc.ABC):
    def __init__(self, name: str, transaction_isolation_level: str, vmin_vmax_index_type: str) -> None:
        self.name = name
        self.transaction_isolation_level = transaction_isolation_level
        self.vmin_vmax_index_type = vmin_vmax_index_type

    @abc.abstractmethod
    def build_drop_db_stmt(self, database: str) -> str:
        pass

    @abc.abstractmethod
    def build_create_db_stmt(self, database: str) -> str:
        pass
