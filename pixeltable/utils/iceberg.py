
from pathlib import Path
from typing import Union

from pyiceberg.catalog.sql import SqlCatalog


def sqlite_catalog(warehouse_path: Union[str, Path]) -> SqlCatalog:
    if isinstance(warehouse_path, str):
        warehouse_path = Path(warehouse_path)
    warehouse_path.mkdir(exist_ok=True)
    return SqlCatalog('default', uri=f'sqlite:///{warehouse_path}/catalog.db', warehouse=f'file://{warehouse_path}')
