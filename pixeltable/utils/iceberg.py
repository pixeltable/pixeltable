from pathlib import Path
from typing import Union

from pyiceberg.catalog.sql import SqlCatalog


def sqlite_catalog(warehouse_path: Union[str, Path], name: str = 'pixeltable') -> SqlCatalog:
    """
    Instantiate a sqlite Iceberg catalog at the specified path. If no catalog exists, one will be created.
    """
    if isinstance(warehouse_path, str):
        warehouse_path = Path(warehouse_path)
    warehouse_path.mkdir(exist_ok=True)
    return SqlCatalog(name, uri=f'sqlite:///{warehouse_path}/catalog.db', warehouse=f'file://{warehouse_path}')
