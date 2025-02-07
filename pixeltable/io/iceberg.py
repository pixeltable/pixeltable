from pathlib import Path
from typing import Union
from pyiceberg.catalog import Catalog
from pyiceberg.catalog.sql import SqlCatalog

import pixeltable as pxt
from pixeltable.utils.arrow import to_arrow_schema, to_pa_tables


def export_iceberg(table: pxt.Table, catalog: Catalog) -> None:
    ancestors = [table] + table._bases
    for t in ancestors:
        # Select only those columns that are defined in this table (columns inherited from ancestor
        # tables will be handled separately)
        # TODO: This is selecting only named columns; do we also want to preserve system columns such as errortype?
        col_refs = [t[col] for col in t._tbl_version.cols_by_name]
        df = t.select(*col_refs)
        namespace = _iceberg_namespace(t)
        catalog.create_namespace_if_not_exists(namespace)
        arrow_schema = to_arrow_schema(df._schema, include_rowid=True)
        iceberg_tbl = catalog.create_table(f'{namespace}.{table._name}', schema=arrow_schema)
        for pa_table in to_pa_tables(df, arrow_schema, include_rowid=True):
            iceberg_tbl.append(pa_table)


def sqlite_catalog(iceberg_path: Union[str, Path]) -> SqlCatalog:
    if isinstance(iceberg_path, str):
        iceberg_path = Path(iceberg_path)
    iceberg_path.mkdir(exist_ok=True)
    return SqlCatalog('default', uri=f'sqlite:///{iceberg_path}/catalog.db', warehouse=f'file://{iceberg_path}')


def _iceberg_namespace(table: pxt.Table) -> str:
    """Iceberg tables must have a namespace, so we prepend `pxt` to the table path."""
    if len(table._parent._path) == 0:
        return 'pxt'
    else:
        return f'pxt.{table._parent._path}'
