from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.table import StaticTable

import pixeltable as pxt
from pixeltable.utils.arrow import to_arrow_schema, to_pa_tables


def export_static_iceberg(table: pxt.Table, iceberg_path: str) -> None:
    """
    Exports a dataframe's data to one or more Parquet files. Requires pyarrow to be installed.

    It additionally writes the pixeltable metadata in a json file, which would otherwise
    not be available in the parquet format.

    Args:
        table_or_df : Table or Dataframe to export.
        parquet_path : Path to directory to write the parquet files to.
        partition_size_bytes : The maximum target size for each chunk. Default 100_000_000 bytes.
        inline_images : If True, images are stored inline in the parquet file. This is useful
                        for small images, to be imported as pytorch dataset. But can be inefficient
                        for large images, and cannot be imported into pixeltable.
                        If False, will raise an error if the Dataframe has any image column.
                        Default False.
    """
    catalog = SqlCatalog('default', uri=f'sqlite:///{iceberg_path}/catalog.db', warehouse=f'file://{iceberg_path}')
    catalog.create_namespace('default')
    iceberg_tbl = catalog.create_table('default.test', schema=to_arrow_schema(table._schema))
    for pa_table in to_pa_tables(table.select()):
        iceberg_tbl.append(pa_table)

    StaticTable.from_metadata
