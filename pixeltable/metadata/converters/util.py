import copy
import logging
from typing import Any, Callable, Optional

import sqlalchemy as sql

from pixeltable.metadata.schema import Table, TableSchemaVersion

__logger = logging.getLogger('pixeltable')


def convert_table_md(
    engine: sql.engine.Engine,
    table_md_updater: Optional[Callable[[dict], None]] = None,
    column_md_updater: Optional[Callable[[dict], None]] = None,
    external_store_md_updater: Optional[Callable[[dict], None]] = None,
    substitution_fn: Optional[Callable[[Optional[str], Any], Optional[tuple[Optional[str], Any]]]] = None
) -> None:
    """
    Converts schema.TableMd dicts based on the specified conversion functions.

    Args:
        engine: The SQLAlchemy engine.
        table_md_updater: A function that updates schema.TableMd dicts in place.
        column_md_updater: A function that updates schema.ColumnMd dicts in place.
        external_store_md_updater: A function that updates the external store metadata in place.
        substitution_fn: A function that substitutes metadata values. If specified, all metadata will be traversed
            recursively, and `substitution_fn` will be called once for each metadata entry. If the entry appears in
            a dict as a `(k, v)` pair, then `substitution_fn(k, v)` will be called. If the entry appears in a list,
            then `substitution_fn(None, v)` will be called. If `substitution_fn` returns a tuple `(k', v')`, then
            the original entry will be replaced, and the traversal will continue with `v'`.
    """
    with engine.begin() as conn:
        for row in conn.execute(sql.select(Table)):
            id = row[0]
            table_md = row[2]
            assert isinstance(table_md, dict)
            updated_table_md = copy.deepcopy(table_md)
            if table_md_updater is not None:
                table_md_updater(updated_table_md)
            if column_md_updater is not None:
                __update_column_md(updated_table_md, column_md_updater)
            if external_store_md_updater is not None:
                __update_external_store_md(updated_table_md, external_store_md_updater)
            if substitution_fn is not None:
                updated_table_md = __substitute_md_rec(updated_table_md, substitution_fn)
            if updated_table_md != table_md:
                __logger.info(f'Updating schema for table: {id}')
                conn.execute(sql.update(Table).where(Table.id == id).values(md=updated_table_md))


def __update_column_md(table_md: dict, column_md_updater: Callable[[dict], None]) -> None:
    columns_md = table_md['column_md']
    assert isinstance(columns_md, dict)
    for column_md in columns_md.values():
        column_md_updater(column_md)


def __update_external_store_md(table_md: dict, external_store_md_updater: Callable[[dict], None]) -> None:
    stores_md = table_md['external_stores']
    assert isinstance(stores_md, list)
    for store_md in stores_md:
        external_store_md_updater(store_md)


def __substitute_md_rec(
    md: Any,
    substitution_fn: Callable[[Optional[str], Any], Optional[tuple[Optional[str], Any]]]
) -> Any:
    if isinstance(md, dict):
        updated_dict: dict[str, Any] = {}
        for k, v in md.items():
            assert isinstance(k, str)
            substitute = substitution_fn(k, v)
            if substitute is not None:
                updated_k, updated_v = substitute
                updated_dict[updated_k] = __substitute_md_rec(updated_v, substitution_fn)
            else:
                updated_dict[k] = __substitute_md_rec(v, substitution_fn)
        return updated_dict
    elif isinstance(md, list):
        updated_list: list[Any] = []
        for v in md:
            substitute = substitution_fn(None, v)
            if substitute is not None:
                _, updated_v = substitute
                updated_list.append(__substitute_md_rec(updated_v, substitution_fn))
            else:
                updated_list.append(__substitute_md_rec(v, substitution_fn))
        return updated_list
    else:
        return md


def convert_table_schema_version_md(
    engine: sql.engine.Engine,
    table_schema_version_md_updater: Optional[Callable[[dict], None]] = None,
    schema_column_updater: Optional[Callable[[dict], None]] = None
) -> None:
    """
    Converts schema.TableSchemaVersionMd dicts based on the specified conversion functions.

    Args:
        engine: The SQLAlchemy engine.
        table_schema_version_md_updater: A function that updates schema.TableSchemaVersionMd dicts in place.
        schema_column_updater: A function that updates schema.SchemaColumn dicts in place.
    """
    with engine.begin() as conn:
        stmt = sql.select(TableSchemaVersion.tbl_id, TableSchemaVersion.schema_version, TableSchemaVersion.md)
        for row in conn.execute(stmt):
            tbl_id, schema_version, md = row[0], row[1], row[2]
            assert isinstance(md, dict)
            updated_md = copy.deepcopy(md)
            if table_schema_version_md_updater is not None:
                table_schema_version_md_updater(updated_md)
            if schema_column_updater is not None:
                __update_schema_column(updated_md, schema_column_updater)
            if updated_md != md:
                __logger.info(f'Updating TableSchemaVersion(tbl_id={tbl_id}, schema_version={schema_version})')
                update_stmt = (
                    sql.update(TableSchemaVersion)
                    .where(TableSchemaVersion.tbl_id == tbl_id)
                    .where(TableSchemaVersion.schema_version == schema_version)
                    .values(md=updated_md)
                )
                conn.execute(update_stmt)


def __update_schema_column(table_schema_version_md: dict, schema_column_updater: Callable[[dict], None]) -> None:
    cols = table_schema_version_md['columns']
    assert isinstance(cols, dict)
    for schema_col in cols.values():
        schema_column_updater(schema_col)
