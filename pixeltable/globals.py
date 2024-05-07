from typing import Any, Optional, Union, Type

import pandas as pd

from pixeltable import catalog, Client
from pixeltable.exprs import Predicate
from pixeltable.iterators import ComponentIterator


def create_table(
        path_str: str, schema: dict[str, Any], *, primary_key: Optional[Union[str, list[str]]] = None,
        num_retained_versions: int = 10, comment: str = ''
) -> catalog.InsertableTable:
    return _client().create_table(
        path_str,
        schema,
        primary_key=primary_key,
        num_retained_versions=num_retained_versions,
        comment=comment
    )


def create_view(
        path_str: str, base: catalog.Table, *, schema: Optional[dict[str, Any]] = None,
        filter: Optional[Predicate] = None,
        is_snapshot: bool = False, iterator_class: Optional[Type[ComponentIterator]] = None,
        iterator_args: Optional[dict[str, Any]] = None, num_retained_versions: int = 10, comment: str = '',
        ignore_errors: bool = False) -> catalog.View:
    return _client().create_view(
        path_str,
        base,
        schema=schema,
        filter=filter,
        is_snapshot=is_snapshot,
        iterator_class=iterator_class,
        iterator_args=iterator_args,
        num_retained_versions=num_retained_versions,
        comment=comment,
        ignore_errors=ignore_errors
    )


def get_table(path: str) -> catalog.Table:
    return _client().get_table(path)


def move(path: str, new_path: str) -> None:
    return _client().move(path, new_path)


def drop_table(path: str, force: bool = False, ignore_errors: bool = False) -> None:
    return _client().drop_table(path, force, ignore_errors)


def list_tables(dir_path: str = '', recursive: bool = True) -> list[str]:
    return _client().list_tables(dir_path, recursive)


def create_dir(path_str: str, ignore_errors: bool = False) -> None:
    return _client().create_dir(path_str, ignore_errors)


def rm_dir(path_str: str) -> None:
    return _client().rm_dir(path_str)


def list_dirs(path_str: str = '', recursive: bool = True) -> list[str]:
    return _client().list_dirs(path_str, recursive)


def list_functions() -> pd.DataFrame:
    return _client().list_functions()


def get_path(schema_obj: catalog.SchemaObject) -> str:
    return _client().get_path(schema_obj)


def import_huggingface_dataset(
    table_path: str,
    dataset: Union['datasets.Dataset', 'datasets.DatasetDict'],
    *,
    column_name_for_split: Optional[str] = 'split',
    schema_override: Optional[dict[str, Any]] = None,
    **kwargs
) -> catalog.InsertableTable:
    return _client().import_huggingface_dataset(
        table_path,
        dataset,
        column_name_for_split=column_name_for_split,
        schema_override=schema_override,
        **kwargs
    )


def import_parquet(
    table_path: str,
    *,
    parquet_path: str,
    schema_override: Optional[dict[str, Any]] = None,
    **kwargs,
) -> catalog.InsertableTable:
    return _client().import_parquet(
        table_path,
        parquet_path=parquet_path,
        schema_override=schema_override,
        **kwargs
    )


def reload() -> None:
    global _default_client
    _default_client = Client(reload=True)


def _client() -> Client:
    global _default_client
    if _default_client is None:
        _default_client = Client()
    return _default_client


_default_client: Optional[Client] = None
